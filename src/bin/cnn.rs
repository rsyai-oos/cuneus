use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CNNParams {
    canvas_size: f32,       
    brush_size: f32,        
    input_resolution: f32,  
    clear_canvas: i32,      
    show_debug: i32,        
    prediction_threshold: f32, 
    canvas_offset_x: f32,   
    canvas_offset_y: f32,
    feature_maps_1: f32,    
    feature_maps_2: f32,    
    num_classes: f32,       
    normalization_mean: f32,
    normalization_std: f32,
    show_frequencies: i32,
    conv1_pool_size: f32,
    conv2_pool_size: f32,
    mouse_x: f32,
    mouse_y: f32,
    mouse_click_x: f32,
    mouse_click_y: f32,
    mouse_buttons: u32,
    _padding: [f32; 3],
}

impl UniformProvider for CNNParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct CNNDigitRecognizer {
    base: RenderKit,
    params_uniform: UniformBinding<CNNParams>,
    compute_shader: ComputeShader,
    frame_count: u32,
}

impl CNNDigitRecognizer {}

impl ShaderManager for CNNDigitRecognizer {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, }, count: None, },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None, },
            ],
            label: Some("texture_bind_group_layout"),
        });
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "CNN Params",
            CNNParams {
                canvas_size: 0.6,
                brush_size: 0.007,
                input_resolution: 28.0,
                clear_canvas: 0,
                show_debug: 0,
                prediction_threshold: 0.1,
                canvas_offset_x: 0.1,
                canvas_offset_y: 0.1,
                feature_maps_1: 8.0,
                feature_maps_2: 5.0,
                num_classes: 10.0,
                normalization_mean: 0.1307,
                normalization_std: 0.3081,
                show_frequencies: 0,
                conv1_pool_size: 12.0,
                conv2_pool_size: 4.0,
                mouse_x: 0.0,
                mouse_y: 0.0,
                mouse_click_x: 0.0,
                mouse_click_y: 0.0,
                mouse_buttons: 0,
                _padding: [0.0; 3],
            },
            &create_bind_group_layout(&core.device, BindGroupLayoutType::CustomUniform, "CNN Params"),
            0,
        );
        
        let base = RenderKit::new(core, include_str!("../../shaders/vertex.wgsl"), include_str!("../../shaders/blit.wgsl"), &[&texture_bind_group_layout], None);
        
        // CNN needs 4 custom storage buffers: canvas, conv1, conv2, fc
        let compute_config = ComputeShaderConfig {
            workgroup_size: [16, 16, 1],
            workgroup_count: None,
            dispatch_once: false,
            storage_texture_format: wgpu::TextureFormat::Rgba16Float,
            enable_atomic_buffer: false,
            atomic_buffer_multiples: 0,
            entry_points: vec![
                "canvas_update".to_string(),
                "conv_layer1".to_string(),
                "conv_layer2".to_string(),
                "fully_connected".to_string(),
                "main_image".to_string(),
            ],
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "CNN".to_string(),
            mouse_bind_group_layout: None,
            enable_fonts: true, // Enable fonts for CNN digit display
            enable_audio_buffer: false,
            audio_buffer_size: 0,
            enable_custom_uniform: true,
            enable_input_texture: false, // CNN doesn't actually use input texture
            custom_storage_buffers: vec![
                CustomStorageBuffer {
                    label: "Canvas Buffer".to_string(),
                    size: (28 * 28 * 4) as u64, // Canvas data for 28x28 pixels
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
                CustomStorageBuffer {
                    label: "Conv1 Buffer".to_string(),
                    size: (12 * 12 * 8 * 4) as u64, // First convolution layer: 12x12x8 feature maps
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
                CustomStorageBuffer {
                    label: "Conv2 Buffer".to_string(),
                    size: (4 * 4 * 5 * 4) as u64, // Second convolution layer: 4x4x5 feature maps
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
                CustomStorageBuffer {
                    label: "FC Buffer".to_string(),
                    size: (10 * 4) as u64, // Fully connected layer: 10 classes
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
            ],
        };
        
        let mut compute_shader = ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/cnn.wgsl"),
            compute_config,
        );

        // Enable hot reload
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CNN Compute Shader Hot Reload"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/cnn.wgsl").into()),
        });
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("shaders/cnn.wgsl"),
            shader_module,
        ) {
            eprintln!("Failed to enable compute shader hot reload: {}", e);
        }
        
        compute_shader.add_custom_uniform_binding(&params_uniform.bind_group);
        
        Self {
            base,
            params_uniform,
            compute_shader,
            frame_count: 0,
        }
    }
    
    fn update(&mut self, _core: &Core) {
        self.base.fps_tracker.update();
    }
    
    fn resize(&mut self, core: &Core) {
        self.compute_shader.resize(core, core.size.width, core.size.height);
    }
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Handle UI and controls like in jfa.rs
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });

                egui::Window::new("CNN Digit Recognizer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        ui.label("Draw a digit in the canvas area and watch the CNN predict it!");
                        ui.separator();
                        ui.label("The CNN will predict the digit using pre-trained weights");
                        ui.separator();
                        
                        egui::CollapsingHeader::new("Canvas Settings")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.canvas_size, 0.3..=0.8).text("Canvas Size")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.brush_size, 0.001..=0.015).text("Brush Size")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.canvas_offset_x, 0.0..=0.5).text("Canvas X")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.canvas_offset_y, 0.0..=0.5).text("Canvas Y")).changed();
                                
                                if ui.button("Clear Canvas").clicked() {
                                    params.clear_canvas = 1;
                                    changed = true;
                                } else {
                                    params.clear_canvas = 0;
                                }
                            });
                        
                        egui::CollapsingHeader::new("CNN Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.prediction_threshold, 0.0..=0.5).text("Prediction Threshold")).changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        
                        ui.separator();
                        ui.label(format!("Frame: {}", self.frame_count));
                        ui.label("CNN Digit Recognition");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        // Create separate encoder for compute passes to avoid texture usage conflicts
        let mut compute_encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CNN Compute Encoder"),
        });

        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
        
        // Execute CNN pipeline stages using dispatch_stage
        self.compute_shader.dispatch_stage(&mut compute_encoder, 0, (28, 28, 1), Some(&self.params_uniform.bind_group)); // canvas_update
        self.compute_shader.dispatch_stage(&mut compute_encoder, 1, (12, 12, 8), Some(&self.params_uniform.bind_group)); // conv_layer1
        self.compute_shader.dispatch_stage(&mut compute_encoder, 2, (4, 4, 5), Some(&self.params_uniform.bind_group)); // conv_layer2
        self.compute_shader.dispatch_stage(&mut compute_encoder, 3, (10, 1, 1), Some(&self.params_uniform.bind_group)); // fully_connected
        self.compute_shader.dispatch_stage(&mut compute_encoder, 4, (core.size.width.div_ceil(16), core.size.height.div_ceil(16), 1), Some(&self.params_uniform.bind_group)); // main_image
        
        // Handle control requests like in jfa.rs
        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_control_request(controls_request);

        // Submit compute commands first
        core.queue.submit(Some(compute_encoder.finish()));
        
        // Create separate encoder for render pass
        let mut render_encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CNN Render Encoder"),
        });
        
        {
            let mut render_pass = Renderer::begin_render_pass(&mut render_encoder, &view, wgpu::LoadOp::Clear(wgpu::Color::BLACK), Some("Display Pass"));
            
            let compute_texture = self.compute_shader.get_output_texture();
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.draw(0..4, 0..1);
        }
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = self.frame_count;
        self.base.time_uniform.update(&core.queue);
        
        self.compute_shader.set_time(current_time, 1.0/60.0, &core.queue);
        self.compute_shader.time_uniform.data.frame = self.frame_count;
        self.compute_shader.time_uniform.update(&core.queue);
        
        // Update mouse parameters for drawing
        params.mouse_x = self.base.mouse_tracker.uniform.position[0];
        params.mouse_y = self.base.mouse_tracker.uniform.position[1];
        params.mouse_click_x = self.base.mouse_tracker.uniform.click_position[0];
        params.mouse_click_y = self.base.mouse_tracker.uniform.click_position[1];
        params.mouse_buttons = self.base.mouse_tracker.uniform.buttons[0];
        changed = true;

        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        self.frame_count += 1;
        
        self.base.handle_render_output(core, &view, full_output, &mut render_encoder);
        core.queue.submit(Some(render_encoder.finish()));
        output.present();
        
        Ok(())
    }
    
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
            return true;
        }
        
        // Handle mouse input for drawing on canvas
        if self.base.handle_mouse_input(core, event, false) {
            return true;
        }
        
        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (app, event_loop) = cuneus::ShaderApp::new("CNN Digit Recognizer", 800, 600);
    
    app.run(event_loop, |core| {
        CNNDigitRecognizer::init(core)
    })
}