use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SpiralParams {
    a: f32,
    b: f32,
    c: f32,
    dof_amount: f32,
    dof_focal_dist: f32,
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    brightness: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
    _padding: u32,
}

impl UniformProvider for SpiralParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct SpiralShader {
    base: RenderKit,
    params_uniform: UniformBinding<SpiralParams>,
    compute_shader: ComputeShader,
    frame_count: u32,
    export_time: Option<f32>,
    export_frame: Option<u32>,
}

impl SpiralShader {
    fn capture_frame(&mut self, core: &Core, time: f32) -> Result<Vec<u8>, wgpu::SurfaceError> {
        let settings = self.base.export_manager.settings();
        let (capture_texture, output_buffer) = self.base.create_capture_texture(
            &core.device,
            settings.width,
            settings.height
        );
        let align = 256;
        let unpadded_bytes_per_row = settings.width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;
        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Capture Encoder"),
        });
        self.base.time_uniform.data.time = time;
        self.base.time_uniform.update(&core.queue);
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &capture_view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Capture Pass"),
            );
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.compute_shader.output_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &capture_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(settings.height),
                },
            },
            wgpu::Extent3d {
                width: settings.width,
                height: settings.height,
                depth_or_array_layers: 1,
            },
        );
        
        core.queue.submit(Some(encoder.finish()));
        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = core.device.poll(wgpu::PollType::Wait).unwrap();
        rx.recv().unwrap().unwrap();
        let padded_data = buffer_slice.get_mapped_range().to_vec();
        let mut unpadded_data = Vec::with_capacity((settings.width * settings.height * 4) as usize);
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        Ok(unpadded_data)
    }
    fn handle_export(&mut self, core: &Core) {
        if let Some((frame, time)) = self.base.export_manager.try_get_next_frame() {
            self.export_time = Some(time);
            self.export_frame = Some(frame);
            
            if let Ok(data) = self.capture_frame(core, time) {
                let settings = self.base.export_manager.settings();
                if let Err(e) = cuneus::save_frame(data, frame, settings) {
                    eprintln!("Error saving frame: {:?}", e);
                }
            }
        } else {
            self.export_time = None;
            self.export_frame = None;
            self.base.export_manager.complete_export();
        }
    }
}

impl ShaderManager for SpiralShader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
            label: Some("texture_bind_group_layout"),
        });
        
        let mut resource_layout = cuneus::compute::ResourceLayout::new();
        resource_layout.add_custom_uniform("spiral_params", std::mem::size_of::<SpiralParams>() as u64);
        let bind_group_layouts = resource_layout.create_bind_group_layouts(&core.device);
        let spiral_params_layout = bind_group_layouts.get(&2).unwrap();

        let params_uniform = UniformBinding::new(
            &core.device,
            "Spiral Params",
            SpiralParams {
                a: 1.0,              
                b: 1.0,            
                c: 1.0,              
                dof_amount: 1.0,    
                dof_focal_dist: 1.0, 
                rotation_x: 0.0,     
                rotation_y: 0.0,     
                click_state: 0,     
                brightness: 0.00004,
                color1_r: 0.0,       
                color1_g: 0.7,       
                color1_b: 1.0,       
                color2_r: 1.0,       
                color2_g: 0.3,       
                color2_b: 0.5,       
                _padding: 0,
            },
            spiral_params_layout,
            0,
        );
        
        let base = RenderKit::new(core, include_str!("../../shaders/vertex.wgsl"), include_str!("../../shaders/blit.wgsl"), &[&texture_bind_group_layout], None);

        let compute_config = ComputeShaderConfig {
            workgroup_size: [16, 16, 1],
            workgroup_count: None,
            dispatch_once: false,
            storage_texture_format: COMPUTE_TEXTURE_FORMAT_RGBA16,
            enable_atomic_buffer: true,
            atomic_buffer_multiples: 2,
            entry_points: vec!["Splat".to_string(), "main_image".to_string()],
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "Spiral Chaos".to_string(),
            mouse_bind_group_layout: None,
            enable_fonts: false,
            enable_audio_buffer: false,
            audio_buffer_size: 0,
            enable_custom_uniform: true,
            enable_input_texture: false,
            custom_storage_buffers: Vec::new(),
        };

        let mut compute_shader = ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/spiralchaos.wgsl"),
            compute_config,
        );

        // Enable hot reload
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spiral Chaos Compute Shader Hot Reload"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/spiralchaos.wgsl").into()),
        });
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("shaders/spiralchaos.wgsl"),
            shader_module,
        ) {
            eprintln!("Failed to enable compute shader hot reload: {}", e);
        }

        // Add custom parameters uniform to the compute shader
        compute_shader.add_custom_uniform_binding(&params_uniform.bind_group);

        Self {
            base,
            params_uniform,
            compute_shader,
            frame_count: 0,
            export_time: None,
            export_frame: None,
        }
    }
    
    fn update(&mut self, core: &Core) {
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
        self.base.fps_tracker.update();
    }
    
    fn resize(&mut self, core: &Core) {
        println!("Resizing to {:?}", core.size);
        self.base.update_resolution(&core.queue, core.size);
        self.compute_shader.resize(core, core.size.width, core.size.height);
    }
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Handle UI interactions
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        
        // Render UI
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });
                
                egui::Window::new("Chaos Spiral")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Spiral")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.a, 0.0..=3.0).text("Tightness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.b, 0.0..=3.0).text("Speed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.c, 0.0..=3.0).text("N Arms")).changed();
                            });
                        
                        egui::CollapsingHeader::new("DOF")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.dof_amount, 0.0..=3.0).text("N")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dof_focal_dist, 0.0..=3.0).text("Focal Distance")).changed();
                                params.click_state = 1;
                            });
                        
                        egui::CollapsingHeader::new("rot")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.rotation_x, -1.0..=1.0).text("X")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.rotation_y, -1.0..=1.0).text("Y")).changed();
                            });
                            
                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.brightness, 0.00001..=0.0001).logarithmic(true).text("Brightness")).changed();
                                
                                ui.horizontal(|ui| {
                                    ui.label("c1:");
                                    let mut color = [params.color1_r, params.color1_g, params.color1_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.color1_r = color[0];
                                        params.color1_g = color[1];
                                        params.color1_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("c2:");
                                    let mut color = [params.color2_r, params.color2_g, params.color2_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.color2_r = color[0];
                                        params.color2_g = color[1];
                                        params.color2_b = color[2];
                                        changed = true;
                                    }
                                });
                            });
                        
                        ui.separator();
                        
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        ui.separator();
                        
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
            self.compute_shader.clear_atomic_buffer(core);
        }
        self.base.apply_control_request(controls_request);
        
        let (current_time, current_frame) = if let (Some(export_time), Some(export_frame)) = (self.export_time, self.export_frame) {
            (export_time, export_frame)
        } else {
            let current_time = self.base.controls.get_time(&self.base.start_time);
            (current_time, self.frame_count)
        };
        
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = current_frame;
        self.base.time_uniform.update(&core.queue);
        
        // Update compute shader with the same time data
        self.compute_shader.set_time(current_time, 1.0/60.0, &core.queue);
        self.compute_shader.time_uniform.data.frame = current_frame;
        self.compute_shader.time_uniform.update(&core.queue);
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        // Check for hot reload
        self.compute_shader.check_hot_reload(&core.device);
        
        // Pass 1: Generate and splat particles (Splat entry point)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Spiral Splat Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_shader.pipelines[0]); // First pipeline is Splat
            compute_pass.set_bind_group(0, &self.compute_shader.time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.compute_shader.storage_bind_group, &[]);
            compute_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            if let Some(atomic_buffer) = &self.compute_shader.atomic_buffer {
                compute_pass.set_bind_group(3, &atomic_buffer.bind_group, &[]);
            }
            compute_pass.dispatch_workgroups(4096, 1, 1);
        }
        
        // Pass 2: Render to screen (main_image entry point)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Spiral Render Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_shader.pipelines[1]); // Second pipeline is main_image
            compute_pass.set_bind_group(0, &self.compute_shader.time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.compute_shader.storage_bind_group, &[]);
            compute_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            if let Some(atomic_buffer) = &self.compute_shader.atomic_buffer {
                compute_pass.set_bind_group(3, &atomic_buffer.bind_group, &[]);
            }
            let width = core.size.width.div_ceil(16);
            let height = core.size.height.div_ceil(16);
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Display Pass"),
            );
            
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.compute_shader.output_texture.bind_group, &[]);
            
            render_pass.draw(0..4, 0..1);
        }
        
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        
        if self.export_time.is_none() {
            self.frame_count = self.frame_count.wrapping_add(1);
        }
        
        Ok(())
    }
    
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
            return true;
        }
        
        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Chaos Spiral", 800, 600);
    
    app.run(event_loop, |core| {
        SpiralShader::init(core)
    })
}