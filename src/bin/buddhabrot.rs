use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BuddhabrotParams {
    max_iterations: u32,      
    escape_radius: f32,       
    zoom: f32,                
    offset_x: f32,            
    offset_y: f32,            
    rotation: f32,            
    exposure: f32,            
    low_iterations: u32, 
    high_iterations: u32,
    motion_speed: f32,        
    color1_r: f32,            
    color1_g: f32,            
    color1_b: f32,            
    color2_r: f32,            
    color2_g: f32,            
    color2_b: f32,            
    sample_density: f32,      
    dithering: f32,           
}


impl UniformProvider for BuddhabrotParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct BuddhabrotShader {
    base: RenderKit,
    params_uniform: UniformBinding<BuddhabrotParams>,
    compute_shader: ComputeShader,
    frame_count: u32,
    accumulated_rendering: bool,
}

impl BuddhabrotShader {
    
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
            if let Ok(data) = self.capture_frame(core, time) {
                let settings = self.base.export_manager.settings();
                if let Err(e) = cuneus::save_frame(data, frame, settings) {
                    eprintln!("Error saving frame: {:?}", e);
                }
            }
        } else {
            self.base.export_manager.complete_export();
        }
    }
    
    fn clear_buffers(&mut self, core: &Core) {
        if !self.accumulated_rendering {
            self.compute_shader.clear_atomic_buffer(core);
        }
        self.accumulated_rendering = false;
    }
}

impl ShaderManager for BuddhabrotShader {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { multisampled: false, sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
            label: Some("texture_bind_group_layout"),
        });
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "Buddhabrot Params",
            BuddhabrotParams {
                max_iterations: 500,
                escape_radius: 4.0,
                zoom: 0.5,
                offset_x: -0.5,
                offset_y: 0.0,
                rotation: 1.5,
                exposure: 0.0005,
                low_iterations: 20,
                high_iterations: 100,
                motion_speed: 0.0,
                color1_r: 1.0,
                color1_g: 0.5,
                color1_b: 0.2,
                color2_r: 0.2,
                color2_g: 0.5,
                color2_b: 1.0,
                sample_density: 0.5,
                dithering: 0.2,
            },
            &create_bind_group_layout(&core.device, BindGroupLayoutType::CustomUniform, "Buddhabrot Params"),
            0,
        );
        
        let base = RenderKit::new(core, include_str!("../../shaders/vertex.wgsl"), include_str!("../../shaders/blit.wgsl"), &[&texture_bind_group_layout], None);

        let compute_config = ComputeShaderConfig {
            workgroup_size: [16, 16, 1],
            workgroup_count: None,
            dispatch_once: false,
            storage_texture_format: COMPUTE_TEXTURE_FORMAT_RGBA16,
            enable_atomic_buffer: true,
            atomic_buffer_multiples: 3,
            entry_points: vec!["Splat".to_string(), "main_image".to_string()],
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "Buddhabrot".to_string(),
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
            include_str!("../../shaders/buddhabrot.wgsl"),
            compute_config,
        );

        // Enable hot reload
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Buddhabrot Compute Shader Hot Reload"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/buddhabrot.wgsl").into()),
        });
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("shaders/buddhabrot.wgsl"),
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
            accumulated_rendering: false,
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
        
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });
                
                egui::Window::new("Buddhabrot Explorer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .min_width(250.0)
                    .max_width(500.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Fractal Parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.max_iterations, 100..=500).text("Max Iterations")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.escape_radius, 2.0..=10.0).text("Escape Radius")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.low_iterations, 5..=50).text("Low Iterations")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.high_iterations, 50..=500).text("High Iterations")).changed();
                            });
                        
                        egui::CollapsingHeader::new("View Controls")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.1..=5.0).logarithmic(true).text("Zoom")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.offset_x, -2.0..=1.0).text("Offset X")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.offset_y, -1.5..=1.5).text("Offset Y")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.rotation, -3.14159..=3.14159).text("Rotation")).changed();
                                ui.add_space(10.0);
                                ui.separator();
                                
                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.00005..=0.001).logarithmic(true).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.sample_density, 0.1..=2.0).text("Sample Density")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dithering, 0.0..=1.0).text("Dithering")).changed();
                            });
                            
                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Color 1:");
                                    let mut color = [params.color1_r, params.color1_g, params.color1_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.color1_r = color[0];
                                        params.color1_g = color[1];
                                        params.color1_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Color 2:");
                                    let mut color = [params.color2_r, params.color2_g, params.color2_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.color2_r = color[0];
                                        params.color2_g = color[1];
                                        params.color2_b = color[2];
                                        changed = true;
                                    }
                                });
                            });
                        
                        egui::CollapsingHeader::new("Rendering Options")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Accumulated?:");
                                    ui.checkbox(&mut self.accumulated_rendering, "");
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
            self.clear_buffers(core);
        }
        self.base.apply_control_request(controls_request);

        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = self.frame_count;
        self.base.time_uniform.update(&core.queue);
        
        // Update compute shader with the same time data
        self.compute_shader.set_time(current_time, 1.0/60.0, &core.queue);
        self.compute_shader.time_uniform.data.frame = self.frame_count;
        self.compute_shader.time_uniform.update(&core.queue);

        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
        
        if changed {
            if !self.accumulated_rendering {
                self.clear_buffers(core);
            }
            
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        // Intelligent dispatch: Only generate new samples if we're not in accumulated mode
        // or if we're still accumulating (frame count < 500)  
        let should_generate_samples = !self.accumulated_rendering || self.frame_count < 500;
        
        // Manual dispatch for selective pass execution
        if should_generate_samples {
            // Pass 1: Generate and splat particles (Splat entry point)
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Buddhabrot Splat Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.compute_shader.pipelines[0]); // First pipeline is Splat
            compute_pass.set_bind_group(0, &self.compute_shader.time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.compute_shader.storage_bind_group, &[]);
            compute_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            if let Some(atomic_buffer) = &self.compute_shader.atomic_buffer {
                compute_pass.set_bind_group(3, &atomic_buffer.bind_group, &[]);
            }
            compute_pass.dispatch_workgroups(2048, 1, 1);
        }
        
        // Pass 2: Render the accumulated data (main_image entry point)  
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Buddhabrot Render Pass"),
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
        self.frame_count = self.frame_count.wrapping_add(1);
        
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
    let (app, event_loop) = cuneus::ShaderApp::new("Buddhabrot", 800, 600);
    
    app.run(event_loop, |core| {
        BuddhabrotShader::init(core)
    })
}