use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager};
use cuneus::compute::{ComputeShader, ComputeShaderConfig, CustomStorageBuffer};
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FFTParams {
    filter_type: i32,     
    filter_strength: f32, 
    filter_direction: f32,
    filter_radius: f32,   
    show_freqs: i32,      
    resolution: u32,      
    _padding1: u32,
    _padding2: u32,
}

impl UniformProvider for FFTParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct FFTShader {
    base: RenderKit,
    params_uniform: UniformBinding<FFTParams>,
    compute_shader: ComputeShader,
    frame_count: u32,
    should_initialize: bool,
}

impl ShaderManager for FFTShader {
    fn init(core: &Core) -> Self {
        // Create texture bind group layout for displaying compute shader output
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("FFT Texture Bind Group Layout"),
        });
        
        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        let initial_params = FFTParams {
            filter_type: 1,
            filter_strength: 0.3,
            filter_direction: 0.0,
            filter_radius: 3.0,
            show_freqs: 0,
            resolution: 1024,
            _padding1: 0,
            _padding2: 0,
        };
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "FFT Params",
            initial_params,
            &cuneus::compute::create_bind_group_layout(&core.device, cuneus::compute::BindGroupLayoutType::CustomUniform, "FFT Params"),
            0,
        );
        
        // FFT requires custom storage buffer for complex algorithm data
        let buffer_size = 1024 * 1024 * 4 * 8; // Complex numbers (2 floats) for FFT data
        let compute_config = ComputeShaderConfig {
            label: "FFT".to_string(),
            enable_input_texture: true, // Enable texture upload capability like user's manual implementation
            enable_custom_uniform: true,
            entry_points: vec![
                "initialize_data".to_string(),      // Stage 0
                "fft_horizontal".to_string(),       // Stage 1 
                "fft_vertical".to_string(),         // Stage 2
                "modify_frequencies".to_string(),   // Stage 3
                "ifft_horizontal".to_string(),      // Stage 4
                "ifft_vertical".to_string(),        // Stage 5
                "main_image".to_string(),           // Stage 6 - main rendering
            ],
            custom_storage_buffers: vec![
                CustomStorageBuffer {
                    label: "FFT Storage".to_string(),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
            ],
            ..Default::default()
        };
        
        let mut compute_shader = ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/fft.wgsl"),
            compute_config,
        );

        // Enable hot reload
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FFT Compute Shader Hot Reload"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fft.wgsl").into()),
        });
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("shaders/fft.wgsl"),
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
            should_initialize: true,
        }
    }
    
    fn update(&mut self, core: &Core) {
        self.base.update_current_texture(core, &core.queue);
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            self.compute_shader.update_input_texture(core, &texture_manager.view, &texture_manager.sampler);
        }
        
        if self.base.export_manager.is_exporting() {
            // Handle export if needed
        }
        
        self.base.fps_tracker.update();
    }
    
    fn resize(&mut self, core: &Core) {
        self.compute_shader.resize(core, core.size.width, core.size.height);
    }
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FFT Render Encoder"),
        });
        
        // Handle UI and controls - using original transparent UI design
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );

        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();
        
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });
                
                egui::Window::new("fourier workflow")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        // Media controls
                        ShaderControls::render_media_panel(
                            ui,
                            &mut controls_request,
                            using_video_texture,
                            video_info,
                            using_hdri_texture,
                            hdri_info,
                            using_webcam_texture,
                            webcam_info
                        );
                        
                        ui.separator();
                        
                        egui::CollapsingHeader::new("FFT Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Resolution:");
                                
                                ui.horizontal(|ui| {
                                    changed |= ui.radio_value(&mut params.resolution, 256, "256").changed();
                                    changed |= ui.radio_value(&mut params.resolution, 512, "512").changed();
                                    changed |= ui.radio_value(&mut params.resolution, 1024, "1024").changed();
                                    changed |= ui.radio_value(&mut params.resolution, 2048, "2048").changed();
                                });
                                
                                if changed {
                                    self.should_initialize = true;
                                }
                                
                                ui.separator();
                                ui.label("View Mode:");
                                changed |= ui.radio_value(&mut params.show_freqs, 0, "Filtered").changed();
                                changed |= ui.radio_value(&mut params.show_freqs, 1, "Frequency Domain").changed();
                                
                                ui.separator();
                            });
                        
                        egui::CollapsingHeader::new("Filter Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Filter Type:");
                                // Keep the improved ComboBox as requested
                                changed |= egui::ComboBox::from_label("")
                                    .selected_text(match params.filter_type {
                                        0 => "LP",
                                        1 => "HP", 
                                        2 => "BP",
                                        3 => "Directional",
                                        _ => "None"
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(&mut params.filter_type, 0, "LP").changed() ||
                                        ui.selectable_value(&mut params.filter_type, 1, "HP").changed() ||
                                        ui.selectable_value(&mut params.filter_type, 2, "BP").changed() ||
                                        ui.selectable_value(&mut params.filter_type, 3, "Directional").changed()
                                    })
                                    .inner.unwrap_or(false);
                                
                                ui.separator();
                                
                                changed |= ui.add(egui::Slider::new(&mut params.filter_strength, 0.0..=1.0)
                                    .text("Filter Strength"))
                                    .changed();
                                
                                if params.filter_type == 2 {
                                    changed |= ui.add(egui::Slider::new(&mut params.filter_radius, 0.0..=6.28)
                                        .text("Band Radius"))
                                        .changed();
                                }
                                
                                if params.filter_type == 3 {
                                    changed |= ui.add(egui::Slider::new(&mut params.filter_direction, 0.0..=6.28)
                                        .text("Direction"))
                                        .changed();
                                }
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

        // Apply parameter changes
        if changed {
            self.params_uniform.data = params;
        }
        
        // Apply controls
        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        self.base.handle_webcam_requests(core, &controls_request);
        
        // Handle export requests
        self.base.export_manager.apply_ui_request(export_request);
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        if controls_request.load_media_path.is_some() {
            self.should_initialize = true;
        }
        if controls_request.start_webcam {
            self.should_initialize = true;
        }
        
        // Update time and params
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = self.frame_count;
        self.base.time_uniform.update(&core.queue);
        
        // Apply parameter changes (original clean approach)
        if changed {
            self.params_uniform.data = params;
            self.should_initialize = true;  // Trigger FFT reprocessing
        }
        
        // Update FFT parameters
        self.params_uniform.update(&core.queue);
        
        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);

        // FFT compute stages - run full pipeline when needed
        let n = self.params_uniform.data.resolution;
        let should_run_fft = self.should_initialize || 
                            self.base.using_video_texture || 
                            self.base.using_webcam_texture;
        
        if should_run_fft {
            // Stage 0: Initialize data from input texture
            self.compute_shader.dispatch_stage(&mut encoder, 0, (
                n.div_ceil(16), 
                n.div_ceil(16), 
                1
            ), Some(&self.params_uniform.bind_group));
            
            // Stage 1: FFT horizontal
            self.compute_shader.dispatch_stage(&mut encoder, 1, (n, 1, 1), Some(&self.params_uniform.bind_group));
            
            // Stage 2: FFT vertical  
            self.compute_shader.dispatch_stage(&mut encoder, 2, (n, 1, 1), Some(&self.params_uniform.bind_group));
            
            // Stage 3: Modify frequencies (apply filter)
            self.compute_shader.dispatch_stage(&mut encoder, 3, (
                n.div_ceil(16), 
                n.div_ceil(16), 
                1
            ), Some(&self.params_uniform.bind_group));
            
            // Stage 4: Inverse FFT horizontal
            self.compute_shader.dispatch_stage(&mut encoder, 4, (n, 1, 1), Some(&self.params_uniform.bind_group));
            
            // Stage 5: Inverse FFT vertical
            self.compute_shader.dispatch_stage(&mut encoder, 5, (n, 1, 1), Some(&self.params_uniform.bind_group));
            
            self.should_initialize = false;
        }
        
        // Stage 6: Main rendering (always run for display)
        self.compute_shader.dispatch_stage(&mut encoder, 6, (
            core.size.width.div_ceil(16),
            core.size.height.div_ceil(16), 
            1
        ), Some(&self.params_uniform.bind_group));
        
        // Display result using ComputeShader
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Display Pass"),
            );
            
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.compute_shader.get_output_texture().bind_group, &[]);
            
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
        
        if let WindowEvent::DroppedFile(path) = event {
            if let Err(e) = self.base.load_media(core, path) {
                eprintln!("Failed to load dropped file: {:?}", e);
            } else {
                self.should_initialize = true;
            }
            return true;
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("FFT", 800, 600);
    app.run(event_loop, |core| {
        FFTShader::init(core)
    })
}