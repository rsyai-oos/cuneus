use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SplattingParams {
    animation_speed: f32,
    splat_size: f32,
    particle_spread: f32,
    intensity: f32,
    particle_density: f32,
    brightness: f32,
    physics_strength: f32,
    _padding: u32,
}

impl UniformProvider for SplattingParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct ColorProjection {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: SplattingParams,
}

impl ColorProjection {
    fn clear_buffers(&mut self, core: &Core) {
        self.compute_shader.clear_all_buffers(core);
    }
}

impl ShaderManager for ColorProjection {
    fn init(core: &Core) -> Self {
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
            label: Some("Color Projection Texture Bind Group Layout"),
        });
        
        let base = RenderKit::new(
            core,
            &[&texture_bind_group_layout],
            None,
        );
        
        let initial_params = SplattingParams {
            animation_speed: 1.0,
            splat_size: 0.8,
            particle_spread: 0.3,
            intensity: 2.0,
            particle_density: 0.4,
            brightness: 24.0,
            physics_strength: 0.5,
            _padding: 0,
        };
        
        // Define the multi-stage passes  
        let passes = vec![
            PassDescription::new("clear_buffer", &[]),       // Stage 0: Clear atomic buffer
            PassDescription::new("project_colors", &[]),     // Stage 1: Project colors to 3D space
            PassDescription::new("generate_image", &[]),     // Stage 2: Generate final image
        ];

        let config = ComputeShader::builder()
            .with_entry_point("clear_buffer") 
            .with_multi_pass(&passes)
            .with_input_texture() // Enable input texture support
            .with_custom_uniforms::<SplattingParams>()
            .with_storage_buffer(StorageBufferSpec::new("atomic_buffer", (core.size.width * core.size.height * 4 * 4) as u64)) 
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Particle Splatting Multi-Pass")
            .build();

        let mut compute_shader = ComputeShader::from_builder(
            core,
            include_str!("shaders/computecolors.wgsl"),
            config,
        );

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("examples/shaders/computecolors.wgsl"),
            core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("ComputeColors Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/computecolors.wgsl").into()),
            }),
        ) {
            eprintln!("Failed to enable ComputeColors hot reload: {}", e);
        }

        compute_shader.set_custom_params(initial_params, &core.queue);
        
        Self {
            base,
            compute_shader,
            current_params: initial_params,
        }
    }
    
    fn update(&mut self, core: &Core) {
        // Update time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0/60.0;
        self.compute_shader.set_time(current_time, delta, &core.queue);
        
        // Update input textures for media processing
        self.base.update_current_texture(core, &core.queue);
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            self.compute_shader.update_input_texture(&texture_manager.view, &texture_manager.sampler, &core.device);
        }
        
        self.base.fps_tracker.update();
    }
    
    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.compute_shader.resize(core, core.size.width, core.size.height);
    }
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Color Projection Render Encoder"),
        });
        
        // Handle UI and controls
        let mut params = self.current_params;
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
                
                egui::Window::new("Particle Splatting")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
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
                        
                        egui::CollapsingHeader::new("Particle Splatting")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.particle_density, 0.1..=1.0).text("Particle Density")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.splat_size, 0.1..=2.0).text("Splat Size")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.intensity, 0.1..=3.0).text("Particle Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.brightness, 20.0..=36.0).text("Brightness")).changed();
                            });
                        
                        egui::CollapsingHeader::new("Artistic Effects")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.animation_speed, 0.0..=3.0).text("Animation Speed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.particle_spread, 0.0..=1.0).text("Scramble Amount")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.physics_strength, 0.0..=1.0).text("Return Force")).changed();
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
        
        // Apply controls
        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
            self.clear_buffers(core);
        }
        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        self.base.handle_webcam_requests(core, &controls_request);
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
        }
        
        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
        // Handle export        
        self.compute_shader.handle_export(core, &mut self.base);
        
        // Color projection multi-stage dispatch - run all stages every frame for animation
        
        // Stage 0: Clear atomic buffer (16x16 workgroups)
        self.compute_shader.dispatch_stage(&mut encoder, core, 0);
        
        // Stage 1: Project colors to 3D space (uses input texture dimensions)
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            let input_workgroups = [
                texture_manager.texture.width().div_ceil(16),
                texture_manager.texture.height().div_ceil(16),
                1
            ];
            self.compute_shader.dispatch_stage_with_workgroups(&mut encoder, 1, input_workgroups);
        } else {
            // Fallback to screen size if no input texture
            self.compute_shader.dispatch_stage(&mut encoder, core, 1);
        }
        
        // Stage 2: Generate final image (16x16 workgroups, screen size)
        self.compute_shader.dispatch_stage(&mut encoder, core, 2);
        
        // Display result
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
            }
            return true;
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Particle Splatting", 800, 600);
    app.run(event_loop, |core| {
        ColorProjection::init(core)
    })
}