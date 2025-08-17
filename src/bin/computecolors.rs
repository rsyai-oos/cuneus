use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager};
use cuneus::compute::{ComputeShader, ComputeShaderConfig, CustomStorageBuffer};
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ColorProjectionParams {
    rotation_speed: f32,
    intensity: f32,
    rot_x: f32,
    rot_y: f32,
    rot_z: f32,
    rot_w: f32,
    scale: f32,
    _padding: u32,
}

impl UniformProvider for ColorProjectionParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
struct ColorProjection {
    base: RenderKit,
    params_uniform: UniformBinding<ColorProjectionParams>,
    compute_shader: ComputeShader,
    frame_count: u32,
}

impl ShaderManager for ColorProjection {
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
            label: Some("Color Projection Texture Bind Group Layout"),
        });
        
        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        let initial_params = ColorProjectionParams {
            rotation_speed: 0.3,
            intensity: 1.2,
            rot_x: 0.0,
            rot_y: 0.0,
            rot_z: 0.0,
            rot_w: 1.0,
            scale: 1.0,
            _padding: 0,
        };
        

        let mut resource_layout = cuneus::compute::ResourceLayout::new();
        resource_layout.add_custom_uniform("color_params", std::mem::size_of::<ColorProjectionParams>() as u64);
        let bind_group_layouts = resource_layout.create_bind_group_layouts(&core.device);
        let color_params_layout = bind_group_layouts.get(&2).unwrap();

        let params_uniform = UniformBinding::new(
            &core.device,
            "Color Projection Params",
            initial_params,
            color_params_layout,
            0,
        );
        
        // Color projection requires atomic buffer for 3D color space accumulation
        let buffer_size = (core.size.width * core.size.height * 4 * 4) as u64;
        let compute_config = ComputeShaderConfig {
            label: "Color Projection".to_string(),
            enable_input_texture: true,
            enable_custom_uniform: true,
            entry_points: vec![
                "clear_buffer".to_string(),    // Stage 0
                "project_colors".to_string(),  // Stage 1 
                "generate_image".to_string(),  // Stage 2
            ],
            custom_storage_buffers: vec![
                CustomStorageBuffer {
                    label: "Atomic Buffer".to_string(),
                    size: buffer_size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                },
            ],
            ..Default::default()
        };
        
        let mut compute_shader = ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/computecolors.wgsl"),
            compute_config,
        );

        // Enable hot reload
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ComputeColors Compute Shader Hot Reload"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/computecolors.wgsl").into()),
        });
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("shaders/computecolors.wgsl"),
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
    
    fn update(&mut self, core: &Core) {
        self.base.update_current_texture(core, &core.queue);
        if let Some(texture_manager) = self.base.get_current_texture_manager() {
            self.compute_shader.update_input_texture(core, &texture_manager.view, &texture_manager.sampler);
        }
        
        if self.base.export_manager.is_exporting() {
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
            label: Some("Color Projection Render Encoder"),
        });
        
        // Handle UI and controls
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
                
                egui::Window::new("color projection")
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
                        
                        egui::CollapsingHeader::new("Visual Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.intensity, 0.1..=12.0).text("Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.scale, 0.5..=4.0).text("Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.rotation_speed, 0.0..=1.0).text("Rotation Speed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.rot_w, 0.0..=1.0).text("bg intensity")).changed();

                                if ui.button("Reset Visual").clicked() {
                                    params.intensity = 1.2;
                                    params.scale = 1.0;
                                    params.rotation_speed = 0.3;
                                    changed = true;
                                }
                            });
                        
                        egui::CollapsingHeader::new("Rotation Axes")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.rot_x, -3.14..=3.14).text("X Rotation")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.rot_y, -3.14..=3.14).text("Y Rotation")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.rot_z, -3.14..=3.14).text("Z Rotation")).changed();
                                if ui.button("Reset Rotation").clicked() {
                                    params.rot_x = 0.0;
                                    params.rot_y = 0.0;
                                    params.rot_z = 0.0;
                                    changed = true;
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
            // For first texture upload, ensure input texture is updated immediately
            if let Some(ref texture_manager) = self.base.texture_manager {
                self.compute_shader.update_input_texture(core, &texture_manager.view, &texture_manager.sampler);
            }
        }
        if controls_request.start_webcam {
            // Webcam started
        }
        
        if changed {
            self.params_uniform.data = params;
        }
        
        // Update time and params
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = self.frame_count;
        self.base.time_uniform.update(&core.queue);
        
        self.compute_shader.set_time(current_time, 1.0/60.0, &core.queue);

        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
        
        // Update color projection parameters
        self.params_uniform.update(&core.queue);
        
        // Color projection must run ALL stages EVERY frame for animation
        // Stage 0: Clear atomic buffer
        self.compute_shader.dispatch_stage(&mut encoder, 0, (
            core.size.width.div_ceil(16),
            core.size.height.div_ceil(16), 
            1
        ), Some(&self.params_uniform.bind_group));
        
        // Get input texture dimensions for proper workgroup dispatch
        let input_dimensions = if let Some(ref texture_manager) = self.base.texture_manager {
            (texture_manager.texture.width().div_ceil(16), texture_manager.texture.height().div_ceil(16))
        } else {
            (core.size.width.div_ceil(16), core.size.height.div_ceil(16))
        };
        
        // Stage 1: Project colors to 3D space
        self.compute_shader.dispatch_stage(&mut encoder, 1, (
            input_dimensions.0,
            input_dimensions.1, 
            1
        ), Some(&self.params_uniform.bind_group));
        
        // Stage 2: Generate final image
        self.compute_shader.dispatch_stage(&mut encoder, 2, (
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
                if let Some(ref texture_manager) = self.base.texture_manager {
                    self.compute_shader.update_input_texture(core, &texture_manager.view, &texture_manager.sampler);
                }
            }
            return true;
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Color Projection", 800, 600);
    app.run(event_loop, |core| {
        ColorProjection::init(core)
    })
}