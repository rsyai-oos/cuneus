use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LorenzParams {
    sigma: f32,
    rho: f32,
    beta: f32,
    step_size: f32,
    motion_speed: f32,
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
    scale: f32,
    dof_amount: f32,
    dof_focal_dist: f32,
    gamma: f32,
    exposure: f32,
    particle_count: f32,
    decay_speed: f32,
}

impl UniformProvider for LorenzParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct LorenzShader {
    base: RenderKit,
    params_uniform: UniformBinding<LorenzParams>,
    compute_shader: ComputeShader,
    frame_count: u32,
    mouse_look_enabled: bool,
}

impl LorenzShader {
    fn clear_atomic_buffer(&mut self, core: &Core) {
        self.compute_shader.clear_atomic_buffer(core);
        self.frame_count = 0;
    }
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
}

impl ShaderManager for LorenzShader {
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
            label: Some("texture_bind_group_layout"),
        });
        let params_uniform = UniformBinding::new(
            &core.device,
            "Lorenz Params",
            LorenzParams {
                sigma: 40.0,
                rho: 33.0,
                beta: 30.0 / 3.0,
                step_size: 0.02,
                motion_speed: 2.2,
                rotation_x: 0.0,
                rotation_y: 0.0,
                click_state: 0,
                brightness: 0.0005,
                color1_r: 1.0,
                color1_g: 0.5,
                color1_b: 0.0,
                color2_r: 0.0,
                color2_g: 0.5,
                color2_b: 1.0,
                scale: 0.013,
                dof_amount: 0.1,
                dof_focal_dist: 0.5,
                gamma: 2.2,
                exposure: 1.0,
                particle_count: 1000.0,
                decay_speed: 8.0,
            },
            &create_bind_group_layout(&core.device, BindGroupLayoutType::CustomUniform, "Lorenz Params"),
            0,
        );

        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        // Mouse data is passed through custom uniform instead of separate mouse uniform bind group
        // base.setup_mouse_uniform(core);

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
            label: "Lorenz".to_string(),
            mouse_bind_group_layout: None, // Mouse data passed through custom uniform instead
            enable_fonts: false,
            enable_audio_buffer: false,
            audio_buffer_size: 0,
            enable_custom_uniform: true,
            enable_input_texture: false,
            custom_storage_buffers: Vec::new(),
        };

        let mut compute_shader = ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/lorenz.wgsl"),
            compute_config,
        );

        // Enable hot reload
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Lorenz Compute Shader Hot Reload"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/lorenz.wgsl").into()),
        });
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("shaders/lorenz.wgsl"),
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
            mouse_look_enabled: false,
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
                
                egui::Window::new("Volumetric Lorenz")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(350.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Attractor Parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.sigma, 0.0..=80.0).text("Sigma (σ)")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.rho, 0.0..=100.0).text("Rho (ρ)")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.beta, 0.0..=10.0).text("Beta (β)")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.step_size, 0.001..=0.02).text("Step Size").logarithmic(true)).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.motion_speed, 0.0..=5.0).text("Motion Speed")).changed();
                            });
                        
                        egui::CollapsingHeader::new("Camera")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.checkbox(&mut self.mouse_look_enabled, "Enable Mouse Look");
                                ui.separator();
                                
                                if !self.mouse_look_enabled {
                                    changed |= ui.add(egui::Slider::new(&mut params.rotation_x, -1.0..=1.0).text("Rotation X")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.rotation_y, -1.0..=1.0).text("Rotation Y")).changed();
                                } else {
                                    ui.label("Mouse Look Active - Move mouse to control camera");
                                }
                                
                                changed |= ui.add(egui::Slider::new(&mut params.scale, 0.001..=0.1).text("Zoom").logarithmic(true)).changed();
                            });
                        
                        egui::CollapsingHeader::new("Rendering")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.brightness, 0.0001..=0.01).text("Brightness").logarithmic(true)).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.1..=5.0).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.5..=4.0).text("Gamma")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.particle_count, 100.0..=5000.0).text("Particle Count")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dof_amount, 0.0..=1.0).text("DOF Amount")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dof_focal_dist, 0.0..=1.0).text("DOF Focal Distance")).changed();
                                
                                ui.separator();
                                ui.label("Trail Settings:");
                                changed |= ui.add(egui::Slider::new(&mut params.decay_speed, 1.0..=50.0).text("Decay Speed (higher = faster fade)")).changed();
                            });
                        
                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                let mut color1 = [params.color1_r, params.color1_g, params.color1_b];
                                let mut color2 = [params.color2_r, params.color2_g, params.color2_b];
                                
                                ui.horizontal(|ui| {
                                    ui.label("Color 1:");
                                    if ui.color_edit_button_rgb(&mut color1).changed() {
                                        params.color1_r = color1[0];
                                        params.color1_g = color1[1];
                                        params.color1_b = color1[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Color 2:");
                                    if ui.color_edit_button_rgb(&mut color2).changed() {
                                        params.color2_r = color2[0];
                                        params.color2_g = color2[1];
                                        params.color2_b = color2[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.separator();
                                ui.label("Presets:");
                                ui.horizontal(|ui| {
                                    if ui.button("Fire").clicked() {
                                        params.color1_r = 1.0; params.color1_g = 0.3; params.color1_b = 0.0;
                                        params.color2_r = 1.0; params.color2_g = 1.0; params.color2_b = 0.0;
                                        changed = true;
                                    }
                                    if ui.button("Ocean").clicked() {
                                        params.color1_r = 0.0; params.color1_g = 0.3; params.color1_b = 1.0;
                                        params.color2_r = 0.0; params.color2_g = 0.8; params.color2_b = 1.0;
                                        changed = true;
                                    }
                                    if ui.button("Purple").clicked() {
                                        params.color1_r = 0.5; params.color1_g = 0.0; params.color1_b = 1.0;
                                        params.color2_r = 1.0; params.color2_g = 0.0; params.color2_b = 0.5;
                                        changed = true;
                                    }
                                });
                            });
                        
                        ui.separator();
                        
                        ui.separator();
        ui.label("Controls:");
        ui.horizontal(|ui| {
            ui.label("• Mouse:");
            if self.mouse_look_enabled {
                ui.colored_label(egui::Color32::GREEN, "Active");
            } else {
                ui.colored_label(egui::Color32::RED, "Disabled");
            }
        });
        ui.label("• Right click: Toggle mouse control");
        ui.label("• H: Toggle UI");
        
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
            self.clear_atomic_buffer(core);
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
        
        // Mouse data is read from tracker and passed through custom uniform parameters
        // self.base.update_mouse_uniform(&core.queue);
        if self.mouse_look_enabled {
            params.rotation_x = self.base.mouse_tracker.uniform.position[0];
            params.rotation_y = self.base.mouse_tracker.uniform.position[1];
        }
        params.click_state = if self.base.mouse_tracker.uniform.buttons[0] & 1 > 0 { 1 } else { 0 };
        changed = true;
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        // Use ComputeShader dispatch
        self.compute_shader.dispatch(&mut encoder, core);
        
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
        if let WindowEvent::MouseInput { state, button, .. } = event {
            if *button == winit::event::MouseButton::Right {
                if *state == winit::event::ElementState::Released {
                    self.mouse_look_enabled = !self.mouse_look_enabled;
                    return true;
                }
            }
        }
        if self.mouse_look_enabled && self.base.handle_mouse_input(core, event, false) {
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
    let (app, event_loop) = cuneus::ShaderApp::new("Volumetric Lorenz", 800, 600);
    
    app.run(event_loop, |core| {
        LorenzShader::init(core)
    })
}