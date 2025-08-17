use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager};
use cuneus::compute::{ComputeShader, ComputeShaderConfig, CustomStorageBuffer};
use winit::event::WindowEvent;
use std::path::PathBuf;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RorschachParams {
    // Matrix 1
    m1_scale: f32,
    m1_y_scale: f32,
    // Matrix 2
    m2_scale: f32,
    m2_shear: f32,
    m2_shift: f32,
    // Matrix 3
    m3_scale: f32,
    m3_shear: f32,
    m3_shift: f32,
    // Matrix 4
    m4_scale: f32,
    m4_shift: f32,
    // Matrix 5
    m5_scale: f32,
    m5_shift: f32,
    time_scale: f32,
    decay: f32,
    intensity: f32,
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    brightness: f32,
    exposure: f32,
    gamma: f32,
    particle_count: f32,
    scale: f32,
    dof_amount: f32,
    dof_focal_dist: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
}

impl UniformProvider for RorschachParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct RorschachShader {
    base: RenderKit,
    params_uniform: UniformBinding<RorschachParams>,
    frame_count: u32,
}

impl RorschachShader {
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
            if let Some(compute_shader) = &self.base.compute_shader {
                render_pass.set_bind_group(0, &compute_shader.get_output_texture().bind_group, &[]);
            }
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

impl ShaderManager for RorschachShader {
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
            label: Some("Rorschach Texture Bind Group Layout"),
        });
        
        let mut base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        base.setup_mouse_uniform(core);

        let initial_params = RorschachParams {
            m1_scale: 0.8,
            m1_y_scale: 0.5,
            m2_scale: 0.4,
            m2_shear: 0.2,
            m2_shift: 0.3,
            m3_scale: 0.4,
            m3_shear: 0.2,
            m3_shift: 0.3,
            m4_scale: 0.3,
            m4_shift: 0.2,
            m5_scale: 0.2,
            m5_shift: 0.4,
            time_scale: 0.5,
            decay: 0.0,
            intensity: 0.0,
            rotation_x: 0.0,
            rotation_y: 0.0,
            click_state: 0,
            brightness: 0.003,
            exposure: 1.5,
            gamma: 0.4,
            particle_count: 100000.0,
            scale: 1.0,
            dof_amount: 0.0,
            dof_focal_dist: 0.5,
            color1_r: 1.0,
            color1_g: 0.3,
            color1_b: 0.1,
            color2_r: 0.1,
            color2_g: 0.5,
            color2_b: 1.0,
        };
        
        let mut resource_layout = cuneus::compute::ResourceLayout::new();
        resource_layout.add_custom_uniform("rorschach_params", std::mem::size_of::<RorschachParams>() as u64);
        let bind_group_layouts = resource_layout.create_bind_group_layouts(&core.device);
        let rorschach_params_layout = bind_group_layouts.get(&2).unwrap();

        let params_uniform = UniformBinding::new(
            &core.device,
            "Rorschach Params",
            initial_params,
            rorschach_params_layout,
            0,
        );
        
        // Rorschach requires atomic buffer for particle accumulation
        let buffer_size = (core.size.width * core.size.height * 4 * 4) as u64;
        let compute_config = ComputeShaderConfig {
            label: "Rorschach".to_string(),
            enable_input_texture: false,
            enable_custom_uniform: true,
            entry_points: vec![
                "Splat".to_string(),      // Stage 0
                "main_image".to_string(), // Stage 1 
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
        
        let mut base = base;
        base.compute_shader = Some(ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/rorschach.wgsl"),
            compute_config,
        ));
        

        if let Some(compute_shader) = &mut base.compute_shader {
            let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Rorschach Compute Shader Hot Reload"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/rorschach.wgsl").into()),
            });
            if let Err(e) = compute_shader.enable_hot_reload(
                core.device.clone(),
                PathBuf::from("shaders/rorschach.wgsl"),
                shader_module,
            ) {
                eprintln!("Failed to enable compute shader hot reload: {}", e);
            }
        }
        
        // Add custom uniform binding
        if let Some(compute_shader) = &mut base.compute_shader {
            compute_shader.add_custom_uniform_binding(&params_uniform.bind_group);
        }
        
        Self {
            base,
            params_uniform,
            frame_count: 0,
        }
    }

    fn update(&mut self, core: &Core) {
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
        
        self.base.update_mouse_uniform(&core.queue);
        self.base.fps_tracker.update();
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        if self.base.mouse_tracker.uniform.buttons[0] & 1 != 0 {
            params.rotation_x = self.base.mouse_tracker.uniform.position[0];
            params.rotation_y = self.base.mouse_tracker.uniform.position[1];
            params.click_state = 1;
            changed = true;
        } else if self.base.mouse_tracker.uniform.buttons[0] & 2 != 0 {
            params.click_state = 0;
        } else {
            params.click_state = 0;
        }

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });
                
                egui::Window::new("Rorschach IFS")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        
                        egui::CollapsingHeader::new("IFS Matrices")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.label("Matrix 1:");
                                changed |= ui.add(egui::Slider::new(&mut params.m1_scale, 0.1..=1.2).text("Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m1_y_scale, 0.1..=1.2).text("Y Scale")).changed();
                                
                                ui.separator();
                                ui.label("Matrix 2:");
                                changed |= ui.add(egui::Slider::new(&mut params.m2_scale, 0.1..=1.0).text("Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m2_shear, -0.5..=0.5).text("Shear")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m2_shift, -0.5..=0.5).text("Shift")).changed();
                                
                                ui.separator();
                                ui.label("Matrix 3:");
                                changed |= ui.add(egui::Slider::new(&mut params.m3_scale, 0.1..=1.0).text("Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m3_shear, -0.5..=0.5).text("Shear")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m3_shift, -0.5..=0.5).text("Shift")).changed();
                                
                                ui.separator();
                                ui.label("Matrix 4 & 5:");
                                changed |= ui.add(egui::Slider::new(&mut params.m4_scale, 0.1..=1.0).text("M4 Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m4_shift, -0.5..=0.5).text("M4 Shift")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m5_scale, 0.1..=1.0).text("M5 Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m5_shift, -0.5..=0.5).text("M5 Shift")).changed();
                            });

                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Primary:");
                                    let mut color1 = [params.color1_r, params.color1_g, params.color1_b];
                                    if ui.color_edit_button_rgb(&mut color1).changed() {
                                        params.color1_r = color1[0];
                                        params.color1_g = color1[1];
                                        params.color1_b = color1[2];
                                        changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Secondary:");
                                    let mut color2 = [params.color2_r, params.color2_g, params.color2_b];
                                    if ui.color_edit_button_rgb(&mut color2).changed() {
                                        params.color2_r = color2[0];
                                        params.color2_g = color2[1];
                                        params.color2_b = color2[2];
                                        changed = true;
                                    }
                                });
                            });

                        egui::CollapsingHeader::new("Rendering")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.particle_count, 10000.0..=200000.0).text("Particles")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.scale, 0.1..=3.0).text("Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.brightness, 0.001..=0.01).text("Brightness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.5..=3.0).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=2.0).text("Gamma")).changed();
                            });

                        egui::CollapsingHeader::new("Depth of Field")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.dof_amount, 0.0..=2.0).text("DOF Amount")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dof_focal_dist, 0.0..=1.0).text("Focal Distance")).changed();
                            });

                        egui::CollapsingHeader::new("Animation")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.time_scale, 0.0..=2.0).text("Animation Speed")).changed();
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
        self.base.apply_control_request(controls_request);
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Update time and dispatch compute shader stages
        if let Some(compute_shader) = &mut self.base.compute_shader {
            let delta = 1.0/60.0;
            compute_shader.set_time(current_time, delta, &core.queue);
            
            // Check for hot reload updates
            compute_shader.check_hot_reload(&core.device);
            
            // Clear atomic buffer for new frame
            if !compute_shader.custom_storage_buffers.is_empty() {
                let buffer = &compute_shader.custom_storage_buffers[0];
                let buffer_size = (core.size.width * core.size.height * 4 * 4) as u64;
                core.queue.write_buffer(buffer, 0, &vec![0u8; buffer_size as usize]);
            }

            // Stage 0: Generate and splat particles
            let workgroups = (self.params_uniform.data.particle_count as u32 / 256).max(1);
            compute_shader.dispatch_stage(&mut encoder, 0, (workgroups, 1, 1), Some(&self.params_uniform.bind_group));

            // Stage 1: Render to screen
            let workgroups_x = (core.size.width as f32 / 16.0).ceil() as u32;
            let workgroups_y = (core.size.height as f32 / 16.0).ceil() as u32;
            compute_shader.dispatch_stage(&mut encoder, 1, (workgroups_x, workgroups_y, 1), Some(&self.params_uniform.bind_group));
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
            if let Some(compute_shader) = &self.base.compute_shader {
                render_pass.set_bind_group(0, &compute_shader.get_output_texture().bind_group, &[]);
            }
            render_pass.draw(0..4, 0..1);
        }
        
        self.frame_count = self.frame_count.wrapping_add(1);

        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        let ui_handled = self.base.egui_state.on_window_event(core.window(), event).consumed;
        
        if ui_handled {
            return true;
        }

        if self.base.handle_mouse_input(core, event, false) {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }

        false
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        if let Some(compute_shader) = &mut self.base.compute_shader {
            compute_shader.resize(core, core.size.width, core.size.height);
        }
    }
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Rorschach IFS", 800, 600);
    app.run(event_loop, |core| {
        RorschachShader::init(core)
    })
}