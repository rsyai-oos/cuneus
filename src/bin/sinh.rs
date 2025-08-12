use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager};
use cuneus::compute::{ComputeShader, ComputeShaderConfig};
use winit::event::WindowEvent;
use std::path::PathBuf;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SinhParams {
    aa: i32,
    camera_x: f32,
    camera_y: f32,
    camera_z: f32,
    orbit_speed: f32,
    magic_number: f32,
    cv_min: f32,
    cv_max: f32,
    os_base: f32,
    os_scale: f32,
    base_color_r: f32,
    base_color_g: f32,
    base_color_b: f32,
    light_color_r: f32,
    light_color_g: f32,
    light_color_b: f32,
    ambient_r: f32,
    ambient_g: f32,
    ambient_b: f32,
    gamma: f32,
    iterations: i32,
    bound: f32,
    fractal_scale: f32,
    vignette_offset: f32,
}

impl UniformProvider for SinhParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct SinhShader {
    base: RenderKit,
    params_uniform: UniformBinding<SinhParams>,
    frame_count: u32,
}

impl SinhShader {

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

impl ShaderManager for SinhShader {
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
            label: Some("Sinh Texture Bind Group Layout"),
        });
        
        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        let initial_params = SinhParams {
            aa: 2,
            camera_x: 0.1,
            camera_y: 10.0,
            camera_z: 10.0,
            orbit_speed: 0.3,
            magic_number: 36.0,
            cv_min: 2.197,
            cv_max: 2.99225,
            os_base: 0.00004,
            os_scale: 0.02040101,
            base_color_r: 0.5,
            base_color_g: 0.25,
            base_color_b: 0.05,
            light_color_r: 0.8,
            light_color_g: 1.0,
            light_color_b: 0.3,
            ambient_r: 1.2,
            ambient_g: 1.0,
            ambient_b: 0.8,
            gamma: 0.4,
            iterations: 65,
            bound: 12.25,
            fractal_scale: 0.05,
            vignette_offset: 0.0,
        };
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "Sinh Params",
            initial_params,
            &cuneus::compute::create_bind_group_layout(&core.device, cuneus::compute::BindGroupLayoutType::CustomUniform, "Sinh Params"),
            0,
        );
        
        // Sinh doesn't need atomic buffers, it's a direct render-to-texture fractal
        let compute_config = ComputeShaderConfig {
            label: "Sinh".to_string(),
            enable_input_texture: false, // Sinh is generative fractal, no input needed
            enable_custom_uniform: true,
            entry_points: vec![
                "main".to_string(), // Single entry point 
            ],
            custom_storage_buffers: vec![], // No atomic buffers needed
            ..Default::default()
        };
        
        // Create compute shader with RenderKit integration
        let mut base = base;
        base.compute_shader = Some(ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/sinh.wgsl"),
            compute_config,
        ));
        
        // Enable hot reload using direct ComputeShader approach (before adding bindings)
        if let Some(compute_shader) = &mut base.compute_shader {
            let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Sinh Compute Shader Hot Reload"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/sinh.wgsl").into()),
            });
            if let Err(e) = compute_shader.enable_hot_reload(
                core.device.clone(),
                PathBuf::from("shaders/sinh.wgsl"),
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
        self.base.fps_tracker.update();
    }
    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        if let Some(compute_shader) = &mut self.base.compute_shader {
            compute_shader.resize(core, core.size.width, core.size.height);
        }
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
                
                egui::Window::new("Sinh")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Rendering")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.aa, 1..=4).text("AA")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.2..=1.1).text("Gamma")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.vignette_offset, 0.0..=1.0).text("Vignette")).changed();
                            });
                        
                        egui::CollapsingHeader::new("Camera")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.camera_x, -1.0..=1.0).text("X")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.camera_y, 5.0..=20.0).text("Y")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.camera_z, 5.0..=20.0).text("Z")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.orbit_speed, 0.0..=1.0).text("speed")).changed();
                            });
                        
                        egui::CollapsingHeader::new("Fractal")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.iterations, 10..=100).text("Iterations")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.bound, 1.0..=25.0).text("Bound")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.magic_number, 1.0..=100.0).text("Magic Number")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cv_min, 1.0..=3.0).text("CV Min")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cv_max, 2.0..=4.0).text("CV Max")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.os_base, 0.00001..=0.001).logarithmic(true).text("OS Base")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.os_scale, 0.001..=0.1).text("OS Scale")).changed();
                                ui.separator();
                                changed |= ui.add(egui::Slider::new(&mut params.fractal_scale, 0.01..=1.0).text("Fractal Scale")).changed();
                            });
                        
                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Base Color:");
                                    let mut color = [params.base_color_r, params.base_color_g, params.base_color_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.base_color_r = color[0];
                                        params.base_color_g = color[1];
                                        params.base_color_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Light Color:");
                                    let mut color = [params.light_color_r, params.light_color_g, params.light_color_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.light_color_r = color[0];
                                        params.light_color_g = color[1];
                                        params.light_color_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Ambient Color:");
                                    let mut color = [params.ambient_r, params.ambient_g, params.ambient_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.ambient_r = color[0];
                                        params.ambient_g = color[1];
                                        params.ambient_b = color[2];
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
        self.base.apply_control_request(controls_request);
        
        // Apply parameter changes (clean pattern)
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        // Update time for ComputeShader
        if let Some(compute_shader) = &mut self.base.compute_shader {
            let delta = 1.0/60.0;
            compute_shader.set_time(current_time, delta, &core.queue);
            
            // Check for hot reload updates
            compute_shader.check_hot_reload(&core.device);
            
            // Compute stage: Render fractal
            let width = core.size.width.div_ceil(16);
            let height = core.size.height.div_ceil(16);
            compute_shader.dispatch_stage(&mut encoder, 0, (width, height, 1), Some(&self.params_uniform.bind_group));
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
        
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        
        self.frame_count += 1;
        
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
    let (app, event_loop) = cuneus::ShaderApp::new("Sinh 3D", 800, 300);
    
    app.run(event_loop, |core| {
        SinhShader::init(core)
    })
}