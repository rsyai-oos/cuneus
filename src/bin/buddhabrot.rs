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
    compute_shader: ComputeShader,
    frame_count: u32,
    accumulated_rendering: bool,
    current_params: BuddhabrotParams,
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
            let compute_texture = self.compute_shader.get_output_texture();
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
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
        // Clear atomic buffer (by recreating it)
        self.compute_shader.clear_atomic_buffer(core);
        
        self.compute_shader.current_frame = 0;
        self.frame_count = 0;
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
        
        let initial_params = BuddhabrotParams {
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
        };
        
        let base = RenderKit::new(core, include_str!("../../shaders/vertex.wgsl"), include_str!("../../shaders/blit.wgsl"), &[&texture_bind_group_layout], None);

        let mut config = ComputeShader::builder()
            .with_entry_point("Splat")
            .with_custom_uniforms::<BuddhabrotParams>()
            .with_atomic_buffer()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("Buddhabrot Unified")
            .build();
            
        // Add second entry point 
        config.entry_points.push("main_image".to_string());
            
        let mut compute_shader = ComputeShader::from_builder(
            core,
            include_str!("../../shaders/buddhabrot.wgsl"),
            config,
        );

        // Enable hot reload
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("shaders/buddhabrot.wgsl"),
            core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Buddhabrot Hot Reload"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/buddhabrot.wgsl").into()),
            }),
        ) {
            eprintln!("Failed to enable hot reload for buddhabrot shader: {}", e);
        }
        
        compute_shader.set_custom_params(initial_params, &core.queue);

        Self {
            base,
            compute_shader,
            frame_count: 0,
            accumulated_rendering: false,
            current_params: initial_params,
        }
    }
    
    fn update(&mut self, core: &Core) {
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
        self.base.fps_tracker.update();
        
        // Check for hot reload updates
        self.compute_shader.check_hot_reload(&core.device);
    }
    
    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.compute_shader.resize(core, core.size.width, core.size.height);
    }
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        let mut params = self.current_params;
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
        
        let delta = 1.0 / 60.0;
        self.compute_shader.set_time(current_time, delta, &core.queue);
        
        if changed {
            self.current_params = params;
            self.compute_shader.set_custom_params(params, &core.queue);
            
            // Clear buffers when parameters change (unless in accumulated mode)
            if !self.accumulated_rendering {
                self.clear_buffers(core);
            }
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        // Only generate new samples if we're not in accumulated mode
        // or if we're still accumulating (frame count < 500) - use frame counter
        let should_generate_samples = !self.accumulated_rendering || self.compute_shader.current_frame < 500;
        
        if should_generate_samples {
            self.compute_shader.dispatch_stage_with_workgroups(&mut encoder, 0, [2048, 1, 1]);
        }
        
        // Always dispatch stage 1 (main_image) for rendering with screen-based workgroups
        // Note: in cuneus, individual stage dispatch methods need manual frame management (if you need of course!)

        self.compute_shader.dispatch_stage(&mut encoder, core, 1);
        
        //Manual frame increment since dispatch_stage() doesn't auto-increment
        self.compute_shader.current_frame += 1;
        
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Display Pass"),
            );
            
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            let compute_texture = self.compute_shader.get_output_texture();
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
            
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