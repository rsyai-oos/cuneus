use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NebulaParams {
    iterations: i32,
    formuparam: f32,
    volsteps: i32,
    stepsize: f32,
    zoom: f32,
    tile: f32,
    speed: f32,
    brightness: f32,
    dust_intensity: f32,
    distfading: f32,
    color_variation: f32,
    n_boxes: f32,
    rotation: i32,
    depth: f32,
    color_mode: i32,
    _padding1: f32,
    
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    scale: f32,
    
    dof_amount: f32,
    dof_focal_dist: f32,
    exposure: f32,
    gamma: f32,
    
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
    _padding8: f32,
    _padding9: f32,
    
    time_scale: f32,
    
    spiral_mode: i32,
    spiral_strength: f32,
    spiral_speed: f32,
    visual_mode: i32,
    _padding2: f32,
    _padding3: f32,
}

impl UniformProvider for NebulaParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct NebulaShader {
    base: RenderKit,
    params_uniform: UniformBinding<NebulaParams>,
    compute_shader: ComputeShader,
    frame_count: u32,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("universe", 800, 600);
    app.run(event_loop, |core| {
        NebulaShader::init(core)
    })
}
impl NebulaShader {
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
        self.base.resolution_uniform.data.dimensions = [settings.width as f32, settings.height as f32];
        self.base.resolution_uniform.update(&core.queue);

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
impl ShaderManager for NebulaShader {
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
            "Nebula Params Uniform",
            NebulaParams {
                iterations: 21,
                formuparam: 0.55,
                volsteps: 10,
                stepsize: 0.21,
                zoom: 1.85,
                tile: 0.8,
                speed: 0.020,
                brightness: 0.00062,
                dust_intensity: 0.3,
                distfading: 1.0,
                color_variation: 1.5,
                n_boxes: 10.0,
                rotation: 1,
                depth: 5.0,
                color_mode: 1,
                _padding1: 0.0,
                
                rotation_x: 0.0,
                rotation_y: 0.0,
                click_state: 0,
                scale: 1.0,
                
                dof_amount: 0.0,
                dof_focal_dist: 0.5,
                exposure: 1.65,
                gamma: 0.400,
                
                _padding4: 0.0,
                _padding5: 0.0,
                _padding6: 0.0,
                _padding7: 0.0,
                _padding8: 0.0,
                _padding9: 0.0,
                
                time_scale: 1.0,
                
                spiral_mode: 0,
                spiral_strength: 2.0,
                spiral_speed: 0.02,
                visual_mode: 0,
                _padding2: 0.0,
                _padding3: 0.0,
            },
            &create_bind_group_layout(&core.device, BindGroupLayoutType::CustomUniform, "Nebula Params"),
            0,
        );

        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );

        let compute_config = ComputeShaderConfig {
            workgroup_size: [16, 16, 1],
            workgroup_count: None,
            dispatch_once: false,
            storage_texture_format: COMPUTE_TEXTURE_FORMAT_RGBA16,
            enable_atomic_buffer: true,
            atomic_buffer_multiples: 2,
            entry_points: vec!["volumetric_render".to_string(), "main_image".to_string()],
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "Nebula".to_string(),
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
            include_str!("../../shaders/nebula.wgsl"),
            compute_config,
        );

        // Enable hot reload
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nebula Compute Shader Hot Reload"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/nebula.wgsl").into()),
        });
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            std::path::PathBuf::from("shaders/nebula.wgsl"),
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
                
                egui::Window::new("universe")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(320.0)
                    .show(ctx, |ui| {
                        
                        egui::CollapsingHeader::new("Volumetric Parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.iterations, 5..=30).text("Iterations")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.formuparam, 0.1..=1.0).text("Form Parameter")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.volsteps, 1..=20).text("Volume Steps")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.stepsize, 0.05..=0.5).text("Step Size")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.1..=112.0).text("Zoom")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.tile, 0.1..=2.0).text("Tile")).changed();
                            });

                        egui::CollapsingHeader::new("Appearance")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.brightness, 0.0005..=0.015).logarithmic(true).text("Brightness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dust_intensity, 0.0..=1.0).text("Dust Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.distfading, 0.1..=3.0).text("Distance Fading")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.color_variation, 0.5..=5.0).text("Color Variation")).changed();
                                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.5..=3.0).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=1.2).text("Gamma")).changed();
                            });

                        egui::CollapsingHeader::new("Visual Modes")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(params.spiral_mode == 0, "Normal").clicked() {
                                        params.spiral_mode = 0;
                                        changed = true;
                                    }
                                    if ui.selectable_label(params.spiral_mode == 1, "Spiral").clicked() {
                                        params.spiral_mode = 1;
                                        changed = true;
                                    }
                                    if ui.selectable_label(params.spiral_mode == 2, "Hole").clicked() {
                                        params.spiral_mode = 2;
                                        changed = true;
                                    }
                                    if ui.selectable_label(params.spiral_mode == 3, "Tunnel").clicked() {
                                        params.spiral_mode = 3;
                                        changed = true;
                                    }
                                });
                                
                                if params.spiral_mode != 0 {
                                    ui.separator();
                                    changed |= ui.add(egui::Slider::new(&mut params.spiral_strength, 0.5..=4.0).text("Spiral Strength")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.spiral_speed, -0.1..=0.1).text("Spiral Speed")).changed();
                                }
                                
                            });

                        egui::CollapsingHeader::new("Depth of Field")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.dof_amount, 0.0..=3.0).text("DOF Amount")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dof_focal_dist, 0.0..=2.0).text("Focal Distance")).changed();
                            });


                        egui::CollapsingHeader::new("Animation")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.speed, -0.1..=0.1).text("Galaxy Speed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.time_scale, 0.1..=2.0).text("Animation Speed")).changed();
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
            self.frame_count = 0;
        }
        self.base.apply_control_request(controls_request);
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = self.frame_count;
        self.base.time_uniform.update(&core.queue);
        
        // Update compute shader with the same time data
        self.compute_shader.set_time(current_time, 1.0/60.0, &core.queue);
        self.compute_shader.time_uniform.data.frame = self.frame_count;
        self.compute_shader.time_uniform.update(&core.queue);
        
        // Mouse data is read from tracker and passed through custom uniform parameters
        if self.base.mouse_tracker.uniform.buttons[0] & 1 != 0 {
            params.rotation_x = self.base.mouse_tracker.uniform.position[0];
            params.rotation_y = self.base.mouse_tracker.uniform.position[1];
            params.click_state = 1;
            changed = true;
        } else {
            params.click_state = 0;
        }
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Use ComputeShader dispatch (handles both volumetric_render and main_image passes)
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
}