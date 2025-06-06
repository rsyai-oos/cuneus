use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager};
use cuneus::compute::{create_bind_group_layout, BindGroupLayoutType};
use winit::event::WindowEvent;
use std::path::PathBuf;

struct CameraMovement {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    speed: f32,
    last_update: std::time::Instant,
    
    yaw: f32,
    pitch: f32,
    mouse_sensitivity: f32,
    
    last_mouse_x: f32,
    last_mouse_y: f32,
    mouse_initialized: bool,
    mouse_look_enabled: bool,
}

impl Default for CameraMovement {
    fn default() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            speed: 1.5,
            last_update: std::time::Instant::now(),
            
            yaw: 0.0,
            pitch: -0.3,
            mouse_sensitivity: 0.004,
            
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            mouse_initialized: false,
            mouse_look_enabled: true,
        }
    }
}

impl CameraMovement {
    fn update_camera(&mut self, params: &mut WaterParams) -> bool {
        let now = std::time::Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        self.last_update = now;
        
        let mut changed = false;
        
        let forward = [
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.sin(),
            self.pitch.cos() * self.yaw.sin(),
        ];
        
        let world_up = [0.0, 1.0, 0.0];
        let right = [
            forward[1] * world_up[2] - forward[2] * world_up[1],
            forward[2] * world_up[0] - forward[0] * world_up[2],
            forward[0] * world_up[1] - forward[1] * world_up[0],
        ];
        
        let right_len = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
        let right = [right[0] / right_len, right[1] / right_len, right[2] / right_len];
        
        let delta = self.speed * dt;
        let mut move_vec = [0.0, 0.0, 0.0];
        
        if self.forward {
            move_vec[0] += forward[0] * delta;
            move_vec[1] += forward[1] * delta;
            move_vec[2] += forward[2] * delta;
            changed = true;
        }
        if self.backward {
            move_vec[0] -= forward[0] * delta;
            move_vec[1] -= forward[1] * delta;
            move_vec[2] -= forward[2] * delta;
            changed = true;
        }
        if self.right {
            move_vec[0] += right[0] * delta;
            move_vec[1] += right[1] * delta;
            move_vec[2] += right[2] * delta;
            changed = true;
        }
        if self.left {
            move_vec[0] -= right[0] * delta;
            move_vec[1] -= right[1] * delta;
            move_vec[2] -= right[2] * delta;
            changed = true;
        }
        if self.up {
            move_vec[1] += delta;
            changed = true;
        }
        if self.down {
            move_vec[1] -= delta;
            changed = true;
        }
        
        params.camera_pos_x += move_vec[0];
        params.camera_pos_y += move_vec[1];
        params.camera_pos_z += move_vec[2];
        
        params.camera_yaw = self.yaw;
        params.camera_pitch = self.pitch;
        
        changed
    }
    
    fn handle_mouse_movement(&mut self, x: f32, y: f32) -> bool {
        if !self.mouse_look_enabled {
            return false;
        }
        
        if !self.mouse_initialized {
            self.last_mouse_x = x;
            self.last_mouse_y = y;
            self.mouse_initialized = true;
            return false;
        }
        
        let dx = x - self.last_mouse_x;
        let dy = y - self.last_mouse_y;
        
        self.last_mouse_x = x;
        self.last_mouse_y = y;
        
        self.yaw += dx * self.mouse_sensitivity;
        self.pitch -= dy * self.mouse_sensitivity;
        
        self.pitch = self.pitch.clamp(-std::f32::consts::PI * 0.49, std::f32::consts::PI * 0.49);
        
        true
    }
    
    fn toggle_mouse_look(&mut self) {
        self.mouse_look_enabled = !self.mouse_look_enabled;
        self.mouse_initialized = false;
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WaterParams {
    camera_pos_x: f32,
    camera_pos_y: f32,
    camera_pos_z: f32,
    camera_yaw: f32,
    camera_pitch: f32,
    
    water_depth: f32,
    drag_mult: f32,
    camera_height: f32,
    
    wave_iterations_raymarch: u32,
    wave_iterations_normal: u32,
    
    time_speed: f32,
    sun_speed: f32,
    
    mouse_x: f32,
    mouse_y: f32,
    
    atmosphere_intensity: f32,
    water_color_r: f32,
    water_color_g: f32,
    water_color_b: f32,
    
    sun_color_r: f32,
    sun_color_g: f32,
    sun_color_b: f32,
    
    cloud_coverage: f32,
    cloud_speed: f32,
    cloud_height: f32,
    
    night_sky_r: f32,
    night_sky_g: f32,
    night_sky_b: f32,
    
    exposure: f32,
    gamma: f32,
    
    fresnel_strength: f32,
    reflection_strength: f32,
}

impl UniformProvider for WaterParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct WaterShader {
    base: RenderKit,
    params_uniform: UniformBinding<WaterParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    
    compute_pipeline: wgpu::ComputePipeline,
    output_texture: cuneus::TextureManager,
    
    compute_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    
    compute_bind_group: wgpu::BindGroup,
    frame_count: u32,
    hot_reload: cuneus::ShaderHotReload,
    camera_movement: CameraMovement,
    
    export_time: Option<f32>,
    export_frame: Option<u32>,
}

impl WaterShader {
    fn recreate_compute_resources(&mut self, core: &Core) {
        self.output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            wgpu::TextureFormat::Rgba16Float,
            &self.base.texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Arctic Water Output Texture",
        );
        
        let view_output = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Arctic Water Compute Bind Group"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_output),
                },
            ],
        });
    }
    
    fn capture_frame(&mut self, core: &Core, time: f32, frame: u32) -> Result<Vec<u8>, wgpu::SurfaceError> {
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
        self.base.time_uniform.data.frame = frame;
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
            render_pass.set_bind_group(0, &self.output_texture.bind_group, &[]);
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
        
        core.device.poll(wgpu::Maintain::Wait);
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
            // Store export timing to freeze time during export
            self.export_time = Some(time);
            self.export_frame = Some(frame);
            
            if let Ok(data) = self.capture_frame(core, time, frame) {
                let settings = self.base.export_manager.settings();
                if let Err(e) = cuneus::save_frame(data, frame, settings) {
                    eprintln!("Error saving frame: {:?}", e);
                }
            }
        } else {
            // Clear export timing when done
            self.export_time = None;
            self.export_frame = None;
            self.base.export_manager.complete_export();
        }
    }
}

impl ShaderManager for WaterShader {
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
        
        let time_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::TimeUniform, 
            "Arctic Water Compute"
        );
        
        let params_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::CustomUniform, 
            "Arctic Water Params"
        );
        
        let compute_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
            label: Some("arctic_water_compute_output_layout"),
        });
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "Arctic Water Params",
            WaterParams {
                camera_pos_x: 0.0,
                camera_pos_y: 2.5,  
                camera_pos_z: 3.0,
                camera_yaw: 0.0,
                camera_pitch: -0.3,
                
                water_depth: 2.0,
                drag_mult: 0.3, 
                camera_height: 2.5,
                
                wave_iterations_raymarch: 14,
                wave_iterations_normal: 32,
                
                time_speed: 0.6,
                sun_speed: 0.05,
                
                mouse_x: 0.5,
                mouse_y: 0.5,
                
                atmosphere_intensity: 1.4,
                water_color_r: 0.02, 
                water_color_g: 0.04,
                water_color_b: 0.08,
                
                sun_color_r: 1.0,
                sun_color_g: 0.85,
                sun_color_b: 0.75,
                
                cloud_coverage: 0.4,
                cloud_speed: 0.15, 
                cloud_height: 30.0, 
                
                night_sky_r: 0.008,
                night_sky_g: 0.012,
                night_sky_b: 0.025,
                
                exposure: 1.8,
                gamma: 0.4,
                
                fresnel_strength: 1.2,
                reflection_strength: 0.9,
            },
            &params_bind_group_layout,
            0,
        );
        
        let compute_time_uniform = UniformBinding::new(
            &core.device,
            "Arctic Compute Time Uniform",
            cuneus::compute::ComputeTimeUniform {
                time: 0.0,
                delta: 0.0,
                frame: 0,
                _padding: 0,
            },
            &time_bind_group_layout,
            0,
        );
        
        let shader_source = std::fs::read_to_string("shaders/water.wgsl")
            .unwrap_or_else(|_| include_str!("../../shaders/water.wgsl").to_string());
        
        let cs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Arctic Water Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let hot_reload = cuneus::ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shaders/water.wgsl"),
            cs_module.clone(),
            "main",
        ).expect("Failed to initialize hot reload");
        
        let mut base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        base.setup_mouse_uniform(core);
        
        let output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.config.width,
            core.config.height,
            wgpu::TextureFormat::Rgba16Float,
            &texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Arctic Water Output Texture",
        );
        
        let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Arctic Water Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,
                &params_bind_group_layout,
                &compute_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Arctic Water Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let view_output = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_output),
                },
            ],
            label: Some("Arctic Water Compute Bind Group"),
        });
        
        let mut result = Self {
            base,
            params_uniform,
            compute_time_uniform,
            compute_pipeline,
            output_texture,
            compute_bind_group_layout,
            time_bind_group_layout,
            params_bind_group_layout,
            compute_bind_group,
            frame_count: 0,
            hot_reload,
            camera_movement: CameraMovement::default(),
            export_time: None,
            export_frame: None,
        };
        
        result.recreate_compute_resources(core);
        result
    }
    
    fn update(&mut self, core: &Core) {
        if let Some(new_shader) = self.hot_reload.reload_compute_shader() {
            println!("Reloading Arctic Water shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Arctic Water Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.compute_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            
            self.compute_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Arctic Water Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        }
        
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
        
        if self.camera_movement.update_camera(&mut self.params_uniform.data) {
            self.params_uniform.update(&core.queue);
        }
        
        self.base.fps_tracker.update();
    }
    
    fn resize(&mut self, core: &Core) {
        println!("Resizing Arctic water to {:?}", core.size);
        self.recreate_compute_resources(core);
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
        
        let current_fps = self.base.fps_tracker.fps();
        
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 200);
                });

                egui::Window::new("Arctic Night Ocean")
                    .collapsible(true)
                    .resizable(false)
                    .default_width(340.0)
                    .show(ctx, |ui| {
                        ui.separator();
                        ui.label("Controls:");
                        ui.label("W/A/S/D&Q/E, mouse");
                        ui.label("Right Click - Toggle mouse look");
                        ui.separator();
                        
                        egui::CollapsingHeader::new("Water")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.water_depth, 1.0..=5.0).text("Depth")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.drag_mult, 0.1..=0.8).text("Wave Calmness")).changed();
                                
                                changed |= ui.add(egui::Slider::new(&mut params.wave_iterations_raymarch, 8..=24).text("Water Quality")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.wave_iterations_normal, 16..=64).text("Surface Detail")).changed();
                                
                                ui.horizontal(|ui| {
                                    ui.label("Water Color:");
                                    let mut color = [params.water_color_r, params.water_color_g, params.water_color_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.water_color_r = color[0];
                                        params.water_color_g = color[1];
                                        params.water_color_b = color[2];
                                        changed = true;
                                    }
                                });
                                changed |= ui.add(egui::Slider::new(&mut params.fresnel_strength, 0.5..=2.0).text("Ice Clarity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.reflection_strength, 0.3..=1.5).text("Reflections")).changed();
                            });

                        egui::CollapsingHeader::new("Sky")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.time_speed, 0.1..=2.0).text("Time Flow")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.sun_speed, 0.01..=0.2).text("Moon Movement")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.atmosphere_intensity, 0.8..=2.5).text("Sky Brightness")).changed();
                                
                                ui.horizontal(|ui| {
                                    ui.label("Moonlight:");
                                    let mut color = [params.sun_color_r, params.sun_color_g, params.sun_color_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.sun_color_r = color[0];
                                        params.sun_color_g = color[1];
                                        params.sun_color_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Sky:");
                                    let mut color = [params.night_sky_r, params.night_sky_g, params.night_sky_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.night_sky_r = color[0];
                                        params.night_sky_g = color[1];
                                        params.night_sky_b = color[2];
                                        changed = true;
                                    }
                                });
                            });

                        egui::CollapsingHeader::new("Aurora Borealis")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.cloud_coverage, 0.0..=1.0).text("Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cloud_speed, 0.05..=0.5).text("Animation Speed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cloud_height, 1.0..=150.0).text("Aurora qual")).changed();
                            });

                        egui::CollapsingHeader::new("Vis Settings")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.1..=3.5).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=2.8).text("Gamma")).changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        ui.separator();
                        ui.label(format!("Resolution: {}x{}", core.size.width, core.size.height));
                        ui.label(format!("FPS: {:.1}", current_fps));
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        self.base.export_manager.apply_ui_request(export_request);
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
        
        self.compute_time_uniform.data.time = current_time;
        self.compute_time_uniform.data.delta = 1.0/60.0;
        self.compute_time_uniform.data.frame = current_frame;
        self.compute_time_uniform.update(&core.queue);
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        self.base.update_mouse_uniform(&core.queue);
        if let Some(mouse_uniform) = &self.base.mouse_uniform {
            self.params_uniform.data.mouse_x = mouse_uniform.data.position[0];
            self.params_uniform.data.mouse_y = mouse_uniform.data.position[1];
            self.params_uniform.update(&core.queue);
        }
        
        // Run the compute shader pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Arctic Water Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            
            let width = core.size.width.div_ceil(8);
            let height = core.size.height.div_ceil(8);
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        // Render the compute output to the screen
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Display Pass"),
            );
            
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &self.output_texture.bind_group, &[]);
            
            render_pass.draw(0..4, 0..1);
        }
        
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        
        if self.export_time.is_none() {
            self.frame_count += 1;
        }
        
        Ok(())
    }
    
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
            return true;
        }
        
        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let winit::keyboard::Key::Character(ch) = &event.logical_key {
                match ch.as_str() {
                    "w" | "W" => {
                        self.camera_movement.forward = event.state == winit::event::ElementState::Pressed;
                        return true;
                    },
                    "s" | "S" => {
                        self.camera_movement.backward = event.state == winit::event::ElementState::Pressed;
                        return true;
                    },
                    "a" | "A" => {
                        self.camera_movement.left = event.state == winit::event::ElementState::Pressed;
                        return true;
                    },
                    "d" | "D" => {
                        self.camera_movement.right = event.state == winit::event::ElementState::Pressed;
                        return true;
                    },
                    "q" | "Q" => {
                        self.camera_movement.down = event.state == winit::event::ElementState::Pressed;
                        return true;
                    },
                    "e" | "E" => {
                        self.camera_movement.up = event.state == winit::event::ElementState::Pressed;
                        return true;
                    },
                    _ => {}
                }
            }
        }
        
        if let WindowEvent::CursorMoved { position, .. } = event {
            let x = position.x as f32;
            let y = position.y as f32;
            
            if self.camera_movement.handle_mouse_movement(x, y) {
                return true;
            }
        }
        
        if let WindowEvent::MouseInput { state, button, .. } = event {
            if *button == winit::event::MouseButton::Right {
                if *state == winit::event::ElementState::Released {
                    self.camera_movement.toggle_mouse_look();
                    return true;
                }
            }
        }
        
        if let WindowEvent::KeyboardInput { event, .. } = event {
            if self.base.key_handler.handle_keyboard_input(core.window(), event) {
                return true;
            }
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Arctic Night Ocean", 800, 800);
    
    app.run(event_loop, |core| {
        WaterShader::init(core)
    })
}