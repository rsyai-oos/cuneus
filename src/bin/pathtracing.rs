
use cuneus::prelude::*;
use cuneus::compute::*;
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
            speed: 2.0,
            last_update: std::time::Instant::now(),
            
            yaw: 0.0,
            pitch: 0.0,
            mouse_sensitivity: 0.005,
            
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            mouse_initialized: false,
            mouse_look_enabled: true,
        }
    }
}

impl CameraMovement {
    fn update_camera(&mut self, params: &mut PathTracingParams) -> bool {
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
        
        let _up = [
            right[1] * forward[2] - right[2] * forward[1],
            right[2] * forward[0] - right[0] * forward[2],
            right[0] * forward[1] - right[1] * forward[0],
        ];
        
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
        
        let look_distance = 1.0;
        params.camera_target_x = params.camera_pos_x + forward[0] * look_distance;
        params.camera_target_y = params.camera_pos_y + forward[1] * look_distance;
        params.camera_target_z = params.camera_pos_z + forward[2] * look_distance;
        
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
struct PathTracingParams {
    camera_pos_x: f32,
    camera_pos_y: f32,
    camera_pos_z: f32,
    camera_target_x: f32,
    camera_target_y: f32,
    camera_target_z: f32,
    fov: f32,
    aperture: f32,
    
    max_bounces: u32,
    samples_per_pixel: u32,
    accumulate: u32,
    
    num_spheres: u32,
    mouse_x: f32,
    mouse_y: f32,
    
    rotation_speed: f32,
    
    exposure: f32,
}

impl UniformProvider for PathTracingParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct PathTracingShader {
    base: RenderKit,
    params_uniform: UniformBinding<PathTracingParams>,
    compute_shader: ComputeShader,
    camera_movement: CameraMovement,
    frame_count: u32,
    should_reset_accumulation: bool,
}

impl PathTracingShader {
    // Helper function to create pathtracing bind group (eliminates duplication)
    fn create_pathtracing_bind_group(&self, core: &Core, label: &str) -> Option<wgpu::BindGroup> {
        if let Some(ref texture_manager) = self.base.texture_manager {
            let output_view = self.compute_shader.get_output_texture().texture.create_view(&wgpu::TextureViewDescriptor::default());
            
            Some(core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: self.compute_shader.get_storage_layout(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&output_view), // Output storage texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&texture_manager.view), // Background texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&texture_manager.sampler), // Background sampler
                    },
                ],
            }))
        } else {
            None
        }
    }

    fn handle_export(&mut self, core: &Core) {
        if let Some((frame, time)) = self.base.export_manager.try_get_next_frame() {
            if let Ok(data) = self.capture_frame(core, time) {
                let settings = self.base.export_manager.settings();
                if let Err(e) = save_frame(data, frame, settings) {
                    eprintln!("Error saving frame: {:?}", e);
                }
            }
        } else {
            self.base.export_manager.complete_export();
        }
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
            let mut render_pass = Renderer::begin_render_pass(
                &mut encoder,
                &capture_view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Capture Pass"),
            );
            
            let compute_texture = self.compute_shader.get_output_texture();
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
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
}

impl ShaderManager for PathTracingShader {
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
            label: Some("Texture Bind Group Layout"),
        });

        let mut base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );

        base.setup_mouse_uniform(core);
        
        let params_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("params_bind_group_layout"),
        });
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "Path Tracing Params",
            PathTracingParams {
                camera_pos_x: 0.0,
                camera_pos_y: 1.0,
                camera_pos_z: 6.0,
                camera_target_x: 0.0,
                camera_target_y: 0.0,
                camera_target_z: -1.0,
                fov: 40.0,
                aperture: 0.00,
                max_bounces: 4,
                samples_per_pixel: 2,
                accumulate: 1,
                num_spheres: 15,
                mouse_x: 0.5,
                mouse_y: 0.5,
                rotation_speed: 0.2,
                exposure: 1.5,
            },
            &params_bind_group_layout,
            0,
        );
        
        let compute_config = ComputeShaderConfig {
            workgroup_size: [16, 16, 1],
            workgroup_count: None,
            dispatch_once: false,
            storage_texture_format: wgpu::TextureFormat::Rgba16Float,
            enable_atomic_buffer: true,
            atomic_buffer_multiples: 3,
            entry_points: vec!["main".to_string()],
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "Path Tracing".to_string(),
            mouse_bind_group_layout: Some(params_bind_group_layout.clone()),
            enable_fonts: false,
            enable_audio_buffer: false,
            audio_buffer_size: 0,
            enable_custom_uniform: true,
            enable_input_texture: true,
            custom_storage_buffers: Vec::new(),
        };
        
        let mut compute_shader = ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/pathtracing.wgsl"),
            compute_config,
        );
        
        compute_shader.add_custom_uniform_binding(&params_uniform.bind_group);
        
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Path Tracing Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/pathtracing.wgsl").into()),
        });
        if let Err(e) = compute_shader.enable_hot_reload(
            core.device.clone(),
            PathBuf::from("shaders/pathtracing.wgsl"),
            shader_module,
        ) {
            eprintln!("Failed to enable compute shader hot reload: {}", e);
        }

        // Create instance to use helper function
        let mut instance = Self {
            base,
            params_uniform,
            compute_shader,
            camera_movement: CameraMovement::default(),
            frame_count: 0,
            should_reset_accumulation: true,
        };
        
        // Create custom storage bind group for pathtracing (background texture + output texture)
        if let Some(bind_group) = instance.create_pathtracing_bind_group(core, "Path Tracing Storage Bind Group") {
            instance.compute_shader.override_storage_bind_group(bind_group);
        }
        
        instance
    }
    
    fn update(&mut self, core: &Core) {
        // Update video/webcam textures
        if self.base.using_video_texture {
            self.base.update_video_texture(core, &core.queue);
        } else if self.base.using_webcam_texture {
            self.base.update_webcam_texture(core, &core.queue);
        }
        
        // Always update ComputeShader input texture when using dynamic textures
        if self.base.using_video_texture {
            if let Some(ref video_manager) = self.base.video_texture_manager {
                let texture_manager = video_manager.texture_manager();
                self.compute_shader.update_input_texture(core, &texture_manager.view, &texture_manager.sampler);
            }
        } else if self.base.using_webcam_texture {
            if let Some(ref webcam_manager) = self.base.webcam_texture_manager {
                let texture_manager = webcam_manager.texture_manager();
                self.compute_shader.update_input_texture(core, &texture_manager.view, &texture_manager.sampler);
            }
        }
        
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
        
        if self.camera_movement.update_camera(&mut self.params_uniform.data) {
            self.params_uniform.update(&core.queue);
            self.should_reset_accumulation = true;
        }
        
        self.base.fps_tracker.update();
    }
    
    fn resize(&mut self, core: &Core) {
        self.compute_shader.resize(core, core.size.width, core.size.height);
        
        // Recreate pathtracing storage bind group after resize using helper function
        if let Some(bind_group) = self.create_pathtracing_bind_group(core, "Path Tracing Resized Storage Bind Group") {
            self.compute_shader.override_storage_bind_group(bind_group);
        }
        
        self.should_reset_accumulation = true;
    }
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Path Tracing Render Encoder"),
        });
        
        // Handle UI and parameter updates
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
        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();
        
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });

                egui::Window::new("Path Tracer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        ui.label("Camera Controls:");
                        ui.label("W/A/S/D - Movements");
                        ui.label("Q/E - down/up");
                        ui.label("Mouse - Look around");
                        ui.label("Right Click - Toggle mouse look");
                        ui.label("Space - Toggle progressive rendering");
                        ui.separator();
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
                        
                        egui::CollapsingHeader::new("Render Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                let old_samples = params.samples_per_pixel;
                                changed |= ui.add(egui::Slider::new(&mut params.samples_per_pixel, 1..=16).text("Samples/pixel")).changed();
                                if params.samples_per_pixel != old_samples {
                                    self.should_reset_accumulation = true;
                                }

                                let old_bounces = params.max_bounces;
                                changed |= ui.add(egui::Slider::new(&mut params.max_bounces, 1..=16).text("Max Bounces")).changed();
                                if params.max_bounces != old_bounces {
                                    self.should_reset_accumulation = true;
                                }

                                let old_accumulate = params.accumulate;
                                let mut accumulate_bool = params.accumulate > 0;
                                changed |= ui.checkbox(&mut accumulate_bool, "Progressive Rendering").changed();
                                params.accumulate = if accumulate_bool { 1 } else { 0 };
                                if params.accumulate != old_accumulate {
                                    self.should_reset_accumulation = true;
                                }

                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.1..=5.0).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.aperture, 0.0..=0.5).text("Depth of Field")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.rotation_speed, 0.0..=2.0).text("Animation Speed")).changed();

                                if ui.button("Reset Accumulation").clicked() {
                                    self.should_reset_accumulation = true;
                                    changed = true;
                                }
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        ui.separator();
                        ui.label(format!("Accumulated Samples: {}", self.frame_count));
                        ui.label(format!("Resolution: {}x{}", core.size.width, core.size.height));
                        ui.label(format!("FPS: {:.1}", current_fps));
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        // Handle control requests
        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers || self.should_reset_accumulation {
            self.compute_shader.clear_atomic_buffer(core);
            self.should_reset_accumulation = false;
            self.frame_count = 0;
        }
        let was_media_loaded = controls_request.load_media_path.is_some();
        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        self.base.handle_webcam_requests(core, &controls_request);
        if was_media_loaded || controls_request.start_webcam {
            if let Some(ref texture_manager) = self.base.texture_manager {
                self.compute_shader.update_input_texture(core, &texture_manager.view, &texture_manager.sampler);
            }
        }
        if self.base.handle_hdri_requests(core, &controls_request) {
            if let Some(ref texture_manager) = self.base.texture_manager {
                self.compute_shader.update_input_texture(core, &texture_manager.view, &texture_manager.sampler);
            }
        }
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        // Update mouse position in params
        self.base.update_mouse_uniform(&core.queue);
        if let Some(mouse_uniform) = &self.base.mouse_uniform {
            self.params_uniform.data.mouse_x = mouse_uniform.data.position[0];
            self.params_uniform.data.mouse_y = mouse_uniform.data.position[1];
            self.params_uniform.update(&core.queue);
        }
        
        // Update time and dispatch compute shader
        let delta = 1.0/60.0;
        self.compute_shader.set_time(current_time, delta, &core.queue);
        
        // Set frame count for random number generation (even when not accumulating)
        self.compute_shader.time_uniform.data.frame = self.frame_count;
        self.compute_shader.time_uniform.update(&core.queue);
        
        self.compute_shader.dispatch(&mut encoder, core);
        
        // Render compute output to screen
        {
            let mut render_pass = Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Path Tracing Display Pass"),
            );
            
            let compute_texture = self.compute_shader.get_output_texture();
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
        
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        
        // Increment frame count for progressive rendering and noise generation
        if self.params_uniform.data.accumulate > 0 {
            self.frame_count += 1;
        } else {
            // Still increment for noise generation when not accumulating
            self.frame_count = (self.frame_count + 1) % 1000; // Keep it reasonable to avoid overflow
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
                        self.should_reset_accumulation = true;
                        return true;
                    },
                    "s" | "S" => {
                        self.camera_movement.backward = event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    },
                    "a" | "A" => {
                        self.camera_movement.left = event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    },
                    "d" | "D" => {
                        self.camera_movement.right = event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    },
                    "q" | "Q" => {
                        self.camera_movement.down = event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    },
                    "e" | "E" => {
                        self.camera_movement.up = event.state == winit::event::ElementState::Pressed;
                        self.should_reset_accumulation = true;
                        return true;
                    },
                    " " => {
                        if event.state == winit::event::ElementState::Released {
                            self.params_uniform.data.accumulate = 1 - self.params_uniform.data.accumulate;
                            self.should_reset_accumulation = true;
                            self.params_uniform.update(&core.queue);
                            return true;
                        }
                    },
                    _ => {}
                }
            }
        }
        
        if let WindowEvent::CursorMoved { position, .. } = event {
            let x = position.x as f32;
            let y = position.y as f32;
            
            if self.camera_movement.handle_mouse_movement(x, y) {
                self.should_reset_accumulation = true;
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
        if let WindowEvent::DroppedFile(path) = event {
            if let Err(e) = self.base.load_media(core, path) {
                eprintln!("Failed to load dropped file: {:?}", e);
            } else {
            }
            return true;
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
    cuneus::gst::init()?;
    let (app, event_loop) = ShaderApp::new("Path Tracer", 800, 600);
    
    app.run(event_loop, |core| {
        PathTracingShader::init(core)
    })
}