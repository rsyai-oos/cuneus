use cuneus::prelude::*;
use winit::event::WindowEvent;
use std::path::PathBuf;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MandelbulbParams {
    mouse_x: f32,
    mouse_y: f32,
    power: f32,
    max_bounces: u32,
    samples_per_pixel: u32,
    accumulate: u32,
    
    animation_speed: f32,
    hold_duration: f32,
    transition_duration: f32,
    
    exposure: f32,
    focal_length: f32,
    dof_strength: f32,
    
    palette_a_r: f32,
    palette_a_g: f32,
    palette_a_b: f32,
    palette_b_r: f32,
    palette_b_g: f32,
    palette_b_b: f32,
    palette_c_r: f32,
    palette_c_g: f32,
    palette_c_b: f32,
    palette_d_r: f32,
    palette_d_g: f32,
    palette_d_b: f32,
    
    manual_rotation_x: f32,
    manual_rotation_y: f32,
    manual_rotation_z: f32,
    use_mouse_rotation: u32,
    
    gamma: f32,
    zoom: f32,
    
    background_r: f32,
    background_g: f32,
    background_b: f32,
    sun_color_r: f32,
    sun_color_g: f32,
    sun_color_b: f32,
    fog_color_r: f32,
    fog_color_g: f32,
    fog_color_b: f32,
    glow_color_r: f32,
    glow_color_g: f32,
    glow_color_b: f32,
}

impl UniformProvider for MandelbulbParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct MouseRotation {
    last_mouse_x: f32,
    last_mouse_y: f32,
    mouse_initialized: bool,
    mouse_enabled: bool,
    rotation_x: f32,
    rotation_y: f32,
    mouse_sensitivity: f32,
}

impl Default for MouseRotation {
    fn default() -> Self {
        Self {
            last_mouse_x: 0.0,
            last_mouse_y: 0.0,
            mouse_initialized: false,
            mouse_enabled: true,
            rotation_x: 0.0,
            rotation_y: 0.0,
            mouse_sensitivity: 0.005,
        }
    }
}

impl MouseRotation {
    fn handle_mouse_movement(&mut self, x: f32, y: f32) -> bool {
        if !self.mouse_enabled {
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
        
        self.rotation_y += dx * self.mouse_sensitivity;
        self.rotation_x += dy * self.mouse_sensitivity;
        
        self.rotation_x = self.rotation_x.clamp(-std::f32::consts::PI, std::f32::consts::PI);
        
        true
    }
    
    fn toggle_mouse_rotation(&mut self) {
        self.mouse_enabled = !self.mouse_enabled;
        self.mouse_initialized = false;
    }
    
    fn get_normalized_position(&self, window_size: &winit::dpi::PhysicalSize<u32>) -> (f32, f32) {
        let norm_x = (self.last_mouse_x / window_size.width as f32).clamp(0.0, 1.0);
        let norm_y = (self.last_mouse_y / window_size.height as f32).clamp(0.0, 1.0);
        (norm_x, norm_y)
    }
}

struct MandelbulbShader {
    base: RenderKit,
    params_uniform: UniformBinding<MandelbulbParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    
    compute_pipeline: wgpu::ComputePipeline,
    
    output_texture: cuneus::TextureManager,
    
    compute_bind_group_layout: wgpu::BindGroupLayout,
    atomic_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    
    compute_bind_group: wgpu::BindGroup,
    
    atomic_buffer: cuneus::AtomicBuffer,
    
    frame_count: u32,
    
    hot_reload: cuneus::ShaderHotReload,
    
    should_reset_accumulation: bool,
    
    mouse_rotation: MouseRotation,
}

impl MandelbulbShader {
    fn recreate_compute_resources(&mut self, core: &Core) {
        self.output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            wgpu::TextureFormat::Rgba16Float,
            &self.base.texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Mandelbulb Output Texture",
        );

        let buffer_size = core.size.width * core.size.height * 3;
        
        self.atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            buffer_size,
            &self.atomic_bind_group_layout,
        );
        
        let view_output = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mandelbulb Compute Bind Group"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_output),
                },
            ],
        });
        
        self.should_reset_accumulation = true;
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
    
    fn clear_atomic_buffer(&mut self, core: &Core) {
        let buffer_size = core.size.width * core.size.height * 3;
        let clear_data = vec![0u32; buffer_size as usize];
        
        core.queue.write_buffer(
            &self.atomic_buffer.buffer,
            0,
            bytemuck::cast_slice(&clear_data),
        );
        
        self.should_reset_accumulation = false;
        self.frame_count = 0;
    }
}

impl ShaderManager for MandelbulbShader {
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
            "Mandelbulb Compute"
        );
        
        let params_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::CustomUniform, 
            "Mandelbulb Params"
        );
        
        let atomic_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::AtomicBuffer, 
            "Mandelbulb Compute"
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
            label: Some("mandelbulb_compute_output_layout"),
        });
        
        let buffer_size = core.config.width * core.config.height * 3;
        
        let atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            buffer_size,
            &atomic_bind_group_layout,
        );
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "Mandelbulb Params",
            MandelbulbParams {
                mouse_x: 0.5,
                mouse_y: 0.5,
                power: 8.0,
                max_bounces: 6,
                samples_per_pixel: 2,
                accumulate: 1,
                
                animation_speed: 1.0,
                hold_duration: 3.0,
                transition_duration: 3.0,
                
                exposure: 1.5,
                focal_length: 6.0,
                dof_strength: 0.02,
                
                palette_a_r: 0.5, palette_a_g: 0.5, palette_a_b: 0.5,
                palette_b_r: 0.5, palette_b_g: 0.1, palette_b_b: 0.1,
                palette_c_r: 1.0, palette_c_g: 1.0, palette_c_b: 1.0,
                palette_d_r: 0.0, palette_d_g: 0.33, palette_d_b: 0.67,
                
                manual_rotation_x: 0.0,
                manual_rotation_y: 0.0,
                manual_rotation_z: 0.0,
                use_mouse_rotation: 1,
                
                gamma: 1.1,
                zoom: 1.0,
                
                background_r: 0.1, background_g: 0.1, background_b: 0.15,
                sun_color_r: 8.10, sun_color_g: 6.00, sun_color_b: 4.20,
                fog_color_r: 0.1, fog_color_g: 0.1, fog_color_b: 0.15,
                glow_color_r: 0.5, glow_color_g: 0.7, glow_color_b: 1.0,
                

            },
            &params_bind_group_layout,
            0,
        );
        
        let compute_time_uniform = UniformBinding::new(
            &core.device,
            "Compute Time Uniform",
            cuneus::compute::ComputeTimeUniform {
                time: 0.0,
                delta: 0.0,
                frame: 0,
                _padding: 0,
            },
            &time_bind_group_layout,
            0,
        );
        
        let shader_source = std::fs::read_to_string("shaders/mandelbulb.wgsl")
            .unwrap_or_else(|_| include_str!("../../shaders/mandelbulb.wgsl").to_string());
        
        let cs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mandelbulb Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let hot_reload = cuneus::ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shaders/mandelbulb.wgsl"),
            cs_module.clone(),
            "main",
        ).expect("Failed to initialize hot reload");
        
        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        let output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.config.width,
            core.config.height,
            wgpu::TextureFormat::Rgba16Float,
            &texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Mandelbulb Output Texture",
        );
        
        let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mandelbulb Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,
                &params_bind_group_layout,
                &compute_bind_group_layout,
                &atomic_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Mandelbulb Compute Pipeline"),
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
            label: Some("Mandelbulb Compute Bind Group"),
        });
        
        let mut result = Self {
            base,
            params_uniform,
            compute_time_uniform,
            compute_pipeline,
            output_texture,
            compute_bind_group_layout,
            atomic_bind_group_layout,
            time_bind_group_layout,
            params_bind_group_layout,
            compute_bind_group,
            atomic_buffer,
            frame_count: 0,
            hot_reload,
            should_reset_accumulation: true,
            mouse_rotation: MouseRotation::default(),
        };
        
        result.recreate_compute_resources(core);
        
        result
    }
    
    fn update(&mut self, core: &Core) {
        if let Some(new_shader) = self.hot_reload.reload_compute_shader() {
            println!("Reloading Mandelbulb shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Mandelbulb Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.compute_bind_group_layout,
                    &self.atomic_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            
            self.compute_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Mandelbulb Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.should_reset_accumulation = true;
        }
        
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
        
        self.base.fps_tracker.update();
    }
    
    fn resize(&mut self, core: &Core) {
        println!("Resizing to {:?}", core.size);
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
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });

                egui::Window::new("Mandelbulb PathTracer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(350.0)
                    .show(ctx, |ui| {
                        ui.label("Left Click + Drag - Rotate bulb");
                        ui.label("Right: - Toggle mouse");
                        ui.separator();
                        
                        egui::CollapsingHeader::new("Camera&View")
                            .default_open(false)
                            
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.1..=5.0).text("Zoom")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.focal_length, 2.0..=20.0).text("Focal Length")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dof_strength, 0.0..=1.0).text("DoF")).changed();
                                
                                ui.separator();
                                ui.label("Rotation:");
                                let mut use_mouse = params.use_mouse_rotation > 0;
                                if ui.checkbox(&mut use_mouse, "Use Mouse Rotation").changed() {
                                    params.use_mouse_rotation = if use_mouse { 1 } else { 0 };
                                    changed = true;
                                    self.should_reset_accumulation = true;
                                }
                                
                                if !use_mouse {
                                    ui.label("Manual");
                                    changed |= ui.add(egui::Slider::new(&mut params.manual_rotation_x, -std::f32::consts::PI..=std::f32::consts::PI).text("X")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.manual_rotation_y, -std::f32::consts::PI..=std::f32::consts::PI).text("Y")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.manual_rotation_z, -std::f32::consts::PI..=std::f32::consts::PI).text("Z")).changed();
                                }
                            });
                        
                        egui::CollapsingHeader::new("Mandelbulb")
                            .default_open(false)
                            .show(ui, |ui| {
                                let old_power = params.power;
                                changed |= ui.add(egui::Slider::new(&mut params.power, 2.0..=12.0).text("Power")).changed();
                                if params.power != old_power {
                                    self.should_reset_accumulation = true;
                                }
                            });
                        
                        egui::CollapsingHeader::new("Render")
                            .default_open(false)
                            .show(ui, |ui| {
                                let old_samples = params.samples_per_pixel;
                                changed |= ui.add(egui::Slider::new(&mut params.samples_per_pixel, 1..=8).text("Samples/pixel")).changed();
                                if params.samples_per_pixel != old_samples {
                                    self.should_reset_accumulation = true;
                                }

                                let old_bounces = params.max_bounces;
                                changed |= ui.add(egui::Slider::new(&mut params.max_bounces, 1..=12).text("Max Bounces")).changed();
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
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=2.0).text("Gamma")).changed();

                                if ui.button("Reset Accumulation").clicked() {
                                    self.should_reset_accumulation = true;
                                    changed = true;
                                }
                            });
                        
                        egui::CollapsingHeader::new("env")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("bg:");
                                    let mut bg_color = [params.background_r, params.background_g, params.background_b];
                                    if ui.color_edit_button_rgb(&mut bg_color).changed() {
                                        params.background_r = bg_color[0];
                                        params.background_g = bg_color[1];
                                        params.background_b = bg_color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Sun:");
                                    let mut sun_color = [params.sun_color_r, params.sun_color_g, params.sun_color_b];
                                    if ui.color_edit_button_rgb(&mut sun_color).changed() {
                                        params.sun_color_r = sun_color[0];
                                        params.sun_color_g = sun_color[1];
                                        params.sun_color_b = sun_color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Fog:");
                                    let mut fog_color = [params.fog_color_r, params.fog_color_g, params.fog_color_b];
                                    if ui.color_edit_button_rgb(&mut fog_color).changed() {
                                        params.fog_color_r = fog_color[0];
                                        params.fog_color_g = fog_color[1];
                                        params.fog_color_b = fog_color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Sky Glow:");
                                    let mut glow_color = [params.glow_color_r, params.glow_color_g, params.glow_color_b];
                                    if ui.color_edit_button_rgb(&mut glow_color).changed() {
                                        params.glow_color_r = glow_color[0];
                                        params.glow_color_g = glow_color[1];
                                        params.glow_color_b = glow_color[2];
                                        changed = true;
                                    }
                                });
                                
                                if ui.button("Reset env cols").clicked() {
                                    params.background_r = 0.1; params.background_g = 0.1; params.background_b = 0.15;
                                    params.sun_color_r = 8.10; params.sun_color_g = 6.00; params.sun_color_b = 4.20;
                                    params.fog_color_r = 0.1; params.fog_color_g = 0.1; params.fog_color_b = 0.15;
                                    params.glow_color_r = 0.5; params.glow_color_g = 0.7; params.glow_color_b = 1.0;
                                    changed = true;
                                }
                            });

                        egui::CollapsingHeader::new("Color Palette")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Base Color:");
                                    let mut color_a = [params.palette_a_r, params.palette_a_g, params.palette_a_b];
                                    if ui.color_edit_button_rgb(&mut color_a).changed() {
                                        params.palette_a_r = color_a[0];
                                        params.palette_a_g = color_a[1];
                                        params.palette_a_b = color_a[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Amplitude:");
                                    let mut color_b = [params.palette_b_r, params.palette_b_g, params.palette_b_b];
                                    if ui.color_edit_button_rgb(&mut color_b).changed() {
                                        params.palette_b_r = color_b[0];
                                        params.palette_b_g = color_b[1];
                                        params.palette_b_b = color_b[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Frequency:");
                                    let mut color_c = [params.palette_c_r, params.palette_c_g, params.palette_c_b];
                                    if ui.color_edit_button_rgb(&mut color_c).changed() {
                                        params.palette_c_r = color_c[0];
                                        params.palette_c_g = color_c[1];
                                        params.palette_c_b = color_c[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Phase:");
                                    let mut color_d = [params.palette_d_r, params.palette_d_g, params.palette_d_b];
                                    if ui.color_edit_button_rgb(&mut color_d).changed() {
                                        params.palette_d_r = color_d[0];
                                        params.palette_d_g = color_d[1];
                                        params.palette_d_b = color_d[2];
                                        changed = true;
                                    }
                                });
                                if ui.button("Reset to Default Palette").clicked() {
                                    params.palette_a_r = 0.5; params.palette_a_g = 0.5; params.palette_a_b = 0.5;
                                    params.palette_b_r = 0.5; params.palette_b_g = 0.1; params.palette_b_b = 0.1;
                                    params.palette_c_r = 1.0; params.palette_c_g = 1.0; params.palette_c_b = 1.0;
                                    params.palette_d_r = 0.0; params.palette_d_g = 0.33; params.palette_d_b = 0.67;
                                    changed = true;
                                }
                                
                                ui.separator();
                            });

                        ui.separator();

                        ShaderControls::render_controls_widget(ui, &mut controls_request);

                        ui.separator();

                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);

                        ui.separator();
                        ui.label(format!("Accumulated Samples: {}", self.frame_count));
                        ui.label(format!("Resolution: {}x{}", core.size.width, core.size.height));
                        ui.label(format!("FPS: {:.1}", current_fps));
                        ui.label(format!("Mouse Rotation: {}", if self.mouse_rotation.mouse_enabled { "ON" } else { "OFF" }));
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers || self.should_reset_accumulation {
            self.clear_atomic_buffer(core);
            self.should_reset_accumulation = false;
        }
        self.base.apply_control_request(controls_request);
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = self.frame_count;
        self.base.time_uniform.update(&core.queue);
        
        self.compute_time_uniform.data.time = current_time;
        self.compute_time_uniform.data.delta = 1.0/60.0;
        self.compute_time_uniform.data.frame = self.frame_count;
        self.compute_time_uniform.update(&core.queue);
        
        if params.use_mouse_rotation > 0 {
            let (norm_x, norm_y) = self.mouse_rotation.get_normalized_position(&core.size);
            params.mouse_x = norm_x;
            params.mouse_y = norm_y;
            changed = true;
        }
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Mandelbulb Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
            let width = core.size.width.div_ceil(16);
            let height = core.size.height.div_ceil(16);
            compute_pass.dispatch_workgroups(width, height, 1);
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
            render_pass.set_bind_group(0, &self.output_texture.bind_group, &[]);
            
            render_pass.draw(0..4, 0..1);
        }
        
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        
        if self.params_uniform.data.accumulate > 0 {
            self.frame_count += 1;
        }
        
        Ok(())
    }
    
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
            return true;
        }
        
        if let WindowEvent::CursorMoved { position, .. } = event {
            let x = position.x as f32;
            let y = position.y as f32;
            
            if self.mouse_rotation.handle_mouse_movement(x, y) && self.params_uniform.data.use_mouse_rotation > 0 {
                self.should_reset_accumulation = true;
                return true;
            }
        }
        
        if let WindowEvent::MouseInput { state, button, .. } = event {
            if *button == winit::event::MouseButton::Right {
                if *state == winit::event::ElementState::Released {
                    self.mouse_rotation.toggle_mouse_rotation();
                    return true;
                }
            }
        }
        
        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let winit::keyboard::Key::Character(ch) = &event.logical_key {
                match ch.as_str() {
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
    let (app, event_loop) = cuneus::ShaderApp::new("Mandelbulb Path Tracer", 800, 600);
    
    app.run(event_loop, |core| {
        MandelbulbShader::init(core)
    })
}