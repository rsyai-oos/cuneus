use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager};
use cuneus::compute::{create_bind_group_layout, BindGroupLayoutType};
use winit::event::WindowEvent;

// Parameter struct for Clifford attractor
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ComputeParams {
    decay: f32,
    speed: f32,
    intensity: f32,
    scale: f32,
    rotation_x: f32,
    rotation_y: f32,
    rotation_z: f32,
    rotation_speed: f32,
    attractor_a: f32,
    attractor_b: f32,
    attractor_c: f32,
    attractor_d: f32,
    attractor_animate_amount: f32,
    num_points: u32,
    iterations_per_point: u32,
    clear_buffer: u32,
    _padding: u32,
}

impl UniformProvider for ComputeParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

// Texture pair for ping-pong buffer 
struct TexturePair {
    a: cuneus::TextureManager,
    b: cuneus::TextureManager,
    bind_group_a: wgpu::BindGroup,
    bind_group_b: wgpu::BindGroup,
}

// Main structure for Clifford compute shader example
struct CliffordCompute {
    // Core components
    base: RenderKit,
    params_uniform: UniformBinding<ComputeParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    
    // Texture ping-pong
    texture_pair: TexturePair,
    frame_count: u32,
    source_is_a: bool,
    
    // Bind group layouts
    compute_bind_group_layout: wgpu::BindGroupLayout,
    atomic_bind_group_layout: wgpu::BindGroupLayout,
    
    // Atomic buffer for point accumulation
    atomic_buffer: cuneus::AtomicBuffer,
    
    // Compute pipelines
    compute_pipeline_points: wgpu::ComputePipeline,
    compute_pipeline_feedback: wgpu::ComputePipeline,
}

impl CliffordCompute {
    // Create a pair of ping-pong textures with necessary bind groups
    fn create_texture_pair(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        compute_bind_group_layout: &wgpu::BindGroupLayout,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> TexturePair {
        // Create two storage textures for ping-pong
        let texture_a = cuneus::compute::create_storage_texture(
            device, 
            width, 
            height, 
            cuneus::compute::COMPUTE_TEXTURE_FORMAT_RGBA16,
            "Compute Texture A"
        );
        
        let texture_b = cuneus::compute::create_storage_texture(
            device, 
            width, 
            height, 
            cuneus::compute::COMPUTE_TEXTURE_FORMAT_RGBA16,
            "Compute Texture B"
        );
        
        // Create views and sampler
        let view_a = texture_a.create_view(&wgpu::TextureViewDescriptor::default());
        let view_b = texture_b.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        // Create texture managers
        let tex_manager_a = cuneus::TextureManager {
            texture: texture_a,
            view: view_a.clone(),
            sampler: sampler.clone(),
            bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_a),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: Some("Texture A Bind Group"),
            }),
        };
        
        let tex_manager_b = cuneus::TextureManager {
            texture: texture_b,
            view: view_b.clone(),
            sampler,
            bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: texture_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view_b),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&tex_manager_a.sampler),
                    },
                ],
                label: Some("Texture B Bind Group"),
            }),
        };
        
        // Create compute bind groups (both directions of ping-pong)
        let compute_bind_group_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_a),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&tex_manager_a.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view_b),
                },
            ],
            label: Some("Compute A->B Bind Group"),
        });
        
        let compute_bind_group_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_b),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&tex_manager_a.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view_a),
                },
            ],
            label: Some("Compute B->A Bind Group"),
        });
        
        TexturePair {
            a: tex_manager_a,
            b: tex_manager_b,
            bind_group_a: compute_bind_group_a,
            bind_group_b: compute_bind_group_b,
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
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &capture_view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Capture Pass"),
            );
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            if self.source_is_a {
                render_pass.set_bind_group(0, &self.texture_pair.a.bind_group, &[]);
            } else {
                render_pass.set_bind_group(0, &self.texture_pair.b.bind_group, &[]);
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
    
    fn recreate_textures(&mut self, core: &Core) {
        let texture_pair = Self::create_texture_pair(
            &core.device,
            core.size.width,
            core.size.height,
            &self.compute_bind_group_layout,
            &self.base.texture_bind_group_layout,
        );
        
        self.texture_pair = texture_pair;
        self.source_is_a = false;
        let buffer_size = core.size.width * core.size.height;
        self.atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            buffer_size,
            &self.atomic_bind_group_layout,
        );
    }
}

impl ShaderManager for CliffordCompute {
    fn init(core: &Core) -> Self {
        // Create the standard fragment texture bind group layout
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
        
        // Create specialized bind group layouts using our utility functions
        let time_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::TimeUniform, 
            "Clifford Compute"
        );
        
        let params_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::CustomUniform, 
            "Clifford Params"
        );
        
        let atomic_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::AtomicBuffer, 
            "Clifford Compute"
        );
        
        // Create compute bind group layout
        let compute_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: cuneus::compute::COMPUTE_TEXTURE_FORMAT_RGBA16,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
            label: Some("compute_bind_group_layout"),
        });
        
        // Create atomic buffer
        let buffer_size = core.config.width * core.config.height;
        let atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            buffer_size,
            &atomic_bind_group_layout,
        );
        
        // Create params uniform
        let params_uniform = UniformBinding::new(
            &core.device,
            "Compute Params Uniform",
            ComputeParams {
                decay: 0.9,
                speed: 1.0,
                intensity: 1.0,
                scale: 1.0,
                rotation_x: 0.0,
                rotation_y: 0.0,
                rotation_z: 0.0,
                rotation_speed: 0.15,
                attractor_a: 1.7,
                attractor_b: 1.7,
                attractor_c: 0.6,
                attractor_d: 1.2,
                attractor_animate_amount: 1.0,
                num_points: 222,
                iterations_per_point: 15,
                clear_buffer: 1,
                _padding: 0,
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
        
        // Create texture pair
        let texture_pair = Self::create_texture_pair(
            &core.device,
            core.config.width,
            core.config.height,
            &compute_bind_group_layout,
            &texture_bind_group_layout,
        );
        
        // Create compute shader module
        let shader_source = include_str!("../../shaders/clifford_compute.wgsl");
        let cs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Clifford Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create compute pipeline layout
        let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,
                &params_bind_group_layout,
                &compute_bind_group_layout,
                &atomic_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        
        // Create compute pipelines
        let compute_pipeline_points = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Points Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("compute_points"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let compute_pipeline_feedback = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Feedback Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("compute_feedback"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Create base RenderKit for UI
        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        Self {
            base,
            params_uniform,
            compute_time_uniform,
            compute_pipeline_points,
            compute_pipeline_feedback,
            texture_pair,
            frame_count: 0,
            source_is_a: false,
            compute_bind_group_layout,
            atomic_bind_group_layout,
            atomic_buffer,
        }
    }
    
    fn update(&mut self, core: &Core) {
        // Handle exports if needed
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
    }
    
    fn resize(&mut self, core: &Core) {
        println!("Resizing to {:?}", core.size);
        self.recreate_textures(core);
    }
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Handle UI
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });
                
                egui::Window::new("Compute Shader Settings")
                    .collapsible(true)
                    .resizable(false)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                    egui::CollapsingHeader::new("Visual Settings")
                        .default_open(true)
                        .show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.intensity, 0.0..=3.0).text("Intensity")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.speed, 0.1..=4.0).text("Speed")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.scale, 0.5..=4.0).text("Scale")).changed();
                            if ui.button("Reset Visual").clicked() {
                                params.intensity = 5.0;
                                params.speed = 1.0;
                                params.scale = 2.0;
                                changed = true;
                            }
                        });
                    
                    egui::CollapsingHeader::new("Camera & Rotation")
                        .default_open(false)
                        .show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.rotation_x, -3.14..=3.14).text("X")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.rotation_y, -3.14..=3.14).text("Y")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.rotation_z, -3.14..=3.14).text("Z")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.rotation_speed, 0.0..=1.0).text("Speed")).changed();
                            if ui.button("Reset Camera").clicked() {
                                params.rotation_x = 0.0;
                                params.rotation_y = 0.0;
                                params.rotation_z = 0.0;
                                params.rotation_speed = 0.15;
                                changed = true;
                            }
                        });
                    
                    egui::CollapsingHeader::new("Attractor Parameters")
                        .default_open(false)
                        .show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.attractor_a, 0.0..=3.0).text("a")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.attractor_b, 0.0..=3.0).text("b")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.attractor_c, 0.0..=3.0).text("c")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.attractor_d, 0.0..=3.0).text("d")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.attractor_animate_amount, 0.0..=2.0).text("Animate")).changed();
                            if ui.button("Reset Attractor").clicked() {
                                params.attractor_a = 1.7;
                                params.attractor_b = 1.7;
                                params.attractor_c = 0.6;
                                params.attractor_d = 1.2;
                                params.attractor_animate_amount = 1.0;
                                changed = true;
                            }
                        });
                    
                    egui::CollapsingHeader::new("Compute Performance")
                        .default_open(false)
                        .show(ui, |ui| {
                            changed |= ui.add(egui::Slider::new(&mut params.num_points, 2..=2048).logarithmic(true).text("Points")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.iterations_per_point, 2..=200).text("Iterations")).changed();
                            if ui.button("Reset Performance").clicked() {
                                params.num_points = 100;
                                params.iterations_per_point = 10;
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
        
        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
            self.recreate_textures(core);
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
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        // Determine ping-pong direction
        let source_bind_group = if self.source_is_a {
            &self.texture_pair.bind_group_a
        } else {
            &self.texture_pair.bind_group_b
        };
        
        // Calculate workgroup dispatch dimensions
        let width = core.size.width.div_ceil(16);
        let height = core.size.height.div_ceil(16);
        
        // First compute pass - Points
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Points Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline_points);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, source_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        // Second compute pass - Feedback
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Feedback Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline_feedback);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, source_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        // Render the result to screen
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Display Pass"),
            );
            
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            
            // Display the current output texture
            if self.source_is_a {
                render_pass.set_bind_group(0, &self.texture_pair.a.bind_group, &[]);
            } else {
                render_pass.set_bind_group(0, &self.texture_pair.b.bind_group, &[]);
            }
            
            render_pass.draw(0..4, 0..1);
        }
        
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        
        // Swap ping-pong buffers
        self.source_is_a = !self.source_is_a;
        
        // Submit work
        core.queue.submit(Some(encoder.finish()));
        output.present();
        
        // Increment frame counter
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
    let (app, event_loop) = cuneus::ShaderApp::new("Clifford Compute", 800, 600);
    
    app.run(event_loop, |core| {
        CliffordCompute::init(core)
    })
}