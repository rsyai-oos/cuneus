use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager, ShaderHotReload};
use std::path::PathBuf;
use winit::event::WindowEvent;

// Parameters for the 3D colorspace projection compute shader
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

// Main structure for Color Projection compute shader example
struct ColorProjection {
    // Core components
    base: RenderKit,
    params_uniform: UniformBinding<ColorProjectionParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    
    // Compute-specific components
    compute_pipeline_clear: wgpu::ComputePipeline,
    compute_pipeline_project: wgpu::ComputePipeline,
    compute_pipeline_generate: wgpu::ComputePipeline,
    
    // Output texture for visualization
    output_texture: cuneus::TextureManager,
    
    // Bind group layouts
    compute_bind_group_layout: wgpu::BindGroupLayout,
    atomic_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    
    // Bind groups
    compute_bind_group: wgpu::BindGroup,
    
    // Atomic buffer for color accumulation
    atomic_buffer: cuneus::AtomicBuffer,
    
    // Frame counter
    frame_count: u32,
    
    // Hot reload for shader
    hot_reload: cuneus::ShaderHotReload,
}

impl ColorProjection {
    // Create a storage texture for compute shader output
    fn create_storage_texture(
        device: &wgpu::Device, 
        width: u32, 
        height: u32, 
        label: &str
    ) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
    }
    
    // Helper method to create a texture manager for output display
    fn create_output_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> cuneus::TextureManager {
        let texture = Self::create_storage_texture(device, width, height, "Output Texture");
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("Output Texture Bind Group"),
        });
        
        cuneus::TextureManager {
            texture,
            view,
            sampler,
            bind_group,
        }
    }
    
    // Creates all compute resources after window resize or texture changes
    fn recreate_compute_resources(&mut self, core: &Core) {
        // Create output texture
        self.output_texture = Self::create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            &self.base.texture_bind_group_layout,
        );
        
        // Create atomic buffer
        let buffer_size = core.size.width * core.size.height;
        self.atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            buffer_size,
            &self.atomic_bind_group_layout,
        );
        
        // Recreate compute bind group based on current input texture
        let view_output = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Determine which texture to use as input
        let input_texture_view;
        let input_sampler;
        
        if self.base.using_video_texture {
            if let Some(ref video_manager) = self.base.video_texture_manager {
                let texture_manager = video_manager.texture_manager();
                input_texture_view = &texture_manager.view;
                input_sampler = &texture_manager.sampler;
            } else if let Some(ref texture_manager) = self.base.texture_manager {
                input_texture_view = &texture_manager.view;
                input_sampler = &texture_manager.sampler;
            } else {
                // This should never happen as we always have a default texture
                panic!("No texture available for compute shader input");
            }
        } else if let Some(ref texture_manager) = self.base.texture_manager {
            input_texture_view = &texture_manager.view;
            input_sampler = &texture_manager.sampler;
        } else {
            // This should never happen as we always have a default texture
            panic!("No texture available for compute shader input");
        }
        self.compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(input_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view_output),
                },
            ],
            label: Some("Compute Bind Group"),
        });
    }
    
    // Capture the current frame for export
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
    
    // Handle export of animation frames
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

impl ShaderManager for ColorProjection {
    fn init(core: &Core) -> Self {
        // Create bind group layouts
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
        
        let time_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("time_bind_group_layout"),
        });
        
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
        
        let atomic_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("atomic_bind_group_layout"),
        });
        
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
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
            label: Some("compute_bind_group_layout"),
        });
        
        // Create uniforms
        let buffer_size = core.config.width * core.config.height;
        let atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            buffer_size,
            &atomic_bind_group_layout,
        );
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "Color Projection Params",
            ColorProjectionParams {
                rotation_speed: 0.3,
                intensity: 1.2,
                rot_x: 0.0,
                rot_y: 0.0,
                rot_z: 0.0,
                rot_w: 0.0,
                scale: 1.0,
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
        
        // Create shader module
        let cs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Color Projection Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/computecolors.wgsl").into()),
        });
        
        // Set up hot reload
        let hot_reload = ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shaders/computecolors.wgsl"),
            cs_module.clone(),
            "project_colors", // Main entry point
        ).expect("Failed to initialize hot reload");
        
        // Create base RenderKit
        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        // Create output texture manager
        let output_texture = Self::create_output_texture(
            &core.device,
            core.config.width,
            core.config.height,
            &texture_bind_group_layout,
        );
        
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
        
        // Create compute pipelines for each pass
        let compute_pipeline_clear = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Clear Buffer Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("clear_buffer"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let compute_pipeline_project = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Project Colors Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("project_colors"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let compute_pipeline_generate = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Generate Image Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("generate_image"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        // Create initial compute bind group (will be updated in recreate_compute_resources)
        let compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base.texture_manager.as_ref().unwrap().view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&base.texture_manager.as_ref().unwrap().sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_texture.view),
                },
            ],
            label: Some("Compute Bind Group"),
        });
        
        // Create the struct
        let mut result = Self {
            base,
            params_uniform,
            compute_time_uniform,
            compute_pipeline_clear,
            compute_pipeline_project,
            compute_pipeline_generate,
            output_texture,
            compute_bind_group_layout,
            atomic_bind_group_layout,
            time_bind_group_layout,
            params_bind_group_layout,
            compute_bind_group,
            atomic_buffer,
            frame_count: 0,
            hot_reload,
        };
        
        // Recreate compute resources to ensure everything is consistent
        result.recreate_compute_resources(core);
        
        result
    }
    
    fn update(&mut self, core: &Core) {
        // Check for shader hot reload
        if let Some(new_shader) = self.hot_reload.reload_compute_shader() {
            println!("Reloading compute shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            // Create compute pipeline layout
            let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.compute_bind_group_layout,
                    &self.atomic_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            
            // Create updated compute pipelines with the new shader
            self.compute_pipeline_clear = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Clear Buffer Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: new_shader,
                entry_point: Some("clear_buffer"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.compute_pipeline_project = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Project Colors Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: new_shader,
                entry_point: Some("project_colors"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.compute_pipeline_generate = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Generate Image Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: new_shader,
                entry_point: Some("generate_image"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        }
        
        // Use a static flag to track texture changes: This is a hack to avoid currently I dont get the source of the problem, will fix it later
        static mut LAST_TEXTURE_ID: usize = 0;
        let current_texture_id = if self.base.using_video_texture {
            if let Some(ref vm) = self.base.video_texture_manager {
                vm as *const _ as usize
            } else {
                0
            }
        } else if let Some(ref tm) = self.base.texture_manager {
            tm as *const _ as usize
        } else {
            0
        };
        
        let mut video_updated = false;
        if self.base.using_video_texture {
            video_updated = self.base.update_video_texture(core, &core.queue);
        }
        
        let texture_changed = unsafe {
            if LAST_TEXTURE_ID != current_texture_id {
                LAST_TEXTURE_ID = current_texture_id;
                true
            } else {
                false
            }
        };
        
        if texture_changed || video_updated {
            self.recreate_compute_resources(core);
        }
        
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
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
        
        // Extract video info before entering the closure
        let using_video_texture = self.base.using_video_texture;
        let video_info = self.base.get_video_info();
        
        // Render UI
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });
                
                egui::Window::new("Color Projection Settings")
                    .collapsible(true)
                    .resizable(false)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        // Media controls
                        ShaderControls::render_media_panel(
                            ui,
                            &mut controls_request,
                            using_video_texture,
                            video_info
                        );
                        
                        ui.separator();
                        
                        egui::CollapsingHeader::new("Visual Settings")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.intensity, 0.1..=3.0).text("Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.scale, 0.5..=4.0).text("Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.rotation_speed, 0.0..=1.0).text("Rotation Speed")).changed();
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
                                changed |= ui.add(egui::Slider::new(&mut params.rot_w, -3.14..=3.14).text("W Rotation")).changed();
                                if ui.button("Reset Rotation").clicked() {
                                    params.rot_x = 0.0;
                                    params.rot_y = 0.0;
                                    params.rot_z = 0.0;
                                    params.rot_w = 0.0;
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
            self.recreate_compute_resources(core);
        }
        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        // Update uniforms
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
        
        // Calculate workgroup dimensions
        let width = core.size.width.div_ceil(16);
        let height = core.size.height.div_ceil(16);
        
        // Pass 1: Clear atomic buffer
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Clear Buffer Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline_clear);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        // Pass 2: Project colors to 3D space
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Project Colors Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline_project);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
            // Use input texture dimensions for workgroup count
            let input_dimensions = if self.base.using_video_texture {
                if let Some(ref vm) = self.base.video_texture_manager {
                    let (w, h) = vm.dimensions();
                    (w.div_ceil(16), h.div_ceil(16))
                } else if let Some(ref tm) = self.base.texture_manager {
                    let dims = wgpu::TextureFormat::Rgba8UnormSrgb.block_dimensions();
                    let w = tm.texture.width() / dims.0;
                    let h = tm.texture.height() / dims.1;
                    (w.div_ceil(16), h.div_ceil(16))
                } else {
                    (width, height)
                }
            } else if let Some(ref tm) = self.base.texture_manager {
                let dims = wgpu::TextureFormat::Rgba8UnormSrgb.block_dimensions();
                let w = tm.texture.width() / dims.0;
                let h = tm.texture.height() / dims.1;
                (w.div_ceil(16), h.div_ceil(16))
            } else {
                (width, height)
            };
            
            compute_pass.dispatch_workgroups(input_dimensions.0, input_dimensions.1, 1);
        }
        
        // Pass 3: Generate final image
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Generate Image Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline_generate);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
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
                self.recreate_compute_resources(core);
            }
            return true;
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GStreamer for video support
    cuneus::gst::init()?;
    env_logger::init();
    
    let (app, event_loop) = cuneus::ShaderApp::new("Color Projection", 800, 600);
    
    app.run(event_loop, |core| {
        ColorProjection::init(core)
    })
}