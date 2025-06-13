use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager};
use cuneus::compute::{create_bind_group_layout, BindGroupLayoutType};
use winit::event::WindowEvent;
use std::path::PathBuf;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FFTParams {
    filter_type: i32,     
    filter_strength: f32, 
    filter_direction: f32,
    filter_radius: f32,   
    show_freqs: i32,      
    resolution: u32,      
    _padding1: u32,
    _padding2: u32,
}

impl UniformProvider for FFTParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct FFTShader {
    base: RenderKit,
    params_uniform: UniformBinding<FFTParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    
    init_pipeline: wgpu::ComputePipeline,
    fft_horizontal_pipeline: wgpu::ComputePipeline,
    fft_vertical_pipeline: wgpu::ComputePipeline,
    modify_freqs_pipeline: wgpu::ComputePipeline,
    ifft_horizontal_pipeline: wgpu::ComputePipeline,
    ifft_vertical_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::ComputePipeline,
    
    output_texture: cuneus::TextureManager,
    
    compute_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    
    compute_bind_group: wgpu::BindGroup,
    
    storage_buffer: wgpu::Buffer,
    storage_bind_group_layout: wgpu::BindGroupLayout,
    storage_bind_group: wgpu::BindGroup,
    
    frame_count: u32,
    
    hot_reload: cuneus::ShaderHotReload,
    
    should_initialize: bool,
}

impl FFTShader {
    fn recreate_compute_resources(&mut self, core: &Core) {
        self.output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            wgpu::TextureFormat::Rgba16Float,
            &self.base.texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "FFT Output Texture",
        );
        
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
                panic!("No texture available for compute shader input");
            }
        } else if self.base.using_webcam_texture {
            if let Some(ref webcam_manager) = self.base.webcam_texture_manager {
                let texture_manager = webcam_manager.texture_manager();
                input_texture_view = &texture_manager.view;
                input_sampler = &texture_manager.sampler;
            } else if let Some(ref texture_manager) = self.base.texture_manager {
                input_texture_view = &texture_manager.view;
                input_sampler = &texture_manager.sampler;
            } else {
                panic!("No texture available for compute shader input");
            }
        } else if let Some(ref texture_manager) = self.base.texture_manager {
            input_texture_view = &texture_manager.view;
            input_sampler = &texture_manager.sampler;
        } else {
            panic!("No texture available for compute shader input");
        }
        
        let view_output = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FFT Compute Bind Group"),
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
        });
        
        let resolution = self.params_uniform.data.resolution;
        let buffer_size = (resolution * resolution * 3 * 2 * 4) as u64;
        
        self.storage_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Storage Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        self.storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FFT Storage Bind Group"),
            layout: &self.storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.storage_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        
        self.should_initialize = true;
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
}

impl ShaderManager for FFTShader {
    fn init(core: &Core) -> Self {
        // Create bind group layouts
        let time_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::TimeUniform, 
            "FFT Compute Time"
        );
        
        let params_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::CustomUniform, 
            "FFT Params"
        );
        
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
            label: Some("fft_compute_bind_group_layout"),
        });
        
        let storage_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("storage_bind_group_layout"),
        });
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "FFT Params",
            FFTParams {
                filter_type: 1,            
                filter_strength: 0.3,      
                filter_direction: 0.0,
                filter_radius: 3.0,  
                show_freqs: 0,  
                resolution: 1024, 
                _padding1: 0,
                _padding2: 0,
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
        
        // Create base kit
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
            "FFT Output Texture",
        );
        
        let view_output = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
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
                    resource: wgpu::BindingResource::TextureView(&view_output),
                },
            ],
            label: Some("FFT Compute Bind Group"),
        });
        
        let resolution = params_uniform.data.resolution;
        let buffer_size = (resolution * resolution * 3 * 2 * 4) as u64;
        
        let storage_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Storage Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FFT Storage Bind Group"),
            layout: &storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &storage_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        
        let shader_source = include_str!("../../shaders/fft.wgsl");
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FFT Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FFT Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,
                &params_bind_group_layout,
                &compute_bind_group_layout,
                &storage_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        
        let init_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFT Initialize Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("initialize_data"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let fft_horizontal_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFT Horizontal Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("fft_horizontal"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let fft_vertical_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFT Vertical Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("fft_vertical"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let modify_freqs_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFT Modify Frequencies Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("modify_frequencies"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let ifft_horizontal_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("IFFT Horizontal Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("ifft_horizontal"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let ifft_vertical_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("IFFT Vertical Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("ifft_vertical"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let render_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFT Render Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("main_image"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let hot_reload = cuneus::ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shaders/fft.wgsl"),
            shader_module.clone(),
            "main_image",
        ).expect("Failed to initialize hot reload");
        
        let mut result = Self {
            base,
            params_uniform,
            compute_time_uniform,
            init_pipeline,
            fft_horizontal_pipeline,
            fft_vertical_pipeline,
            modify_freqs_pipeline,
            ifft_horizontal_pipeline,
            ifft_vertical_pipeline,
            render_pipeline,
            output_texture,
            compute_bind_group_layout,
            time_bind_group_layout,
            params_bind_group_layout,
            compute_bind_group,
            storage_buffer,
            storage_bind_group_layout,
            storage_bind_group,
            frame_count: 0,
            hot_reload,
            should_initialize: true,
        };
        
        result.recreate_compute_resources(core);
        
        result
    }
    
    fn update(&mut self, core: &Core) {
        // Check for hot reload of the shader
        if let Some(new_shader) = self.hot_reload.reload_compute_shader() {
            println!("Reloading FFT shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            // Create compute pipeline layout
            let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated FFT Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.compute_bind_group_layout,
                    &self.storage_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            
            // Recreate all pipelines with the updated shader
            self.init_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated FFT Initialize Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("initialize_data"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.fft_horizontal_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated FFT Horizontal Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("fft_horizontal"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.fft_vertical_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated FFT Vertical Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("fft_vertical"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.modify_freqs_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated FFT Modify Frequencies Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("modify_frequencies"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.ifft_horizontal_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated IFFT Horizontal Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("ifft_horizontal"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.ifft_vertical_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated IFFT Vertical Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("ifft_vertical"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.render_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated FFT Render Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("main_image"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            // We need to reinitialize the data after shader reload
            self.should_initialize = true;
        }
        
        let video_updated = if self.base.using_video_texture {
            self.base.update_video_texture(core, &core.queue)
        } else {
            false
        };
        let webcam_updated = if self.base.using_webcam_texture {
            self.base.update_webcam_texture(core, &core.queue)
        } else {
            false
        };
        
        if video_updated || webcam_updated {
            self.recreate_compute_resources(core);
        }
        
        // Handle export if needed
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

        let using_video_texture = self.base.using_video_texture;
        let using_hdri_texture = self.base.using_hdri_texture;
        let using_webcam_texture = self.base.using_webcam_texture;
        let video_info = self.base.get_video_info();
        let hdri_info = self.base.get_hdri_info();
        let webcam_info = self.base.get_webcam_info();
        
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });
                
                egui::Window::new("fourier workflow")
                    .collapsible(true)
                    .resizable(false)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        // Media controls
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
                        
                        egui::CollapsingHeader::new("FFT Settings")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("Resolution:");
                                
                                ui.horizontal(|ui| {
                                    changed |= ui.radio_value(&mut params.resolution, 256, "256").changed();
                                    changed |= ui.radio_value(&mut params.resolution, 512, "512").changed();
                                    changed |= ui.radio_value(&mut params.resolution, 1024, "1024").changed();
                                    changed |= ui.radio_value(&mut params.resolution, 2048, "2048").changed();
                                });
                                
                                if changed {
                                    self.should_initialize = true;
                                }
                                
                                ui.separator();
                                ui.label("View Mode:");
                                changed |= ui.radio_value(&mut params.show_freqs, 0, "Filtered").changed();
                                changed |= ui.radio_value(&mut params.show_freqs, 1, "Frequency Domain").changed();
                                
                                ui.separator();
                            });
                        
                        egui::CollapsingHeader::new("Filter Settings")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("Filter Type:");
                                changed |= ui.radio_value(&mut params.filter_type, 0, "LP").changed();
                                changed |= ui.radio_value(&mut params.filter_type, 1, "HP").changed();
                                changed |= ui.radio_value(&mut params.filter_type, 2, "BP").changed();
                                changed |= ui.radio_value(&mut params.filter_type, 3, "Directional").changed();
                                
                                ui.separator();
                                
                                changed |= ui.add(egui::Slider::new(&mut params.filter_strength, 0.0..=1.0)
                                    .text("Filter Strength"))
                                    .changed();
                                
                                if params.filter_type == 2 {
                                    changed |= ui.add(egui::Slider::new(&mut params.filter_radius, 0.0..=6.28)
                                        .text("Band Radius"))
                                        .changed();
                                }
                                
                                if params.filter_type == 3 {
                                    changed |= ui.add(egui::Slider::new(&mut params.filter_direction, 0.0..=6.28)
                                        .text("Direction"))
                                        .changed();
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
            self.should_initialize = true;
        }
        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        self.base.handle_webcam_requests(core, &controls_request);
        if controls_request.load_media_path.is_some() {
            self.recreate_compute_resources(core);
            self.should_initialize = true;
        }
        if self.base.handle_hdri_requests(core, &controls_request) {
            self.recreate_compute_resources(core);
        }
        if self.base.handle_hdri_requests(core, &controls_request) {
            self.recreate_compute_resources(core);
        }
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
            
            // Add this line - it's the key fix:
            self.should_initialize = true;
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        if self.should_initialize {
            let mut init_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFT Initialize Pass"),
                timestamp_writes: None,
            });
            
            init_pass.set_pipeline(&self.init_pipeline);
            init_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            init_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            init_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            init_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            let resolution = params.resolution;
            let width = resolution.div_ceil(16);
            let height = resolution.div_ceil(16);
            init_pass.dispatch_workgroups(width, height, 1);
            
            self.should_initialize = false;
        }
        
        // Horizontal FFT
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFT Horizontal Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.fft_horizontal_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            let resolution = params.resolution;
            compute_pass.dispatch_workgroups(resolution, 1, 1);
        }
        
        // Vertical FFT
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFT Vertical Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.fft_vertical_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            let resolution = params.resolution;
            compute_pass.dispatch_workgroups(resolution, 1, 1);
        }
        
        // Modify frequencies
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Modify Frequencies Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.modify_freqs_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            let resolution = params.resolution;
            let width = resolution.div_ceil(16);
            let height = resolution.div_ceil(16);
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        // Horizontal IFFT
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IFFT Horizontal Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.ifft_horizontal_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            let resolution = params.resolution;
            compute_pass.dispatch_workgroups(resolution, 1, 1);
        }
        
        // Vertical IFFT
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IFFT Vertical Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.ifft_vertical_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            let resolution = params.resolution;
            compute_pass.dispatch_workgroups(resolution, 1, 1);
        }
        
        // Render to output texture
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFT Render Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.render_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            let width = core.size.width.div_ceil(16);
            let height = core.size.height.div_ceil(16);
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        // Display the output texture
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
                self.should_initialize = true;
            }
            return true;
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("FFT", 800, 600);
    
    app.run(event_loop, |core| {
        FFTShader::init(core)
    })
}