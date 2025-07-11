use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;
use std::path::PathBuf;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct JfaParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    scale: f32,
    n: f32,
    gamma: f32,
    color_intensity: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    color_w: f32,
    accumulation_speed: f32,
    fade_speed: f32,
    freeze_accumulation: f32,
    pattern_floor_add: f32,
    pattern_temp_add: f32,
    pattern_v_offset: f32,
    pattern_temp_mul1: f32,
    pattern_temp_mul2_3: f32,
    _padding0: f32,
    _padding1: f32,
    _padding2: f32,
}

impl Default for JfaParams {
    fn default() -> Self {
        Self {
            a: -2.7,
            b: 0.7,
            c: 0.2,
            d: 0.2,
            scale: 0.3,
            n: 10.0,
            gamma: 2.1,
            color_intensity: 1.0,
            color_r: 1.0,
            color_g: 2.0,
            color_b: 3.0,
            color_w: 4.0,
            accumulation_speed: 0.1,
            fade_speed: 0.99,
            freeze_accumulation: 0.0,
            // Pattern parameters
            pattern_floor_add: 1.0,
            pattern_temp_add: 0.1,
            pattern_v_offset: 0.7,
            pattern_temp_mul1: 0.7,
            pattern_temp_mul2_3: 3.0,
            _padding0: 0.0,
            _padding1: 0.0,
            _padding2: 0.0,
        }
    }
}

impl UniformProvider for JfaParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct JfaShader {
    base: RenderKit,
    params_uniform: UniformBinding<JfaParams>,
    compute_time_uniform: UniformBinding<ComputeTimeUniform>,
    
    buffer_a_pipeline: wgpu::ComputePipeline,
    buffer_b_pipeline: wgpu::ComputePipeline,
    buffer_c_pipeline: wgpu::ComputePipeline,
    main_image_pipeline: wgpu::ComputePipeline,
    
    buffer_a_textures: (wgpu::Texture, wgpu::Texture),
    buffer_b_textures: (wgpu::Texture, wgpu::Texture),
    buffer_c_textures: (wgpu::Texture, wgpu::Texture),
    output_texture: wgpu::Texture,
    
    buffer_a_bind_groups: (wgpu::BindGroup, wgpu::BindGroup),
    buffer_b_bind_groups: (wgpu::BindGroup, wgpu::BindGroup),
    buffer_c_bind_groups: (wgpu::BindGroup, wgpu::BindGroup),
    output_bind_group: wgpu::BindGroup,
    
    multi_texture_layout: wgpu::BindGroupLayout,
    
    frame_count: u32,
    buffer_flip: bool,
    hot_reload: ShaderHotReload,
}

impl JfaShader {
    fn create_storage_texture(device: &wgpu::Device, width: u32, height: u32, label: &str) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    }
    
    fn create_storage_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture: &wgpu::Texture,
        label: &str,
    ) -> wgpu::BindGroup {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
            label: Some(label),
        })
    }
    
    fn create_multi_texture_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        texture0: &wgpu::Texture,
        texture1: &wgpu::Texture, 
        texture2: &wgpu::Texture,
        sampler: &wgpu::Sampler,
        label: &str,
    ) -> wgpu::BindGroup {
        let view0 = texture0.create_view(&wgpu::TextureViewDescriptor::default());
        let view1 = texture1.create_view(&wgpu::TextureViewDescriptor::default());
        let view2 = texture2.create_view(&wgpu::TextureViewDescriptor::default());
        
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view0),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view1),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&view2),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
            label: Some(label),
        })
    }
}

impl ShaderManager for JfaShader {
    fn init(core: &Core) -> Self {
        let storage_layout = create_bind_group_layout(
            &core.device,
            BindGroupLayoutType::StorageTexture,
            "Storage Texture",
        );
        
        let time_layout = create_bind_group_layout(
            &core.device,
            BindGroupLayoutType::TimeUniform,
            "Time",
        );
        
        let params_layout = create_bind_group_layout(
            &core.device,
            BindGroupLayoutType::CustomUniform,
            "Params",
        );
        
        let multi_texture_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("Multi-Texture Layout"),
        });
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "JFA Params",
            JfaParams::default(),
            &params_layout,
            0,
        );
        
        let compute_time_uniform = UniformBinding::new(
            &core.device,
            "Compute Time",
            ComputeTimeUniform {
                time: 0.0,
                delta: 1.0/60.0,
                frame: 0,
                _padding: 0,
            },
            &time_layout,
            0,
        );
        
        let buffer_a_textures = (
            Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer A0"),
            Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer A1"),
        );
        
        let buffer_b_textures = (
            Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer B0"),
            Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer B1"),
        );
        
        let buffer_c_textures = (
            Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer C0"),
            Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer C1"),
        );
        
        let output_texture = Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Output");
        
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
        
        let _sampler = core.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("JFA Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        let buffer_a_bind_groups = (
            Self::create_storage_bind_group(&core.device, &storage_layout, &buffer_a_textures.0, "Buffer A0 Bind Group"),
            Self::create_storage_bind_group(&core.device, &storage_layout, &buffer_a_textures.1, "Buffer A1 Bind Group"),
        );
        
        let buffer_b_bind_groups = (
            Self::create_storage_bind_group(&core.device, &storage_layout, &buffer_b_textures.0, "Buffer B0 Bind Group"),
            Self::create_storage_bind_group(&core.device, &storage_layout, &buffer_b_textures.1, "Buffer B1 Bind Group"),
        );
        
        let buffer_c_bind_groups = (
            Self::create_storage_bind_group(&core.device, &storage_layout, &buffer_c_textures.0, "Buffer C0 Bind Group"),
            Self::create_storage_bind_group(&core.device, &storage_layout, &buffer_c_textures.1, "Buffer C1 Bind Group"),
        );
        
        let output_bind_group = Self::create_storage_bind_group(&core.device, &storage_layout, &output_texture, "Output Bind Group");
        
        let cs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("JFA Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/jfa.wgsl").into()),
        });

        let hot_reload = ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shaders/jfa.wgsl"),
            cs_module.clone(),
            "buffer_a",
        ).expect("Failed to initialize hot reload");
        
        let buffer_a_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&time_layout, &params_layout, &storage_layout, &multi_texture_layout],
            push_constant_ranges: &[],
        });
        
        let buffer_b_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&time_layout, &params_layout, &storage_layout, &multi_texture_layout],
            push_constant_ranges: &[],
        });
        
        let buffer_c_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&time_layout, &params_layout, &storage_layout, &multi_texture_layout],
            push_constant_ranges: &[],
        });
        
        let main_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&time_layout, &params_layout, &storage_layout, &multi_texture_layout],
            push_constant_ranges: &[],
        });
        
        let buffer_a_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&buffer_a_layout),
            module: &cs_module,
            entry_point: Some("buffer_a"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let buffer_b_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&buffer_b_layout),
            module: &cs_module,
            entry_point: Some("buffer_b"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let buffer_c_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&buffer_c_layout),
            module: &cs_module,
            entry_point: Some("buffer_c"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let main_image_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&main_layout),
            module: &cs_module,
            entry_point: Some("main_image"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
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
            buffer_a_pipeline,
            buffer_b_pipeline,
            buffer_c_pipeline,
            main_image_pipeline,
            buffer_a_textures,
            buffer_b_textures,
            buffer_c_textures,
            output_texture,
            buffer_a_bind_groups,
            buffer_b_bind_groups,
            buffer_c_bind_groups,
            output_bind_group,
            multi_texture_layout,
            frame_count: 0,
            buffer_flip: false,
            hot_reload,
        }
    }
    
    fn update(&mut self, core: &Core) {
        if let Some(new_shader) = self.hot_reload.reload_compute_shader() {
            println!("Reloading JFA shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            let time_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::TimeUniform, "JFA Time");
            let params_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::CustomUniform, "JFA Params");
            let storage_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::StorageTexture, "JFA Storage");
            
            let buffer_a_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Buffer A Layout"),
                bind_group_layouts: &[&time_layout, &params_layout, &storage_layout, &self.multi_texture_layout],
                push_constant_ranges: &[],
            });
            
            let buffer_b_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Buffer B Layout"),
                bind_group_layouts: &[&time_layout, &params_layout, &storage_layout, &self.multi_texture_layout],
                push_constant_ranges: &[],
            });
            
            let buffer_c_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Buffer C Layout"),
                bind_group_layouts: &[&time_layout, &params_layout, &storage_layout, &self.multi_texture_layout],
                push_constant_ranges: &[],
            });
            
            let main_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Main Layout"),
                bind_group_layouts: &[&time_layout, &params_layout, &storage_layout, &self.multi_texture_layout],
                push_constant_ranges: &[],
            });

            self.buffer_a_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Buffer A Pipeline"),
                layout: Some(&buffer_a_layout),
                module: &new_shader,
                entry_point: Some("buffer_a"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.buffer_b_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Buffer B Pipeline"),
                layout: Some(&buffer_b_layout),
                module: &new_shader,
                entry_point: Some("buffer_b"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.buffer_c_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Buffer C Pipeline"),
                layout: Some(&buffer_c_layout),
                module: &new_shader,
                entry_point: Some("buffer_c"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.main_image_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Main Image Pipeline"),
                layout: Some(&main_layout),
                module: &new_shader,
                entry_point: Some("main_image"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        }
        
        self.base.fps_tracker.update();
    }
    
    fn resize(&mut self, _core: &Core) {
        self.frame_count = 0;
    }
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        self.compute_time_uniform.data.time = current_time;
        self.compute_time_uniform.data.delta = 1.0/60.0;
        self.compute_time_uniform.data.frame = self.frame_count;
        self.compute_time_uniform.update(&core.queue);
        
        let width = core.size.width.div_ceil(16);
        let height = core.size.height.div_ceil(16);
        
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("JFA Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let (read_idx, write_idx) = if self.buffer_flip { (0, 1) } else { (1, 0) };
        
        // Get read and write bind groups
        let write_a = if write_idx == 0 { &self.buffer_a_bind_groups.0 } else { &self.buffer_a_bind_groups.1 };
        let write_b = if write_idx == 0 { &self.buffer_b_bind_groups.0 } else { &self.buffer_b_bind_groups.1 };
        let write_c = if write_idx == 0 { &self.buffer_c_bind_groups.0 } else { &self.buffer_c_bind_groups.1 };

        // Get read textures (previous frame)
        let read_a_tex = if read_idx == 0 { &self.buffer_a_textures.0 } else { &self.buffer_a_textures.1 };
        let read_b_tex = if read_idx == 0 { &self.buffer_b_textures.0 } else { &self.buffer_b_textures.1 };
        let read_c_tex = if read_idx == 0 { &self.buffer_c_textures.0 } else { &self.buffer_c_textures.1 };
        
        // Get write textures (current frame) 
        let write_a_tex = if write_idx == 0 { &self.buffer_a_textures.0 } else { &self.buffer_a_textures.1 };
        let write_b_tex = if write_idx == 0 { &self.buffer_b_textures.0 } else { &self.buffer_b_textures.1 };
        let write_c_tex = if write_idx == 0 { &self.buffer_c_textures.0 } else { &self.buffer_c_textures.1 };
        
        // BufferA: ichannel0=BufferA (self-reference)
        let buffer_a_input = Self::create_multi_texture_bind_group(
            &core.device, &self.multi_texture_layout,
            read_a_tex, read_a_tex, read_a_tex, &sampler, "BufferA Input"
        );
        
        // BufferB: ichannel0=BufferA, ichannel1=BufferB (current A + previous B)
        let buffer_b_input = Self::create_multi_texture_bind_group(
            &core.device, &self.multi_texture_layout,
            write_a_tex, read_b_tex, read_b_tex, &sampler, "BufferB Input"
        );
        
        // BufferC: ichannel0=BufferA, ichannel1=BufferB, ichannel2=BufferC (current A + CURRENT B + previous C)
        let buffer_c_input = Self::create_multi_texture_bind_group(
            &core.device, &self.multi_texture_layout,
            write_a_tex, write_b_tex, read_c_tex, &sampler, "BufferC Input"
        );
        
        // MainImage: ichannel2=BufferC
let main_input = Self::create_multi_texture_bind_group(
    &core.device, &self.multi_texture_layout,
    write_c_tex, // The new Buffer C from this frame
    write_c_tex, // Unused
    write_c_tex, // Unused
    &sampler, "Main Input"
);
        
        // Execute Buffer A
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.buffer_a_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, write_a, &[]);
            compute_pass.set_bind_group(3, &buffer_a_input, &[]);
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        // Execute Buffer B
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.buffer_b_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, write_b, &[]);
            compute_pass.set_bind_group(3, &buffer_b_input, &[]);
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        // Execute Buffer C
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.buffer_c_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, write_c, &[]);
            compute_pass.set_bind_group(3, &buffer_c_input, &[]);
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        // Execute Main Image
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.main_image_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.output_bind_group, &[]);
            compute_pass.set_bind_group(3, &main_input, &[]);
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
        // Render to screen
        let output_view = self.output_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
        let display_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.base.renderer.render_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: None,
        });
        
        {
            let mut render_pass = cuneus::Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                None,
            );
            
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &display_bind_group, &[]);
            
            render_pass.draw(0..4, 0..1);
        }
        
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
                
                egui::Window::new("JFA")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("JFA")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.n, 1.0..=50.0).text("N (Frame Cycle)")).changed();
                                ui.separator();
                                changed |= ui.add(egui::Slider::new(&mut params.accumulation_speed, 0.0..=3.0).text("Accumulation Speed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.fade_speed, 0.9..=1.0).text("Fade Speed")).changed();
                            });


                        egui::CollapsingHeader::new("Pattern")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.pattern_floor_add, 0.0..=100.0).text("Floor Add")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.pattern_temp_add, -1.0..=1.0).text("Temp Add")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.pattern_v_offset, -1.0..=1.0).text("V Offset")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.pattern_temp_mul1, -3.2..=3.2).text("Temp Mul1")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.pattern_temp_mul2_3, -3.2..=3.2).text("Temp Mul2/3")).changed();
                            });

                        egui::CollapsingHeader::new("Clifford")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.separator();
                                changed |= ui.add(egui::Slider::new(&mut params.a, -5.0..=5.0).text("a")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.b, -5.0..=5.0).text("b")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.c, -5.0..=5.0).text("c")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.d, -5.0..=5.0).text("d")).changed();
                                ui.separator();
                                changed |= ui.add(egui::Slider::new(&mut params.scale, 0.1..=1.0).text("Scale")).changed();
                            });

                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Color Pattern:");
                                    let mut color = [params.color_r, params.color_g, params.color_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.color_r = color[0];
                                        params.color_g = color[1];
                                        params.color_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                changed |= ui.add(egui::Slider::new(&mut params.color_w, 0.0..=10.0).text("Color W")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.color_intensity, 0.1..=3.0).text("Color Intensity")).changed();
                                
                                ui.separator();
                                ui.label("Post-Processing:");
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=4.0).text("Gamma")).changed();
                            });

 
                        
                        ui.separator();
                        
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        ui.separator();
                        
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        
                        ui.separator();
                        ui.label(format!("Frame: {}", self.frame_count));
                        ui.label(format!("N Cycle: {}", params.n as i32));
                        ui.label(format!("Frame in Cycle: {}", self.frame_count % params.n as u32));
                        ui.label("JFA with Clifford Attractor");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        if controls_request.is_paused != (params.freeze_accumulation > 0.5) {
            params.freeze_accumulation = if controls_request.is_paused { 1.0 } else { 0.0 };
            changed = true;
        }

        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
            self.buffer_a_textures = (
                Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer A0 Clear"),
                Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer A1 Clear"),
            );
            
            self.buffer_b_textures = (
                Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer B0 Clear"),
                Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer B1 Clear"),
            );
            
            self.buffer_c_textures = (
                Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer C0 Clear"),
                Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer C1 Clear"),
            );
            
            self.output_texture = Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Output Clear");
            
            let storage_layout = create_bind_group_layout(
                &core.device,
                BindGroupLayoutType::StorageTexture,
                "Storage Texture",
            );
            
            self.buffer_a_bind_groups = (
                Self::create_storage_bind_group(&core.device, &storage_layout, &self.buffer_a_textures.0, "Buffer A0 Clear Bind Group"),
                Self::create_storage_bind_group(&core.device, &storage_layout, &self.buffer_a_textures.1, "Buffer A1 Clear Bind Group"),
            );
            
            self.buffer_b_bind_groups = (
                Self::create_storage_bind_group(&core.device, &storage_layout, &self.buffer_b_textures.0, "Buffer B0 Clear Bind Group"),
                Self::create_storage_bind_group(&core.device, &storage_layout, &self.buffer_b_textures.1, "Buffer B1 Clear Bind Group"),
            );
            
            self.buffer_c_bind_groups = (
                Self::create_storage_bind_group(&core.device, &storage_layout, &self.buffer_c_textures.0, "Buffer C0 Clear Bind Group"),
                Self::create_storage_bind_group(&core.device, &storage_layout, &self.buffer_c_textures.1, "Buffer C1 Clear Bind Group"),
            );
            
            self.output_bind_group = Self::create_storage_bind_group(&core.device, &storage_layout, &self.output_texture, "Output Clear Bind Group");
            
            self.frame_count = 0;
            self.buffer_flip = false;
        }
        self.base.apply_control_request(controls_request);

        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        
        core.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        self.frame_count += 1;
        self.buffer_flip = !self.buffer_flip;
        
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
    let (app, event_loop) = cuneus::ShaderApp::new("JFA", 800, 600);
    
    app.run(event_loop, |core| {
        JfaShader::init(core)
    })
}