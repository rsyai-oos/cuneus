// Very complex example demonstrating multi-buffer ping-pong computation
// I hope this example is useful for those who came from the Shadertoy, I tried to use same terminology (bufferA, ichannels etc)
// I used the all buffers (buffera,b,c,d,mainimage) and complex ping-pong logic 
use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CurrentsParams {
    sphere_radius: f32,
    sphere_pos_x: f32,
    sphere_pos_y: f32,
    critic2_interval: f32,
    critic2_pause: f32,
    critic3_interval: f32,
    metallic_reflection: f32,
    line_intensity: f32,
    pattern_scale: f32,
    noise_strength: f32,
    gradient_r: f32,
    gradient_g: f32,
    gradient_b: f32,
    line_color_r: f32,
    line_color_g: f32,
    line_color_b: f32,
    gamma: f32,
}

impl Default for CurrentsParams {
    fn default() -> Self {
        Self {
            sphere_radius: 0.2,
            sphere_pos_x: 0.0,
            sphere_pos_y: -0.2,
            critic2_interval: 10.0,
            critic2_pause: 5.0,
            critic3_interval: 10.0,
            metallic_reflection: 1.8,
            line_intensity: 0.8,
            pattern_scale: 150.0,
            noise_strength: 1.0,
            gradient_r: 0.92,
            gradient_g: 0.16,
            gradient_b: 0.20,
            line_color_r: 0.8,
            line_color_g: 0.68,
            line_color_b: 0.82,
            gamma: 2.1,
        }
    }
}

impl UniformProvider for CurrentsParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct CurrentsShader {
    base: RenderKit,
    params_uniform: UniformBinding<CurrentsParams>,
    compute_time_uniform: UniformBinding<ComputeTimeUniform>,
    
    buffer_a_pipeline: wgpu::ComputePipeline,
    buffer_b_pipeline: wgpu::ComputePipeline,
    buffer_c_pipeline: wgpu::ComputePipeline,
    buffer_d_pipeline: wgpu::ComputePipeline,
    main_image_pipeline: wgpu::ComputePipeline,
    
    buffer_a_textures: (wgpu::Texture, wgpu::Texture),
    buffer_b_textures: (wgpu::Texture, wgpu::Texture),
    buffer_c_textures: (wgpu::Texture, wgpu::Texture),
    buffer_d_textures: (wgpu::Texture, wgpu::Texture),
    output_texture: wgpu::Texture,
    
    buffer_a_bind_groups: (wgpu::BindGroup, wgpu::BindGroup),
    buffer_b_bind_groups: (wgpu::BindGroup, wgpu::BindGroup),
    buffer_c_bind_groups: (wgpu::BindGroup, wgpu::BindGroup),
    buffer_d_bind_groups: (wgpu::BindGroup, wgpu::BindGroup),
    output_bind_group: wgpu::BindGroup,
    
    multi_texture_layout: wgpu::BindGroupLayout,
    
    frame_count: u32,
    buffer_flip: bool,
}

impl CurrentsShader {
    
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

impl ShaderManager for CurrentsShader {
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
            "Currents Params",
            CurrentsParams::default(),
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
        
        let buffer_d_textures = (
            Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer D0"),
            Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer D1"),
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
            label: Some("Currents Sampler"),
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
        
        let buffer_d_bind_groups = (
            Self::create_storage_bind_group(&core.device, &storage_layout, &buffer_d_textures.0, "Buffer D0 Bind Group"),
            Self::create_storage_bind_group(&core.device, &storage_layout, &buffer_d_textures.1, "Buffer D1 Bind Group"),
        );
        
        let output_bind_group = Self::create_storage_bind_group(&core.device, &storage_layout, &output_texture, "Output Bind Group");
        
        let cs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Currents Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/currents.wgsl").into()),
        });
        
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
        
        let buffer_d_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
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
        
        let buffer_d_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&buffer_d_layout),
            module: &cs_module,
            entry_point: Some("buffer_d"),
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
            buffer_d_pipeline,
            main_image_pipeline,
            buffer_a_textures,
            buffer_b_textures,
            buffer_c_textures,
            buffer_d_textures,
            output_texture,
            buffer_a_bind_groups,
            buffer_b_bind_groups,
            buffer_c_bind_groups,
            buffer_d_bind_groups,
            output_bind_group,
            multi_texture_layout,
            frame_count: 0,
            buffer_flip: false,
        }
    }
    
    fn update(&mut self, _core: &Core) {
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
        
        // Ping-pong buffer selection
        let (_read_a, write_a) = if self.buffer_flip {
            (&self.buffer_a_bind_groups.0, &self.buffer_a_bind_groups.1)
        } else {
            (&self.buffer_a_bind_groups.1, &self.buffer_a_bind_groups.0)
        };
        
        let (_read_b, write_b) = if self.buffer_flip {
            (&self.buffer_b_bind_groups.0, &self.buffer_b_bind_groups.1)
        } else {
            (&self.buffer_b_bind_groups.1, &self.buffer_b_bind_groups.0)
        };
        
        let (_read_c, write_c) = if self.buffer_flip {
            (&self.buffer_c_bind_groups.0, &self.buffer_c_bind_groups.1)
        } else {
            (&self.buffer_c_bind_groups.1, &self.buffer_c_bind_groups.0)
        };
        
        let (_read_d, write_d) = if self.buffer_flip {
            (&self.buffer_d_bind_groups.0, &self.buffer_d_bind_groups.1)
        } else {
            (&self.buffer_d_bind_groups.1, &self.buffer_d_bind_groups.0)
        };
        
        // Create input bind groups for cross-buffer dependencies
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Currents Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        let buffer_a_texture = if self.buffer_flip { &self.buffer_a_textures.0 } else { &self.buffer_a_textures.1 };
        let buffer_b_texture = if self.buffer_flip { &self.buffer_b_textures.0 } else { &self.buffer_b_textures.1 };
        let buffer_c_texture = if self.buffer_flip { &self.buffer_c_textures.0 } else { &self.buffer_c_textures.1 };
        let buffer_d_texture = if self.buffer_flip { &self.buffer_d_textures.0 } else { &self.buffer_d_textures.1 };
        
        let new_buffer_a_texture = if self.buffer_flip { &self.buffer_a_textures.1 } else { &self.buffer_a_textures.0 };
        let new_buffer_b_texture = if self.buffer_flip { &self.buffer_b_textures.1 } else { &self.buffer_b_textures.0 };
        let new_buffer_c_texture = if self.buffer_flip { &self.buffer_c_textures.1 } else { &self.buffer_c_textures.0 };
        let new_buffer_d_texture = if self.buffer_flip { &self.buffer_d_textures.1 } else { &self.buffer_d_textures.0 };
        
        // Buffer A: self-feedback
        let buffer_a_input = Self::create_multi_texture_bind_group(
            &core.device,
            &self.multi_texture_layout,
            buffer_a_texture,
            buffer_a_texture,
            buffer_a_texture,
            &sampler,
            "Buffer A Input Dynamic",
        );
        
        // Buffer B: reads BufferB + BufferA
        let buffer_b_input = Self::create_multi_texture_bind_group(
            &core.device,
            &self.multi_texture_layout,
            buffer_b_texture,
            new_buffer_a_texture, // Use the NEWLY computed Buffer A
            buffer_a_texture,
            &sampler,
            "Buffer B Input Dynamic",
        );
        
        // Buffer C: reads BufferC + BufferA
        let buffer_c_input = Self::create_multi_texture_bind_group(
            &core.device,
            &self.multi_texture_layout,
            buffer_c_texture,
            new_buffer_a_texture, // Use the NEWLY computed Buffer A
            buffer_a_texture,
            &sampler,
            "Buffer C Input Dynamic",
        );
        
        // Buffer D: reads BufferD + BufferC + BufferB
        let buffer_d_input = Self::create_multi_texture_bind_group(
            &core.device,
            &self.multi_texture_layout,
            buffer_d_texture,
            new_buffer_c_texture, // Use NEWLY computed Buffer C
            new_buffer_b_texture, // Use NEWLY computed Buffer B  
            &sampler,
            "Buffer D Input Dynamic",
        );
        
        // Main Image: reads BufferD
        let main_input = Self::create_multi_texture_bind_group(
            &core.device,
            &self.multi_texture_layout,
            new_buffer_d_texture, // Use NEWLY computed Buffer D
            buffer_d_texture,
            buffer_d_texture,
            &sampler,
            "Main Input Dynamic",
        );
        
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
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.buffer_d_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, write_d, &[]);
            compute_pass.set_bind_group(3, &buffer_d_input, &[]);
            compute_pass.dispatch_workgroups(width, height, 1);
        }
        
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
                
                egui::Window::new("Multi-Buffer Ping-Pong Example")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Sphere Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.sphere_radius, 0.05..=0.5).text("Sphere Radius")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.sphere_pos_x, -1.0..=1.0).text("Sphere X")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.sphere_pos_y, -1.0..=1.0).text("Sphere Y")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.metallic_reflection, 0.5..=3.0).text("Metallic Reflection")).changed();
                            });

                        egui::CollapsingHeader::new("Pattern Control")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.pattern_scale, 50.0..=300.0).text("Pattern Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.critic2_interval, 5.0..=20.0).text("Flow Interval")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.critic2_pause, 1.0..=10.0).text("Flow Pause")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.critic3_interval, 5.0..=20.0).text("Scale Interval")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.noise_strength, 0.5..=5.0).text("Noise Strength")).changed();
                            });

                        egui::CollapsingHeader::new("Colors & Post-Processing")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Gradient:");
                                    let mut color = [params.gradient_r, params.gradient_g, params.gradient_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.gradient_r = color[0];
                                        params.gradient_g = color[1];
                                        params.gradient_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Lines:");
                                    let mut color = [params.line_color_r, params.line_color_g, params.line_color_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.line_color_r = color[0];
                                        params.line_color_g = color[1];
                                        params.line_color_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.separator();
                                changed |= ui.add(egui::Slider::new(&mut params.line_intensity, 0.1..=3.0).text("Line Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=4.0).text("Gamma Correction")).changed();
                            });
                        
                        ui.separator();
                        
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        ui.separator();
                        
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        
                        ui.separator();
                        ui.label(format!("Frame: {}", self.frame_count));
                        ui.label("Multi-buffer system with ping-pong textures");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
            // Recreate storage textures to clear buffer data
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
            
            self.buffer_d_textures = (
                Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer D0 Clear"),
                Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Buffer D1 Clear"),
            );
            
            self.output_texture = Self::create_storage_texture(&core.device, core.size.width, core.size.height, "Output Clear");
            
            // Recreate bind groups with fresh textures
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
            
            self.buffer_d_bind_groups = (
                Self::create_storage_bind_group(&core.device, &storage_layout, &self.buffer_d_textures.0, "Buffer D0 Clear Bind Group"),
                Self::create_storage_bind_group(&core.device, &storage_layout, &self.buffer_d_textures.1, "Buffer D1 Clear Bind Group"),
            );
            
            self.output_bind_group = Self::create_storage_bind_group(&core.device, &storage_layout, &self.output_texture, "Output Clear Bind Group");
            
            // Reset frame count and ping-pong state
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
    let (app, event_loop) = cuneus::ShaderApp::new("Multi-Buffer Ping-Pong", 800, 600);
    
    app.run(event_loop, |core| {
        CurrentsShader::init(core)
    })
}