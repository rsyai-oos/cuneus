use crate::{Core, UniformProvider, UniformBinding, TextureManager};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComputeTimeUniform {
    pub time: f32,
    pub delta: f32,
    pub frame: u32,
    pub _padding: u32,
}

impl UniformProvider for ComputeTimeUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

pub struct ComputeShader {
    pub pipeline: wgpu::ComputePipeline,
    pub output_texture: TextureManager,
    pub workgroup_size: [u32; 3],
    pub workgroup_count: Option<[u32; 3]>,
    pub dispatch_once: bool,
    pub current_frame: u32,
    pub time_uniform: UniformBinding<ComputeTimeUniform>,
    pub time_bind_group_layout: wgpu::BindGroupLayout,
    pub storage_texture_layout: wgpu::BindGroupLayout,
    pub storage_bind_group: wgpu::BindGroup,
}

impl ComputeShader {
    pub fn new(
        core: &Core,
        shader_source: &str,
        entry_point: &str,
        workgroup_size: [u32; 3],
        workgroup_count: Option<[u32; 3]>,
        dispatch_once: bool,
    ) -> Self {
        // 1. Create time uniform and layout
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
            label: Some("compute_time_bind_group_layout"),
        });
        
        let time_uniform = UniformBinding::new(
            &core.device,
            "Compute Time",
            ComputeTimeUniform {
                time: 0.0,
                delta: 0.0,
                frame: 0,
                _padding: 0,
            },
            &time_bind_group_layout,
            0,
        );
        
        let storage_texture_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Storage Texture Layout"),
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
        });
        
        let texture = core.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Compute Output Texture"),
            size: wgpu::Extent3d {
                width: core.size.width,
                height: core.size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
        
        let storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Storage Texture Bind Group"),
            layout: &storage_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Display Layout"),
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
        });
        
        let bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Display Bind Group"),
            layout: &texture_bind_group_layout,
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
        });
        
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,
                &storage_texture_layout,
            ],
            push_constant_ranges: &[],
        });
        
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let output_texture = TextureManager {
            texture,
            view,
            sampler,
            bind_group,
        };
        
        Self {
            pipeline,
            output_texture,
            workgroup_size,
            workgroup_count,
            dispatch_once,
            current_frame: 0,
            time_uniform,
            time_bind_group_layout,
            storage_texture_layout,
            storage_bind_group,
        }
    }

    pub fn set_time(&mut self, elapsed: f32, delta: f32, queue: &wgpu::Queue) {
        self.time_uniform.data.time = elapsed;
        self.time_uniform.data.delta = delta;
        self.time_uniform.data.frame = self.current_frame;
        self.time_uniform.update(queue);
    }
    
    pub fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder, core: &Core) {
        if self.dispatch_once && self.current_frame > 0 {
            return;
        }
        
        let workgroup_count = self.workgroup_count.unwrap_or([
            core.size.width.div_ceil(self.workgroup_size[0]),
            core.size.height.div_ceil(self.workgroup_size[1]),
            1,
        ]);
        
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.time_uniform.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.storage_bind_group, &[]);
        
        compute_pass.dispatch_workgroups(
            workgroup_count[0],
            workgroup_count[1],
            workgroup_count[2],
        );
        
        self.current_frame += 1;
    }
    
    pub fn resize(&mut self, core: &Core, width: u32, height: u32) {
        let texture = core.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Compute Output Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
        
        let storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Storage Texture Bind Group"),
            layout: &self.storage_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Display Layout"),
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
        });
        let bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Display Bind Group"),
            layout: &texture_bind_group_layout,
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
        });
        self.output_texture = TextureManager {
            texture,
            view,
            sampler,
            bind_group,
        };
        
        self.storage_bind_group = storage_bind_group;
    }
    
    pub fn get_output_texture(&self) -> &TextureManager {
        &self.output_texture
    }
}