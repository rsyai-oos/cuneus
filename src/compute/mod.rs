use crate::{Core, UniformProvider, UniformBinding, TextureManager, ShaderHotReload, AtomicBuffer, FontSystem};
use std::sync::Arc;
use std::path::PathBuf;
use log::{info, warn};

pub const COMPUTE_TEXTURE_FORMAT_RGBA16: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const COMPUTE_TEXTURE_FORMAT_RGBA8: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

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

pub struct ComputeShaderConfig {
    pub workgroup_size: [u32; 3],
    pub workgroup_count: Option<[u32; 3]>,
    pub dispatch_once: bool,
    pub storage_texture_format: wgpu::TextureFormat,
    pub enable_atomic_buffer: bool,
    pub atomic_buffer_multiples: usize,
    pub entry_points: Vec<String>,
    pub sampler_address_mode: wgpu::AddressMode,
    pub sampler_filter_mode: wgpu::FilterMode,
    pub label: String,
    pub mouse_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub enable_fonts: bool,
    pub enable_audio_buffer: bool,
    pub audio_buffer_size: usize,
}

impl Default for ComputeShaderConfig {
    fn default() -> Self {
        Self {
            workgroup_size: [16, 16, 1],
            workgroup_count: None,
            dispatch_once: false,
            storage_texture_format: COMPUTE_TEXTURE_FORMAT_RGBA16,
            enable_atomic_buffer: false,
            atomic_buffer_multiples: 4,
            entry_points: vec!["main".to_string()],
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "Compute Shader".to_string(),
            mouse_bind_group_layout: None,
            enable_fonts: false,
            enable_audio_buffer: false,
            audio_buffer_size: 1024,
        }
    }
}

//bind group layout types for different shader needs
pub enum BindGroupLayoutType {
    StorageTexture,
    TimeUniform,
    CustomUniform,
    AtomicBuffer,
    ExternalTexture,
    MouseUniform,
    FontTexture,
    FontWithAudio,
    AudioBuffer,
}

pub fn create_storage_texture(
    device: &wgpu::Device, 
    width: u32, 
    height: u32, 
    format: wgpu::TextureFormat,
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
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    })
}

pub fn create_output_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    address_mode: wgpu::AddressMode,
    filter_mode: wgpu::FilterMode,
    label: &str,
) -> TextureManager {
    let texture = create_storage_texture(device, width, height, format, label);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        address_mode_w: address_mode,
        mag_filter: filter_mode,
        min_filter: filter_mode,
        mipmap_filter: filter_mode,
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
        label: Some(&format!("{} Bind Group", label)),
    });
    
    TextureManager {
        texture,
        view,
        sampler,
        bind_group,
    }
}

pub fn create_bind_group_layout(
    device: &wgpu::Device,
    layout_type: BindGroupLayoutType,
    label: &str,
) -> wgpu::BindGroupLayout {
    match layout_type {
        BindGroupLayoutType::StorageTexture => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} Storage Texture Layout", label)),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: COMPUTE_TEXTURE_FORMAT_RGBA16,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            })
        },
        BindGroupLayoutType::TimeUniform => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some(&format!("{} Time Uniform Layout", label)),
            })
        },
        BindGroupLayoutType::MouseUniform => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some(&format!("{} Mouse Uniform Layout", label)),
            })
        },
        BindGroupLayoutType::AtomicBuffer => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some(&format!("{} Atomic Buffer Layout", label)),
            })
        },
        BindGroupLayoutType::ExternalTexture => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                            format: COMPUTE_TEXTURE_FORMAT_RGBA16,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
                label: Some(&format!("{} External Texture Layout", label)),
            })
        },
        BindGroupLayoutType::CustomUniform => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some(&format!("{} Custom Uniform Layout", label)),
            })
        },
        BindGroupLayoutType::FontTexture => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Font uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Font atlas texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Font atlas sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some(&format!("{} Font Layout", label)),
            })
        }
        BindGroupLayoutType::FontWithAudio => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // Font uniforms
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Font atlas texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Font atlas sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Audio buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some(&format!("{} Font+Audio Layout", label)),
            })
        }
        BindGroupLayoutType::AudioBuffer => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                label: Some(&format!("{} Audio Buffer Layout", label)),
            })
        }
    }
}

pub fn create_external_texture_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    input_texture_view: &wgpu::TextureView,
    input_sampler: &wgpu::Sampler,
    output_texture_view: &wgpu::TextureView,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
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
                resource: wgpu::BindingResource::TextureView(output_texture_view),
            },
        ],
        label: Some(&format!("{} Bind Group", label)),
    })
}

pub struct ComputeShader {
    pub pipelines: Vec<wgpu::ComputePipeline>,
    pub output_texture: TextureManager,
    pub workgroup_size: [u32; 3],
    pub workgroup_count: Option<[u32; 3]>,
    pub dispatch_once: bool,
    pub current_frame: u32,
    pub time_uniform: UniformBinding<ComputeTimeUniform>,
    pub time_bind_group_layout: wgpu::BindGroupLayout,
    pub storage_texture_layout: wgpu::BindGroupLayout,
    pub storage_bind_group: wgpu::BindGroup,
    pub hot_reload: Option<ShaderHotReload>,
    pub pipeline_layout: wgpu::PipelineLayout,
    pub entry_points: Vec<String>,
    pub atomic_buffer: Option<AtomicBuffer>,
    pub atomic_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub external_texture_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub external_texture_bind_group: Option<wgpu::BindGroup>,
    pub config: Option<ComputeShaderConfig>,
    pub mouse_bind_group: Option<wgpu::BindGroup>,
    pub mouse_bind_group_index: Option<u32>,
    pub font_system: Option<FontSystem>,
    pub font_bind_group: Option<wgpu::BindGroup>,
    pub font_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub audio_buffer: Option<wgpu::Buffer>,
    pub audio_bind_group: Option<wgpu::BindGroup>,
    pub audio_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub audio_staging_buffer: Option<wgpu::Buffer>,
}

impl ComputeShader {
    // Backward compatible constructor
    pub fn new(
        core: &Core,
        shader_source: &str,
        entry_point: &str,
        workgroup_size: [u32; 3],
        workgroup_count: Option<[u32; 3]>,
        dispatch_once: bool,
    ) -> Self {
        let config = ComputeShaderConfig {
            workgroup_size,
            workgroup_count,
            dispatch_once,
            entry_points: vec![entry_point.to_string()],
            ..Default::default()
        };
        
        Self::new_with_config(core, shader_source, config)
    }
    
    pub fn new_with_config(
        core: &Core,
        shader_source: &str,
        config: ComputeShaderConfig,
    ) -> Self {
        let time_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::TimeUniform,
            &config.label
        );
        
        let time_uniform = UniformBinding::new(
            &core.device,
            &format!("{} Time Uniform", config.label),
            ComputeTimeUniform {
                time: 0.0,
                delta: 0.0,
                frame: 0,
                _padding: 0,
            },
            &time_bind_group_layout,
            0,
        );
        
        let storage_texture_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::StorageTexture,
            &config.label
        );
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
        
        let output_texture = create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            config.storage_texture_format,
            &texture_bind_group_layout,
            config.sampler_address_mode,
            config.sampler_filter_mode,
            &format!("{} Output Texture", config.label),
        );
        
        // Create external texture layout if needed
        let external_texture_bind_group_layout = if config.entry_points.len() > 1 {
            Some(create_bind_group_layout(
                &core.device, 
                BindGroupLayoutType::ExternalTexture,
                &config.label
            ))
        } else {
            None
        };
        
        // Create atomic buffer layout if needed
        let atomic_bind_group_layout = if config.enable_atomic_buffer {
            Some(create_bind_group_layout(
                &core.device, 
                BindGroupLayoutType::AtomicBuffer,
                &config.label
            ))
        } else {
            None
        };
        

        let (font_system, font_bind_group_layout) = if config.enable_fonts {
            let font_data = include_bytes!("../../assets/fonts/Courier Prime Bold.ttf");
            let font_system = FontSystem::new(core, font_data);
            let layout = create_bind_group_layout(
                &core.device,
                if config.enable_audio_buffer {
                    BindGroupLayoutType::FontWithAudio
                } else {
                    BindGroupLayoutType::FontTexture
                },
                &config.label
            );
            (Some(font_system), Some(layout))
        } else {
            (None, None)
        };
        
        let audio_bind_group_layout = if config.enable_audio_buffer && !config.enable_fonts {
            // Only create separate audio layout if fonts are not enabled
            // If fonts are enabled, audio is included in the font layout
            Some(create_bind_group_layout(
                &core.device, 
                BindGroupLayoutType::AudioBuffer,
                &config.label
            ))
        } else {
            None
        };
        
        let atomic_buffer = if config.enable_atomic_buffer {
            let buffer_size = core.size.width * core.size.height;
            Some(AtomicBuffer::new(
                &core.device,
                buffer_size,
                atomic_bind_group_layout.as_ref().unwrap(),
            ))
        } else {
            None
        };
        
        let (audio_buffer, audio_staging_buffer, audio_bind_group) = if config.enable_audio_buffer {
            let buffer_size = config.audio_buffer_size * std::mem::size_of::<f32>();
            
            let audio_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Audio Buffer", config.label)),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            
            let staging_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{} Audio Staging Buffer", config.label)),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            // Create bind group - use font layout if fonts are enabled, otherwise use audio layout
            let bind_group = if config.enable_fonts {
                // Audio is combined with fonts in group 3
                None // Will be created with font bind group later
            } else {
                // Audio has its own separate bind group
                Some(core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("{} Audio Bind Group", config.label)),
                    layout: audio_bind_group_layout.as_ref().unwrap(),
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: audio_buffer.as_entire_binding(),
                    }],
                }))
            };
            
            (Some(audio_buffer), Some(staging_buffer), bind_group)
        } else {
            (None, None, None)
        };
        
        let view = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Storage Bind Group", config.label)),
            layout: &storage_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        
        let external_texture_bind_group = if let Some(layout) = &external_texture_bind_group_layout {
            // Temporary solution - using the same texture for input/output
            Some(create_external_texture_bind_group(
                &core.device,
                layout,
                &output_texture.view,
                &output_texture.sampler,
                &view,
                &format!("{} External Texture", config.label),
            ))
        } else {
            None
        };
        
        // Create the shader module
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{} Module", config.label)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create pipeline layout
        let mut bind_group_layouts: Vec<&wgpu::BindGroupLayout> = vec![&time_bind_group_layout, &storage_texture_layout];

        if let Some(ref mouse_layout) = config.mouse_bind_group_layout {
            bind_group_layouts.push(mouse_layout);
        }

        if let Some(layout) = &font_bind_group_layout {
            bind_group_layouts.push(layout);
        }
        
        if let Some(layout) = &external_texture_bind_group_layout {
            bind_group_layouts.push(layout);
        }
        
        if let Some(layout) = &atomic_bind_group_layout {
            bind_group_layouts.push(layout);
        }
        
        if let Some(layout) = &audio_bind_group_layout {
            bind_group_layouts.push(layout);
        }
        
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{} Pipeline Layout", config.label)),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        });
        
        // Create pipelines for each entry point
        let mut pipelines = Vec::new();
        for entry_point in &config.entry_points {
            let pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} Pipeline - {}", config.label, entry_point)),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            pipelines.push(pipeline);
        }
        
        let font_bind_group = if let (Some(fs), Some(layout)) = (&font_system, &font_bind_group_layout) {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: fs.font_uniforms.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&fs.atlas_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&fs.atlas_texture.sampler),
                },
            ];
            
            // Add audio buffer if both fonts and audio are enabled
            if config.enable_audio_buffer {
                if let Some(ref audio_buf) = audio_buffer {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 3,
                        resource: audio_buf.as_entire_binding(),
                    });
                }
            }
            
            Some(core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout,
                entries: &entries,
                label: Some(&format!("{} Font+Audio Bind Group", config.label)),
            }))
        } else {
            None
        };
        
        // Determine final audio bind group before moving values
        let final_audio_bind_group = if config.enable_audio_buffer && config.enable_fonts {
            // When both fonts and audio are enabled, audio is in the font bind group
            font_bind_group.clone()
        } else {
            audio_bind_group
        };
        
        Self {
            pipelines,
            output_texture,
            workgroup_size: config.workgroup_size,
            workgroup_count: config.workgroup_count,
            dispatch_once: config.dispatch_once,
            current_frame: 0,
            time_uniform,
            time_bind_group_layout,
            storage_texture_layout,
            storage_bind_group,
            hot_reload: None,
            pipeline_layout,
            entry_points: config.entry_points.clone(),
            atomic_buffer,
            atomic_bind_group_layout,
            external_texture_bind_group_layout,
            external_texture_bind_group,
            config: Some(config),
            mouse_bind_group: None,
            mouse_bind_group_index: None,
            font_system,
            font_bind_group,
            font_bind_group_layout,
            audio_buffer,
            audio_bind_group: final_audio_bind_group,
            audio_bind_group_layout,
            audio_staging_buffer,
        }
    }
    pub fn add_mouse_uniform_binding(
        &mut self,
        mouse_bind_group: &wgpu::BindGroup,
        bind_group_index: u32
    ) {
        self.mouse_bind_group = Some(mouse_bind_group.clone());
        self.mouse_bind_group_index = Some(bind_group_index);
        
        info!("Added mouse uniform binding at group {} for compute shader", bind_group_index);
    }
    
    // Recreate compute resources after window resize or texture changes
    pub fn recreate_compute_resources(
        &mut self,
        core: &Core,
        input_texture_view: Option<&wgpu::TextureView>,
        input_sampler: Option<&wgpu::Sampler>,
    ) {
        let default_config = ComputeShaderConfig::default();
        let config = self.config.as_ref().unwrap_or(&default_config);
        
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
        
        let output_texture = create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            config.storage_texture_format,
            &texture_bind_group_layout,
            config.sampler_address_mode,
            config.sampler_filter_mode,
            &format!("{} Output Texture", config.label),
        );
        
        self.output_texture = output_texture;
        
        // Recreate atomic buffer if needed
        if let Some(atomic_bind_group_layout) = &self.atomic_bind_group_layout {
            let buffer_size = core.size.width * core.size.height;
            self.atomic_buffer = Some(AtomicBuffer::new(
                &core.device,
                buffer_size,
                atomic_bind_group_layout,
            ));
        }
        
        // Recreate storage bind group
        let view = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Storage Bind Group", config.label)),
            layout: &self.storage_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        
        // Recreate external texture bind group if needed
        if let (Some(layout), Some(in_view), Some(in_sampler)) = (
            &self.external_texture_bind_group_layout,
            input_texture_view,
            input_sampler,
        ) {
            self.external_texture_bind_group = Some(create_external_texture_bind_group(
                &core.device,
                layout,
                in_view,
                in_sampler,
                &view,
                &format!("{} External Texture", config.label),
            ));
        }
    }
    
    pub fn enable_hot_reload(&mut self, 
        device: Arc<wgpu::Device>, 
        shader_path: PathBuf, 
        shader_module: wgpu::ShaderModule,
    ) -> Result<(), notify::Error> {
        let entry_point = self.entry_points.first().cloned().unwrap_or_else(|| "main".to_string());
        let hot_reload = ShaderHotReload::new_compute(
            device,
            shader_path,
            shader_module,
            &entry_point,
        )?;
        
        self.hot_reload = Some(hot_reload);
        Ok(())
    }
    
    pub fn check_hot_reload(&mut self, device: &wgpu::Device) -> bool {
        if let Some(hot_reload) = &mut self.hot_reload {
            // Get the entry point BEFORE we mutably borrow for check_and_reload
            let entry_point = hot_reload.entry_point().map(String::from);
            
            // Call reload_compute_shader directly for compute shaders
            if let Some(new_module) = hot_reload.reload_compute_shader() {
                let entry_points = if let Some(ep) = entry_point {
                    vec![ep]
                } else {
                    self.entry_points.clone()
                };
                
                // Create new pipelines with the updated shader
                let mut new_pipelines = Vec::new();
                for entry_point in &entry_points {
                    let new_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Updated Compute Pipeline"),
                        layout: Some(&self.pipeline_layout),
                        module: &new_module,
                        entry_point: Some(entry_point),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    });
                    new_pipelines.push(new_pipeline);
                }
                
                self.pipelines = new_pipelines;
                info!("Compute shader hot-reloaded at frame: {}", self.current_frame);
                return true;
            }
        }
        false
    }

    pub fn set_time(&mut self, elapsed: f32, delta: f32, queue: &wgpu::Queue) {
        self.time_uniform.data.time = elapsed;
        self.time_uniform.data.delta = delta;
        self.time_uniform.data.frame = self.current_frame;
        self.time_uniform.update(queue);
    }
    
    pub fn dispatch(&mut self, encoder: &mut wgpu::CommandEncoder, core: &Core) {
        self.check_hot_reload(&core.device);
        if self.dispatch_once && self.current_frame > 0 {
            return;
        }
        
        let workgroup_count = self.workgroup_count.unwrap_or([
            core.size.width.div_ceil(self.workgroup_size[0]),
            core.size.height.div_ceil(self.workgroup_size[1]),
            1,
        ]);
        
        // For multi-pass compute shaders (e.g., clear -> process -> generate)
        for (i, pipeline) in self.pipelines.iter().enumerate() {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Compute Pass {}", i)),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &self.time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.storage_bind_group, &[]);
            
            if let (Some(mouse_bind_group), Some(bind_idx)) = (&self.mouse_bind_group, self.mouse_bind_group_index) {
                compute_pass.set_bind_group(bind_idx, mouse_bind_group, &[]);
            }
            
            // If this is a multi-stage compute shader with external textures
            if let Some(external_bind_group) = &self.external_texture_bind_group {
                let external_idx = if self.mouse_bind_group_index == Some(2) { 3 } else { 2 };
                compute_pass.set_bind_group(external_idx, external_bind_group, &[]);
            }
            
            // If atomic buffer is used
            if let Some(atomic_buffer) = &self.atomic_buffer {
                let mut bind_idx = if self.external_texture_bind_group.is_some() { 3 } else { 2 };
                if self.mouse_bind_group_index == Some(2) {
                    bind_idx = if self.external_texture_bind_group.is_some() { 4 } else { 3 };
                }
                
                compute_pass.set_bind_group(bind_idx, &atomic_buffer.bind_group, &[]);
            }
            
            if let Some(font_bind_group) = &self.font_bind_group {
                // Font system uses bind group 3 (time=0, storage=1, mouse=2, font=3)
                compute_pass.set_bind_group(3, font_bind_group, &[]);
            }
            
            // Only bind audio separately if fonts are not enabled
            // When fonts are enabled, audio is included in the font bind group
            if let Some(audio_bind_group) = &self.audio_bind_group {
                if self.font_bind_group.is_none() {
                    // Audio has its own separate bind group only when fonts are disabled
                    let mut audio_idx = 2; // Default after time=0, storage=1
                    if self.mouse_bind_group_index == Some(2) { audio_idx += 1; }
                    if self.external_texture_bind_group.is_some() { audio_idx += 1; }
                    if self.atomic_buffer.is_some() { audio_idx += 1; }
                    
                    compute_pass.set_bind_group(audio_idx, audio_bind_group, &[]);
                }
                // If fonts are enabled, audio is already bound with the font bind group at index 3
            }
            
            compute_pass.dispatch_workgroups(
                workgroup_count[0],
                workgroup_count[1],
                workgroup_count[2],
            );
        }
        
        self.current_frame += 1;
    }
    
    pub fn resize(&mut self, core: &Core, width: u32, height: u32) {
        let default_config = ComputeShaderConfig::default();
        let config = self.config.as_ref().unwrap_or(&default_config);
        
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
        
        let output_texture = create_output_texture(
            &core.device,
            width,
            height,
            config.storage_texture_format,
            &texture_bind_group_layout,
            config.sampler_address_mode,
            config.sampler_filter_mode,
            &format!("{} Output Texture", config.label),
        );
        
        let view = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} Storage Bind Group", config.label)),
            layout: &self.storage_texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        
        // Update atomic buffer if needed
        if let Some(atomic_bind_group_layout) = &self.atomic_bind_group_layout {
            let buffer_size = width * height;
            self.atomic_buffer = Some(AtomicBuffer::new(
                &core.device,
                buffer_size,
                atomic_bind_group_layout,
            ));
        }
        
        self.output_texture = output_texture;
        self.storage_bind_group = storage_bind_group;
    }
    
    pub fn get_output_texture(&self) -> &TextureManager {
        &self.output_texture
    }
    /// NOTE: This buffer reading approach caused crackling audio on macOS when used for real-time playback.
    /// Read GPU-computed audio parameters from shader's audio_buffer
    /// Reduced blocking operations and faster polling for GPU↔CPU parameter communication
    /// GPU shaders write computed frequencies/amplitudes to audio_buffer, CPU reads for real-time synthesis
    pub async fn read_audio_samples(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        if let (Some(audio_buffer), Some(staging_buffer)) = (&self.audio_buffer, &self.audio_staging_buffer) {
            let config = self.config.as_ref().unwrap();
            let buffer_size = config.audio_buffer_size * std::mem::size_of::<f32>();
            
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Audio Buffer Copy"),
            });
            
            encoder.copy_buffer_to_buffer(
                audio_buffer,
                0,
                staging_buffer,
                0,
                buffer_size as u64,
            );
            
            queue.submit(std::iter::once(encoder.finish()));
            
            let buffer_slice = staging_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            
            // A minimal polling to avoid blocking
            device.poll(wgpu::Maintain::Poll);
            
            match rx.try_recv() {
                Ok(Ok(())) => {
                    let data = buffer_slice.get_mapped_range();
                    let samples: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                    drop(data);
                    staging_buffer.unmap();
                    Ok(samples)
                },
                Ok(Err(_)) | Err(_) => {
                    // Mapping failed or not ready - return empty data to avoid blocking
                    Ok(vec![0.0; config.audio_buffer_size])
                },
            }
        } else {
            Ok(Vec::new())
        }
    }
    
    pub fn dispatch_pipeline(&mut self, encoder: &mut wgpu::CommandEncoder, core: &Core, pipeline_index: usize) {
        if pipeline_index >= self.pipelines.len() {
            warn!("Pipeline index {} out of bounds (max: {})", pipeline_index, self.pipelines.len() - 1);
            return;
        }
        
        self.check_hot_reload(&core.device);
        
        if self.dispatch_once && self.current_frame > 0 {
            return;
        }
        
        let workgroup_count = self.workgroup_count.unwrap_or([
            core.size.width.div_ceil(self.workgroup_size[0]),
            core.size.height.div_ceil(self.workgroup_size[1]),
            1,
        ]);
        
        let pipeline = &self.pipelines[pipeline_index];
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("Compute Pass {}", pipeline_index)),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &self.time_uniform.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.storage_bind_group, &[]);
        
        if let (Some(mouse_bind_group), Some(bind_idx)) = (&self.mouse_bind_group, self.mouse_bind_group_index) {
            compute_pass.set_bind_group(bind_idx, mouse_bind_group, &[]);
        }
        
        // If this is a multi-stage compute shader with external textures
        if let Some(external_bind_group) = &self.external_texture_bind_group {
            let external_idx = if self.mouse_bind_group_index == Some(2) { 3 } else { 2 };
            compute_pass.set_bind_group(external_idx, external_bind_group, &[]);
        }
        
        // If atomic buffer is used
        if let Some(atomic_buffer) = &self.atomic_buffer {
            let mut bind_idx = if self.external_texture_bind_group.is_some() { 3 } else { 2 };
            
            if self.mouse_bind_group_index == Some(2) {
                bind_idx = if self.external_texture_bind_group.is_some() { 4 } else { 3 };
            }
            
            compute_pass.set_bind_group(bind_idx, &atomic_buffer.bind_group, &[]);
        }
        
        // If font system is used
        if let Some(font_bind_group) = &self.font_bind_group {
            // Font system uses bind group 3 (time=0, storage=1, mouse=2, font=3)
            compute_pass.set_bind_group(3, font_bind_group, &[]);
        }
        
        compute_pass.dispatch_workgroups(
            workgroup_count[0],
            workgroup_count[1],
            workgroup_count[2],
        );
        
        // Only increment the frame counter if this is the last pipeline in sequence
        if pipeline_index == self.pipelines.len() - 1 {
            self.current_frame += 1;
        }
    }
}