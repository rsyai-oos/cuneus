use crate::{Core, UniformProvider, UniformBinding, TextureManager, ShaderHotReload, AtomicBuffer, FontSystem};
use std::sync::Arc;
use std::path::PathBuf;
use std::collections::HashMap;
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
    pub enable_custom_uniform: bool,
    pub enable_input_texture: bool,
    pub custom_storage_buffers: Vec<CustomStorageBuffer>,
}

#[derive(Clone)]
pub struct CustomStorageBuffer {
    pub label: String,
    pub size: u64,
    pub usage: wgpu::BufferUsages,
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
            enable_custom_uniform: false,
            enable_input_texture: false,
            custom_storage_buffers: Vec::new(),
        }
    }
}

//bind group layout types for different shader needs
pub enum BindGroupLayoutType {
    StorageTexture,
    StorageTextureWithInput,
    StorageTextureWithFonts, // New layout for CNN-style shaders (output texture + font atlas + font sampler)
    TextureWithInputAndFonts, // Legacy layout
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
        BindGroupLayoutType::StorageTextureWithInput => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} Storage Texture with Input Layout", label)),
                entries: &[
                    // Output storage texture at binding 0 (pathtracing expects this order)
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
                    // Input texture at binding 1
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
                    // Input sampler at binding 2
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            })
        },
        BindGroupLayoutType::StorageTextureWithFonts => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} Storage Texture with Fonts Layout", label)),
                entries: &[
                    // Output storage texture at binding 0
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
                    // Font atlas texture at binding 1
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
                    // Font sampler at binding 2
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            })
        },
        BindGroupLayoutType::TextureWithInputAndFonts => {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} Texture with Input and Fonts Layout", label)),
                entries: &[
                    // Input texture at binding 0 (CNN expects this order)
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
                    // Input sampler at binding 1
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // Output storage texture at binding 2
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
                    // Font atlas texture at binding 3
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Font sampler at binding 4
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
    pub custom_uniform_bind_group: Option<wgpu::BindGroup>,
    pub custom_uniform_bind_group_layout: Option<wgpu::BindGroupLayout>,
    pub custom_storage_buffers: Vec<wgpu::Buffer>,
    pub custom_storage_bind_group: Option<wgpu::BindGroup>,
    pub custom_storage_bind_group_layout: Option<wgpu::BindGroupLayout>,
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
            if config.enable_input_texture && config.enable_fonts {
                BindGroupLayoutType::TextureWithInputAndFonts
            } else if config.enable_input_texture { 
                BindGroupLayoutType::StorageTextureWithInput 
            } else if config.enable_fonts {
                BindGroupLayoutType::StorageTextureWithFonts
            } else { 
                BindGroupLayoutType::StorageTexture 
            },
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
        
        // Create external texture layout if needed (but not if input texture is already handled or atomic buffer is used)
        // For particle systems with atomic buffers, we don't need external textures since atomic accumulation handles multi-pass
        let external_texture_bind_group_layout = if config.entry_points.len() > 1 && !config.enable_input_texture && !config.enable_atomic_buffer {
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
        let storage_bind_group = if !config.enable_input_texture && config.enable_fonts {
            // For StorageTextureWithFonts layout, we need output storage, font atlas, font sampler
            // Get font textures - use font system if available, otherwise use output as fallback
            let (font_view, font_sampler) = if let Some(ref font_system) = font_system {
                (&font_system.atlas_texture.view, &font_system.atlas_texture.sampler)
            } else {
                (&output_texture.view, &output_texture.sampler) // Fallback
            };
            
            core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group with Fonts", config.label)),
                layout: &storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view), // Output storage texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(font_view), // Font atlas
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(font_sampler), // Font sampler
                    },
                ],
            })
        } else if config.enable_input_texture && config.enable_fonts {
            // For TextureWithInputAndFonts layout, we need input texture, input sampler, output storage, font atlas, font sampler
            // Create dummy textures to avoid usage conflicts during initialization
            let dummy_texture = core.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy Input Texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let dummy_sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
            
            // Get font textures - use font system if available, otherwise use dummy
            let (font_view, font_sampler) = if let Some(ref font_system) = font_system {
                (&font_system.atlas_texture.view, &font_system.atlas_texture.sampler)
            } else {
                (&dummy_view, &dummy_sampler) // Fallback to dummy
            };
            
            core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group with Input and Fonts", config.label)),
                layout: &storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&dummy_view), // Input texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&dummy_sampler), // Input sampler
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view), // Output storage texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(font_view), // Font atlas
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(font_sampler), // Font sampler
                    },
                ],
            })
        } else if config.enable_input_texture {
            // For StorageTextureWithInput layout, we need output texture, input texture, and sampler
            // Create a separate dummy texture to avoid usage conflicts
            let dummy_texture = core.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy Input Texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let dummy_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let dummy_sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
            
            core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group with Input", config.label)),
                layout: &storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view), // Output texture first
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&dummy_view), // Dummy input texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&dummy_sampler), // Dummy sampler
                    },
                ],
            })
        } else {
            // Standard case - just output texture
            core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group", config.label)),
                layout: &storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            })
        };
        
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
        
        // Create custom uniform layout if needed
        let custom_uniform_bind_group_layout = if config.enable_custom_uniform {
            Some(create_bind_group_layout(
                &core.device,
                BindGroupLayoutType::CustomUniform,
                &config.label,
            ))
        } else {
            None
        };
        
        // Create dummy layout for Group 2 if needed to maintain contiguous bind group indices
        let dummy_group2_layout = if !config.enable_custom_uniform && config.mouse_bind_group_layout.is_none() &&
                                    (config.enable_fonts || config.enable_audio_buffer || 
                                     !config.custom_storage_buffers.is_empty() || config.enable_atomic_buffer) {
            Some(create_bind_group_layout(
                &core.device, 
                BindGroupLayoutType::TimeUniform,  // Reuse time uniform layout as dummy
                "Dummy Group 2"
            ))
        } else {
            None
        };

        // Create custom storage buffers if specified
        let mut custom_storage_buffers = Vec::new();
        let custom_storage_bind_group_layout = if !config.custom_storage_buffers.is_empty() {
            // Create storage buffers
            for buffer_config in &config.custom_storage_buffers {
                let buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&buffer_config.label),
                    size: buffer_config.size,
                    usage: buffer_config.usage,
                    mapped_at_creation: false,
                });
                custom_storage_buffers.push(buffer);
            }

            // Create bind group layout for custom storage buffers
            let mut entries = Vec::new();
            for (i, _) in config.custom_storage_buffers.iter().enumerate() {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }

            Some(core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("{} Custom Storage Layout", config.label)),
                entries: &entries,
            }))
        } else {
            None
        };

        // Create pipeline layout following traditional Cuneus shader layout:
        // Group 0: Time uniform
        // Group 1: Storage texture (output) 
        // Group 2: Custom uniforms (mouse, params, etc.)
        // Group 3: Fonts and/or Audio (combined when both enabled)
        let mut bind_group_layouts: Vec<&wgpu::BindGroupLayout> = vec![];
        
        // Group 0: Time uniform (always present)
        bind_group_layouts.push(&time_bind_group_layout);
        
        // Group 1: Storage texture (always present - this is what shaders expect at group 1)
        bind_group_layouts.push(&storage_texture_layout);
        
        // Group 2: Custom uniform OR mouse uniform OR dummy (to maintain contiguous indices)
        if config.enable_custom_uniform {
            if let Some(ref custom_layout) = custom_uniform_bind_group_layout {
                bind_group_layouts.push(custom_layout);
            }
        } else if let Some(ref mouse_layout) = config.mouse_bind_group_layout {
            bind_group_layouts.push(mouse_layout);
        } else if let Some(ref dummy_layout) = dummy_group2_layout {
            // Use dummy layout to maintain contiguous bind group indices
            bind_group_layouts.push(dummy_layout);
        }
        
        // Group 3: Fonts and/or Audio, OR custom storage buffers (prioritize custom storage for CNN-style shaders)
        if let Some(layout) = &custom_storage_bind_group_layout {
            // CNN-style shaders with custom storage buffers
            bind_group_layouts.push(layout);
        } else if let Some(layout) = &font_bind_group_layout {
            // Standard shaders with fonts (may include audio buffer at binding 3)
            bind_group_layouts.push(layout);
        } else if let Some(layout) = &audio_bind_group_layout {
            // Audio-only shaders
            bind_group_layouts.push(layout);
        } else if let Some(layout) = &atomic_bind_group_layout {
            // Particle system shaders with atomic buffers
            bind_group_layouts.push(layout);
        } else if let Some(layout) = &external_texture_bind_group_layout {
            // Multi-pass shaders
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
        
        // Create custom storage bind group if we have custom storage buffers
        let custom_storage_bind_group = if !custom_storage_buffers.is_empty() && custom_storage_bind_group_layout.is_some() {
            let mut entries = Vec::new();
            for (i, buffer) in custom_storage_buffers.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: None,
                    }),
                });
            }

            Some(core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Custom Storage Bind Group", config.label)),
                layout: custom_storage_bind_group_layout.as_ref().unwrap(),
                entries: &entries,
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
            custom_uniform_bind_group: None,
            custom_uniform_bind_group_layout,
            custom_storage_buffers,
            custom_storage_bind_group,
            custom_storage_bind_group_layout,
        }
    }
    pub fn add_mouse_uniform_binding(
        &mut self,
        mouse_bind_group: &wgpu::BindGroup,
        bind_group_index: u32
    ) {
        self.mouse_bind_group = Some(mouse_bind_group.clone());
        self.mouse_bind_group_index = Some(bind_group_index);
    }
    
    pub fn add_custom_uniform_binding(&mut self, custom_bind_group: &wgpu::BindGroup) {
        self.custom_uniform_bind_group = Some(custom_bind_group.clone());
    }
    
    pub fn override_storage_bind_group(&mut self, bind_group: wgpu::BindGroup) {
        self.storage_bind_group = bind_group;
    }
    
    pub fn get_storage_layout(&self) -> &wgpu::BindGroupLayout {
        &self.storage_texture_layout
    }
    
    pub fn clear_atomic_buffer(&self, core: &Core) {
        if let Some(atomic_buffer) = &self.atomic_buffer {
            // The atomic buffer size is determined by the config when created
            // Use 3 as default multiplier (RGB channels) which matches mandelbulb usage
            let multiplier = self.config.as_ref().map(|c| c.atomic_buffer_multiples).unwrap_or(3);
            let buffer_size = core.size.width * core.size.height * multiplier as u32;
            let clear_data = vec![0u32; buffer_size as usize];
            
            core.queue.write_buffer(
                &atomic_buffer.buffer,
                0,
                bytemuck::cast_slice(&clear_data),
            );
        }
    }
    
    pub fn update_input_texture(&mut self, core: &Core, input_view: &wgpu::TextureView, input_sampler: &wgpu::Sampler) {
        if let Some(config) = &self.config {
            if config.enable_input_texture {
                let output_view = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
                
                // CRITICAL: Ensure input and output textures are truly separate to prevent WebGPU usage conflicts
                // This is especially important for complex multi-stage algorithms like FFT
                self.storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("{} Storage Bind Group with Separated Input/Output", config.label)),
                    layout: &self.storage_texture_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&output_view), // Output storage texture (write-only)
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(input_view), // Input texture (read-only, must be different from output)
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(input_sampler), // Input sampler
                        },
                    ],
                });
            }
        }
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
        
        // Recreate storage bind group - handle all layout types
        let view = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        if !config.enable_input_texture && config.enable_fonts {
            // For StorageTextureWithFonts layout, we need output storage, font atlas, font sampler
            // Get font textures - use font system if available, otherwise use output as fallback
            let (font_view, font_sampler) = if let Some(ref font_system) = self.font_system {
                (&font_system.atlas_texture.view, &font_system.atlas_texture.sampler)
            } else {
                (&self.output_texture.view, &self.output_texture.sampler) // Fallback
            };
            
            self.storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group with Fonts", config.label)),
                layout: &self.storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view), // Output storage texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(font_view), // Font atlas
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(font_sampler), // Font sampler
                    },
                ],
            });
        } else if config.enable_input_texture && config.enable_fonts {
            // For TextureWithInputAndFonts layout, we need input texture, input sampler, output storage, font atlas, font sampler
            // Create separate dummy texture to avoid usage conflicts
            let dummy_texture = core.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy Input Texture for Resize"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let default_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let default_sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
            
            // Get font textures - use defaults if font system not available
            let (font_view, font_sampler) = if let Some(ref font_system) = self.font_system {
                (&font_system.atlas_texture.view, &font_system.atlas_texture.sampler)
            } else {
                (&default_view, &default_sampler) // Fallback to default
            };
            
            self.storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group with Input and Fonts", config.label)),
                layout: &self.storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&default_view), // Input texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&default_sampler), // Input sampler
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view), // Output storage texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(font_view), // Font atlas
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(font_sampler), // Font sampler
                    },
                ],
            });
        } else if config.enable_input_texture {
            // For StorageTextureWithInput layout, we need output texture, input texture, and sampler
            // Create separate dummy texture to avoid usage conflicts - never use output as input
            let dummy_texture = core.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy Input Texture for Default"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let default_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());
            let default_sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
            
            self.storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group with Input", config.label)),
                layout: &self.storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view), // Output storage texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&default_view), // Input texture (separate dummy)
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&default_sampler), // Sampler
                    },
                ],
            });
        } else {
            // For regular StorageTexture layout, only output texture
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
        }
        
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
            
            // Call reload_compute_shader directly for compute shaders
            if let Some(new_module) = hot_reload.reload_compute_shader() {
                // Always use all original entry points for multi-stage shaders
                let entry_points = self.entry_points.clone();
                
                // Create new pipelines with the updated shader
                let mut new_pipelines = Vec::new();
                for entry_point in &entry_points {
                    let new_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(&format!("Updated Compute Pipeline - {}", entry_point)),
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
            
            // Group 0: Time uniform (always present)
            compute_pass.set_bind_group(0, &self.time_uniform.bind_group, &[]);
            
            // Group 1: Storage texture (always present) 
            compute_pass.set_bind_group(1, &self.storage_bind_group, &[]);
            
            // Group 2: Custom uniform OR Mouse uniform (depending on configuration)
            // Only bind if we have actual data - dummy Group 2 layouts don't need binding
            if let Some(custom_bind_group) = &self.custom_uniform_bind_group {
                compute_pass.set_bind_group(2, custom_bind_group, &[]);
            } else if let Some(mouse_bind_group) = &self.mouse_bind_group {
                compute_pass.set_bind_group(2, mouse_bind_group, &[]);
            }
            // Note: If neither exists, Group 2 is a dummy layout for contiguous indices and doesn't need binding
            
            // Group 3: Custom storage buffers OR Fonts+Audio OR Atomic buffer OR External texture
            // (same priority order as pipeline layout creation)
            if let Some(custom_storage_bind_group) = &self.custom_storage_bind_group {
                compute_pass.set_bind_group(3, custom_storage_bind_group, &[]);
            } else if let Some(font_bind_group) = &self.font_bind_group {
                // Font bind group includes audio buffer at binding 3 if both are enabled
                compute_pass.set_bind_group(3, font_bind_group, &[]);
            } else if let Some(audio_bind_group) = &self.audio_bind_group {
                // Audio-only bind group
                compute_pass.set_bind_group(3, audio_bind_group, &[]);
            } else if let Some(atomic_buffer) = &self.atomic_buffer {
                compute_pass.set_bind_group(3, &atomic_buffer.bind_group, &[]);
            } else if let Some(external_bind_group) = &self.external_texture_bind_group {
                compute_pass.set_bind_group(3, external_bind_group, &[]);
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
        
        let storage_bind_group = if !config.enable_input_texture && config.enable_fonts {
            // For StorageTextureWithFonts layout, we need output storage, font atlas, font sampler
            // Get font textures - use font system if available, otherwise use output as fallback
            let (font_view, font_sampler) = if let Some(ref font_system) = self.font_system {
                (&font_system.atlas_texture.view, &font_system.atlas_texture.sampler)
            } else {
                (&output_texture.view, &output_texture.sampler) // Fallback
            };
            
            core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group with Fonts", config.label)),
                layout: &self.storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view), // Output storage texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(font_view), // Font atlas
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(font_sampler), // Font sampler
                    },
                ],
            })
        } else if config.enable_input_texture && config.enable_fonts {
            // For TextureWithInputAndFonts layout, we need input texture, input sampler, output storage, font atlas, font sampler
            let default_view = &output_texture.view;  // Use output as default input
            let default_sampler = &output_texture.sampler;
            
            // Get font textures - use defaults if font system not available
            let (font_view, font_sampler) = if let Some(ref font_system) = self.font_system {
                (&font_system.atlas_texture.view, &font_system.atlas_texture.sampler)
            } else {
                (default_view, default_sampler) // Fallback to default
            };
            
            core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group with Input and Fonts", config.label)),
                layout: &self.storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(default_view), // Input texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(default_sampler), // Input sampler
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&view), // Output storage texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(font_view), // Font atlas
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(font_sampler), // Font sampler
                    },
                ],
            })
        } else if config.enable_input_texture {
            // For StorageTextureWithInput layout, we need output texture, input texture, and sampler
            // Use default texture manager if no specific input texture is set
            let default_view = &output_texture.view;  // Use output as default input
            let default_sampler = &output_texture.sampler;
            
            core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group with Input", config.label)),
                layout: &self.storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view), // Output storage texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(default_view), // Input texture
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(default_sampler), // Sampler
                    },
                ],
            })
        } else {
            // For regular StorageTexture layout, only output texture
            core.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("{} Storage Bind Group", config.label)),
                layout: &self.storage_texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                ],
            })
        };
        
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
            
            let _ = device.poll(wgpu::PollType::Wait);
            
            match rx.recv() {
                Ok(Ok(())) => {},
                Ok(Err(e)) => return Err(e.into()),
                Err(_) => return Err("Buffer mapping failed".into()),
            }
            
            let samples = {
                let data = buffer_slice.get_mapped_range();
                let samples: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                samples
            };
            
            staging_buffer.unmap();
            
            Ok(samples)
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
        
        let mut current_bind_idx = 1;
        
        // Custom uniform at bind group 1 if enabled
        if let Some(custom_bind_group) = &self.custom_uniform_bind_group {
            compute_pass.set_bind_group(current_bind_idx, custom_bind_group, &[]);
            current_bind_idx += 1;
        }
        
        // Storage texture comes after custom uniform
        compute_pass.set_bind_group(current_bind_idx, &self.storage_bind_group, &[]);
        current_bind_idx += 1;
        
        // Mouse uniform at next available slot
        if let (Some(mouse_bind_group), Some(_)) = (&self.mouse_bind_group, self.mouse_bind_group_index) {
            compute_pass.set_bind_group(current_bind_idx, mouse_bind_group, &[]);
            current_bind_idx += 1;
        }
        
        // If this is a multi-stage compute shader with external textures
        if let Some(external_bind_group) = &self.external_texture_bind_group {
            compute_pass.set_bind_group(current_bind_idx, external_bind_group, &[]);
            current_bind_idx += 1;
        }
        
        // If atomic buffer is used
        if let Some(atomic_buffer) = &self.atomic_buffer {
            compute_pass.set_bind_group(current_bind_idx, &atomic_buffer.bind_group, &[]);
            current_bind_idx += 1;
        }
        
        // If font system is used
        if let Some(font_bind_group) = &self.font_bind_group {
            compute_pass.set_bind_group(current_bind_idx, font_bind_group, &[]);
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

    /// Dispatch a specific stage by entry point index for multi-stage pipelines
    pub fn dispatch_stage(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        stage_index: usize,
        workgroups: (u32, u32, u32),
        custom_uniforms: Option<&wgpu::BindGroup>,
    ) {
        if stage_index >= self.pipelines.len() {
            eprintln!("Warning: Stage index {} out of bounds for {} pipelines", stage_index, self.pipelines.len());
            return;
        }

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("Compute Pass - Stage {}", stage_index)),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&self.pipelines[stage_index]);

        // Bind group 0: Time uniform (always present)
        compute_pass.set_bind_group(0, &self.time_uniform.bind_group, &[]);

        // Bind group 1: Storage texture (always present)
        compute_pass.set_bind_group(1, &self.storage_bind_group, &[]);

        // Bind group 2: Custom uniform or mouse uniform (depending on configuration)
        // Only bind if we have actual data - dummy Group 2 layouts don't need binding
        if let Some(custom_bind_group) = custom_uniforms {
            compute_pass.set_bind_group(2, custom_bind_group, &[]);
        } else if let Some(ref custom_uniform_bind_group) = self.custom_uniform_bind_group {
            compute_pass.set_bind_group(2, custom_uniform_bind_group, &[]);
        } else if let Some(ref mouse_bind_group) = self.mouse_bind_group {
            compute_pass.set_bind_group(2, mouse_bind_group, &[]);
        }
        // Note: If none exist, Group 2 is a dummy layout for contiguous indices and doesn't need binding

        // Bind group 3: Custom storage, atomic buffer, or other optional buffers
        if let Some(ref custom_storage_bind_group) = self.custom_storage_bind_group {
            compute_pass.set_bind_group(3, custom_storage_bind_group, &[]);
        } else if let Some(ref atomic_buffer) = self.atomic_buffer {
            compute_pass.set_bind_group(3, &atomic_buffer.bind_group, &[]);
        } else if let Some(ref external_texture_bind_group) = self.external_texture_bind_group {
            compute_pass.set_bind_group(3, external_texture_bind_group, &[]);
        } else if let Some(ref font_bind_group) = self.font_bind_group {
            compute_pass.set_bind_group(3, font_bind_group, &[]);
        } else if let Some(ref audio_bind_group) = self.audio_bind_group {
            compute_pass.set_bind_group(3, audio_bind_group, &[]);
        }

        compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }
}

/// Multi-buffer manager for ping-pong buffer workflows
pub struct MultiBufferManager {
    buffers: HashMap<String, (wgpu::Texture, wgpu::Texture)>,
    bind_groups: HashMap<String, (wgpu::BindGroup, wgpu::BindGroup)>,
    output_texture: wgpu::Texture,
    output_bind_group: wgpu::BindGroup,
    storage_layout: wgpu::BindGroupLayout,
    multi_texture_layout: wgpu::BindGroupLayout,
    frame_flip: bool,
    width: u32,
    height: u32,
}

impl MultiBufferManager {
    pub fn new(
        core: &Core,
        buffer_names: &[&str],
        texture_format: wgpu::TextureFormat,
    ) -> Self {
        let storage_layout = create_bind_group_layout(
            &core.device,
            BindGroupLayoutType::StorageTexture,
            "Multi-Buffer Storage",
        );

        // Create multi-texture layout for reading multiple buffers
        let multi_texture_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // Up to 3 textures + 3 samplers (ichannel0, ichannel1, ichannel2)
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
            label: Some("Multi-Buffer Input Layout"),
        });

        let width = core.size.width;
        let height = core.size.height;

        let mut buffers = HashMap::new();
        let mut bind_groups = HashMap::new();

        // Create ping-pong texture pairs for each buffer
        for &name in buffer_names {
            let texture0 = Self::create_storage_texture(&core.device, width, height, texture_format, &format!("{}_0", name));
            let texture1 = Self::create_storage_texture(&core.device, width, height, texture_format, &format!("{}_1", name));
            
            let bind_group0 = Self::create_storage_bind_group(&core.device, &storage_layout, &texture0, &format!("{}_0_bind", name));
            let bind_group1 = Self::create_storage_bind_group(&core.device, &storage_layout, &texture1, &format!("{}_1_bind", name));

            buffers.insert(name.to_string(), (texture0, texture1));
            bind_groups.insert(name.to_string(), (bind_group0, bind_group1));
        }

        // Create output texture
        let output_texture = Self::create_storage_texture(&core.device, width, height, texture_format, "multi_buffer_output");
        let output_bind_group = Self::create_storage_bind_group(&core.device, &storage_layout, &output_texture, "output_bind");

        Self {
            buffers,
            bind_groups,
            output_texture,
            output_bind_group,
            storage_layout,
            multi_texture_layout,
            frame_flip: false,
            width,
            height,
        }
    }

    fn create_storage_texture(device: &wgpu::Device, width: u32, height: u32, format: wgpu::TextureFormat, label: &str) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
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

    /// Get the bind group for writing to a buffer (current frame)
    pub fn get_write_bind_group(&self, buffer_name: &str) -> &wgpu::BindGroup {
        let bind_groups = self.bind_groups.get(buffer_name).expect("Buffer not found");
        if self.frame_flip {
            &bind_groups.1
        } else {
            &bind_groups.0
        }
    }

    /// Get the texture for reading from a buffer (previous frame)
    pub fn get_read_texture(&self, buffer_name: &str) -> &wgpu::Texture {
        let textures = self.buffers.get(buffer_name).expect("Buffer not found");
        if self.frame_flip {
            &textures.0
        } else {
            &textures.1
        }
    }

    /// Get the texture for writing to a buffer (current frame)
    pub fn get_write_texture(&self, buffer_name: &str) -> &wgpu::Texture {
        let textures = self.buffers.get(buffer_name).expect("Buffer not found");
        if self.frame_flip {
            &textures.1
        } else {
            &textures.0
        }
    }

    /// Create a bind group for reading multiple textures (ichannel pattern)
    pub fn create_input_bind_group(&self, device: &wgpu::Device, sampler: &wgpu::Sampler, channels: &[&str]) -> wgpu::BindGroup {
        // Create views that will live long enough
        let mut views = Vec::new();
        
        // Fill up to 3 channels, using the first channel for unused slots
        for i in 0..3 {
            let channel_name = channels.get(i).unwrap_or(&channels[0]);
            let texture = self.get_read_texture(channel_name);
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            views.push(view);
        }
        
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&views[0]),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&views[1]),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&views[2]),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ];

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.multi_texture_layout,
            entries: &entries,
            label: Some("Multi-Buffer Input"),
        })
    }

    /// Get the output texture bind group
    pub fn get_output_bind_group(&self) -> &wgpu::BindGroup {
        &self.output_bind_group
    }

    /// Get the output texture for display
    pub fn get_output_texture(&self) -> &wgpu::Texture {
        &self.output_texture
    }

    /// Flip ping-pong buffers
    pub fn flip_buffers(&mut self) {
        self.frame_flip = !self.frame_flip;
    }

    /// Clear all buffers by recreating them
    pub fn clear_all(&mut self, core: &Core, texture_format: wgpu::TextureFormat) {
        // Recreate all buffer textures
        for (name, textures) in &mut self.buffers {
            textures.0 = Self::create_storage_texture(&core.device, self.width, self.height, texture_format, &format!("{}_0", name));
            textures.1 = Self::create_storage_texture(&core.device, self.width, self.height, texture_format, &format!("{}_1", name));
        }

        // Recreate all bind groups
        for (name, bind_groups) in &mut self.bind_groups {
            let textures = self.buffers.get(name).unwrap();
            bind_groups.0 = Self::create_storage_bind_group(&core.device, &self.storage_layout, &textures.0, &format!("{}_0_bind", name));
            bind_groups.1 = Self::create_storage_bind_group(&core.device, &self.storage_layout, &textures.1, &format!("{}_1_bind", name));
        }

        // Recreate output texture and bind group
        self.output_texture = Self::create_storage_texture(&core.device, self.width, self.height, texture_format, "multi_buffer_output");
        self.output_bind_group = Self::create_storage_bind_group(&core.device, &self.storage_layout, &self.output_texture, "output_bind");

        self.frame_flip = false;
    }

    /// Resize all buffers
    pub fn resize(&mut self, core: &Core, width: u32, height: u32, texture_format: wgpu::TextureFormat) {
        self.width = width;
        self.height = height;
        self.clear_all(core, texture_format);
    }

    /// Get the multi-texture layout for pipeline creation
    pub fn get_multi_texture_layout(&self) -> &wgpu::BindGroupLayout {
        &self.multi_texture_layout
    }

    /// Get the storage layout for pipeline creation
    pub fn get_storage_layout(&self) -> &wgpu::BindGroupLayout {
        &self.storage_layout
    }
}

/// Multi-buffer compute pipeline manager
pub struct MultiBufferCompute<P: UniformProvider> {
    pub buffer_manager: MultiBufferManager,
    pub params_uniform: UniformBinding<P>,
    pub time_uniform: UniformBinding<ComputeTimeUniform>,
    pub pipelines: HashMap<String, wgpu::ComputePipeline>,
    pub hot_reload: ShaderHotReload,
    pub frame_count: u32,
    pub custom_buffers: Vec<(String, wgpu::Buffer)>,
    pub atomic_buffer: Option<AtomicBuffer>,
    pub atomic_layout: Option<wgpu::BindGroupLayout>,
}

impl<P: UniformProvider> MultiBufferCompute<P> {
    pub fn new(
        core: &Core,
        buffer_names: &[&str],
        shader_path: &str,
        entry_points: &[&str],
        params: P,
    ) -> Self {
        Self::new_with_storage_buffer(core, buffer_names, shader_path, entry_points, params, None)
    }
    
    /// Create with optional storage buffer for FFT-style workflows
    pub fn new_with_storage_buffer(
        core: &Core,
        buffer_names: &[&str],
        shader_path: &str,
        entry_points: &[&str],
        params: P,
        _storage_buffer_size: Option<u64>,
    ) -> Self {
        let buffer_manager = MultiBufferManager::new(core, buffer_names, COMPUTE_TEXTURE_FORMAT_RGBA16);

        // Create uniforms
        let time_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::TimeUniform, "Multi-Buffer Time");
        let params_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::CustomUniform, "Multi-Buffer Params");

        let params_uniform = UniformBinding::new(
            &core.device,
            "Multi-Buffer Params",
            params,
            &params_layout,
            0,
        );

        let time_uniform = UniformBinding::new(
            &core.device,
            "Multi-Buffer Time",
            ComputeTimeUniform {
                time: 0.0,
                delta: 1.0/60.0,
                frame: 0,
                _padding: 0,
            },
            &time_layout,
            0,
        );

        // Load shader
        let shader_source = std::fs::read_to_string(shader_path).expect("Failed to read shader file");
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Multi-Buffer Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let hot_reload = ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from(shader_path),
            shader_module.clone(),
            entry_points[0],
        ).expect("Failed to initialize hot reload");

        // Create pipeline layout
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Multi-Buffer Pipeline Layout"),
            bind_group_layouts: &[
                &time_layout,
                &params_layout,
                buffer_manager.get_storage_layout(),
                buffer_manager.get_multi_texture_layout(),
            ],
            push_constant_ranges: &[],
        });

        // Create pipelines for each entry point
        let mut pipelines = HashMap::new();
        for &entry_point in entry_points {
            let pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Multi-Buffer Pipeline - {}", entry_point)),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            pipelines.insert(entry_point.to_string(), pipeline);
        }

        Self {
            buffer_manager,
            params_uniform,
            time_uniform,
            pipelines,
            hot_reload,
            frame_count: 0,
            custom_buffers: Vec::new(),
            atomic_buffer: None,
            atomic_layout: None,
        }
    }

    /// Dispatch a specific buffer computation
    pub fn dispatch_buffer(&mut self, encoder: &mut wgpu::CommandEncoder, core: &Core, buffer_name: &str, input_channels: &[&str]) {
        if let Some(pipeline) = self.pipelines.get(buffer_name) {
            let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
            let input_bind_group = self.buffer_manager.create_input_bind_group(&core.device, &sampler, input_channels);
            let write_bind_group = self.buffer_manager.get_write_bind_group(buffer_name);

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Multi-Buffer Pass - {}", buffer_name)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &self.time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, write_bind_group, &[]);
            compute_pass.set_bind_group(3, &input_bind_group, &[]);

            let width = core.size.width.div_ceil(16);
            let height = core.size.height.div_ceil(16);
            compute_pass.dispatch_workgroups(width, height, 1);
        }
    }
    
    /// Create MultiBufferCompute with custom storage buffers for specialized shaders (particle systems, etc)
    pub fn new_with_custom_buffers(
        core: &Core,
        buffer_names: &[&str],
        shader_path: &str,
        entry_points: &[&str],
        params: P,
        custom_buffers: &[(String, u64)], // (label, size) pairs
    ) -> Self {
        let buffer_manager = MultiBufferManager::new(core, buffer_names, COMPUTE_TEXTURE_FORMAT_RGBA16);

        // Create uniforms
        let time_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::TimeUniform, "Custom Multi-Buffer Time");
        let params_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::CustomUniform, "Custom Multi-Buffer Params");

        let params_uniform = UniformBinding::new(
            &core.device,
            "Custom Multi-Buffer Params",
            params,
            &params_layout,
            0,
        );

        let time_uniform = UniformBinding::new(
            &core.device,
            "Custom Multi-Buffer Time",
            ComputeTimeUniform {
                time: 0.0,
                delta: 1.0/60.0,
                frame: 0,
                _padding: 0,
            },
            &time_layout,
            0,
        );

        // Create custom storage buffers
        let mut storage_buffers = Vec::new();
        for (label, size) in custom_buffers {
            let buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: *size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            storage_buffers.push((label.clone(), buffer));
        }

        // Create atomic buffer for advanced compute operations
        let atomic_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::AtomicBuffer, "Custom Atomic");
        let buffer_size = core.size.width * core.size.height * 4;
        let atomic_buffer = AtomicBuffer::new(&core.device, buffer_size, &atomic_layout);

        // Create shader module
        let shader_source = std::fs::read_to_string(shader_path)
            .unwrap_or_else(|_| panic!("Failed to read shader file: {}", shader_path));
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Custom Multi-Buffer Shader - {}", shader_path)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create pipeline layout with custom storage buffers
        let bind_group_layouts = vec![
            &time_layout,
            &params_layout,
            buffer_manager.get_storage_layout(),
            &atomic_layout,
        ];
        
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("Custom Multi-Buffer Pipeline Layout - {}", shader_path)),
            bind_group_layouts: &bind_group_layouts,
            push_constant_ranges: &[],
        });

        // Create pipelines for each entry point
        let mut pipelines = HashMap::new();
        for &entry_point in entry_points {
            let pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Custom Multi-Buffer Pipeline - {}", entry_point)),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            pipelines.insert(entry_point.to_string(), pipeline);
        }

        // Set up hot reload
        let hot_reload = ShaderHotReload::new_compute(
            core.device.clone(),
            std::path::PathBuf::from(shader_path),
            shader_module,
            entry_points.first().unwrap_or(&"main"),
        ).unwrap_or_else(|e| {
            eprintln!("Failed to set up hot reload for {}: {}", shader_path, e);
            // Create a dummy hot reload that won't work
            ShaderHotReload::new_compute(
                core.device.clone(),
                std::path::PathBuf::from("dummy.wgsl"),
                core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Dummy"),
                    source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
                }),
                "main",
            ).unwrap()
        });

        Self {
            buffer_manager,
            params_uniform,
            time_uniform,
            pipelines,
            hot_reload,
            frame_count: 0,
            custom_buffers: storage_buffers,
            atomic_buffer: Some(atomic_buffer),
            atomic_layout: Some(atomic_layout),
        }
    }

    /// Create MultiBufferCompute with atomic buffer support (for particle systems like Lorenz)
    pub fn new_with_atomic_buffer(
        core: &Core,
        buffer_names: &[&str],
        shader_path: &str,
        entry_points: &[&str],
        params: P,
    ) -> Self {
        let buffer_manager = MultiBufferManager::new(core, buffer_names, COMPUTE_TEXTURE_FORMAT_RGBA16);

        // Create uniforms
        let time_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::TimeUniform, "Multi-Buffer Time");
        let params_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::CustomUniform, "Multi-Buffer Params");

        let params_uniform = UniformBinding::new(
            &core.device,
            "Multi-Buffer Params",
            params,
            &params_layout,
            0,
        );

        let time_uniform = UniformBinding::new(
            &core.device,
            "Multi-Buffer Time",
            ComputeTimeUniform {
                time: 0.0,
                delta: 1.0/60.0,
                frame: 0,
                _padding: 0,
            },
            &time_layout,
            0,
        );

        // Create atomic buffer for particle accumulation
        let atomic_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::AtomicBuffer, "Atomic Buffer");
        let buffer_size = core.size.width * core.size.height * 4;
        let atomic_buffer = AtomicBuffer::new(&core.device, buffer_size, &atomic_layout);

        // Create shader module
        let shader_source = std::fs::read_to_string(shader_path)
            .unwrap_or_else(|_| panic!("Failed to read shader file: {}", shader_path));
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("Multi-Buffer Shader - {}", shader_path)),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create pipeline layout with atomic buffer (optimize to use only 4 bind groups)
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Multi-Buffer Pipeline Layout with Atomic"),
            bind_group_layouts: &[
                &time_layout,
                &params_layout,
                buffer_manager.get_storage_layout(),
                &atomic_layout,
            ],
            push_constant_ranges: &[],
        });

        // Create pipelines for each entry point
        let mut pipelines = HashMap::new();
        for &entry_point in entry_points {
            let pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Multi-Buffer Pipeline with Atomic - {}", entry_point)),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            pipelines.insert(entry_point.to_string(), pipeline);
        }

        // Set up hot reload
        let hot_reload = ShaderHotReload::new_compute(
            core.device.clone(),
            std::path::PathBuf::from(shader_path),
            shader_module,
            entry_points.first().unwrap_or(&"main"),
        ).unwrap_or_else(|e| {
            eprintln!("Failed to set up hot reload for {}: {}", shader_path, e);
            // Create a dummy hot reload that won't work
            ShaderHotReload::new_compute(
                core.device.clone(),
                std::path::PathBuf::from("dummy.wgsl"),
                core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Dummy"),
                    source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
                }),
                "main",
            ).unwrap()
        });

        Self {
            buffer_manager,
            params_uniform,
            time_uniform,
            pipelines,
            hot_reload,
            frame_count: 0,
            custom_buffers: Vec::new(),
            atomic_buffer: Some(atomic_buffer),
            atomic_layout: Some(atomic_layout),
        }
    }

    /// Clear atomic buffer (for particle systems)
    pub fn clear_atomic_buffer(&mut self, core: &Core) {
        if let Some(atomic_buffer) = &self.atomic_buffer {
            atomic_buffer.clear(&core.queue);
        }
    }

    /// Dispatch with atomic buffer support
    pub fn dispatch_with_atomic(&mut self, encoder: &mut wgpu::CommandEncoder, core: &Core, entry_point: &str) {
        if let Some(pipeline) = self.pipelines.get(entry_point) {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("Multi-Buffer Atomic Pass - {}", entry_point)),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &self.time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, self.buffer_manager.get_output_bind_group(), &[]);

            // Add atomic buffer
            if let Some(atomic_buffer) = &self.atomic_buffer {
                compute_pass.set_bind_group(3, &atomic_buffer.bind_group, &[]);
            }

            let width = if entry_point == "Splat" { 
                // Particle dispatch
                (self.params_uniform.data.as_bytes().len() / 4).max(256).div_ceil(256) as u32
            } else {
                // Screen dispatch  
                core.size.width.div_ceil(16)
            };
            let height = if entry_point == "Splat" {
                1
            } else {
                core.size.height.div_ceil(16) 
            };
            compute_pass.dispatch_workgroups(width, height, 1);
        }
    }

    /// Update time uniform
    pub fn update_time(&mut self, queue: &wgpu::Queue, current_time: f32) {
        self.time_uniform.data.time = current_time;
        self.time_uniform.data.delta = 1.0/60.0;
        self.time_uniform.data.frame = self.frame_count;
        self.time_uniform.update(queue);
    }
}



/// Helper function to create a display bind group for the final output
pub fn create_display_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    texture: &wgpu::Texture,
) -> wgpu::BindGroup {
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout,
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
        label: Some("Display Bind Group"),
    })
}