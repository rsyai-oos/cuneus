use crate::{Core, UniformProvider, UniformBinding, TextureManager};
use fontdue::{Font, FontSettings};
use std::collections::HashMap;
use bytemuck::{Pod, Zeroable};

//note that: I always use following:
// _ATLAS_SIZE: u32 = 1024;
// _CELL_SIZE: u32 = 64;
// _GRID_SIZE: u32 = 16;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct FontUniforms {
    pub atlas_size: [f32; 2],
    pub char_size: [f32; 2],
    pub screen_size: [f32; 2],
    pub _padding: [f32; 2],
}

impl UniformProvider for FontUniforms {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[derive(Clone, Copy)]
pub struct CharInfo {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    pub size: [f32; 2],
    pub bearing: [f32; 2],
    pub advance: f32,
}

pub struct FontSystem {
    pub font: Font,
    pub atlas_texture: TextureManager,
    pub char_map: HashMap<char, CharInfo>,
    pub font_uniforms: UniformBinding<FontUniforms>,
    pub font_bind_group_layout: wgpu::BindGroupLayout,
}


impl FontSystem {
    fn generate_simple_atlas(font: &Font) -> (Vec<u8>, u32, u32) {
        let atlas_size = 1024u32;  
        let cell_size = 64u32;
        let grid_size = 16u32;
        let mut atlas_data = vec![0u8; (atlas_size * atlas_size * 4) as usize];
        
        
        for ascii_code in 32u32..127u32 {
            let ch = char::from(ascii_code as u8);
            let grid_x = ascii_code % grid_size;
            let grid_y = ascii_code / grid_size;
            let cell_x = grid_x * cell_size;
            let cell_y = grid_y * cell_size;
            //note: larger, more quality:
            let font_size = 48.0;
            let (metrics, bitmap) = font.rasterize(ch, font_size);
            
            if bitmap.is_empty() {
                continue;
            }
            
            let padding = 4u32;
            let available_width = cell_size - padding * 2;
            let available_height = cell_size - padding * 2;
            
            let scale_x = available_width as f32 / metrics.width as f32;
            let scale_y = available_height as f32 / metrics.height as f32;
            let scale = scale_x.min(scale_y).min(1.0);
            
            let scaled_width = (metrics.width as f32 * scale) as u32;
            let scaled_height = (metrics.height as f32 * scale) as u32;
            
            let offset_x = cell_x + padding + (available_width - scaled_width) / 2;
            let offset_y = cell_y + padding + (available_height - scaled_height) / 2;
            
            for y in 0..scaled_height {
                for x in 0..scaled_width {
                    let src_x = (x as f32 / scale) as usize;
                    let src_y = (y as f32 / scale) as usize;
                    
                    if src_x < metrics.width && src_y < metrics.height {
                        let atlas_x = offset_x + x;
                        let atlas_y = offset_y + y;
                        
                        if atlas_x < atlas_size && atlas_y < atlas_size {
                            let atlas_idx = ((atlas_y * atlas_size + atlas_x) * 4) as usize;
                            let src_idx = src_y * metrics.width + src_x;
                            
                            if atlas_idx + 3 < atlas_data.len() && src_idx < bitmap.len() {
                                let alpha = bitmap[src_idx];

                                let corrected_alpha = ((alpha as f32 / 255.0).powf(0.8) * 255.0) as u8;
                                atlas_data[atlas_idx] = 255;     // R
                                atlas_data[atlas_idx + 1] = 255; // G
                                atlas_data[atlas_idx + 2] = 255; // B
                                atlas_data[atlas_idx + 3] = corrected_alpha; // A
                            }
                        }
                    }
                }
            }
        }
        
        (atlas_data, atlas_size, atlas_size)
    }


    pub fn new(core: &Core, font_data: &[u8]) -> Self {
        let font = Font::from_bytes(font_data, FontSettings::default()).unwrap();
        
        let font_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("Font Bind Group Layout"),
        });

        let font_uniforms_data = FontUniforms {
            atlas_size: [512.0, 512.0],
            char_size: [32.0, 32.0],
            screen_size: [core.size.width as f32, core.size.height as f32],
            _padding: [0.0, 0.0],
        };

        //separate uniform layout for just the font uniforms buffer
        let font_uniform_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
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
            ],
            label: Some("Font Uniforms Layout"),
        });

        let font_uniforms = UniformBinding::new(
            &core.device,
            "Font Uniforms",
            font_uniforms_data,
            &font_uniform_layout,
            0,
        );

        let (atlas_texture, char_map, atlas_width, atlas_height) = Self::create_font_atlas(core, &font);
        
        //font uniforms update with actual atlas dimensions
        let mut font_system = Self {
            font,
            atlas_texture,
            char_map,
            font_uniforms,
            font_bind_group_layout,
        };
        
        font_system.font_uniforms.data.atlas_size = [atlas_width as f32, atlas_height as f32];
        font_system.font_uniforms.data.char_size = [atlas_width as f32 / 16.0, atlas_height as f32 / 16.0];
        font_system.font_uniforms.update(&core.queue);
        
        font_system
    }

    fn create_font_atlas(core: &Core, font: &Font) -> (TextureManager, HashMap<char, CharInfo>, u32, u32) {
        let (atlas_data, width, height) = Self::generate_simple_atlas(font);
        let mut char_map = HashMap::new();
        
        let grid_size = 16;
        for ascii_code in 32u32..127u32 {
            let ch = char::from(ascii_code as u8);
            let grid_x = ascii_code % grid_size;
            let grid_y = ascii_code / grid_size;
            
            let char_info = CharInfo {
                uv_min: [grid_x as f32 / grid_size as f32, grid_y as f32 / grid_size as f32],
                uv_max: [(grid_x + 1) as f32 / grid_size as f32, (grid_y + 1) as f32 / grid_size as f32],
                size: [width as f32 / grid_size as f32, height as f32 / grid_size as f32],
                bearing: [0.0, 0.0],
                advance: width as f32 / grid_size as f32,
            };
            
            char_map.insert(ch, char_info);
        }
        
        let texture = core.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadertoy Font Atlas"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        
        core.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &atlas_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Font Texture Display Layout"),
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
            label: Some("Font Atlas Bind Group"),
        });
        
        let atlas_texture = TextureManager {
            texture,
            view,
            sampler,
            bind_group,
        };
        
        (atlas_texture, char_map, width, height)
    }

    pub fn update_screen_size(&mut self, width: u32, height: u32, queue: &wgpu::Queue) {
        self.font_uniforms.data.screen_size = [width as f32, height as f32];
        self.font_uniforms.update(queue);
    }

    pub fn get_char_info(&self, ch: char) -> Option<&CharInfo> {
        self.char_map.get(&ch)
    }

    pub fn create_font_bind_group(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.font_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.font_uniforms.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.atlas_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_texture.sampler),
                },
            ],
            label: Some("Font Bind Group"),
        })
    }
}

