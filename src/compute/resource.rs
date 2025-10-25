use std::collections::HashMap;
use wgpu;

#[derive(Debug, Clone)]
pub enum ResourceType {
    UniformBuffer {
        size: u64,
    },
    StorageBuffer {
        size: u64,
        read_only: bool,
    },
    StorageTexture {
        format: wgpu::TextureFormat,
        access: wgpu::StorageTextureAccess,
    },
    InputTexture,
    ChannelTexture, // External texture channels (channel0, channel1, etc.)
    Sampler,
}

#[derive(Debug, Clone)]
pub struct ResourceBinding {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub resource_type: ResourceType,
}

#[derive(Debug, Default)]
pub struct ResourceLayout {
    pub bindings: Vec<ResourceBinding>,
}

impl ResourceLayout {
    pub fn new() -> Self {
        log::info!("ResourceLayout::new");
        Self {
            bindings: Vec::new(),
        }
    }

    pub fn add_resource(&mut self, group: u32, name: &str, resource_type: ResourceType) {
        log::info!(
            "ResrouceLayout::add_resource, group: {}, name: {}, resource_type: {:?}",
            group,
            name,
            resource_type,
        );
        let binding = self.next_binding_in_group(group);
        self.bindings.push(ResourceBinding {
            group,
            binding,
            name: name.to_string(),
            resource_type,
        });
    }

    fn next_binding_in_group(&self, group: u32) -> u32 {
        let next_binding_id = self
            .bindings
            .iter()
            .filter(|b| b.group == group)
            .map(|b| b.binding)
            .max()
            .map(|max| max + 1)
            .unwrap_or(0);
        log::info!(
            "ResrouceLayout::next_binding_in_group, group: {}, next_binding_id: {}",
            group,
            next_binding_id
        );
        next_binding_id
    }

    pub fn create_bind_group_layouts(
        &self,
        device: &wgpu::Device,
    ) -> HashMap<u32, wgpu::BindGroupLayout> {
        log::info!("ResrouceLayout::create_bind_group_layouts");
        let mut groups: HashMap<u32, Vec<&ResourceBinding>> = HashMap::new();
        for binding in &self.bindings {
            groups.entry(binding.group).or_default().push(binding);
        }

        // layout for each group
        groups
            .into_iter()
            .map(|(group_idx, bindings)| {
                let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
                    .iter()
                    .map(|binding| self.create_layout_entry(binding))
                    .collect();

                let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("Dynamic Group {} Layout", group_idx)),
                    entries: &entries,
                });
                log::info!("\n group_idx: {:?}, layout: {:?}", group_idx, layout);
                (group_idx, layout)
            })
            .collect()
    }

    fn create_layout_entry(&self, binding: &ResourceBinding) -> wgpu::BindGroupLayoutEntry {
        log::info!(
            "ResrouceLayout::create_layout_entry, binding.resource_type: {:?}",
            binding.resource_type
        );
        wgpu::BindGroupLayoutEntry {
            binding: binding.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: match &binding.resource_type {
                ResourceType::UniformBuffer { .. } => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                ResourceType::StorageBuffer { read_only, .. } => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: *read_only,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                ResourceType::StorageTexture { format, access } => {
                    wgpu::BindingType::StorageTexture {
                        access: *access,
                        format: *format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }
                }
                ResourceType::InputTexture => wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                ResourceType::ChannelTexture => wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                ResourceType::Sampler => {
                    wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)
                }
            },
            count: None,
        }
    }

    /// Get all bindings for a specific group
    pub fn get_bindings_for_group(&self, group: u32) -> Vec<&ResourceBinding> {
        log::info!("ResrouceLayout::get_bindings_for_group, group: {}", group);
        self.bindings.iter().filter(|b| b.group == group).collect()
    }

    /// Get binding by name
    pub fn get_binding_by_name(&self, name: &str) -> Option<&ResourceBinding> {
        log::info!("ResrouceLayout::get_bindings_for_group, name: {}", name);
        self.bindings.iter().find(|b| b.name == name)
    }
}

///resource sizing
#[derive(Debug, Clone)]
pub enum DynamicSize {
    Fixed(u64),
    ResolutionSquared(u32),      // resolution² × multiplier
    ResolutionLinear(u32),       // resolution × multiplier
    Custom(fn(u32, u32) -> u64), // Custom function(width, height) -> size
}

impl DynamicSize {
    pub fn calculate(&self, width: u32, height: u32) -> u64 {
        log::info!(
            "DynamicSize::calculate, width: {}, height: {}",
            width,
            height
        );
        match self {
            DynamicSize::Fixed(size) => *size,
            DynamicSize::ResolutionSquared(multiplier) => {
                let res = width.max(height);
                (res * res * multiplier) as u64
            }
            DynamicSize::ResolutionLinear(multiplier) => (width * height * multiplier) as u64,
            DynamicSize::Custom(func) => func(width, height),
        }
    }
}

/// 4-Group Convention Implementation:
/// @group(0): Per-Frame Resources (TimeUniform)
/// @group(1): Primary Pass I/O & Parameters (output texture, shader params, input textures)
/// @group(2): Global Engine Resources (fonts, audio, atomics, mouse)
/// @group(3): User-Defined Data Buffers (custom storage buffers)
impl ResourceLayout {
    // GROUP 0: Per-Frame Resources
    pub fn add_time_uniform(&mut self) {
        log::info!("ResourceLayout::add_time_uniform");
        self.add_resource(
            0,
            "time",
            ResourceType::UniformBuffer {
                size: std::mem::size_of::<super::ComputeTimeUniform>() as u64,
            },
        );
    }

    // GROUP 1: Primary Pass I/O & Parameters
    pub fn add_output_texture(&mut self, format: wgpu::TextureFormat) {
        log::info!("ResourceLayout::add_output_texture, format: {:?}", format);
        self.add_resource(
            1,
            "output",
            ResourceType::StorageTexture {
                format,
                access: wgpu::StorageTextureAccess::WriteOnly,
            },
        );
    }

    pub fn add_input_texture(&mut self) {
        log::info!("ResourceLayout::add_input_texture");
        self.add_resource(1, "input_texture", ResourceType::InputTexture);
        self.add_resource(1, "input_sampler", ResourceType::Sampler);
    }

    /// Add multi-pass input textures to Group 3 (up to 3 input textures with samplers)
    // GROUP 2: Engine Resources including Channels
    /// Add channel textures (channel0-channel3) for external media accessible from all passes
    pub fn add_channel_textures(&mut self, num_channels: u32) {
        log::info!(
            "ResourceLayout::add_channel_textures, num_channels: {}",
            num_channels
        );
        for i in 0..num_channels {
            let channel_name = format!("channel{}", i);
            let sampler_name = format!("channel{}_sampler", i);

            self.add_resource(2, &channel_name, ResourceType::ChannelTexture);
            self.add_resource(2, &sampler_name, ResourceType::Sampler);
        }
    }

    pub fn add_multipass_input_textures(&mut self) {
        log::info!("ResourceLayout::add_multipass_input_textures");
        // Add 3 input texture pairs for multi-pass dependencies
        for i in 0..3 {
            self.add_resource(
                3,
                &format!("input_texture{}", i),
                ResourceType::InputTexture,
            );
            self.add_resource(3, &format!("input_sampler{}", i), ResourceType::Sampler);
        }
    }

    pub fn add_custom_uniform(&mut self, name: &str, size: u64) {
        log::info!(
            "ResourceLayout::add_custom_uniform, name: {}, size: {}",
            name,
            size
        );
        self.add_resource(1, name, ResourceType::UniformBuffer { size });
    }

    // GROUP 2: Global Engine Resources
    pub fn add_mouse_uniform(&mut self) {
        log::info!("ResourceLayout::add_mouse_uniform");
        self.add_resource(
            2,
            "mouse",
            ResourceType::UniformBuffer {
                size: std::mem::size_of::<crate::MouseUniform>() as u64,
            },
        );
    }

    pub fn add_font_resources(&mut self) {
        log::info!("ResourceLayout::add_font_resources");
        self.add_resource(
            2,
            "font_texture_uniform",
            ResourceType::UniformBuffer {
                size: std::mem::size_of::<crate::FontUniforms>() as u64,
            },
        );
        self.add_resource(2, "font_texture_atlas", ResourceType::InputTexture);
    }

    pub fn add_audio_buffer(&mut self, size: usize) {
        log::info!("ResourceLayout::add_audio_buffer, size: {}", size);
        self.add_resource(
            2,
            "audio_buffer",
            ResourceType::StorageBuffer {
                size: (size * std::mem::size_of::<f32>()) as u64,
                read_only: false,
            },
        );
    }

    pub fn add_audio_spectrum_buffer(&mut self, size: usize) {
        log::info!("ResourceLayout::add_audio_spectrum_buffer, size: {}", size);
        self.add_resource(
            2,
            "audio_spectrum",
            ResourceType::StorageBuffer {
                size: (size * std::mem::size_of::<f32>()) as u64,
                read_only: true,
            },
        );
    }

    pub fn add_atomic_buffer(&mut self, size: u64) {
        log::info!("ResourceLayout::add_atomic_buffer, size: {}", size);
        self.add_resource(
            2,
            "atomic_buffer",
            ResourceType::StorageBuffer {
                size,
                read_only: false,
            },
        );
    }

    // GROUP 3: User-Defined Data Buffers
    pub fn add_storage_buffer(&mut self, name: &str, size: u64) {
        log::info!(
            "ResourceLayout::add_storage_buffer, name: {}, size: {}",
            name,
            size
        );
        self.add_resource(
            3,
            name,
            ResourceType::StorageBuffer {
                size,
                read_only: false,
            },
        );
    }

    pub fn add_readonly_storage_buffer(&mut self, name: &str, size: u64) {
        log::info!(
            "ResourceLayout::add_readonly_storage_buffer, name: {}, size: {}",
            name,
            size
        );
        self.add_resource(
            3,
            name,
            ResourceType::StorageBuffer {
                size,
                read_only: true,
            },
        );
    }

    /// Examples:
    /// - Algorithm data: add_dynamic_storage_buffer("data", DynamicSize::ResolutionSquared(8)) // 8 bytes per pixel
    /// - Particle system: add_dynamic_storage_buffer("particles", DynamicSize::Fixed(1000 * 64)) // Fixed count
    /// - Grid simulation: add_dynamic_storage_buffer("grid", DynamicSize::ResolutionLinear(16)) // 16 bytes per cell
    pub fn add_dynamic_storage_buffer(
        &mut self,
        name: &str,
        dynamic_size: DynamicSize,
        width: u32,
        height: u32,
    ) {
        log::info!(
            "ResourceLayout::add_dynamic_storage_buffer, name: {}, dynamic_size: {:?}, width: {}, height: {}",
            name,
            dynamic_size,
            width,
            height
        );
        let calculated_size = dynamic_size.calculate(width, height);
        self.add_resource(
            3,
            name,
            ResourceType::StorageBuffer {
                size: calculated_size,
                read_only: false,
            },
        );
    }
}

/// standard layouts for common shader patterns
pub fn create_basic_layout() -> ResourceLayout {
    log::info!("resources::create_basic_layout");
    let mut layout = ResourceLayout::new();
    layout.add_time_uniform(); // Group 0
    layout.add_output_texture(wgpu::TextureFormat::Rgba16Float); // Group 1
    layout
}

pub fn create_layout_with_input() -> ResourceLayout {
    log::info!("resources::create_layout_with_input");
    let mut layout = create_basic_layout();
    layout.add_input_texture(); // Group 1
    layout
}

pub fn create_layout_with_uniform(uniform_size: u64) -> ResourceLayout {
    log::info!(
        "resources::create_layout_with_uniform, uniform_size: {}",
        uniform_size
    );
    let mut layout = create_basic_layout();
    layout.add_custom_uniform("params", uniform_size); // Group 1
    layout
}

/// Create layout for algorithms requiring resolution-dependent storage
/// Useful for: FFT, convolution, image processing etc
pub fn create_algorithm_layout(
    uniform_size: u64,
    resolution: u32,
    bytes_per_pixel: u32,
) -> ResourceLayout {
    log::info!(
        "resources::create_algorithm_layout, uniform_size: {}, resolution: {}, bytes_per_pixel: {}",
        uniform_size,
        resolution,
        bytes_per_pixel
    );
    let mut layout = create_basic_layout();
    layout.add_input_texture(); // Group 1 (for media input)
    layout.add_custom_uniform("params", uniform_size); // Group 1
                                                       // Algorithm needs resolution² × bytes_per_pixel storage
    layout.add_dynamic_storage_buffer(
        "algorithm_data",
        DynamicSize::ResolutionSquared(bytes_per_pixel),
        resolution,
        resolution,
    );
    layout
}

/// for particle systems
pub fn create_particle_layout(uniform_size: u64, particle_count: u32) -> ResourceLayout {
    log::info!(
        "resources::create_particle_layout, uniform_size: {}, particle_count: {}",
        uniform_size,
        particle_count
    );
    let mut layout = create_basic_layout();
    layout.add_custom_uniform("params", uniform_size); // Group 1
                                                       // Particles need fixed count × bytes per particle
    let bytes_per_particle = 64; // Position + velocity + color + life
    layout.add_dynamic_storage_buffer(
        "particles",
        DynamicSize::Fixed((particle_count * bytes_per_particle) as u64),
        0,
        0,
    );
    layout
}

/// Create layout for grid-based simulations (resolution-dependent)
pub fn create_grid_layout(
    uniform_size: u64,
    width: u32,
    height: u32,
    bytes_per_cell: u32,
) -> ResourceLayout {
    log::info!(
        "resources::create_grid_layout, uniform_size: {}, width: {}, height: {}, bytes_per_cell: {}",
        uniform_size,
        width,
        height,
        bytes_per_cell
    );
    let mut layout = create_basic_layout();
    layout.add_custom_uniform("params", uniform_size); // Group 1
                                                       // Grid needs width × height × bytes per cell
    layout.add_dynamic_storage_buffer(
        "grid_data",
        DynamicSize::ResolutionLinear(bytes_per_cell),
        width,
        height,
    );
    layout
}
