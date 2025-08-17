use std::collections::HashMap;
use wgpu;

#[derive(Debug, Clone)]
pub enum ResourceType {
    UniformBuffer { 
        size: u64 
    },
    StorageBuffer { 
        size: u64, 
        read_only: bool 
    },
    StorageTexture { 
        format: wgpu::TextureFormat, 
        access: wgpu::StorageTextureAccess 
    },
    InputTexture,
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
        Self {
            bindings: Vec::new(),
        }
    }
    
    pub fn add_resource(&mut self, group: u32, name: &str, resource_type: ResourceType) {
        let binding = self.next_binding_in_group(group);
        self.bindings.push(ResourceBinding {
            group,
            binding,
            name: name.to_string(),
            resource_type,
        });
    }
    
    fn next_binding_in_group(&self, group: u32) -> u32 {
        self.bindings
            .iter()
            .filter(|b| b.group == group)
            .map(|b| b.binding)
            .max()
            .map(|max| max + 1)
            .unwrap_or(0)
    }
    
    pub fn create_bind_group_layouts(&self, device: &wgpu::Device) -> HashMap<u32, wgpu::BindGroupLayout> {
        let mut groups: HashMap<u32, Vec<&ResourceBinding>> = HashMap::new();
        for binding in &self.bindings {
            groups.entry(binding.group).or_default().push(binding);
        }
        
        // layout for each group
        groups.into_iter().map(|(group_idx, bindings)| {
            let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings.iter()
                .map(|binding| self.create_layout_entry(binding))
                .collect();
                
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(&format!("Dynamic Group {} Layout", group_idx)),
                entries: &entries,
            });
            
            (group_idx, layout)
        }).collect()
    }
    
    fn create_layout_entry(&self, binding: &ResourceBinding) -> wgpu::BindGroupLayoutEntry {
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
                    ty: wgpu::BufferBindingType::Storage { read_only: *read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                ResourceType::StorageTexture { format, access } => wgpu::BindingType::StorageTexture {
                    access: *access,
                    format: *format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                ResourceType::InputTexture => wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                ResourceType::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            },
            count: None,
        }
    }
    
    /// Get all bindings for a specific group
    pub fn get_bindings_for_group(&self, group: u32) -> Vec<&ResourceBinding> {
        self.bindings.iter().filter(|b| b.group == group).collect()
    }
    
    /// Get binding by name
    pub fn get_binding_by_name(&self, name: &str) -> Option<&ResourceBinding> {
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
        match self {
            DynamicSize::Fixed(size) => *size,
            DynamicSize::ResolutionSquared(multiplier) => {
                let res = width.max(height);
                (res * res * multiplier) as u64
            },
            DynamicSize::ResolutionLinear(multiplier) => {
                (width * height * multiplier) as u64
            },
            DynamicSize::Custom(func) => func(width, height),
        }
    }
}

/// Helper functions to create common resource layouts
/// time uniform to group 0 (standard)
/// output texture to group 1 (standard)
/// input texture + sampler pair to group 1
/// custom uniform to group 2 (standard for user parameters)
/// storage buffer to group 3 (standard for custom data)
/// dynamic storage buffer that resizes based on resolution
impl ResourceLayout {
    pub fn add_time_uniform(&mut self) {
        self.add_resource(0, "time", ResourceType::UniformBuffer { 
            size: std::mem::size_of::<super::ComputeTimeUniform>() as u64 
        });
    }
    
    pub fn add_output_texture(&mut self, format: wgpu::TextureFormat) {
        self.add_resource(1, "output", ResourceType::StorageTexture { 
            format,
            access: wgpu::StorageTextureAccess::WriteOnly,
        });
    }

    pub fn add_input_texture(&mut self) {
        self.add_resource(1, "input_texture", ResourceType::InputTexture);
        self.add_resource(1, "input_sampler", ResourceType::Sampler);
    }
    
    pub fn add_custom_uniform(&mut self, name: &str, size: u64) {
        self.add_resource(2, name, ResourceType::UniformBuffer { size });
    }
    
    pub fn add_storage_buffer(&mut self, name: &str, size: u64) {
        self.add_resource(3, name, ResourceType::StorageBuffer { 
            size, 
            read_only: false 
        });
    }
    
    /// Examples:
    /// - Algorithm data: add_dynamic_storage_buffer("data", DynamicSize::ResolutionSquared(8)) // 8 bytes per pixel
    /// - Particle system: add_dynamic_storage_buffer("particles", DynamicSize::Fixed(1000 * 64)) // Fixed count  
    /// - Grid simulation: add_dynamic_storage_buffer("grid", DynamicSize::ResolutionLinear(16)) // 16 bytes per cell
    pub fn add_dynamic_storage_buffer(&mut self, name: &str, dynamic_size: DynamicSize, width: u32, height: u32) {
        let calculated_size = dynamic_size.calculate(width, height);
        self.add_resource(3, name, ResourceType::StorageBuffer {
            size: calculated_size,
            read_only: false,
        });
    }
}

/// standard layouts for common shader patterns
pub fn create_basic_layout() -> ResourceLayout {
    let mut layout = ResourceLayout::new();
    layout.add_time_uniform();  // Group 0
    layout.add_output_texture(wgpu::TextureFormat::Rgba16Float);  // Group 1
    layout
}

pub fn create_layout_with_input() -> ResourceLayout {
    let mut layout = create_basic_layout();
    layout.add_input_texture();  // Group 1
    layout
}

pub fn create_layout_with_uniform(uniform_size: u64) -> ResourceLayout {
    let mut layout = create_basic_layout();
    layout.add_custom_uniform("params", uniform_size);  // Group 2
    layout
}

/// Create layout for algorithms requiring resolution-dependent storage
/// Useful for: FFT, convolution, image processing etc
pub fn create_algorithm_layout(uniform_size: u64, resolution: u32, bytes_per_pixel: u32) -> ResourceLayout {
    let mut layout = create_basic_layout();
    layout.add_input_texture();  // Group 1 (for media input)
    layout.add_custom_uniform("params", uniform_size);  // Group 2
    // Algorithm needs resolution² × bytes_per_pixel storage
    layout.add_dynamic_storage_buffer("algorithm_data", DynamicSize::ResolutionSquared(bytes_per_pixel), resolution, resolution);
    layout
}

/// for particle systems
pub fn create_particle_layout(uniform_size: u64, particle_count: u32) -> ResourceLayout {
    let mut layout = create_basic_layout();
    layout.add_custom_uniform("params", uniform_size);  // Group 2
    // Particles need fixed count × bytes per particle
    let bytes_per_particle = 64; // Position + velocity + color + life
    layout.add_dynamic_storage_buffer("particles", DynamicSize::Fixed((particle_count * bytes_per_particle) as u64), 0, 0);
    layout
}

/// Create layout for grid-based simulations (resolution-dependent)
pub fn create_grid_layout(uniform_size: u64, width: u32, height: u32, bytes_per_cell: u32) -> ResourceLayout {
    let mut layout = create_basic_layout();
    layout.add_custom_uniform("params", uniform_size);  // Group 2
    // Grid needs width × height × bytes per cell
    layout.add_dynamic_storage_buffer("grid_data", DynamicSize::ResolutionLinear(bytes_per_cell), width, height);
    layout
}