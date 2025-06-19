use cuneus::prelude::*;
use cuneus::FontSystem;
use winit::event::WindowEvent;
use std::path::PathBuf;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CNNParams {
    canvas_size: f32,       
    brush_size: f32,        
    input_resolution: f32,  
    clear_canvas: i32,      
    show_debug: i32,        
    prediction_threshold: f32, 
    canvas_offset_x: f32,   
    canvas_offset_y: f32,
    feature_maps_1: f32,    
    feature_maps_2: f32,    
    num_classes: f32,       
    normalization_mean: f32,
    normalization_std: f32,
    show_frequencies: i32,
    conv1_pool_size: f32,
    conv2_pool_size: f32,
    mouse_x: f32,
    mouse_y: f32,
    mouse_click_x: f32,
    mouse_click_y: f32,
    mouse_buttons: u32,
    _padding: [f32; 3],
}

impl UniformProvider for CNNParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct CNNDigitRecognizer {
    base: RenderKit,
    params_uniform: UniformBinding<CNNParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    
    canvas_update_pipeline: wgpu::ComputePipeline,     
    conv_layer1_pipeline: wgpu::ComputePipeline,       
    conv_layer2_pipeline: wgpu::ComputePipeline,       
    fully_connected_pipeline: wgpu::ComputePipeline,   
    visualization_pipeline: wgpu::ComputePipeline,     
    
    output_texture: cuneus::TextureManager,
    
    compute_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    storage_bind_group_layout: wgpu::BindGroupLayout,
    
    compute_bind_group: wgpu::BindGroup,
    storage_bind_group: wgpu::BindGroup,
    
    canvas_buffer: wgpu::Buffer,      
    conv1_buffer: wgpu::Buffer,       
    conv2_buffer: wgpu::Buffer,       
    fc_buffer: wgpu::Buffer,          
    
    font_system: FontSystem,

    frame_count: u32,
    hot_reload: cuneus::ShaderHotReload,
    should_initialize: bool,
}

impl CNNDigitRecognizer {
    fn recreate_compute_resources(&mut self, core: &Core) {
        self.output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            wgpu::TextureFormat::Rgba16Float,
            &self.base.texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "CNN Output Texture",
        );
        
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
        } else if let Some(ref texture_manager) = self.base.texture_manager {
            input_texture_view = &texture_manager.view;
            input_sampler = &texture_manager.sampler;
        } else {
            panic!("No texture available for compute shader input");
        }
        
        let output_view = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        self.compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CNN Compute Bind Group"),
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
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&self.font_system.atlas_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.font_system.atlas_texture.sampler),
                },
            ],
        });
        
        // CNN storage buffers
        let canvas_size = (28 * 28 * 4) as u64;           
        let conv1_size = (12 * 12 * 8 * 4) as u64; 
        let conv2_size = (4 * 4 * 5 * 4) as u64; 
        let fc_size = (10 * 4) as u64;
        
        self.canvas_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Canvas Buffer"),
            size: canvas_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        self.conv1_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conv1 Buffer"),
            size: conv1_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        self.conv2_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conv2 Buffer"),
            size: conv2_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        self.fc_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FC Buffer"),
            size: fc_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        self.storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CNN Storage Bind Group"),
            layout: &self.storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.canvas_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.conv1_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.conv2_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.fc_buffer,
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

impl ShaderManager for CNNDigitRecognizer {
    fn init(core: &Core) -> Self {
        let time_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::TimeUniform, 
            "CNN Compute Time"
        );
        
        let params_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::CustomUniform, 
            "CNN Params"
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("cnn_compute_bind_group_layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
            label: Some("storage_bind_group_layout"),
        });
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "CNN Params",
            CNNParams {
                canvas_size: 0.6,          
                brush_size: 0.007,         
                input_resolution: 28.0,    
                clear_canvas: 0,
                show_debug: 0,
                prediction_threshold: 0.1,
                canvas_offset_x: 0.1,       
                canvas_offset_y: 0.1,
                feature_maps_1: 8.0,        
                feature_maps_2: 5.0,
                num_classes: 10.0,          
                normalization_mean: 0.1307, 
                normalization_std: 0.3081,
                show_frequencies: 0,       
                conv1_pool_size: 12.0,   
                conv2_pool_size: 4.0,
                mouse_x: 0.0,
                mouse_y: 0.0,
                mouse_click_x: 0.0,
                mouse_click_y: 0.0,
                mouse_buttons: 0,
                _padding: [0.0; 3],
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
        
        let font_data = include_bytes!("../../assets/fonts/Courier Prime Bold.ttf");
        let font_system = FontSystem::new(core, font_data);

        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        let output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            wgpu::TextureFormat::Rgba16Float,
            &texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "CNN Output Texture",
        );
        
        // initial compute
        let output_view = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CNN Compute Bind Group"),
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
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&font_system.atlas_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&font_system.atlas_texture.sampler),
                },
            ],
        });
        
        // CNN storage buffers
        let canvas_size = (28 * 28 * 4) as u64;           
        let conv1_size = (12 * 12 * 8 * 4) as u64;        
        let conv2_size = (4 * 4 * 5 * 4) as u64;          
        let fc_size = (10 * 4) as u64;                     
        
        let canvas_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Canvas Buffer"),
            size: canvas_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let conv1_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conv1 Buffer"),
            size: conv1_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let conv2_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conv2 Buffer"),
            size: conv2_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let fc_buffer = core.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FC Buffer"),
            size: fc_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        let storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CNN Storage Bind Group"),
            layout: &storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &canvas_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &conv1_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &conv2_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &fc_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        
        let shader_source = include_str!("../../shaders/cnn.wgsl");
        let cs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CNN Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let hot_reload = cuneus::ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shaders/cnn.wgsl"),
            cs_module.clone(),
            "main_image",
        ).expect("Failed to initialize hot reload");

        let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CNN Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,
                &params_bind_group_layout,
                &compute_bind_group_layout,
                &storage_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        
        // Create all CNN pipelines
        let canvas_update_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Canvas Update Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("canvas_update"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let conv_layer1_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Conv Layer 1 Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("conv_layer1"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let conv_layer2_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Conv Layer 2 Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("conv_layer2"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let fully_connected_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Fully Connected Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("fully_connected"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let visualization_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Visualization Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("main_image"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        

        let mut result = Self {
            base,
            params_uniform,
            compute_time_uniform,
            canvas_update_pipeline,
            conv_layer1_pipeline,
            conv_layer2_pipeline,
            fully_connected_pipeline,
            visualization_pipeline,
            output_texture,
            compute_bind_group_layout,
            time_bind_group_layout,
            params_bind_group_layout,
            storage_bind_group_layout,
            compute_bind_group,
            storage_bind_group,
            canvas_buffer,
            conv1_buffer,
            conv2_buffer,
            fc_buffer,
            font_system,
            frame_count: 0,
            hot_reload,
            should_initialize: true,
        };
        
        result.recreate_compute_resources(core);
        result
    }
    
    fn update(&mut self, core: &Core) {
        if let Some(new_shader) = self.hot_reload.reload_compute_shader() {
            println!("Reloading CNN shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated CNN Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.compute_bind_group_layout,
                    &self.storage_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            
            // Recreate all pipelines
            self.canvas_update_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Canvas Update Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("canvas_update"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.conv_layer1_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Conv Layer 1 Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("conv_layer1"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.conv_layer2_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Conv Layer 2 Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("conv_layer2"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.fully_connected_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Fully Connected Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("fully_connected"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.visualization_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Visualization Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("main_image"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        }
        
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
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        
let full_output = if self.base.key_handler.show_ui {
    self.base.render_ui(core, |ctx| {
        ctx.style_mut(|style| {
            style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
            style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
            style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
        });
        egui::Window::new("CNN Digit Recognizer")
            .collapsible(true)
            .resizable(true)
            .default_width(400.0)
            .show(ctx, |ui| {
                ui.heading("Instructions");
                ui.label("Draw a digit (0-9) in the canvas area with your mouse, right click clear canv.");
                ui.label("The CNN will predict the digit using pre-trained weights");
                ui.separator();
                egui::CollapsingHeader::new("Canvas Set")
                    .default_open(false)
                    .show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.canvas_size, 0.3..=0.8).text("Canvas Size")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.brush_size, 0.001..=0.015).text("Brush Size")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.canvas_offset_x, 0.0..=0.5).text("Canvas X")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.canvas_offset_y, 0.0..=0.5).text("Canvas Y")).changed();
                        
                        if ui.button("Clear Canvas").clicked() {
                            params.clear_canvas = 1;
                            changed = true;
                        } else if params.clear_canvas == 1 {
                            params.clear_canvas = 0;
                            changed = true;
                        }
                    });
                
                egui::CollapsingHeader::new("CNN Settings")
                    .default_open(false)
                    .show(ui, |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.prediction_threshold, 0.0..=0.5).text("Prediction Threshold")).changed();
                    });
                ShaderControls::render_controls_widget(ui, &mut controls_request);
                
                ui.separator();
                
                should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
            });
    })
} else {
    self.base.render_ui(core, |_ctx| {})
};
        
        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_control_request(controls_request);
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = self.frame_count;
        self.base.time_uniform.update(&core.queue);
        
        self.compute_time_uniform.data.time = current_time;
        self.compute_time_uniform.data.delta = 1.0/60.0;
        self.compute_time_uniform.data.frame = self.frame_count;
        self.compute_time_uniform.update(&core.queue);
        
        params.mouse_x = self.base.mouse_tracker.uniform.position[0];
        params.mouse_y = self.base.mouse_tracker.uniform.position[1];
        params.mouse_click_x = self.base.mouse_tracker.uniform.click_position[0];
        params.mouse_click_y = self.base.mouse_tracker.uniform.click_position[1];
        params.mouse_buttons = self.base.mouse_tracker.uniform.buttons[0];
        changed = true;
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        if self.should_initialize {
            self.should_initialize = false;
        }
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Canvas Update Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.canvas_update_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            compute_pass.dispatch_workgroups(28, 28, 1);
        }
        
        // Pass 2: First convolution layer (28x28 → 12x12x8)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Conv Layer 1 Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.conv_layer1_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            compute_pass.dispatch_workgroups(12, 12, 8);
        }
        
        // Pass 3: Second convolution layer (12x12x8 → 4x4x5)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Conv Layer 2 Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.conv_layer2_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            compute_pass.dispatch_workgroups(4, 4, 5);
        }
        
        // Pass 4: Fully connected layer (80 → 10)
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fully Connected Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.fully_connected_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            compute_pass.dispatch_workgroups(10, 1, 1);
        }
        
        // Pass 5: main
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Visualization Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.visualization_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.storage_bind_group, &[]);
            
            let width = core.size.width.div_ceil(16);
            let height = core.size.height.div_ceil(16);
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
        
        if self.base.handle_mouse_input(core, event, false) {
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
    let (app, event_loop) = cuneus::ShaderApp::new("CNN Digit Recognizer", 800, 600);
    
    app.run(event_loop, |core| {
        CNNDigitRecognizer::init(core)
    })
}