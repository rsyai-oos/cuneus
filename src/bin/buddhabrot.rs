use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager};
use cuneus::compute::{create_bind_group_layout, BindGroupLayoutType};
use winit::event::WindowEvent;
use std::path::PathBuf;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BuddhabrotParams {
    max_iterations: u32,      
    escape_radius: f32,       
    zoom: f32,                
    offset_x: f32,            
    offset_y: f32,            
    rotation: f32,            
    exposure: f32,            
    low_iterations: u32, 
    high_iterations: u32,
    motion_speed: f32,        
    color1_r: f32,            
    color1_g: f32,            
    color1_b: f32,            
    color2_r: f32,            
    color2_g: f32,            
    color2_b: f32,            
    sample_density: f32,      
    dithering: f32,           
}

impl UniformProvider for BuddhabrotParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct BuddhabrotShader {
    // Core components
    base: RenderKit,
    params_uniform: UniformBinding<BuddhabrotParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    
    // Compute pipelines
    compute_pipeline_splat: wgpu::ComputePipeline,
    compute_pipeline_render: wgpu::ComputePipeline,
    
    // Output texture
    output_texture: cuneus::TextureManager,
    
    // Bind group layouts
    compute_bind_group_layout: wgpu::BindGroupLayout,
    atomic_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    
    // Bind groups
    compute_bind_group: wgpu::BindGroup,
    
    // Atomic buffer for point accumulation
    atomic_buffer: cuneus::AtomicBuffer,
    
    // Frame counter
    frame_count: u32,
    
    // Hot reload for shader
    hot_reload: cuneus::ShaderHotReload,
    
    // Accumulated rendering flag
    accumulated_rendering: bool,
}

impl BuddhabrotShader {
    fn recreate_compute_resources(&mut self, core: &Core) {
        self.output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            wgpu::TextureFormat::Rgba16Float,
            &self.base.texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Buddhabrot Output Texture",
        );

        let buffer_size = core.size.width * core.size.height * 3;
        
        self.atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            buffer_size,
            &self.atomic_bind_group_layout,
        );
        
        let view_output = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Buddhabrot Compute Bind Group"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_output),
                },
            ],
        });

        self.accumulated_rendering = false;
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
    
    fn clear_buffers(&mut self, core: &Core) {
        if !self.accumulated_rendering {
            self.atomic_buffer.clear(&core.queue);
        }
        self.accumulated_rendering = false;
    }
}

impl ShaderManager for BuddhabrotShader {
    fn init(core: &Core) -> Self {
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
        
        let time_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::TimeUniform, 
            "Buddhabrot Compute"
        );
        
        let params_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::CustomUniform, 
            "Buddhabrot Params"
        );
        
        let atomic_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::AtomicBuffer, 
            "Buddhabrot Compute"
        );
        
        let compute_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("buddhabrot_compute_output_layout"),
        });
        
        let buffer_size = core.config.width * core.config.height * 3;
        
        let atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            buffer_size,
            &atomic_bind_group_layout,
        );
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "Buddhabrot Params",
            BuddhabrotParams {
                max_iterations: 500,
                escape_radius: 4.0,
                zoom: 0.5,
                offset_x: -0.5,
                offset_y: 0.0,
                rotation: 1.5,
                exposure: 0.0005,
                low_iterations: 20,
                high_iterations: 100,
                motion_speed: 0.0,
                color1_r: 1.0,
                color1_g: 0.5,
                color1_b: 0.2,
                color2_r: 0.2,
                color2_g: 0.5,
                color2_b: 1.0,
                sample_density: 0.5,
                dithering: 0.2,
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
        
        let shader_source = std::fs::read_to_string("shaders/buddhabrot.wgsl")
            .unwrap_or_else(|_| include_str!("../../shaders/buddhabrot.wgsl").to_string());
        
        let cs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Buddhabrot Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let hot_reload = cuneus::ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shaders/buddhabrot.wgsl"),
            cs_module.clone(),
            "Splat",
        ).expect("Failed to initialize hot reload");
        
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
            "Buddhabrot Output Texture",
        );
        
        let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Buddhabrot Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,
                &params_bind_group_layout,
                &compute_bind_group_layout,
                &atomic_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline_splat = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Splat Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("Splat"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let compute_pipeline_render = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Main Image Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("main_image"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let view_output = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_output),
                },
            ],
            label: Some("Buddhabrot Compute Bind Group"),
        });
        
        let mut result = Self {
            base,
            params_uniform,
            compute_time_uniform,
            compute_pipeline_splat,
            compute_pipeline_render,
            output_texture,
            compute_bind_group_layout,
            atomic_bind_group_layout,
            time_bind_group_layout,
            params_bind_group_layout,
            compute_bind_group,
            atomic_buffer,
            frame_count: 0,
            hot_reload,
            accumulated_rendering: false,
        };
        
        result.recreate_compute_resources(core);
        
        result
    }
    
    fn update(&mut self, core: &Core) {
        if let Some(new_shader) = self.hot_reload.reload_compute_shader() {
            println!("Reloading Buddhabrot shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Buddhabrot Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.compute_bind_group_layout,
                    &self.atomic_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            
            self.compute_pipeline_splat = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Splat Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("Splat"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.compute_pipeline_render = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Main Image Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("main_image"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.clear_buffers(core);
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
                
                egui::Window::new("Buddhabrot Explorer")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .min_width(250.0)
                    .max_width(500.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Fractal Parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.max_iterations, 100..=500).text("Max Iterations")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.escape_radius, 2.0..=10.0).text("Escape Radius")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.low_iterations, 5..=50).text("Low Iterations")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.high_iterations, 50..=500).text("High Iterations")).changed();
                            });
                        
                        egui::CollapsingHeader::new("View Controls")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.1..=5.0).logarithmic(true).text("Zoom")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.offset_x, -2.0..=1.0).text("Offset X")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.offset_y, -1.5..=1.5).text("Offset Y")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.rotation, -3.14159..=3.14159).text("Rotation")).changed();
                                ui.add_space(10.0);
                                ui.separator();
                                
                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.00005..=0.001).logarithmic(true).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.sample_density, 0.1..=2.0).text("Sample Density")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dithering, 0.0..=1.0).text("Dithering")).changed();
                            });
                            
                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Color 1:");
                                    let mut color = [params.color1_r, params.color1_g, params.color1_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.color1_r = color[0];
                                        params.color1_g = color[1];
                                        params.color1_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Color 2:");
                                    let mut color = [params.color2_r, params.color2_g, params.color2_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.color2_r = color[0];
                                        params.color2_g = color[1];
                                        params.color2_b = color[2];
                                        changed = true;
                                    }
                                });
                            });
                        
                        egui::CollapsingHeader::new("Rendering Options")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Accumulated?:");
                                    ui.checkbox(&mut self.accumulated_rendering, "");
                                });
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
            self.clear_buffers(core);
        }
        self.base.apply_control_request(controls_request);

        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = self.frame_count;
        self.base.time_uniform.update(&core.queue);
        
        self.compute_time_uniform.data.time = current_time;
        self.compute_time_uniform.data.delta = 1.0/60.0;
        self.compute_time_uniform.data.frame = self.frame_count;
        self.compute_time_uniform.update(&core.queue);
        
        if changed {
            if !self.accumulated_rendering {
                self.clear_buffers(core);
            }
            
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        // Only generate new samples if we're not in accumulated mode
        // or if we're still accumulating (frame count < 500)
        let should_generate_samples = !self.accumulated_rendering || self.frame_count < 500;
        
        if should_generate_samples {
            // Pass 1: Generate and splat particles
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Particle Generation Pass"),
                    timestamp_writes: None,
                });
                
                compute_pass.set_pipeline(&self.compute_pipeline_splat);
                compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
                compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
                compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
                compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
                
                compute_pass.dispatch_workgroups(2048, 1, 1);
            }
        }
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Render Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline_render);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
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
        
        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = cuneus::ShaderApp::new("Buddhabrot", 800, 600);
    
    app.run(event_loop, |core| {
        BuddhabrotShader::init(core)
    })
}