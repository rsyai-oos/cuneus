use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager, ShaderApp};
use cuneus::compute::{create_bind_group_layout, BindGroupLayoutType};
use winit::event::WindowEvent;
use std::path::PathBuf;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RorschachParams {
    // Matrix 1
    m1_scale: f32,
    m1_y_scale: f32,
    // Matrix 2
    m2_scale: f32,
    m2_shear: f32,
    m2_shift: f32,
    // Matrix 3
    m3_scale: f32,
    m3_shear: f32,
    m3_shift: f32,
    // Matrix 4
    m4_scale: f32,
    m4_shift: f32,
    // Matrix 5
    m5_scale: f32,
    m5_shift: f32,
    time_scale: f32,
    decay: f32,
    intensity: f32,
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    brightness: f32,
    exposure: f32,
    gamma: f32,
    particle_count: f32,
    scale: f32,
    dof_amount: f32,
    dof_focal_dist: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
}

impl UniformProvider for RorschachParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct Shader {
    base: RenderKit,
    params_uniform: UniformBinding<RorschachParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    
    splat_pipeline: wgpu::ComputePipeline,
    main_image_pipeline: wgpu::ComputePipeline,
    
    output_texture: cuneus::TextureManager,
    
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    storage_bind_group_layout: wgpu::BindGroupLayout,
    atomic_bind_group_layout: wgpu::BindGroupLayout,
    
    storage_bind_group: wgpu::BindGroup,
    
    atomic_buffer: cuneus::AtomicBuffer,
    
    frame_count: u32,
    
    hot_reload: cuneus::ShaderHotReload,
}

impl Shader {
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

impl ShaderManager for Shader {
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
        
        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );

        let time_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::TimeUniform, 
            "Rorschach Time"
        );
        
        let params_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::CustomUniform, 
            "Rorschach Params"
        );
        
        let storage_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::StorageTexture, 
            "Rorschach Storage"
        );
        
        let atomic_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::AtomicBuffer, 
            "Rorschach Atomic"
        );

        let params_uniform = UniformBinding::new(
            &core.device,
            "Rorschach Params Uniform",
            RorschachParams {
                m1_scale: 0.8,
                m1_y_scale: 0.5,
                m2_scale: 0.4,
                m2_shear: 0.2,
                m2_shift: 0.3,
                m3_scale: 0.4,
                m3_shear: 0.2,
                m3_shift: 0.3,
                m4_scale: 0.3,
                m4_shift: 0.2,
                m5_scale: 0.2,
                m5_shift: 0.4,
                time_scale: 0.5,
                decay: 0.0,
                intensity: 0.0,
                rotation_x: 0.0,
                rotation_y: 0.0,
                click_state: 0,
                brightness: 0.003,
                exposure: 1.5,
                gamma: 0.4,
                particle_count: 100000.0,
                scale: 1.0,
                dof_amount: 0.0,
                dof_focal_dist: 0.5,
                color1_r: 1.0,
                color1_g: 0.3,
                color1_b: 0.1,
                color2_r: 0.1,
                color2_g: 0.5,
                color2_b: 1.0,
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

        let atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            core.config.width * core.config.height * 2,
            &atomic_bind_group_layout,
        );

        let output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.config.width,
            core.config.height,
            wgpu::TextureFormat::Rgba16Float,
            &texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Rorschach Output Texture",
        );

        let storage_view = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rorschach Storage Bind Group"),
            layout: &storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
            ],
        });
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Rorschach Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/rorschach.wgsl").into()),
        });
        let hot_reload = cuneus::ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shaders/rorschach.wgsl"),
            shader_module.clone(),
            "Splat",
        ).expect("Failed to initialize hot reload");

        let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Rorschach Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,      
                &params_bind_group_layout,    
                &storage_bind_group_layout,   
                &atomic_bind_group_layout,    
            ],
            push_constant_ranges: &[],
        });

        let splat_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Splat Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("Splat"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let main_image_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Main Image Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("main_image"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            base,
            params_uniform,
            compute_time_uniform,
            splat_pipeline,
            main_image_pipeline,
            output_texture,
            time_bind_group_layout,
            params_bind_group_layout,
            storage_bind_group_layout,
            atomic_bind_group_layout,
            storage_bind_group,
            atomic_buffer,
            frame_count: 0,
            hot_reload,
        }
    }

    fn update(&mut self, core: &Core) {
        if let Some(new_shader) = self.hot_reload.reload_compute_shader() {
            println!("Reloading Rorschach shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Rorschach Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.storage_bind_group_layout,
                    &self.atomic_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            self.splat_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Splat Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("Splat"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.main_image_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Main Image Pipeline"),
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
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
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
                
                egui::Window::new("Rorschach IFS")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        
                        egui::CollapsingHeader::new("IFS Matrices")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("Matrix 1:");
                                changed |= ui.add(egui::Slider::new(&mut params.m1_scale, 0.1..=1.2).text("Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m1_y_scale, 0.1..=1.2).text("Y Scale")).changed();
                                
                                ui.separator();
                                ui.label("Matrix 2:");
                                changed |= ui.add(egui::Slider::new(&mut params.m2_scale, 0.1..=1.0).text("Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m2_shear, -0.5..=0.5).text("Shear")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m2_shift, -0.5..=0.5).text("Shift")).changed();
                                
                                ui.separator();
                                ui.label("Matrix 3:");
                                changed |= ui.add(egui::Slider::new(&mut params.m3_scale, 0.1..=1.0).text("Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m3_shear, -0.5..=0.5).text("Shear")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m3_shift, -0.5..=0.5).text("Shift")).changed();
                                
                                ui.separator();
                                ui.label("Matrix 4 & 5:");
                                changed |= ui.add(egui::Slider::new(&mut params.m4_scale, 0.1..=1.0).text("M4 Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m4_shift, -0.5..=0.5).text("M4 Shift")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m5_scale, 0.1..=1.0).text("M5 Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.m5_shift, -0.5..=0.5).text("M5 Shift")).changed();
                            });

                        egui::CollapsingHeader::new("Colors")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Primary:");
                                    let mut color1 = [params.color1_r, params.color1_g, params.color1_b];
                                    if ui.color_edit_button_rgb(&mut color1).changed() {
                                        params.color1_r = color1[0];
                                        params.color1_g = color1[1];
                                        params.color1_b = color1[2];
                                        changed = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    ui.label("Secondary:");
                                    let mut color2 = [params.color2_r, params.color2_g, params.color2_b];
                                    if ui.color_edit_button_rgb(&mut color2).changed() {
                                        params.color2_r = color2[0];
                                        params.color2_g = color2[1];
                                        params.color2_b = color2[2];
                                        changed = true;
                                    }
                                });
                            });

                        egui::CollapsingHeader::new("Rendering")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.particle_count, 10000.0..=200000.0).text("Particles")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.scale, 0.1..=3.0).text("Scale")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.brightness, 0.001..=0.01).text("Brightness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.5..=3.0).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=2.0).text("Gamma")).changed();
                            });

                        egui::CollapsingHeader::new("Depth of Field")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.dof_amount, 0.0..=2.0).text("DOF Amount")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dof_focal_dist, 0.0..=1.0).text("Focal Distance")).changed();
                            });

                        egui::CollapsingHeader::new("Animation")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.time_scale, 0.0..=2.0).text("Animation Speed")).changed();
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
        self.base.apply_control_request(controls_request);
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        let current_time = self.base.controls.get_time(&self.base.start_time);
        
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        self.compute_time_uniform.data.time = current_time;
        self.compute_time_uniform.data.delta = 1.0/60.0;
        self.compute_time_uniform.data.frame = self.frame_count;
        self.compute_time_uniform.update(&core.queue);

        // Clear atomic buffer
        self.atomic_buffer.clear(&core.queue);

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Splat Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.splat_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.storage_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
            let workgroups = (self.params_uniform.data.particle_count as u32 / 256).max(1);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Main image pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Main Image Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.main_image_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.storage_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
            let workgroups_x = (core.config.width as f32 / 16.0).ceil() as u32;
            let workgroups_y = (core.config.height as f32 / 16.0).ceil() as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
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
        
        self.frame_count = self.frame_count.wrapping_add(1);

        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();

        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        let ui_handled = self.base.egui_state.on_window_event(core.window(), event).consumed;
        
        if ui_handled {
            return true;
        }

        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }

        false
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        
        // Recreate output texture and atomic buffer for new size
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
        
        self.output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.config.width,
            core.config.height,
            wgpu::TextureFormat::Rgba16Float,
            &texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Rorschach Output Texture",
        );
        
        self.atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            core.config.width * core.config.height * 2,
            &self.atomic_bind_group_layout,
        );
        
        let storage_view = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Rorschach Storage Bind Group"),
            layout: &self.storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
            ],
        });
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Rorschach IFS", 800, 600);
    app.run(event_loop, |core| {
        Shader::init(core)
    })
}