use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager};
use cuneus::compute::{create_bind_group_layout, BindGroupLayoutType};
use winit::event::WindowEvent;
use std::path::PathBuf;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SinhParams {
    aa: i32,
    camera_x: f32,
    camera_y: f32,
    camera_z: f32,
    orbit_speed: f32,
    magic_number: f32,
    cv_min: f32,
    cv_max: f32,
    os_base: f32,
    os_scale: f32,
    base_color_r: f32,
    base_color_g: f32,
    base_color_b: f32,
    light_color_r: f32,
    light_color_g: f32,
    light_color_b: f32,
    ambient_r: f32,
    ambient_g: f32,
    ambient_b: f32,
    gamma: f32,
    iterations: i32,
    bound: f32,
    fractal_scale: f32,
    vignette_offset: f32,
}

impl UniformProvider for SinhParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct SinhShader {
    base: RenderKit,
    params_uniform: UniformBinding<SinhParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    compute_pipeline: wgpu::ComputePipeline,
    output_texture: cuneus::TextureManager,
    compute_bind_group_layout: wgpu::BindGroupLayout,
    atomic_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
    compute_bind_group: wgpu::BindGroup,
    atomic_buffer: cuneus::AtomicBuffer,
    frame_count: u32,
    hot_reload: cuneus::ShaderHotReload,
}

impl SinhShader {
    fn recreate_compute_resources(&mut self, core: &Core) {
        self.output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.size.width,
            core.size.height,
            wgpu::TextureFormat::Rgba16Float,
            &self.base.texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Sinh Output Texture",
        );
        let buffer_size = core.size.width * core.size.height * 2;
        self.atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            buffer_size,
            &self.atomic_bind_group_layout,
        );
        let view_output = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.compute_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sinh Compute Bind Group"),
            layout: &self.compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_output),
                },
            ],
        });
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
        let _ = core.device.poll(wgpu::PollType::Wait).unwrap();
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

impl ShaderManager for SinhShader {
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
            "Sinh Compute"
        );
        
        let params_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::CustomUniform, 
            "Sinh Params"
        );
        
        let atomic_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::AtomicBuffer, 
            "Sinh Compute"
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
            label: Some("sinh_compute_output_layout"),
        });
        
        let buffer_size = core.config.width * core.config.height * 2;
        let atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            buffer_size,
            &atomic_bind_group_layout,
        );
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "Sinh Params",
            SinhParams {
                aa: 2,
                camera_x: 0.1,
                camera_y: 10.0,
                camera_z: 10.0,
                orbit_speed: 0.3,
                magic_number: 36.0,
                cv_min: 2.197,
                cv_max: 2.99225,
                os_base: 0.00004,
                os_scale: 0.02040101,
                base_color_r: 0.5,
                base_color_g: 0.25,
                base_color_b: 0.05,
                light_color_r: 0.8,
                light_color_g: 1.0,
                light_color_b: 0.3,
                ambient_r: 1.2,
                ambient_g: 1.0,
                ambient_b: 0.8,
                gamma: 0.4,
                iterations: 65,
                bound: 12.25,
                fractal_scale: 0.05,
                vignette_offset: 0.0,
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
        
        let shader_source = include_str!("../../shaders/sinh.wgsl");
        let cs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sinh Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        let hot_reload = cuneus::ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shaders/sinh.wgsl"),
            cs_module.clone(),
            "main",
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
            "Sinh Output Texture",
        );
        
        let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sinh Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,
                &params_bind_group_layout,
                &compute_bind_group_layout,
                &atomic_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sinh Compute Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &cs_module,
            entry_point: Some("main"),
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
            label: Some("Sinh Compute Bind Group"),
        });
        
        let mut result = Self {
            base,
            params_uniform,
            compute_time_uniform,
            compute_pipeline,
            output_texture,
            compute_bind_group_layout,
            atomic_bind_group_layout,
            time_bind_group_layout,
            params_bind_group_layout,
            compute_bind_group,
            atomic_buffer,
            frame_count: 0,
            hot_reload,
        };
        
        result.recreate_compute_resources(core);
        result
    }
    
    fn update(&mut self, core: &Core) {
        if let Some(new_shader) = self.hot_reload.reload_compute_shader() {
            println!("Reloading Sinh shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Sinh Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.compute_bind_group_layout,
                    &self.atomic_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            
            self.compute_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Sinh Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("main"),
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
                
                egui::Window::new("Sinh")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Rendering")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.aa, 1..=4).text("AA")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.2..=1.1).text("Gamma")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.vignette_offset, 0.0..=1.0).text("Vignette")).changed();
                            });
                        
                        egui::CollapsingHeader::new("Camera")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.camera_x, -1.0..=1.0).text("X")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.camera_y, 5.0..=20.0).text("Y")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.camera_z, 5.0..=20.0).text("Z")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.orbit_speed, 0.0..=1.0).text("speed")).changed();
                            });
                        
                        egui::CollapsingHeader::new("Fractal")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.iterations, 10..=100).text("Iterations")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.bound, 1.0..=25.0).text("Bound")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.magic_number, 1.0..=100.0).text("Magic Number")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cv_min, 1.0..=3.0).text("CV Min")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.cv_max, 2.0..=4.0).text("CV Max")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.os_base, 0.00001..=0.001).logarithmic(true).text("OS Base")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.os_scale, 0.001..=0.1).text("OS Scale")).changed();
                                ui.separator();
                                changed |= ui.add(egui::Slider::new(&mut params.fractal_scale, 0.01..=1.0).text("Fractal Scale")).changed();
                            });
                        
                        egui::CollapsingHeader::new("Colors")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Base Color:");
                                    let mut color = [params.base_color_r, params.base_color_g, params.base_color_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.base_color_r = color[0];
                                        params.base_color_g = color[1];
                                        params.base_color_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Light Color:");
                                    let mut color = [params.light_color_r, params.light_color_g, params.light_color_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.light_color_r = color[0];
                                        params.light_color_g = color[1];
                                        params.light_color_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Ambient Color:");
                                    let mut color = [params.ambient_r, params.ambient_g, params.ambient_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.ambient_r = color[0];
                                        params.ambient_g = color[1];
                                        params.ambient_b = color[2];
                                        changed = true;
                                    }
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
            self.recreate_compute_resources(core);
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
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sinh Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.compute_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.compute_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
            let width = core.size.width.div_ceil(8);
            let height = core.size.height.div_ceil(8);
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
        
        self.frame_count += 1;
        
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
    let (app, event_loop) = cuneus::ShaderApp::new("Sinh 3D", 800, 300);
    
    app.run(event_loop, |core| {
        SinhShader::init(core)
    })
}