use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, RenderKit, ShaderControls, ExportManager, ShaderApp};
use cuneus::compute::{create_bind_group_layout, BindGroupLayoutType};
use winit::event::WindowEvent;
use std::path::PathBuf;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct NebulaParams {
    iterations: i32,
    formuparam: f32,
    volsteps: i32,
    stepsize: f32,
    zoom: f32,
    tile: f32,
    speed: f32,
    brightness: f32,
    dust_intensity: f32,
    distfading: f32,
    color_variation: f32,
    n_boxes: f32,
    rotation: i32,
    depth: f32,
    color_mode: i32,
    _padding1: f32,
    
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    scale: f32,
    
    dof_amount: f32,
    dof_focal_dist: f32,
    exposure: f32,
    gamma: f32,
    
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
    _padding8: f32,
    _padding9: f32,
    
    time_scale: f32,
    
    spiral_mode: i32,
    spiral_strength: f32,
    spiral_speed: f32,
    visual_mode: i32,
    _padding2: f32,
    _padding3: f32,
}

impl UniformProvider for NebulaParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct NebulaShader {
    base: RenderKit,
    params_uniform: UniformBinding<NebulaParams>,
    compute_time_uniform: UniformBinding<cuneus::compute::ComputeTimeUniform>,
    
    volumetric_pipeline: wgpu::ComputePipeline,
    composition_pipeline: wgpu::ComputePipeline,
    
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
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("universe", 800, 600);
    app.run(event_loop, |core| {
        NebulaShader::init(core)
    })
}
impl NebulaShader {
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
        self.base.resolution_uniform.data.dimensions = [settings.width as f32, settings.height as f32];
        self.base.resolution_uniform.update(&core.queue);

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
impl ShaderManager for NebulaShader {
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
        
        let mut base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        base.setup_mouse_uniform(core);

        let time_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::TimeUniform, 
            "Nebula Time"
        );
        
        let params_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::CustomUniform, 
            "Nebula Params"
        );
        
        let storage_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::StorageTexture, 
            "Nebula Storage"
        );
        
        let atomic_bind_group_layout = create_bind_group_layout(
            &core.device, 
            BindGroupLayoutType::AtomicBuffer, 
            "Nebula Atomic"
        );

        let params_uniform = UniformBinding::new(
            &core.device,
            "Nebula Params Uniform",
            NebulaParams {
                iterations: 21,
                formuparam: 0.55,
                volsteps: 10,
                stepsize: 0.21,
                zoom: 1.85,
                tile: 0.8,
                speed: 0.020,
                brightness: 0.00062,
                dust_intensity: 0.3,
                distfading: 1.0,
                color_variation: 1.5,
                n_boxes: 10.0,
                rotation: 1,
                depth: 5.0,
                color_mode: 1,
                _padding1: 0.0,
                
                rotation_x: 0.0,
                rotation_y: 0.0,
                click_state: 0,
                scale: 1.0,
                
                dof_amount: 0.0,
                dof_focal_dist: 0.5,
                exposure: 1.65,
                gamma: 0.400,
                
                _padding4: 0.0,
                _padding5: 0.0,
                _padding6: 0.0,
                _padding7: 0.0,
                _padding8: 0.0,
                _padding9: 0.0,
                
                time_scale: 1.0,
                
                spiral_mode: 0,
                spiral_strength: 2.0,
                spiral_speed: 0.02,
                visual_mode: 0,
                _padding2: 0.0,
                _padding3: 0.0,
            },
            &params_bind_group_layout,
            0,
        );

        let compute_time_uniform = UniformBinding::new(
            &core.device,
            "Nebula Compute Time Uniform",
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
            "Nebula Output Texture",
        );

        let storage_view = output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nebula Storage Bind Group"),
            layout: &storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
            ],
        });
        
        let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Nebula Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/nebula.wgsl").into()),
        });
        
        let hot_reload = cuneus::ShaderHotReload::new_compute(
            core.device.clone(),
            PathBuf::from("shaders/nebula.wgsl"),
            shader_module.clone(),
            "volumetric_render",
        ).expect("Failed to initialize hot reload");

        let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Nebula Compute Pipeline Layout"),
            bind_group_layouts: &[
                &time_bind_group_layout,      
                &params_bind_group_layout,    
                &storage_bind_group_layout,   
                &atomic_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });

        let volumetric_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Volumetric Render Pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &shader_module,
            entry_point: Some("volumetric_render"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let composition_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Composition Pipeline"),
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
            volumetric_pipeline,
            composition_pipeline,
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
            println!("Reloading Nebula shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            let compute_pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Nebula Compute Pipeline Layout"),
                bind_group_layouts: &[
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                    &self.storage_bind_group_layout,
                    &self.atomic_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            
            self.volumetric_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Volumetric Pipeline"),
                layout: Some(&compute_pipeline_layout),
                module: &new_shader,
                entry_point: Some("volumetric_render"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            
            self.composition_pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Updated Composition Pipeline"),
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
        
        self.base.update_mouse_uniform(&core.queue);
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
        
        if self.base.mouse_tracker.uniform.buttons[0] & 1 != 0 {
            params.rotation_x = self.base.mouse_tracker.uniform.position[0];
            params.rotation_y = self.base.mouse_tracker.uniform.position[1];
            params.click_state = 1;
            changed = true;
        } else {
            params.click_state = 0;
        }

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });
                
                egui::Window::new("universe")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(320.0)
                    .show(ctx, |ui| {
                        
                        egui::CollapsingHeader::new("Volumetric Parameters")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.iterations, 5..=30).text("Iterations")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.formuparam, 0.1..=1.0).text("Form Parameter")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.volsteps, 1..=20).text("Volume Steps")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.stepsize, 0.05..=0.5).text("Step Size")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.zoom, 0.1..=112.0).text("Zoom")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.tile, 0.1..=2.0).text("Tile")).changed();
                            });

                        egui::CollapsingHeader::new("Appearance")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.brightness, 0.0005..=0.015).logarithmic(true).text("Brightness")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dust_intensity, 0.0..=1.0).text("Dust Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.distfading, 0.1..=3.0).text("Distance Fading")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.color_variation, 0.5..=5.0).text("Color Variation")).changed();
                                                changed |= ui.add(egui::Slider::new(&mut params.exposure, 0.5..=3.0).text("Exposure")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=1.2).text("Gamma")).changed();
                            });

                        egui::CollapsingHeader::new("Visual Modes")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    if ui.selectable_label(params.spiral_mode == 0, "Normal").clicked() {
                                        params.spiral_mode = 0;
                                        changed = true;
                                    }
                                    if ui.selectable_label(params.spiral_mode == 1, "Spiral").clicked() {
                                        params.spiral_mode = 1;
                                        changed = true;
                                    }
                                    if ui.selectable_label(params.spiral_mode == 2, "Hole").clicked() {
                                        params.spiral_mode = 2;
                                        changed = true;
                                    }
                                    if ui.selectable_label(params.spiral_mode == 3, "Tunnel").clicked() {
                                        params.spiral_mode = 3;
                                        changed = true;
                                    }
                                });
                                
                                if params.spiral_mode != 0 {
                                    ui.separator();
                                    changed |= ui.add(egui::Slider::new(&mut params.spiral_strength, 0.5..=4.0).text("Spiral Strength")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.spiral_speed, -0.1..=0.1).text("Spiral Speed")).changed();
                                }
                                
                            });

                        egui::CollapsingHeader::new("Depth of Field")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.dof_amount, 0.0..=3.0).text("DOF Amount")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.dof_focal_dist, 0.0..=2.0).text("Focal Distance")).changed();
                            });


                        egui::CollapsingHeader::new("Animation")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.speed, -0.1..=0.1).text("Galaxy Speed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.time_scale, 0.1..=2.0).text("Animation Speed")).changed();
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

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Particle Accumulation Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.volumetric_pipeline);
            compute_pass.set_bind_group(0, &self.compute_time_uniform.bind_group, &[]);
            compute_pass.set_bind_group(1, &self.params_uniform.bind_group, &[]);
            compute_pass.set_bind_group(2, &self.storage_bind_group, &[]);
            compute_pass.set_bind_group(3, &self.atomic_buffer.bind_group, &[]);
            
            let workgroups_x = (core.config.width as f32 / 16.0).ceil() as u32;
            let workgroups_y = (core.config.height as f32 / 16.0).ceil() as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Nebula Render Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.composition_pipeline);
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
    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        
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
        
        self.atomic_buffer = cuneus::AtomicBuffer::new(
            &core.device,
            core.config.width * core.config.height * 2,
            &self.atomic_bind_group_layout,
        );

        self.output_texture = cuneus::compute::create_output_texture(
            &core.device,
            core.config.width,
            core.config.height,
            wgpu::TextureFormat::Rgba16Float,
            &texture_bind_group_layout,
            wgpu::AddressMode::ClampToEdge,
            wgpu::FilterMode::Linear,
            "Nebula Output Texture",
        );
        
        
        let storage_view = self.output_texture.texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.storage_bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Nebula Storage Bind Group"),
            layout: &self.storage_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&storage_view),
                },
            ],
        });
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        let ui_handled = self.base.egui_state.on_window_event(core.window(), event).consumed;
        
        if ui_handled {
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