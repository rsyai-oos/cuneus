use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LichParams {
    cloud_density: f32,
    lightning_intensity: f32,
    branch_count: f32,
    feedback_decay: f32,
    base_color: [f32; 3],
    _pad1: f32,
    color_shift: f32,
    spectrum_mix: f32,
    _pad2: [f32; 2],
}

impl UniformProvider for LichParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct LichShader {
    base: RenderKit,
    multi_buffer: MultiBufferCompute<LichParams>,
}

impl ShaderManager for LichShader {
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
            label: Some("Texture Bind Group Layout"),
        });

        let base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );

        let multi_buffer = MultiBufferCompute::new(
            core,
            &["buffer_a", "buffer_b"],
            "shaders/lich.wgsl",
            &["buffer_a", "buffer_b", "main_image"],
            LichParams {
                cloud_density: 3.0,
                lightning_intensity: 1.0,
                branch_count: 1.0,
                feedback_decay: 0.98,
                base_color: [1.0, 1.0, 1.0],
                _pad1: 0.0,
                color_shift: 2.0,
                spectrum_mix: 0.5,
                _pad2: [0.0; 2],
            },
        );

        Self { base, multi_buffer }
    }

    fn update(&mut self, core: &Core) {
        if let Some(new_shader) = self.multi_buffer.hot_reload.reload_compute_shader() {
            println!("Reloading Lich shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            let mut resource_layout = cuneus::compute::ResourceLayout::new();
            resource_layout.add_time_uniform();
            resource_layout.add_custom_uniform("lich_params", std::mem::size_of::<LichParams>() as u64);
            let bind_group_layouts = resource_layout.create_bind_group_layouts(&core.device);
            let time_layout = bind_group_layouts.get(&0).unwrap();
            let params_layout = bind_group_layouts.get(&2).unwrap();
            
            let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Lich Pipeline Layout"),
                bind_group_layouts: &[
                    &time_layout,
                    &params_layout,
                    self.multi_buffer.buffer_manager.get_storage_layout(),
                    self.multi_buffer.buffer_manager.get_multi_texture_layout(),
                ],
                push_constant_ranges: &[],
            });

            for entry_point in &["buffer_a", "buffer_b", "main_image"] {
                let pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("Updated Lich Pipeline - {}", entry_point)),
                    layout: Some(&pipeline_layout),
                    module: &new_shader,
                    entry_point: Some(entry_point),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
                self.multi_buffer.pipelines.insert(entry_point.to_string(), pipeline);
            }
        }

        self.base.fps_tracker.update();
    }

    fn resize(&mut self, core: &Core) {
        self.multi_buffer.buffer_manager.resize(core, core.size.width, core.size.height, COMPUTE_TEXTURE_FORMAT_RGBA16);
        self.multi_buffer.frame_count = 0;
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Lich Render Encoder"),
        });

        let mut params = self.multi_buffer.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());

        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });

                egui::Window::new("Lich Lightning")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(300.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Lightning Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.cloud_density, 0.0..=24.0).text("Seed")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.lightning_intensity, 0.1..=6.0).text("Lightning")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.branch_count, 0.0..=2.0).text("Branch")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.feedback_decay, 0.1..=1.5).text("Decay")).changed();
                            });

                        egui::CollapsingHeader::new("Color Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                let mut color = params.base_color;
                                if ui.color_edit_button_rgb(&mut color).changed() {
                                    params.base_color = color;
                                    changed = true;
                                }
                                changed |= ui.add(egui::Slider::new(&mut params.color_shift, 0.1..=20.0).text("Temperature")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.spectrum_mix, 0.0..=1.0).text("Spectral")).changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        
                        ui.separator();
                        ui.label("Electric lightning with atomic buffer accumulation");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.multi_buffer.update_time(&core.queue, current_time);

        // Lightning generation: dispatch lightning_gen entry point to buffer_a
        self.multi_buffer.dispatch_buffer(&mut encoder, core, "buffer_a", &["buffer_a"]);
        
        // Feedback accumulation: dispatch feedback_accumulate entry point to buffer_b, reading from buffer_a and buffer_b  
        self.multi_buffer.dispatch_buffer(&mut encoder, core, "buffer_b", &["buffer_a", "buffer_b"]);
        
        // Main image: Final gamma correction
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
        let main_input_bind_group = self.multi_buffer.buffer_manager.create_input_bind_group(&core.device, &sampler, &["buffer_b"]);
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Lich Main Image Pass"),
                timestamp_writes: None,
            });

            if let Some(pipeline) = self.multi_buffer.pipelines.get("main_image") {
                compute_pass.set_pipeline(pipeline);
                compute_pass.set_bind_group(0, &self.multi_buffer.time_uniform.bind_group, &[]);
                compute_pass.set_bind_group(1, &self.multi_buffer.params_uniform.bind_group, &[]);
                compute_pass.set_bind_group(2, self.multi_buffer.buffer_manager.get_output_bind_group(), &[]);
                compute_pass.set_bind_group(3, &main_input_bind_group, &[]);

                let width = core.size.width.div_ceil(16);
                let height = core.size.height.div_ceil(16);
                compute_pass.dispatch_workgroups(width, height, 1);
            }
        }

        let display_bind_group = create_display_bind_group(
            &core.device,
            &self.base.renderer.render_pipeline.get_bind_group_layout(0),
            self.multi_buffer.buffer_manager.get_output_texture(),
        );

        {
            let mut render_pass = Renderer::begin_render_pass(
                &mut encoder,
                &view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Lich Display Pass"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &display_bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
            self.multi_buffer.buffer_manager.clear_all(core, COMPUTE_TEXTURE_FORMAT_RGBA16);
            self.multi_buffer.frame_count = 0;
        }
        self.base.apply_control_request(controls_request);

        if changed {
            self.multi_buffer.params_uniform.data = params;
            self.multi_buffer.params_uniform.update(&core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }

        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        self.multi_buffer.frame_count += 1;
        self.multi_buffer.buffer_manager.flip_buffers();

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
    let (app, event_loop) = ShaderApp::new("Lich Lightning", 800, 600);
    app.run(event_loop, |core| {
        LichShader::init(core)
    })
}