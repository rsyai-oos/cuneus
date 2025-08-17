use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TreeParams {
    pixel_offset: f32,
    pixel_offset2: f32,
    lights: f32,
    exp: f32,
    frame: f32,
    col1: f32,
    col2: f32,
    decay: f32,
}

impl UniformProvider for TreeParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct TreeShader {
    base: RenderKit,
    multi_buffer: MultiBufferCompute<TreeParams>,
}

impl ShaderManager for TreeShader {
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
            &["buffer_a", "buffer_b", "buffer_c"],
            "shaders/tree.wgsl",
            &["buffer_a", "buffer_b", "buffer_c", "main_image"],
            TreeParams {
                pixel_offset: 1.35,
                pixel_offset2: 1.0,
                lights: 2.2,
                exp: 4.0,
                frame: 0.5,
                col1: 205.0,
                col2: 5.5,
                decay: 0.96,
            },
        );

        Self { base, multi_buffer }
    }

    fn update(&mut self, core: &Core) {
        if let Some(new_shader) = self.multi_buffer.hot_reload.reload_compute_shader() {
            println!("Reloading Tree shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            let mut resource_layout = cuneus::compute::ResourceLayout::new();
            resource_layout.add_time_uniform();
            resource_layout.add_custom_uniform("tree_params", std::mem::size_of::<TreeParams>() as u64);
            let bind_group_layouts = resource_layout.create_bind_group_layouts(&core.device);
            let time_layout = bind_group_layouts.get(&0).unwrap();
            let params_layout = bind_group_layouts.get(&2).unwrap();
            
            let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Tree Pipeline Layout"),
                bind_group_layouts: &[
                    &time_layout,
                    &params_layout,
                    self.multi_buffer.buffer_manager.get_storage_layout(),
                    self.multi_buffer.buffer_manager.get_multi_texture_layout(),
                ],
                push_constant_ranges: &[],
            });

            for entry_point in &["buffer_a", "buffer_b", "buffer_c", "main_image"] {
                let pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("Updated Tree Pipeline - {}", entry_point)),
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
            label: Some("Tree Render Encoder"),
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

                egui::Window::new("Fractal Tree")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Fractal Parameters")
                            .default_open(true)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.pixel_offset, -3.14..=3.14).text("Pixel Offset Y")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.pixel_offset2, -3.14..=3.14).text("Pixel Offset X")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.lights, 0.0..=12.2).text("Lights")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.exp, 1.0..=120.0).text("Exp")).changed();
                            });

                        egui::CollapsingHeader::new("Visual Settings")
                            .default_open(false)
                            .show(ui, |ui| {
                                changed |= ui.add(egui::Slider::new(&mut params.frame, 0.0..=2.2).text("Frame")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.col1, 0.0..=300.0).text("Iterations")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.col2, 0.0..=10.0).text("Color 2")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.decay, 0.0..=1.0).text("Feedback")).changed();
                            });

                        ui.separator();
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        ui.separator();
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        
                        ui.separator();
                        ui.label(format!("Frame: {}", self.multi_buffer.frame_count));
                        ui.label("Multi-buffer fractal tree with particle tracing");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.multi_buffer.update_time(&core.queue, current_time);

        // Buffer A: Fractal calculation with self-feedback
        self.multi_buffer.dispatch_buffer(&mut encoder, core, "buffer_a", &["buffer_a"]);
        
        // Buffer B: Gradient computation from Buffer A
        self.multi_buffer.dispatch_buffer(&mut encoder, core, "buffer_b", &["buffer_a"]);
        
        // Buffer C: Particle tracing with self-feedback + Buffer B input
        self.multi_buffer.dispatch_buffer(&mut encoder, core, "buffer_c", &["buffer_c", "buffer_b"]);
        
        // Main image: Final gamma correction from Buffer C
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
        let main_input_bind_group = self.multi_buffer.buffer_manager.create_input_bind_group(&core.device, &sampler, &["buffer_c"]);
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Tree Main Image Pass"),
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
                Some("Tree Display Pass"),
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
    let (app, event_loop) = ShaderApp::new("Fractal Tree", 800, 600);
    app.run(event_loop, |core| {
        TreeShader::init(core)
    })
}