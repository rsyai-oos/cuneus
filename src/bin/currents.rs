// Photon tracing: currents
// Very complex example demonstrating multi-buffer ping-pong computation
// I hope this example is useful for those who came from the Shadertoy, I tried to use same terminology (bufferA, ichannels etc)
// I used the all buffers (buffera,b,c,d,mainimage) and complex ping-pong logic 
use cuneus::prelude::*;
use cuneus::compute::*;
use winit::event::WindowEvent;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CurrentsParams {
    sphere_radius: f32,
    sphere_pos_x: f32,
    sphere_pos_y: f32,
    critic2_interval: f32,
    critic2_pause: f32,
    critic3_interval: f32,
    metallic_reflection: f32,
    line_intensity: f32,
    pattern_scale: f32,
    noise_strength: f32,
    gradient_r: f32,
    gradient_g: f32,
    gradient_b: f32,
    gradient_w: f32,
    line_color_r: f32,
    line_color_g: f32,
    line_color_b: f32,
    line_color_w: f32,
    gradient_intensity: f32,
    line_intensity_final: f32,
    c2_min: f32,
    c2_max: f32,
    c3_min: f32,
    c3_max: f32,
    fbm_scale: f32,
    fbm_offset: f32,
    pattern_mode: f32, // 0.0 = currents, 1.0 = mandelbrot
    // Mandelbrot-specific parameters
    mandel_zoom_min: f32,
    mandel_zoom_max: f32,
    mandel_pan_x: f32,
    mandel_pan_y: f32,
    mandel_trap1_x: f32,
    mandel_trap1_y: f32,
    mandel_trap2_x: f32,
    mandel_trap2_y: f32,
    gamma: f32,
}

impl Default for CurrentsParams {
    fn default() -> Self {
        Self {
            sphere_radius: 0.2,
            sphere_pos_x: 0.0,
            sphere_pos_y: -0.2,
            critic2_interval: 10.0,
            critic2_pause: 5.0,
            critic3_interval: 10.0,
            metallic_reflection: 1.8,
            line_intensity: 0.8,
            pattern_scale: 150.0,
            noise_strength: 1.0,
            gradient_r: 1.0,
            gradient_g: 2.0,
            gradient_b: 3.0,
            gradient_w: 4.0,
            line_color_r: 1.0,
            line_color_g: 2.0,
            line_color_b: 3.0,
            line_color_w: 4.0,
            gradient_intensity: 1.5,
            line_intensity_final: 1.5,
            c2_min: 333.0,
            c2_max: 1.0,
            c3_min: 1.0,
            c3_max: 3.0,
            fbm_scale: 4.0,
            fbm_offset: 1.0,
            pattern_mode: 0.0, // default to currents
            // Mandelbrot defaults
            mandel_zoom_min: 0.0008,
            mandel_zoom_max: 0.0008,
            mandel_pan_x: 0.8086,
            mandel_pan_y: 0.2607,
            mandel_trap1_x: 0.0,
            mandel_trap1_y: 1.0,
            mandel_trap2_x: -0.5,
            mandel_trap2_y: 2.0,
            gamma: 2.1,
        }
    }
}

impl UniformProvider for CurrentsParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct CurrentsShader {
    base: RenderKit,
    multi_buffer: MultiBufferCompute<CurrentsParams>,
}


impl ShaderManager for CurrentsShader {
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

        // Create multi-buffer compute system
        let multi_buffer = MultiBufferCompute::new(
            core,
            &["buffer_a", "buffer_b", "buffer_c", "buffer_d"],
            "shaders/currents.wgsl",
            &["buffer_a", "buffer_b", "buffer_c", "buffer_d", "main_image"],
            CurrentsParams::default(),
        );

        Self { base, multi_buffer }
    }

    fn update(&mut self, core: &Core) {
        // Check hot reload
        if let Some(new_shader) = self.multi_buffer.hot_reload.reload_compute_shader() {
            println!("Reloading Currents shader at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            
            // Recreate all pipelines with updated shader
            let time_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::TimeUniform, "Currents Time");
            let params_layout = create_bind_group_layout(&core.device, BindGroupLayoutType::CustomUniform, "Currents Params");
            
            let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Updated Currents Pipeline Layout"),
                bind_group_layouts: &[
                    &time_layout,
                    &params_layout,
                    self.multi_buffer.buffer_manager.get_storage_layout(),
                    self.multi_buffer.buffer_manager.get_multi_texture_layout(),
                ],
                push_constant_ranges: &[],
            });

            // Update pipelines
            for entry_point in &["buffer_a", "buffer_b", "buffer_c", "buffer_d", "main_image"] {
                let pipeline = core.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&format!("Updated Currents Pipeline - {}", entry_point)),
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
            label: Some("Currents Render Encoder"),
        });

        // Update time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.multi_buffer.update_time(&core.queue, current_time);

        // Execute multi-buffer compute passes in dependency order
        // Buffer A: self-feedback
        self.multi_buffer.dispatch_buffer(&mut encoder, core, "buffer_a", &["buffer_a"]);
        
        // Buffer B: reads BufferB + BufferA  
        self.multi_buffer.dispatch_buffer(&mut encoder, core, "buffer_b", &["buffer_b", "buffer_a"]);
        
        // Buffer C: reads BufferC + BufferA
        self.multi_buffer.dispatch_buffer(&mut encoder, core, "buffer_c", &["buffer_c", "buffer_a"]);
        
        // Buffer D: reads BufferD + BufferC + BufferB
        self.multi_buffer.dispatch_buffer(&mut encoder, core, "buffer_d", &["buffer_d", "buffer_c", "buffer_b"]);
        
        // Main image uses buffer_d output
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());
        let main_input_bind_group = self.multi_buffer.buffer_manager.create_input_bind_group(&core.device, &sampler, &["buffer_d"]);
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Currents Main Image Pass"),
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

        // Render to screen
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
                Some("Currents Display Pass"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &display_bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        // Handle UI and controls
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
                
                egui::Window::new("Multi-Buffer Ping-Pong Example")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(280.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Pattern Mode")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Mode:");
                                    if ui.selectable_label(params.pattern_mode < 0.5, "Currents").clicked() {
                                        params.pattern_mode = 0.0;
                                        changed = true;
                                    }
                                    if ui.selectable_label(params.pattern_mode >= 0.5, "Mandelbrot").clicked() {
                                        params.pattern_mode = 1.0;
                                        changed = true;
                                    }
                                });
                            });

                        // Show different UI sections based on mode
                        if params.pattern_mode < 0.5 {
                            // CURRENTS MODE UI
                            egui::CollapsingHeader::new("Sphere Settings")
                                .default_open(false)
                                .show(ui, |ui| {
                                    changed |= ui.add(egui::Slider::new(&mut params.sphere_radius, 0.05..=0.5).text("Sphere Radius")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.sphere_pos_x, -1.0..=1.0).text("Sphere X")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.sphere_pos_y, -1.0..=1.0).text("Sphere Y")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.metallic_reflection, 0.5..=3.0).text("Metallic Reflection")).changed();
                                });

                            egui::CollapsingHeader::new("Pattern Control")
                                .default_open(false)
                                .show(ui, |ui| {
                                    changed |= ui.add(egui::Slider::new(&mut params.pattern_scale, 50.0..=300.0).text("Pattern Scale")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.critic2_interval, 5.0..=20.0).text("Flow Interval")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.critic2_pause, 1.0..=10.0).text("Flow Pause")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.critic3_interval, 5.0..=20.0).text("Scale Interval")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.noise_strength, 0.5..=5.0).text("Noise Strength")).changed();
                                });

                            egui::CollapsingHeader::new("Noise")
                                .default_open(false)
                                .show(ui, |ui| {
                                    ui.label("Oscillator 2 (c2):");
                                    changed |= ui.add(egui::Slider::new(&mut params.c2_min, 1.0..=500.0).text("C2 Min")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.c2_max, 0.1..=10.0).text("C2 Max")).changed();
                                    
                                    ui.separator();
                                    ui.label("Oscillator 3 (c3):");
                                    changed |= ui.add(egui::Slider::new(&mut params.c3_min, 0.1..=10.0).text("C3 Min")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.c3_max, 0.5..=10.0).text("C3 Max")).changed();
                                    
                                    ui.separator();
                                    ui.label("FBM Noise:");
                                    changed |= ui.add(egui::Slider::new(&mut params.fbm_scale, 1.0..=10.0).text("FBM Scale")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.fbm_offset, 0.1..=5.0).text("FBM Offset")).changed();
                                });
                        } else {
                            // MANDELBROT MODE UI
                            egui::CollapsingHeader::new("Mandelbrot Settings")
                                .default_open(false)
                                .show(ui, |ui| {
                                    ui.label("Zoom Animation:");
                                    changed |= ui.add(egui::Slider::new(&mut params.mandel_zoom_min, 0.0001..=0.01).logarithmic(true).text("Zoom Min")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.mandel_zoom_max, 0.0001..=0.01).logarithmic(true).text("Zoom Max")).changed();
                                    
                                    ui.separator();
                                    ui.label("View Position:");
                                    changed |= ui.add(egui::Slider::new(&mut params.mandel_pan_x, -2.0..=2.0).text("Pan X")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.mandel_pan_y, -2.0..=2.0).text("Pan Y")).changed();
                                });

                            egui::CollapsingHeader::new("Orbit Traps")
                                .default_open(false)
                                .show(ui, |ui| {
                                    ui.label("Trap 1 Position:");
                                    changed |= ui.add(egui::Slider::new(&mut params.mandel_trap1_x, -2.0..=2.0).text("Trap1 X")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.mandel_trap1_y, -2.0..=2.0).text("Trap1 Y")).changed();
                                    
                                    ui.separator();
                                    ui.label("Trap 2 Position:");
                                    changed |= ui.add(egui::Slider::new(&mut params.mandel_trap2_x, -2.0..=2.0).text("Trap2 X")).changed();
                                    changed |= ui.add(egui::Slider::new(&mut params.mandel_trap2_y, -2.0..=2.0).text("Trap2 Y")).changed();
                                });
                        }

                        egui::CollapsingHeader::new("Colors & Post-Processing")
                            .default_open(false)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label("Gradient:");
                                    let mut color = [params.gradient_r, params.gradient_g, params.gradient_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.gradient_r = color[0];
                                        params.gradient_g = color[1];
                                        params.gradient_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.horizontal(|ui| {
                                    ui.label("Lines:");
                                    let mut color = [params.line_color_r, params.line_color_g, params.line_color_b];
                                    if ui.color_edit_button_rgb(&mut color).changed() {
                                        params.line_color_r = color[0];
                                        params.line_color_g = color[1];
                                        params.line_color_b = color[2];
                                        changed = true;
                                    }
                                });
                                
                                ui.separator();
                                changed |= ui.add(egui::Slider::new(&mut params.gradient_intensity, 0.1..=2.0).text("Gradient Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.line_intensity_final, 0.1..=2.0).text("Line Final Intensity")).changed();
                                
                                ui.separator();
                                changed |= ui.add(egui::Slider::new(&mut params.line_intensity, 0.1..=3.0).text("Line Intensity")).changed();
                                changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.1..=4.0).text("Gamma Correction")).changed();
                            });
                        
                        ui.separator();
                        
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        ui.separator();
                        
                        should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                        
                        ui.separator();
                        ui.label(format!("Frame: {}", self.multi_buffer.frame_count));
                        ui.label("Multi-buffer system with ping-pong textures - Simplified");
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };

        // Handle control requests
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

        // Update frame count and flip buffers
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
    let (app, event_loop) = cuneus::ShaderApp::new("Multi-Buffer Ping-Pong", 800, 600);
    
    app.run(event_loop, |core| {
        CurrentsShader::init(core)
    })
}