use cuneus::{Core, ShaderApp, ShaderManager, RenderKit, ShaderControls};
use cuneus::compute::{ComputeShaderConfig, COMPUTE_TEXTURE_FORMAT_RGBA16};
use cuneus::SynthesisManager;
use winit::event::*;
use std::path::PathBuf;

struct VeridisQuo {
    base: RenderKit,
    audio_synthesis: Option<SynthesisManager>,
    
    volume: f32,
    octave_shift: f32,
    waveform_type: u32,
    tempo_multiplier: f32,
}

impl ShaderManager for VeridisQuo {
    fn init(core: &Core) -> Self {
        let texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Texture Bind Group Layout"),
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
        });
        
        let mut base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        let mouse_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("mouse_bind_group_layout"),
        });
        
        let mouse_uniform = cuneus::UniformBinding::new(
            &core.device,
            "Mouse Uniform",
            cuneus::MouseUniform::default(),
            &mouse_bind_group_layout,
            0,
        );
        
        base.mouse_bind_group_layout = Some(mouse_bind_group_layout.clone());
        base.mouse_uniform = Some(mouse_uniform);
        
        let compute_config = ComputeShaderConfig {
            workgroup_size: [16, 16, 1],
            workgroup_count: None,
            dispatch_once: false,
            storage_texture_format: COMPUTE_TEXTURE_FORMAT_RGBA16,
            enable_atomic_buffer: false,
            atomic_buffer_multiples: 4,
            entry_points: vec!["main".to_string()],
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "Veridis Quo".to_string(),
            mouse_bind_group_layout: Some(mouse_bind_group_layout),
            enable_fonts: true,
            enable_audio_buffer: true,
            audio_buffer_size: 4096,
        };
        
        base.compute_shader = Some(cuneus::compute::ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/veridisquo.wgsl"),
            compute_config,
        ));
        
        if let (Some(compute_shader), Some(mouse_uniform)) = (&mut base.compute_shader, &base.mouse_uniform) {
            compute_shader.add_mouse_uniform_binding(
                &mouse_uniform.bind_group,
                2
            );
        }
        
        if let Some(compute_shader) = &mut base.compute_shader {
            let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Veridis Quo Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/veridisquo.wgsl").into()),
            });
            if let Err(e) = compute_shader.enable_hot_reload(
                core.device.clone(),
                PathBuf::from("shaders/veridisquo.wgsl"),
                shader_module,
            ) {
                eprintln!("Failed to enable compute shader hot reload: {}", e);
            }
        }
        
        let audio_synthesis = match SynthesisManager::new() {
            Ok(mut synth) => {
                if let Err(_e) = synth.start_gpu_synthesis() {
                    None
                } else {
                    println!("🎵 Veridis Quo synthesis started - enjoy the melody!");
                    Some(synth)
                }
            },
            Err(_e) => {
                None
            }
        };
        
        Self { 
            base,
            audio_synthesis,
            volume: 0.4,
            octave_shift: 0.0,
            waveform_type: 0,
            tempo_multiplier: 1.0,
        }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0/60.0;
        self.base.update_compute_shader_time(current_time, delta, &core.queue);
        self.base.update_mouse_uniform(&core.queue);
        self.base.fps_tracker.update();
        

        if self.base.time_uniform.data.frame % 5 == 0 {
            if let Some(compute_shader) = &self.base.compute_shader {
                if let Ok(gpu_samples) = pollster::block_on(compute_shader.read_audio_samples(&core.device, &core.queue)) {
                    if gpu_samples.len() >= 3 {
                        let frequency = gpu_samples[0];
                        let amplitude = gpu_samples[1];  
                        
                        if let Some(ref mut synth) = self.audio_synthesis {
                            if amplitude > 0.01 {
                                let adjusted_frequency = frequency * 2.0_f32.powf(self.octave_shift);
                                let adjusted_amplitude = amplitude * self.volume;
                                synth.update_synth_params(adjusted_frequency, adjusted_amplitude, self.waveform_type);
                            }
                        }
                    }
                }
            }
        }
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        
        let song_time = self.base.controls.get_time(&self.base.start_time);
        let is_playing = self.audio_synthesis.as_ref().map(|s| s.is_gpu_synthesis_enabled()).unwrap_or(false);
        
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(30, 20, 50, 220);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 13.0;
                });

                egui::Window::new("🎵 Veridis Quo - Daft Punk")
                    .default_size([350.0, 250.0])
                    .show(ctx, |ui| {
                        ui.heading("Veridis Quo");
                        
                        ui.separator();
                        ui.label("🎶 Song Status:");
                        ui.colored_label(
                            if is_playing { egui::Color32::from_rgb(100, 255, 100) } else { egui::Color32::from_rgb(255, 100, 100) },
                            format!("♪ Playing: {} | Time: {:.1}s", if is_playing { "YES" } else { "NO" }, song_time)
                        );

                        ui.separator();
                        ui.label("🎛️ Audio Controls:");
                        
                        ui.horizontal(|ui| {
                            ui.label("Volume:");
                            ui.add(egui::Slider::new(&mut self.volume, 0.0..=0.8).text(""));
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Octave Shift:");
                            ui.add(egui::Slider::new(&mut self.octave_shift, -2.0..=2.0).text(""));
                        });
                        
                        ui.horizontal(|ui| {
                            ui.label("Tempo:");
                            ui.add(egui::Slider::new(&mut self.tempo_multiplier, 0.5..=2.0).text("x"));
                        });
                        
                        ui.separator();
                        ui.label("🌊 Waveform:");
                        ui.horizontal(|ui| {
                            let waveform_names = [("🌊 Sine", 0), ("⚡ Sawtooth", 1), ("⬛ Square", 2)];
                            for (name, wave_type) in waveform_names.iter() {
                                let selected = self.waveform_type == *wave_type;
                                if ui.selectable_label(selected, *name).clicked() {
                                    self.waveform_type = *wave_type;
                                }
                            }
                        });
                        
                        ui.separator();
                        ui.label("ℹ️ About:");
                        ui.label("• Melody from Daft Punk's \"Veridis Quo\"");
                        ui.separator();
                        ui.label("⌨️ Controls:");
                        ui.label("H = Hide/Show UI");
                        ui.label("F = Fullscreen");
                        ui.label("R = Restart song");
                        ui.separator();
                        
                        ui.separator();
                        ui.heading("Playback Controls");
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        self.base.apply_control_request(controls_request);
        
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Veridis Quo Render Encoder"),
        });
        
        self.base.dispatch_compute_shader(&mut encoder, core);
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Veridis Quo Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            
            if let Some(compute_texture) = self.base.get_compute_output_texture() {
                render_pass.set_pipeline(&self.base.renderer.render_pipeline);
                render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
        }

        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        
        Ok(())
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
        self.base.resize_compute_shader(core);
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        let ui_handled = self.base.egui_state.on_window_event(core.window(), event).consumed;
        
        if ui_handled {
            return true;
        }
        
        if self.base.handle_mouse_input(core, event, ui_handled) {
            return true;
        }
        
        if let WindowEvent::KeyboardInput { event, .. } = event {
            if event.state == winit::event::ElementState::Pressed {
                if let winit::keyboard::Key::Character(ref s) = event.logical_key {
                    match s.as_str() {
                        "r" | "R" => {
                            self.base.start_time = std::time::Instant::now();
                            return true;
                        }
                        _ => {}
                    }
                }
            }
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Veridis Quo", 1200, 800);
    
    app.run(event_loop, |core| {
        VeridisQuo::init(core)
    })
}