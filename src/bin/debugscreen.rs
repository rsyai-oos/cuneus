use cuneus::{Core, ShaderApp, ShaderManager, RenderKit, ShaderControls};
use cuneus::compute::{ComputeShaderConfig, COMPUTE_TEXTURE_FORMAT_RGBA16, create_bind_group_layout, BindGroupLayoutType};
use cuneus::audio::SynthesisManager;
use winit::event::*;
use std::path::PathBuf;

struct DebugScreen {
    base: RenderKit,
    audio_synthesis: Option<SynthesisManager>,
    generate_note: bool,
}

impl ShaderManager for DebugScreen {
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
        
        let mouse_bind_group_layout = create_bind_group_layout(
            &core.device,
            BindGroupLayoutType::MouseUniform,
            "Debug Screen Mouse"
        );
        
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
            workgroup_count: None,  // Auto-determine from texture size
            dispatch_once: false,   // Run every frame
            storage_texture_format: COMPUTE_TEXTURE_FORMAT_RGBA16,
            enable_atomic_buffer: false,  // Not needed for this simple shader
            atomic_buffer_multiples: 4,
            entry_points: vec!["main".to_string()],  // Single entry point
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "Basic Compute".to_string(),
            mouse_bind_group_layout: None,  // Don't pass here, add separately
            enable_fonts: true,
            enable_audio_buffer: true,
            audio_buffer_size: 1024,
            enable_custom_uniform: false,
            enable_input_texture: false,
            custom_storage_buffers: Vec::new(),
        };
        
        // Create compute shader with our backend
        base.compute_shader = Some(cuneus::compute::ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/debugscreen.wgsl"),
            compute_config,
        ));
        
        if let (Some(compute_shader), Some(mouse_uniform)) = (&mut base.compute_shader, &base.mouse_uniform) {
            compute_shader.add_mouse_uniform_binding(
                &mouse_uniform.bind_group,
                2
            );
        }
        
        // Enable hot reload if desired
        if let Some(compute_shader) = &mut base.compute_shader {
            // Create shader module for hot reload
            let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Debug Screen Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/debugscreen.wgsl").into()),
            });
            if let Err(e) = compute_shader.enable_hot_reload(
                core.device.clone(),
                PathBuf::from("shaders/debugscreen.wgsl"),
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
            generate_note: false,
        }
    }

    fn update(&mut self, core: &Core) {
        // Update compute shader time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0/60.0; // Approximate delta time
        self.base.update_compute_shader_time(current_time, delta, &core.queue);
        self.base.update_mouse_uniform(&core.queue);
        self.base.fps_tracker.update();
        // Handle audio generation when note is requested
        if self.generate_note {
            // Read GPU-generated audio parameters for CPU synthesis (check every few frames to reduce overhead)
            if self.base.time_uniform.data.frame % 60 == 0 {
                if let Some(compute_shader) = &self.base.compute_shader {
                    if let Ok(gpu_samples) = pollster::block_on(compute_shader.read_audio_samples(&core.device, &core.queue)) {
                        if gpu_samples.len() >= 12 { // Use full polyphonic system
                            let amplitude = gpu_samples[1];
                            let waveform_type = gpu_samples[2] as u32;
                            
                            if let Some(ref mut synth) = self.audio_synthesis {
                                // Update global waveform
                                synth.update_waveform(waveform_type);
                                
                                // Use voice 0 for simple debug audio
                                let frequency = gpu_samples[3]; // Use first shader-generated frequency
                                let active = amplitude > 0.01;
                                let amp = if active { amplitude * 0.3 } else { 0.0 };
                                synth.set_voice(0, frequency, amp, active);
                            }
                        }
                    }
                }
            }
        } else {
            // Stop audio when not requested using unified API
            if let Some(ref mut synth) = self.audio_synthesis {
                synth.set_voice(0, 440.0, 0.0, false);
            }
        }
        
        if let Some(ref mut synth) = self.audio_synthesis {
            synth.update();
        }
    }
    fn resize(&mut self, core: &Core) {
        // Update resolution uniform
        self.base.update_resolution(&core.queue, core.size);
        // Resize compute shader resources
        self.base.resize_compute_shader(core);
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        let mouse_pos = self.base.mouse_tracker.uniform.position;
        let raw_pos = self.base.mouse_tracker.raw_position;
        let mouse_buttons = self.base.mouse_tracker.uniform.buttons[0];
        let mouse_wheel = self.base.mouse_tracker.uniform.wheel;
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });
                
                egui::Window::new("Debug Screen Controls")
                    .show(ctx, |ui| {
                        // Time controls (play/pause/reset)
                        ui.heading("Controls");
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        ui.separator();
                        ui.heading("Mouse Debug");
                        ui.label(format!("Position (normalized): {:.3}, {:.3}", mouse_pos[0], mouse_pos[1]));
                        ui.label(format!("Position (pixels): {:.1}, {:.1}", raw_pos[0], raw_pos[1]));
                        ui.label(format!("Buttons: {:#b}", mouse_buttons));
                        ui.label(format!("Wheel: {:.2}, {:.2}", mouse_wheel[0], mouse_wheel[1]));
                        
                        ui.separator();
                        ui.heading("Audio Test");
                        if ui.button("Press 5 to generate a simple note").clicked() {
                            self.generate_note = !self.generate_note;
                        }
                        
                        // Check for '5' key in egui context (this handles it properly)
                        if ui.input(|i| i.key_pressed(egui::Key::Num5)) {
                            self.generate_note = !self.generate_note;
                        }
                        
                        let audio_status = if self.generate_note { 
                            "🔊 Note playing" 
                        } else { 
                            "🔇 No audio" 
                        };
                        ui.label(audio_status);
                        
                        if let Some(ref synth) = self.audio_synthesis {
                            if synth.is_gpu_synthesis_enabled() {
                                ui.label("✓ Audio synthesis ready");
                            } else {
                                ui.label("⚠ Audio synthesis not active");
                            }
                        } else {
                            ui.label("❌ Audio synthesis unavailable");
                        }
                        
                        ui.separator();
                        ui.label("Controls:");
                        ui.label("• Left-click to invert colors");
                        ui.label("• Scroll wheel to create pulse effect");
                        ui.label("• Press 'H' to toggle this UI");
                        ui.label("• Press 'F' to toggle fullscreen");
                        ui.label("• Press '5' to generate audio note");
                    });
            })
        } else {
            // Empty UI if hidden
            self.base.render_ui(core, |_ctx| {})
        };
        
        // Apply control requests (play/pause/etc)
        self.base.apply_control_request(controls_request);
        
        // Create command encoder
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        
        // Run compute shader
        self.base.dispatch_compute_shader(&mut encoder, core);
        
        // Render compute output to screen
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
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
            // Draw the compute shader output
            if let Some(compute_texture) = self.base.get_compute_output_texture() {
                render_pass.set_pipeline(&self.base.renderer.render_pipeline);
                render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
        }
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        // Submit work and present
        core.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        // Handle egui events
        let ui_handled = self.base.egui_state.on_window_event(core.window(), event).consumed;
        
        // Handle mouse input for shader if UI didn't consume it
        if self.base.handle_mouse_input(core, event, ui_handled) {
            return true;
        }
        // Handle keyboard input
        if let WindowEvent::KeyboardInput { event, .. } = event {
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }
        
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    cuneus::gst::init()?;
    
    let (app, event_loop) = ShaderApp::new("Debug Screen", 800, 600);
    
    app.run(event_loop, |core| {
        DebugScreen::init(core)
    })
}