// This example demonstrates a how to generate audio using cunes via compute shaders
use cuneus::{Core, ShaderApp, ShaderManager, RenderKit, UniformProvider, UniformBinding, ShaderControls};
use cuneus::SynthesisManager;
use cuneus::compute::{ComputeShaderConfig, COMPUTE_TEXTURE_FORMAT_RGBA16};
use winit::event::*;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let (app, event_loop) = ShaderApp::new("Synth", 800, 600);
    app.run(event_loop, |core| {
        SynthManager::init(core)
    })
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SynthParams {
    tempo: f32,
    waveform_type: u32,
    octave: f32,
    volume: f32,
    beat_enabled: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    key_states: [[f32; 4]; 3],
    key_decay: [[f32; 4]; 3],
}

impl UniformProvider for SynthParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct SynthManager {
    base: RenderKit,
    params_uniform: UniformBinding<SynthParams>,
    gpu_synthesis: Option<SynthesisManager>,
    key_press_times: [Option<std::time::Instant>; 9],
}

impl SynthManager {
    fn update_synthesis_visualization(&mut self, _queue: &wgpu::Queue) {}
    
    fn set_key_state(&mut self, key_index: usize, state: f32) {
        if key_index < 9 {
            let vec_idx = key_index / 4;
            let comp_idx = key_index % 4;
            self.params_uniform.data.key_states[vec_idx][comp_idx] = state;
        }
    }
    
    fn set_key_decay(&mut self, key_index: usize, decay: f32) {
        if key_index < 9 {
            let vec_idx = key_index / 4;
            let comp_idx = key_index % 4;
            self.params_uniform.data.key_decay[vec_idx][comp_idx] = decay;
        }
    }
}

impl ShaderManager for SynthManager {
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
        
        let params_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("params_bind_group_layout"),
        });
        
        let mut base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        

        let config = ComputeShaderConfig {
            workgroup_size: [16, 16, 1],
            workgroup_count: None,
            dispatch_once: false,
            storage_texture_format: COMPUTE_TEXTURE_FORMAT_RGBA16,
            enable_atomic_buffer: false,
            atomic_buffer_multiples: 4,
            entry_points: vec!["main".to_string()],
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "Synth".to_string(),
            mouse_bind_group_layout: Some(params_bind_group_layout.clone()),
            enable_fonts: false,
            enable_audio_buffer: true,
            audio_buffer_size: 1024,
        };
        
        let params_uniform = UniformBinding::new(
            &core.device,
            "Synth Params",
            SynthParams {
                tempo: 120.0,
                waveform_type: 0,
                octave: 4.0,
                volume: 0.3,
                beat_enabled: 1,
                _padding1: 0,
                _padding2: 0,
                _padding3: 0,
                key_states: [[0.0; 4]; 3],
                key_decay: [[0.0; 4]; 3],
            },
            &params_bind_group_layout,
            0,
        );

        base.compute_shader = Some(cuneus::compute::ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/synth.wgsl"),
            config,
        ));
        
        if let Some(compute_shader) = &mut base.compute_shader {
            compute_shader.add_mouse_uniform_binding(&params_uniform.bind_group, 2);
        }
        
        if let Some(compute_shader) = &mut base.compute_shader {
            let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Synth Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/synth.wgsl").into()),
            });
            if let Err(_e) = compute_shader.enable_hot_reload(
                core.device.clone(),
                PathBuf::from("shaders/synth.wgsl"),
                shader_module,
            ) {
            }
        }
        
        let gpu_synthesis = match SynthesisManager::new() {
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
            params_uniform,
            gpu_synthesis,
            key_press_times: [None; 9],
        }
    }
    
    fn update(&mut self, core: &Core) {
        self.base.fps_tracker.update();
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.base.update_compute_shader_time(current_time, delta, &core.queue);
        
        let mut keys_updated = false;
        for i in 0..9 {
            if let Some(_press_time) = self.key_press_times[i] {
                // Key is currently held - maintain full intensity for sustain
                self.set_key_decay(i, 1.0);
                keys_updated = true;
            } else {
                // Key released - apply smooth exponential fade-out
                let current_decay = self.params_uniform.data.key_decay[i / 4][i % 4];
                if current_decay > 0.01 {
                    let fade_speed = 0.95; // Faster fade but still smooth
                    let new_decay = current_decay * fade_speed;
                    self.set_key_decay(i, new_decay);
                    keys_updated = true;
                } else {
                    // Fade complete
                    self.set_key_state(i, 0.0);
                    self.set_key_decay(i, 0.0);
                }
            }
        }
        
        if keys_updated {
            self.params_uniform.update(&core.queue);
        }
        
        
        // Read shader-generated audio parameters and control polyphonic voices
        if self.base.time_uniform.data.frame % 60 == 0 { // Check every second at 60fps
            if let Some(compute_shader) = &self.base.compute_shader {
                if let Ok(gpu_samples) = pollster::block_on(compute_shader.read_audio_samples(&core.device, &core.queue)) {
                    if gpu_samples.len() >= 12 { // Need at least 12 values (3 base + 9 frequencies)
                        let amplitude = gpu_samples[1];
                        let waveform_type = gpu_samples[2] as u32;
                        
                        // Read shader-generated frequencies for all 9 keys (positions 3-11)
                        let mut shader_frequencies = [440.0; 9];
                        for i in 0..9 {
                            shader_frequencies[i] = gpu_samples[3 + i];
                        }
                        
                        if let Some(ref mut synth) = self.gpu_synthesis {
                            // Update global waveform type from shader
                            synth.update_synth_params(440.0, amplitude, waveform_type);
                            
                            // Control individual voices using SHADER-GENERATED frequencies
                            for i in 0..9 {
                                let vec_idx = i / 4;
                                let comp_idx = i % 4;
                                let key_state = self.params_uniform.data.key_states[vec_idx][comp_idx];
                                let key_decay = self.params_uniform.data.key_decay[vec_idx][comp_idx];
                                
                                if key_state > 0.5 {
                                    // Key is pressed - use SHADER-GENERATED frequency, this coming from our shader 
                                    let frequency = shader_frequencies[i];
                                    let note_amplitude = amplitude * key_decay * 0.3;
                                    
                                    synth.start_voice(i, frequency, note_amplitude);
                                } else if key_decay < 0.1 {
                                    // Key released and fade complete
                                    synth.release_voice(i);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        self.update_synthesis_visualization(&core.queue);
        
        if let Some(ref mut synth) = self.gpu_synthesis {
            synth.update();
        }
    }
    
    
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Synth Render Encoder"),
        });
        
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut controls_request = self.base.controls.get_ui_request(&self.base.start_time, &core.size);
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });
                
                egui::Window::new("Cuneus Synth")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(250.0)
                    .show(ctx, |ui| {
                        ui.vertical(|ui| {
                            ui.label("GPU Synthesizer");
                            ui.separator();
                            
                            ui.label("Piano Controls:");
                            ui.label("Press keys 1-9 for C D E F G A B C D");
                            
                            ui.separator();
                            ui.label("Settings:");
                            
                            let mut beat_enabled = params.beat_enabled > 0;
                            if ui.checkbox(&mut beat_enabled, "Background Beat").changed() {
                                params.beat_enabled = if beat_enabled { 1 } else { 0 };
                                changed = true;
                            }
                            
                            changed |= ui.add(egui::Slider::new(&mut params.tempo, 60.0..=180.0).text("Beat Tempo")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.octave, 3.0..=6.0).text("Octave")).changed();
                            changed |= ui.add(egui::Slider::new(&mut params.volume, 0.0..=1.0).text("Volume")).changed();
                            
                            ui.separator();
                            ui.label("Waveform:");
                            ui.horizontal(|ui| {
                                let waveform_names = ["Sine", "Saw", "Square"];
                                for (i, name) in waveform_names.iter().enumerate() {
                                    let selected = params.waveform_type == i as u32;
                                    if ui.selectable_label(selected, *name).clicked() {
                                        params.waveform_type = i as u32;
                                        changed = true;
                                    }
                                }
                            });
                            
                            ui.separator();
                            ui.label("How it works:");
                            ui.label("• Press 1-9 keys to trigger musical notes");
                            ui.label("• Toggle background beat on/off");
                            ui.label("• Waveform changes tone color");
                            ui.label("• Visual bars show key activity");
                            ui.label("• All audio generated on GPU");
                            
                            ui.separator();
                            ShaderControls::render_controls_widget(ui, &mut controls_request);
                        });
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.base.update_compute_shader_time(current_time, delta, &core.queue);
        
        self.update_synthesis_visualization(&core.queue);
        
        self.base.dispatch_compute_shader(&mut encoder, core);
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Display Pass"),
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
        
        self.base.apply_control_request(controls_request);
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
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
            return true;
        }
        
        if let WindowEvent::KeyboardInput { event, .. } = event {
            if event.state == winit::event::ElementState::Pressed {
                if let winit::keyboard::Key::Character(ref s) = event.logical_key {
                    if let Some(key_index) = s.chars().next().and_then(|c| c.to_digit(10)) {
                        if key_index >= 1 && key_index <= 9 {
                            let index = (key_index - 1) as usize;
                            
                            // Only start if not already pressed (prevent retriggering)
                            if self.key_press_times[index].is_none() {
                                self.key_press_times[index] = Some(std::time::Instant::now());
                                self.set_key_state(index, 1.0);
                                self.set_key_decay(index, 1.0);
                                self.params_uniform.update(&core.queue);
                            }
                            return true;
                        }
                    }
                }
            } else if event.state == winit::event::ElementState::Released {
                // Handle key release for smooth fade-out
                if let winit::keyboard::Key::Character(ref s) = event.logical_key {
                    if let Some(key_index) = s.chars().next().and_then(|c| c.to_digit(10)) {
                        if key_index >= 1 && key_index <= 9 {
                            let index = (key_index - 1) as usize;
                            
                            // Start fade-out process
                            if self.key_press_times[index].is_some() {
                                self.key_press_times[index] = None; // This triggers fade-out in update()
                                self.set_key_state(index, 0.0);
                                // Keep current decay value for smooth fade
                                self.params_uniform.update(&core.queue);
                            }
                            return true;
                        }
                    }
                }
            }
            return self.base.key_handler.handle_keyboard_input(core.window(), event);
        }
        
        false
    }
}