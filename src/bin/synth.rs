use cuneus::{Core, ShaderApp, ShaderManager, RenderKit};
use cuneus::{SynthesisManager, SynthesisUniform};
use cuneus::compute::{ComputeShaderConfig, COMPUTE_TEXTURE_FORMAT_RGBA16};
use winit::event::*;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let (app, event_loop) = ShaderApp::new("Synth", 600, 300);
    app.run(event_loop, |core| {
        SynthManager::init(core)
    })
}

struct SynthManager {
    base: RenderKit,
    synthesis_uniform: cuneus::UniformBinding<SynthesisUniform>,
    gpu_synthesis: Option<SynthesisManager>,
}

impl SynthManager {
    
    fn update_synthesis_visualization(&mut self, queue: &wgpu::Queue) {
        self.synthesis_uniform.data.master_volume = 0.3;
        self.synthesis_uniform.update(queue);
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
        
        let mut base = RenderKit::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/blit.wgsl"),
            &[&texture_bind_group_layout],
            None,
        );
        
        let synthesis_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("synthesis_bind_group_layout"),
        });

        let config = ComputeShaderConfig {
            workgroup_size: [16, 16, 1],
            workgroup_count: Some([64, 4, 1]), // Fixed workgroups: 64*16=1024 samples in X, independent of window size
            dispatch_once: false,
            storage_texture_format: COMPUTE_TEXTURE_FORMAT_RGBA16,
            enable_atomic_buffer: false,
            atomic_buffer_multiples: 4,
            entry_points: vec!["main".to_string()],
            sampler_address_mode: wgpu::AddressMode::ClampToEdge,
            sampler_filter_mode: wgpu::FilterMode::Linear,
            label: "Synth".to_string(),
            mouse_bind_group_layout: Some(synthesis_bind_group_layout.clone()),
            enable_fonts: false,
            enable_audio_buffer: true,
            audio_buffer_size: 1024,
        };
        
        let synthesis_uniform = cuneus::UniformBinding::new(
            &core.device,
            "Synthesis Uniform",
            SynthesisUniform::new(),
            &synthesis_bind_group_layout,
            0,
        );

        base.compute_shader = Some(cuneus::compute::ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/synth.wgsl"),
            config,
        ));
        
        if let Some(compute_shader) = &mut base.compute_shader {
            compute_shader.add_mouse_uniform_binding(&synthesis_uniform.bind_group, 2);
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
            synthesis_uniform,
            gpu_synthesis,
        }
    }
    
    fn update(&mut self, core: &Core) {
        self.base.fps_tracker.update();
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.base.update_compute_shader_time(current_time, delta, &core.queue);
        
        
        if self.base.time_uniform.data.frame % 180 == 0 {
            if let Some(compute_shader) = &self.base.compute_shader {
                if let Ok(gpu_samples) = pollster::block_on(compute_shader.read_audio_samples(&core.device, &core.queue)) {
                    if gpu_samples.len() >= 3 {
                        let frequency = gpu_samples[0];
                        let amplitude = gpu_samples[1];
                        let waveform_type = gpu_samples[2] as u32;
                        
                        if let Some(ref mut synth) = self.gpu_synthesis {
                            synth.update_synth_params(frequency, amplitude, waveform_type);
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
        
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        controls_request.current_fps = Some(self.base.fps_tracker.fps());
        
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GPU Synth Render Encoder"),
        });
        
        self.base.dispatch_compute_shader(&mut encoder, core);
        
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
            
            if let Some(compute_texture) = self.base.get_compute_output_texture() {
                render_pass.set_pipeline(&self.base.renderer.render_pipeline);
                render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
        }
        
        let full_output = self.base.render_ui(core, |_ctx| {});
        
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
        self.base.egui_state.on_window_event(core.window(), event).consumed;
        false
    }
}