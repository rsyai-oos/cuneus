use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, BaseShader,TextureManager};
use winit::event::WindowEvent;
use cuneus::ShaderApp;
use cuneus::Renderer;
use cuneus::create_feedback_texture_pair;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FeedbackParams {
    feedback: f32,
    speed: f32,
    scale: f32,
}
impl UniformProvider for FeedbackParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
struct FeedbackShader {
    base: BaseShader,
    renderer_pass2: Renderer,
    params_uniform: UniformBinding<FeedbackParams>,
    texture_a: Option<TextureManager>,
    texture_b: Option<TextureManager>,
    frame_count: u32,
}

impl ShaderManager for FeedbackShader {
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

        let time_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("time_bind_group_layout"),
        });
        let params_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("params_bind_group_layout"),
        });
        let params_uniform = UniformBinding::new(
            &core.device,
            "Params Uniform",
            FeedbackParams {
                feedback: 0.95,
                speed: 1.0,
                scale: 1.0,
            },
            &params_bind_group_layout,
            0,
        );
        let (texture_a, texture_b) = create_feedback_texture_pair(
            core,
            core.config.width,
            core.config.height,
            &texture_bind_group_layout,
        );
        let vs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/vertex.wgsl").into()),
        });
        let fs_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/feedback.wgsl").into()),
        });
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &time_bind_group_layout,
                &params_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let base = BaseShader::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/feedback.wgsl"),
            &[
                &texture_bind_group_layout,
                &time_bind_group_layout,
                &params_bind_group_layout,
            ],
            Some("fs_pass1"),
        );
        let renderer_pass2 = Renderer::new(
            &core.device,
            &vs_module,
            &fs_module,
            core.config.format,
            &pipeline_layout,
            Some("fs_pass2"),
        );
        Self {
            base,
            renderer_pass2,
            params_uniform,
            texture_a: Some(texture_a),
            texture_b: Some(texture_b),
            frame_count: 0,
        }
    }
    fn update(&mut self, core: &Core) {
        self.base.update_time(&core.queue);
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                egui::Window::new("Feedback Settings").show(ctx, |ui| {
                    changed |= ui.add(egui::Slider::new(&mut params.feedback, 0.0..=0.99).text("Feedback")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.speed, 0.1..=5.0).text("Speed")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.scale, 0.1..=2.0).text("Scale")).changed();
                });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        if let (Some(ref texture_a), Some(ref texture_b)) = (&self.texture_a, &self.texture_b) {
            let (source_texture, target_texture) = if self.frame_count % 2 == 0 {
                (texture_b, texture_a)
            } else {
                (texture_a, texture_b)
            };
            {
                let mut render_pass = Renderer::begin_render_pass(
                    &mut encoder,
                    &target_texture.view,
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    Some("Feedback Pass"),
                );
                render_pass.set_pipeline(&self.base.renderer.render_pipeline);
                render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &source_texture.bind_group, &[]);
                render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
                render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
            {
                let mut render_pass = Renderer::begin_render_pass(
                    &mut encoder,
                    &view,
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    Some("Display Pass"),
                );
                render_pass.set_pipeline(&self.renderer_pass2.render_pipeline);
                render_pass.set_vertex_buffer(0, self.renderer_pass2.vertex_buffer.slice(..));
                render_pass.set_bind_group(0, &target_texture.bind_group, &[]);
                render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
                render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
            self.frame_count = self.frame_count.wrapping_add(1);
        }
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        encoder.insert_debug_marker("Transition to Present");
        core.queue.submit(Some(encoder.finish()));
        output.present();
    
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
    let (app, event_loop) = ShaderApp::new("Feedback Shader", 800, 600);
    let shader = FeedbackShader::init(app.core());
    app.run(event_loop, shader)
}
