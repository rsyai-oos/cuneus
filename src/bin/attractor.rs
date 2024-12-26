use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, BaseShader, TextureManager, create_feedback_texture_pair};
use winit::event::WindowEvent;
use cuneus::ShaderApp;
use cuneus::Renderer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TimeUniform {
    time: f32,
}

impl UniformProvider for TimeUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AttractorParams {
    min_radius: f32,
    max_radius: f32,
    size: f32,
    decay: f32,
}

impl UniformProvider for AttractorParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct AttractorShader {
    base: BaseShader,
    renderer_pass2: Renderer,
    renderer_pass3: Renderer,
    time_uniform: UniformBinding<TimeUniform>,
    params_uniform: UniformBinding<AttractorParams>,
    texture_pair1: (TextureManager, TextureManager),
    texture_pair2: (TextureManager, TextureManager),
    frame_count: u32,
}

impl ShaderManager for AttractorShader {
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

        let time_uniform = UniformBinding::new(
            &core.device,
            "Time Uniform",
            TimeUniform { time: 0.0 },
            &time_bind_group_layout,
            0,
        );

        let params_uniform = UniformBinding::new(
            &core.device,
            "Params Uniform",
            AttractorParams {
                min_radius: 4.5,
                max_radius: 4.5,
                size: 0.07,
                decay: 0.95,
            },
            &params_bind_group_layout,
            0,
        );

        let texture_pair1 = create_feedback_texture_pair(
            core,
            core.config.width,
            core.config.height,
            &texture_bind_group_layout,
        );

        let texture_pair2 = create_feedback_texture_pair(
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/attractor.wgsl").into()),
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
            include_str!("../../shaders/attractor.wgsl"),
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

        let renderer_pass3 = Renderer::new(
            &core.device,
            &vs_module,
            &fs_module,
            core.config.format,
            &pipeline_layout,
            Some("fs_pass3"),
        );

        Self {
            base,
            renderer_pass2,
            renderer_pass3,
            time_uniform,
            params_uniform,
            texture_pair1,
            texture_pair2,
            frame_count: 0,
        }
    }

    fn update(&mut self, core: &Core) {
        self.time_uniform.data.time = self.base.start_time.elapsed().as_secs_f32();
        self.time_uniform.update(&core.queue);
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
                egui::Window::new("Attractor Settings").show(ctx, |ui| {
                    changed |= ui.add(egui::Slider::new(&mut params.min_radius, 1.0..=10.0).text("Min Radius")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.max_radius, 1.0..=10.0).text("Max Radius")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.size, 0.01..=0.2).text("Size")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.decay, 0.8..=0.99).text("Decay")).changed();
                });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})  // Empty UI when hidden
        };

        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }

 {
    let source_tex = if self.frame_count % 2 == 0 {
        &self.texture_pair1.0
    } else {
        &self.texture_pair1.1
    };
    
    let target_tex = if self.frame_count % 2 == 0 {
        &self.texture_pair1.1
    } else {
        &self.texture_pair1.0
    };

    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Pass 1"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &target_tex.view,
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

    render_pass.set_pipeline(&self.base.renderer.render_pipeline);
    render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
    render_pass.set_bind_group(0, &source_tex.bind_group, &[]);
    render_pass.set_bind_group(1, &self.time_uniform.bind_group, &[]);
    render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
    render_pass.draw(0..4, 0..1);
}

{
    let source_tex = if self.frame_count % 2 == 0 {
        &self.texture_pair1.1
    } else {
        &self.texture_pair1.0
    };
    
    let target_tex = if self.frame_count % 2 == 0 {
        &self.texture_pair2.0
    } else {
        &self.texture_pair2.1
    };

    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Pass 2"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &target_tex.view,
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

    render_pass.set_pipeline(&self.renderer_pass2.render_pipeline);
    render_pass.set_vertex_buffer(0, self.renderer_pass2.vertex_buffer.slice(..));
    render_pass.set_bind_group(0, &source_tex.bind_group, &[]);
    render_pass.set_bind_group(1, &self.time_uniform.bind_group, &[]);
    render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
    render_pass.draw(0..4, 0..1);
}

{
    let source_tex = if self.frame_count % 2 == 0 {
        &self.texture_pair2.0
    } else {
        &self.texture_pair2.1
    };

    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Pass 3"),
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

    render_pass.set_pipeline(&self.renderer_pass3.render_pipeline);
    render_pass.set_vertex_buffer(0, self.renderer_pass3.vertex_buffer.slice(..));
    render_pass.set_bind_group(0, &source_tex.bind_group, &[]);
    render_pass.set_bind_group(1, &self.time_uniform.bind_group, &[]);
    render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
    render_pass.draw(0..4, 0..1);
}

self.frame_count = self.frame_count.wrapping_add(1);

self.base.handle_render_output(core, &view, full_output, &mut encoder);

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
let (app, event_loop) = ShaderApp::new("Attractor Shader", 800, 600);
let shader = AttractorShader::init(app.core());
app.run(event_loop, shader)
}