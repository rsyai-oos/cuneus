use cuneus::{ShaderApp, ShaderManager, UniformProvider, UniformBinding, BaseShader};
use winit::event::*;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    lambda: f32,
    theta: f32,
    alpha: f32,
    sigma: f32,
    gamma: f32,
    blue: f32,
    use_texture_colors: f32,
}
impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Spiral Shader", 800, 600);
    let shader = SpiralShader::init(app.core());
    app.run(event_loop, shader)
}
struct SpiralShader {
    base: BaseShader,
    params_uniform: UniformBinding<ShaderParams>,
}
impl ShaderManager for SpiralShader {
    fn init(core: &cuneus::Core) -> Self {
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
                binding: 2,
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
            ShaderParams {
                lambda: 35.0,
                theta: 0.7,
                alpha: 0.7,
                sigma: 0.1,
                gamma: 0.1,
                blue: 0.1,
                use_texture_colors: 0.0,
            },
            &params_bind_group_layout,
            2,
        );
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
        let bind_group_layouts = vec![
            &texture_bind_group_layout,
            &time_bind_group_layout,
            &params_bind_group_layout,
        ];
        let base = BaseShader::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/spiral.wgsl"),
            &bind_group_layouts,
            None,
        );

        Self {
            base,
            params_uniform,
        }
    }
    fn update(&mut self, core: &cuneus::Core) {
        self.base.update_time(&core.queue);
    }
    fn render(&mut self, core: &cuneus::Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut reload_image = false;
        let mut selected_path = None;
        // Local copies of the parameters to fight the borrow checker
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let full_output = {
            let base = &mut self.base;
            base.render_ui(core, |ctx| {
                egui::Window::new("Shader Settings").show(ctx, |ui| {
                    if ui.button("Load Image").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Image", &["png", "jpg", "jpeg"])
                            .pick_file() 
                        {
                            selected_path = Some(path);
                            reload_image = true;
                        }
                    }
                    changed |= ui.add(egui::Slider::new(&mut params.lambda, 1.0..=360.0).text("Lambda")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.theta, -6.2..=6.2).text("Theta")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.alpha, 0.0..=1.0).text("Alpha")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.sigma, 0.0..=1.0).text("Sigma")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.gamma, 0.0..=1.0).text("Gamma")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.blue, 0.0..=1.0).text("Blue")).changed();
                    let mut use_texture = params.use_texture_colors > 0.5;
                    if ui.checkbox(&mut use_texture, "Use Texture Colors").changed() {
                        changed = true;
                        params.use_texture_colors = if use_texture { 1.0 } else { 0.0 };
                    }
                });
            })
        };
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        if reload_image {
            if let Some(path) = selected_path {
                self.base.load_image(core, path);
            }
        }
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
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
            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            if let Some(texture_manager) = &self.base.texture_manager {
                render_pass.set_bind_group(0, &texture_manager.bind_group, &[]);
            }
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
        self.base.handle_render_output(core, &view, full_output, &mut encoder);
        core.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
    fn handle_input(&mut self, core: &cuneus::Core, event: &WindowEvent) -> bool {
        self.base.egui_state.on_window_event(core.window(), event).consumed
    }
}