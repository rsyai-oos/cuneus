use cuneus::{Core, ShaderApp, ShaderManager, RenderKit, ShaderControls};
use cuneus::compute::{ComputeShaderConfig, COMPUTE_TEXTURE_FORMAT_RGBA16};
use winit::event::*;
use std::path::PathBuf;

struct ComputeExample {
    base: RenderKit,
}

impl ShaderManager for ComputeExample {
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
        };
        
        // Create compute shader with our backend
        base.compute_shader = Some(cuneus::compute::ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/compute_basic.wgsl"),
            compute_config,
        ));
        
        // Enable hot reload if desired
        if let Some(compute_shader) = &mut base.compute_shader {
            // Create shader module for hot reload
            let shader_module = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Basic Compute Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/compute_basic.wgsl").into()),
            });
            if let Err(e) = compute_shader.enable_hot_reload(
                core.device.clone(),
                PathBuf::from("shaders/compute_basic.wgsl"),
                shader_module,
            ) {
                eprintln!("Failed to enable compute shader hot reload: {}", e);
            }
        }
        
        Self { base }
    }

    fn update(&mut self, core: &Core) {
        // Update compute shader time
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0/60.0; // Approximate delta time
        self.base.update_compute_shader_time(current_time, delta, &core.queue);
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
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });
                
                egui::Window::new("Compute Shader Controls")
                    .show(ctx, |ui| {
                        // Time controls (play/pause/reset)
                        ui.heading("Controls");
                        ShaderControls::render_controls_widget(ui, &mut controls_request);
                        
                        // Add any shader-specific controls here
                        ui.separator();
                        ui.label("Press 'H' to toggle this UI");
                        ui.label("Press 'F' to toggle fullscreen");
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
        if self.base.egui_state.on_window_event(core.window(), event).consumed {
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
    let (app, event_loop) = ShaderApp::new("Compute Shader Example", 800, 600);
    
    app.run(event_loop, |core| {
        ComputeExample::init(core)
    })
}