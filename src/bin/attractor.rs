use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, BaseShader, TextureManager, create_feedback_texture_pair,ExportSettings, ExportError,ExportManager,ShaderHotReload,ShaderControls};
use winit::event::WindowEvent;
use cuneus::ShaderApp;
use cuneus::Renderer;
use image::ImageError;
use std::path::PathBuf;


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
    params_uniform: UniformBinding<AttractorParams>,
    texture_pair1: (TextureManager, TextureManager),
    texture_pair2: (TextureManager, TextureManager),
    frame_count: u32,
    hot_reload: ShaderHotReload,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
}

impl AttractorShader {
    fn capture_frame(&mut self, core: &Core, time: f32) -> Result<Vec<u8>, wgpu::SurfaceError> {
        let settings = self.base.export_manager.settings();
        let (capture_texture, output_buffer) = self.base.create_capture_texture(
            &core.device,
            settings.width,
            settings.height
        );
        
        let align = 256;
        let unpadded_bytes_per_row = settings.width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;

        let capture_view = capture_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Capture Encoder"),
        });
        // Update time uniform for this frame
        self.base.time_uniform.data.time = time;
        self.base.time_uniform.update(&core.queue);

        // First Pass
        let temp_tex1 = if self.frame_count % 2 == 0 {
            &self.texture_pair1.1
        } else {
            &self.texture_pair1.0
        };

        {
            let mut render_pass = Renderer::begin_render_pass(
                &mut encoder,
                &temp_tex1.view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Capture Pass 1"),
            );

            render_pass.set_pipeline(&self.base.renderer.render_pipeline);
            render_pass.set_vertex_buffer(0, self.base.renderer.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &if self.frame_count % 2 == 0 { &self.texture_pair1.0 } else { &self.texture_pair1.1 }.bind_group, &[]);
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
        let temp_tex2 = if self.frame_count % 2 == 0 {
            &self.texture_pair2.0
        } else {
            &self.texture_pair2.1
        };

        {
            let mut render_pass = Renderer::begin_render_pass(
                &mut encoder,
                &temp_tex2.view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Capture Pass 2"),
            );
            render_pass.set_pipeline(&self.renderer_pass2.render_pipeline);
            render_pass.set_vertex_buffer(0, self.renderer_pass2.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &temp_tex1.bind_group, &[]);
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        {
            let mut render_pass = Renderer::begin_render_pass(
                &mut encoder,
                &capture_view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Capture Pass 3"),
            );

            render_pass.set_pipeline(&self.renderer_pass3.render_pipeline);
            render_pass.set_vertex_buffer(0, self.renderer_pass3.vertex_buffer.slice(..));
            render_pass.set_bind_group(0, &temp_tex2.bind_group, &[]);
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &capture_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &output_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(settings.height),
                },
            },
            wgpu::Extent3d {
                width: settings.width,
                height: settings.height,
                depth_or_array_layers: 1,
            },
        );

        core.queue.submit(Some(encoder.finish()));

        let buffer_slice = output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        core.device.poll(wgpu::Maintain::Wait);
        
        rx.recv().unwrap().unwrap();
        let padded_data = buffer_slice.get_mapped_range().to_vec();

        let mut unpadded_data = Vec::with_capacity((settings.width * settings.height * 4) as usize);
        for chunk in padded_data.chunks(padded_bytes_per_row as usize) {
            unpadded_data.extend_from_slice(&chunk[..unpadded_bytes_per_row as usize]);
        }
        
        Ok(unpadded_data)
    }

    #[allow(unused_mut)]
    fn save_frame(&self, mut data: Vec<u8>, frame: u32, settings: &ExportSettings) -> Result<(), ExportError> {
        let frame_path = settings.export_path
            .join(format!("frame_{:05}.png", frame));
        
        if let Some(parent) = frame_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        #[cfg(target_os = "macos")]
        {
            for chunk in data.chunks_mut(4) {
                chunk.swap(0, 2);
            }
        }
        let image = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
            settings.width,
            settings.height,
            data
        ).ok_or_else(|| ImageError::Parameter(
            image::error::ParameterError::from_kind(
                image::error::ParameterErrorKind::Generic(
                    "Failed to create image buffer".to_string()
                )
            )
        ))?;
        
        image.save(&frame_path)?;
        Ok(())
    }

    fn handle_export(&mut self, core: &Core) {
        if let Some((frame, time)) = self.base.export_manager.try_get_next_frame() {
            if let Ok(data) = self.capture_frame(core, time) {
                let settings = self.base.export_manager.settings();
                if let Err(e) = self.save_frame(data, frame, settings) {
                    eprintln!("Error saving frame: {:?}", e);
                }
            }
        } else {
            self.base.export_manager.complete_export();
        }
    }
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


        let params_uniform = UniformBinding::new(
            &core.device,
            "Params Uniform",
            AttractorParams {
                min_radius: 4.5,
                max_radius: 4.5,
                size: 0.99,
                decay: 0.98,
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
        let shader_paths = vec![
            PathBuf::from("shaders/vertex.wgsl"),
            PathBuf::from("shaders/attractor.wgsl"),
        ];
        let hot_reload = ShaderHotReload::new(
            core.device.clone(),
            shader_paths,
            vs_module,
            fs_module,
        ).expect("Failed to initialize hot reload");
        let renderer_pass2 = Renderer::new(
            &core.device,
            &hot_reload.vs_module,
            &hot_reload.fs_module,
            core.config.format,
            &pipeline_layout,
            Some("fs_pass2"),
        );

        let renderer_pass3 = Renderer::new(
            &core.device,
            &hot_reload.vs_module,
            &hot_reload.fs_module,
            core.config.format,
            &pipeline_layout,
            Some("fs_pass3"),
        );

        Self {
            base,
            renderer_pass2,
            renderer_pass3,
            params_uniform,
            texture_pair1,
            texture_pair2,
            frame_count: 0,
            hot_reload,
            texture_bind_group_layout,
            time_bind_group_layout,
            params_bind_group_layout,
        }
    }

    fn update(&mut self, core: &Core) {
        if let Some((new_vs, new_fs)) = self.hot_reload.check_and_reload() {
            println!("Reloading shaders at time: {:.2}s", self.base.start_time.elapsed().as_secs_f32());
            let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &self.texture_bind_group_layout,
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            self.base.renderer = Renderer::new(
                &core.device,
                new_vs,
                new_fs,
                core.config.format,
                &pipeline_layout,
                Some("fs_pass1"),
            );
    
            self.renderer_pass2 = Renderer::new(
                &core.device,
                new_vs,
                new_fs,
                core.config.format,
                &pipeline_layout,
                Some("fs_pass2"),
            );
    
            self.renderer_pass3 = Renderer::new(
                &core.device,
                new_vs,
                new_fs,
                core.config.format,
                &pipeline_layout,
                Some("fs_pass3"),
            );
        }
    
        if self.base.export_manager.is_exporting() {
            self.handle_export(core);
        }
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let output = core.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut should_start_export = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                // transparent
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });
                egui::Window::new("Attractor Settings").show(ctx, |ui| {
                    changed |= ui.add(egui::Slider::new(&mut params.min_radius, 1.0..=10.0).text("Min Radius")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.max_radius, 1.0..=10.0).text("Max Radius")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.size, 0.01..=1.2).text("color")).changed();
                    changed |= ui.add(egui::Slider::new(&mut params.decay, 0.0..=0.99).text("feedback")).changed();
                    ui.separator();
                    ShaderControls::render_controls_widget(ui, &mut controls_request);
                    ui.separator();
                    should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        self.base.export_manager.apply_ui_request(export_request);
        if controls_request.should_clear_buffers {
            let (texture_pair1, texture_pair2) = (
                create_feedback_texture_pair(
                    core,
                    core.config.width,
                    core.config.height,
                    &self.texture_bind_group_layout,
                ),
                create_feedback_texture_pair(
                    core,
                    core.config.width,
                    core.config.height,
                    &self.texture_bind_group_layout,
                )
            );
            self.texture_pair1 = texture_pair1;
            self.texture_pair2 = texture_pair2;
        }
        self.base.apply_control_request(controls_request);
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.update(&core.queue);
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }

        if should_start_export {
            self.base.export_manager.start_export();
        }
        {
            let (source_tex, target_tex) = if self.frame_count % 2 == 0 {
                (&self.texture_pair1.1, &self.texture_pair1.0)
            } else {
                (&self.texture_pair1.0, &self.texture_pair1.1)
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
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
    
        {
            let source_tex = if self.frame_count % 2 == 0 {
                &self.texture_pair1.0  // Result from Pass 1
            } else {
                &self.texture_pair1.1
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
            render_pass.set_bind_group(0, &source_tex.bind_group, &[]); // Using the result from Pass 1
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(2, &self.params_uniform.bind_group, &[]);
            render_pass.draw(0..4, 0..1);
        }
    
        // Pass 3
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
            render_pass.set_bind_group(1, &self.base.time_uniform.bind_group, &[]);
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
let (app, event_loop) = ShaderApp::new("Attractor", 800, 600);
let shader = AttractorShader::init(app.core());
app.run(event_loop, shader)
}