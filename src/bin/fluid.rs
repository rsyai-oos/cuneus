use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, BaseShader, TextureManager, create_feedback_texture_pair,ExportSettings, ExportError, ExportManager,ShaderHotReload,ShaderControls};
use winit::event::WindowEvent;
use cuneus::ShaderApp;
use cuneus::Renderer;
use image::ImageError;
use std::path::PathBuf;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct FluidParams {
    rotation_speed: f32,
    motor_strength: f32,
    distortion: f32,
    feedback: f32,
}
impl UniformProvider for FluidParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
struct FluidShader {
    base: BaseShader,
    renderer_pass2: Renderer,
    params_uniform: UniformBinding<FluidParams>,
    texture_a: Option<TextureManager>,
    texture_b: Option<TextureManager>,
    input_texture: Option<TextureManager>,
    frame_count: u32,
    hot_reload: ShaderHotReload,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    input_texture_bind_group_layout: wgpu::BindGroupLayout,
    time_bind_group_layout: wgpu::BindGroupLayout,
    params_bind_group_layout: wgpu::BindGroupLayout,
}
impl FluidShader {
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
        self.base.time_uniform.data.time = time;
        self.base.time_uniform.data.frame= self.frame_count;
        self.base.time_uniform.update(&core.queue);
        {
            let mut render_pass = Renderer::begin_render_pass(
                &mut encoder,
                &capture_view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                Some("Capture Pass"),
            );
            render_pass.set_pipeline(&self.renderer_pass2.render_pipeline);
            render_pass.set_vertex_buffer(0, self.renderer_pass2.vertex_buffer.slice(..));
            // First binding (feedback texture)
            if let Some(ref texture_a) = self.texture_a {
                render_pass.set_bind_group(0, &texture_a.bind_group, &[]);
            }
            // Second binding (input texture) - could be image, video, or default
            if let Some(ref input_texture) = self.input_texture {
                render_pass.set_bind_group(1, &input_texture.bind_group, &[]);
            } else if self.base.using_video_texture {
                if let Some(video_manager) = &self.base.video_texture_manager {
                    render_pass.set_bind_group(1, &video_manager.texture_manager().bind_group, &[]);
                } else if let Some(ref default_texture) = self.base.texture_manager {
                    render_pass.set_bind_group(1, &default_texture.bind_group, &[]);
                }
            } else if let Some(ref default_texture) = self.base.texture_manager {
                render_pass.set_bind_group(1, &default_texture.bind_group, &[]);
            }
            render_pass.set_bind_group(2, &self.base.time_uniform.bind_group, &[]);
            render_pass.set_bind_group(3, &self.params_uniform.bind_group, &[]);
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

impl ShaderManager for FluidShader {
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
        let input_texture_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: Some("input_texture_bind_group_layout"),
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
            FluidParams {
                rotation_speed: 2.0,
                motor_strength: 0.01,
                distortion: 10.0,
                feedback: 0.95,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fluid.wgsl").into()),
        });
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[
                &texture_bind_group_layout,
                &input_texture_bind_group_layout,
                &time_bind_group_layout,
                &params_bind_group_layout,
            ],
            push_constant_ranges: &[],
        });
        let shader_paths = vec![
            PathBuf::from("shaders/vertex.wgsl"),
            PathBuf::from("shaders/fluid.wgsl"),
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
        let base = BaseShader::new(
            core,
            include_str!("../../shaders/vertex.wgsl"),
            include_str!("../../shaders/fluid.wgsl"),
            &[
                &texture_bind_group_layout,
                &input_texture_bind_group_layout,
                &time_bind_group_layout,
                &params_bind_group_layout,
            ],
            Some("fs_pass1"),
        );
        Self {
            base,
            renderer_pass2,
            params_uniform,
            texture_a: Some(texture_a),
            texture_b: Some(texture_b),
            input_texture: None,
            frame_count: 0,
            hot_reload,
            texture_bind_group_layout,
            input_texture_bind_group_layout,
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
                    &self.input_texture_bind_group_layout,
                    &self.time_bind_group_layout,
                    &self.params_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });
            self.renderer_pass2 = Renderer::new(
                &core.device,
                new_vs,
                new_fs,
                core.config.format,
                &pipeline_layout,
                Some("fs_pass2"),
            );
            self.base.renderer = Renderer::new(
                &core.device,
                new_vs,
                new_fs,
                core.config.format,
                &pipeline_layout,
                Some("fs_pass1"),
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
        
        // Local copies of parameters to fight the borrow checker
        let mut params = self.params_uniform.data;
        let mut changed = false;
        let mut export_request = self.base.export_manager.get_ui_request();
        let mut controls_request = self.base.controls.get_ui_request(
            &self.base.start_time,
            &core.size
        );
        let mut should_start_export = false;
        
        // Extract video info before entering the closure to avoid borrow checker issues
        let using_video_texture = self.base.using_video_texture;
        let video_info = self.base.get_video_info();
        
        // Update video texture if one is loaded
        if self.base.using_video_texture {
            self.base.update_video_texture(core, &core.queue);
        }
        
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                });                egui::Window::new("Fluid Settings").show(ctx, |ui| {
                    // Media controls in collapsible section
                    ShaderControls::render_media_panel(
                        ui,
                        &mut controls_request,
                        using_video_texture,
                        video_info
                    );
                    
                    ui.separator();
                    
                    // Shader-specific parameters
                    ui.collapsing("Fluid Parameters", |ui| {
                        changed |= ui.add(egui::Slider::new(&mut params.rotation_speed, -5.0..=5.0).text("Rotation Speed")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.motor_strength, -0.2..=0.2).text("Motor Strength")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.distortion, 1.0..=20.0).text("Distortion")).changed();
                        changed |= ui.add(egui::Slider::new(&mut params.feedback, 0.0..=1.01).text("Feedback")).changed();
                    });
                    
                    ui.separator();
                    ShaderControls::render_controls_widget(ui, &mut controls_request);
                    ui.separator();
                    should_start_export = ExportManager::render_export_ui_widget(ui, &mut export_request);
                });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        if controls_request.should_clear_buffers {
            let (texture_a, texture_b) = create_feedback_texture_pair(
                core,
                core.config.width,
                core.config.height,
                &self.texture_bind_group_layout,
            );
            self.texture_a = Some(texture_a);
            self.texture_b = Some(texture_b);
        }
        
        // Apply control requests
        self.base.export_manager.apply_ui_request(export_request);
        self.base.apply_control_request(controls_request.clone());
        self.base.handle_video_requests(core, &controls_request);
        
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let current_frame = self.base.controls.get_frame();
        self.base.time_uniform.data.time = current_time;
        self.base.time_uniform.data.frame = current_frame;
        self.base.time_uniform.update(&core.queue);
        
        if should_start_export {
            self.base.export_manager.start_export();
        }
        
        if changed {
            self.params_uniform.data = params;
            self.params_uniform.update(&core.queue);
        }
        
        // Process feedback textures
        if let (Some(ref texture_a), Some(ref texture_b)) = (&self.texture_a, &self.texture_b) {
            let (source_texture, target_texture) = if current_frame % 2 == 0 {
                (texture_b, texture_a)
            } else {
                (texture_a, texture_b)
            };
            
            // First render pass - fluid simulation to target texture
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Fluid Pass 1"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_texture.view,
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
                
                // Source texture from feedback loop
                render_pass.set_bind_group(0, &source_texture.bind_group, &[]);
                
                // Input texture - could be image, video, or default
                if let Some(ref input_texture) = self.input_texture {
                    render_pass.set_bind_group(1, &input_texture.bind_group, &[]);
                } else if self.base.using_video_texture {
                    if let Some(video_manager) = &self.base.video_texture_manager {
                        render_pass.set_bind_group(1, &video_manager.texture_manager().bind_group, &[]);
                    } else if let Some(ref default_texture) = self.base.texture_manager {
                        render_pass.set_bind_group(1, &default_texture.bind_group, &[]);
                    }
                } else if let Some(ref default_texture) = self.base.texture_manager {
                    render_pass.set_bind_group(1, &default_texture.bind_group, &[]);
                }
                
                // Time and parameters
                render_pass.set_bind_group(2, &self.base.time_uniform.bind_group, &[]);
                render_pass.set_bind_group(3, &self.params_uniform.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
            
            // Second render pass - visualization to screen
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Fluid Pass 2"),
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
                render_pass.set_pipeline(&self.renderer_pass2.render_pipeline);
                render_pass.set_vertex_buffer(0, self.renderer_pass2.vertex_buffer.slice(..));
                
                render_pass.set_bind_group(0, &target_texture.bind_group, &[]);
                
                if let Some(ref input_texture) = self.input_texture {
                    render_pass.set_bind_group(1, &input_texture.bind_group, &[]);
                } else if self.base.using_video_texture {
                    if let Some(video_manager) = &self.base.video_texture_manager {
                        render_pass.set_bind_group(1, &video_manager.texture_manager().bind_group, &[]);
                    } else if let Some(ref default_texture) = self.base.texture_manager {
                        render_pass.set_bind_group(1, &default_texture.bind_group, &[]);
                    }
                } else if let Some(ref default_texture) = self.base.texture_manager {
                    render_pass.set_bind_group(1, &default_texture.bind_group, &[]);
                }
                render_pass.set_bind_group(2, &self.base.time_uniform.bind_group, &[]);
                render_pass.set_bind_group(3, &self.params_uniform.bind_group, &[]);
                render_pass.draw(0..4, 0..1);
            }
            
            self.frame_count = self.frame_count.wrapping_add(1);
        }
        
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
    cuneus::gst::init()?;
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Fluid", 800, 600);
    let shader = FluidShader::init(app.core());
    app.run(event_loop, shader)
}