// Block Game, Enes Altun, 2025, MIT License

use cuneus::{Core, ShaderApp, ShaderManager, RenderKit, UniformProvider};
use cuneus::compute::{ComputeShaderConfig, COMPUTE_TEXTURE_FORMAT_RGBA16};
use winit::event::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BlockGameParams {
    // 0=menu, 1=playing, 2=game_over
    game_state: i32,
    score: u32,
    current_block: u32,
    total_blocks: u32,
    
    block_x: f32,
    block_y: f32,
    block_z: f32,
    
    block_width: f32,
    block_height: f32,
    block_depth: f32,
    
    movement_speed: f32,
    movement_range: f32,
    drop_triggered: i32,
    

    camera_height: f32,
    camera_angle: f32,
    
    // Game mech
    perfect_placement: i32,
    game_over: i32,
    
    _padding: [f32; 3],
}

impl Default for BlockGameParams {
    fn default() -> Self {
        Self {
            game_state: 0,
            score: 0,
            current_block: 0,
            total_blocks: 1,
            
            block_x: 0.0,
            block_y: 1.0,
            block_z: 0.0,
            
            block_width: 3.0,
            block_height: 0.6,
            block_depth: 3.0,
            
            movement_speed: 2.0,
            movement_range: 2.5,
            drop_triggered: 0,
            
            camera_height: 0.0,
            camera_angle: 0.0,
            
            perfect_placement: 0,
            game_over: 0,
            
            _padding: [0.0; 3],
        }
    }
}

impl UniformProvider for BlockGameParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct BlockTowerGame {
    base: RenderKit,
    last_mouse_click: bool,
    game_params: BlockGameParams,
}

impl ShaderManager for BlockTowerGame {
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
        
        // Create mouse uniform for game interactions
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
            workgroup_size: [8, 8, 1],
            storage_texture_format: COMPUTE_TEXTURE_FORMAT_RGBA16,
            enable_fonts: true,
            // Used for game state storage, not audio (sorry for being lazy)
            enable_audio_buffer: true,
            // Storage for game state and blocks
            audio_buffer_size: 1024,
            mouse_bind_group_layout: Some(mouse_bind_group_layout),
            entry_points: vec!["main".to_string()],
            label: "Block Tower Game".to_string(),
            ..Default::default()
        };
        
        base.compute_shader = Some(cuneus::compute::ComputeShader::new_with_config(
            core,
            include_str!("../../shaders/blockgame.wgsl"),
            compute_config,
        ));
        
        if let (Some(compute_shader), Some(mouse_uniform)) = (&mut base.compute_shader, &base.mouse_uniform) {
            compute_shader.add_mouse_uniform_binding(
                &mouse_uniform.bind_group,
                2
            );
        }
        
        Self {
            base,
            last_mouse_click: false,
            game_params: BlockGameParams::default(),
        }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0 / 60.0;
        self.base.update_compute_shader_time(current_time, delta, &core.queue);
        self.base.update_mouse_uniform(&core.queue);
        self.base.fps_tracker.update();
        self.update_camera_in_shader(&core.queue);
        let mouse_buttons = self.base.mouse_tracker.uniform.buttons[0];
        let mouse_pressed = mouse_buttons & 1 != 0;
        self.last_mouse_click = mouse_pressed;
    }

    fn resize(&mut self, core: &Core) {
        self.base.update_resolution(&core.queue, core.size);
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
        
        let full_output = if self.base.key_handler.show_ui {
            self.base.render_ui(core, |ctx| {
                ctx.style_mut(|style| {
                    style.visuals.window_fill = egui::Color32::from_rgba_premultiplied(0, 0, 0, 180);
                    style.text_styles.get_mut(&egui::TextStyle::Body).unwrap().size = 11.0;
                    style.text_styles.get_mut(&egui::TextStyle::Button).unwrap().size = 10.0;
                });
                egui::Window::new("Block Tower")
                    .collapsible(true)
                    .resizable(true)
                    .default_width(220.0)
                    .show(ctx, |ui| {
                        egui::CollapsingHeader::new("Camera")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.add(egui::Slider::new(&mut self.game_params.camera_height, 0.0..=20.0).text("Height"));
                                ui.add(egui::Slider::new(&mut self.game_params.camera_angle, -3.14159..=3.14159).text("Angle"));
                                
                                ui.separator();
                                ui.label("Controls:");
                                ui.label("Q/E: Move up/down");
                                ui.label("W/S: Rotate left/right");
                                
                                if ui.button("Reset Camera").clicked() {
                                    self.game_params.camera_height = 8.0;
                                    self.game_params.camera_angle = 0.0;
                                }
                            });
                    });
            })
        } else {
            self.base.render_ui(core, |_ctx| {})
        };
        
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Block Game Render Encoder"),
        });
        
        self.base.dispatch_compute_shader(&mut encoder, core);
        
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Block Game Render Pass"),
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

    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        let ui_handled = self.base.egui_state.on_window_event(core.window(), event).consumed;
        
        if self.base.handle_mouse_input(core, event, ui_handled) {
            return true;
        }
        
        if let WindowEvent::KeyboardInput { event, .. } = event {
            if let winit::keyboard::PhysicalKey::Code(key_code) = event.physical_key {
                if event.state == ElementState::Pressed {
                    let camera_speed = 0.5;
                    
                    match key_code {
                        winit::keyboard::KeyCode::KeyQ => {
                            self.game_params.camera_height += camera_speed;
                            return true;
                        }
                        winit::keyboard::KeyCode::KeyE => {
                            self.game_params.camera_height -= camera_speed;
                            return true;
                        }
                        winit::keyboard::KeyCode::KeyW => {
                            self.game_params.camera_angle += 0.1;
                            return true;
                        }
                        winit::keyboard::KeyCode::KeyS => {
                            self.game_params.camera_angle -= 0.1;
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

impl BlockTowerGame {
    fn update_camera_in_shader(&self, queue: &wgpu::Queue) {
        if let Some(compute_shader) = &self.base.compute_shader {
            if let Some(audio_buffer) = &compute_shader.audio_buffer {
                let camera_data = [
                    self.game_params.camera_height,
                    self.game_params.camera_angle,
                ];
                
                let camera_data_bytes = bytemuck::cast_slice(&camera_data);
                let offset = 5 * std::mem::size_of::<f32>();
                
                queue.write_buffer(audio_buffer, offset as u64, camera_data_bytes);
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    cuneus::gst::init()?;
    
    let (app, event_loop) = ShaderApp::new("Block Tower Game", 600, 800);
    
    app.run(event_loop, |core| {
        BlockTowerGame::init(core)
    })
}