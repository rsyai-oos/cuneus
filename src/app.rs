use crate::{Core, ShaderManager};
use winit::{
    event::*,
    event_loop::EventLoop,
    window::WindowAttributes,
    dpi::LogicalSize,
};
pub struct ShaderApp {
    core: Core,
}
impl ShaderApp {
    pub fn new(window_title: &str, width: u32, height: u32) -> (Self, EventLoop<()>) {
        let event_loop = EventLoop::builder()
            .build()
            .expect("Failed to create event loop");
        let mut window_attributes = WindowAttributes::default();
        window_attributes.inner_size = Some(LogicalSize::new(width, height).into());
        window_attributes.title = String::from(window_title);
        window_attributes.resizable = true;
        let window = event_loop
            .create_window(window_attributes)
            .expect("Failed to create window");
        window.set_window_level(winit::window::WindowLevel::AlwaysOnTop);
        let core = pollster::block_on(Core::new(window));
        (Self { core }, event_loop)
    }
    pub fn run<S: ShaderManager + 'static>(
        mut self,
        event_loop: EventLoop<()>,
        mut shader: S,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut first_render = true;
        event_loop.run(move |event, window_target| {
            match event {
                Event::WindowEvent {
                    window_id,
                    ref event,
                } if window_id == self.core.window().id() => {
                    if !shader.handle_input(&self.core, event) {
                        match event {
                            WindowEvent::CloseRequested => {
                                window_target.exit();
                            }
                            WindowEvent::Resized(size) => {
                                self.core.resize(*size);
                                shader.resize(&self.core);
                            }
                            WindowEvent::RedrawRequested => {
                                shader.update(&self.core);
                                match shader.render(&self.core) {
                                    Ok(_) => {
                                        if first_render {
                                            first_render = false;
                                        }
                                    }
                                    Err(wgpu::SurfaceError::Lost) => self.core.resize(self.core.size),
                                    Err(wgpu::SurfaceError::OutOfMemory) => window_target.exit(),
                                    Err(e) => eprintln!("Render error: {:?}", e),
                                }
                            }
                            _ => {}
                        }
                    }
                }
                Event::AboutToWait => {
                    self.core.window().request_redraw();
                }
                _ => {}
            }
        })?;

        Ok(())
    }

    pub fn core(&self) -> &Core {
        &self.core
    }
}