use std::time::Instant;

#[derive(Clone)]
pub struct ControlsRequest {
    pub is_paused: bool,
    pub should_reset: bool,
    pub should_clear_buffers: bool,  
    pub current_time: Option<f32>, 
    pub window_size: Option<(u32, u32)>, 
}

pub struct ShaderControls {
    is_paused: bool,
    pause_start: Option<Instant>,
    total_pause_duration: f32,
    current_frame: u32,
    
}

impl Default for ShaderControls {
    fn default() -> Self {
        Self {
            is_paused: false,
            pause_start: None,
            total_pause_duration: 0.0,
            current_frame: 0,
        }
    }
}
impl Default for ControlsRequest {
    fn default() -> Self {
        Self {
            is_paused: false,
            should_reset: false,
            should_clear_buffers: false,
            current_time: None,
            window_size: None,
        }
    }
}
impl ShaderControls {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn get_frame(&mut self) -> u32 {
        if !self.is_paused {
            self.current_frame = self.current_frame.wrapping_add(1);
        }
        self.current_frame
    }

        pub fn get_time(&self, start_time: &Instant) -> f32 {
            let raw_time = start_time.elapsed().as_secs_f32();
            if self.is_paused {
                if let Some(pause_start) = self.pause_start {
                    raw_time - self.total_pause_duration - pause_start.elapsed().as_secs_f32()
                } else {
                    raw_time - self.total_pause_duration
                }
            } else {
                raw_time - self.total_pause_duration
            }
        }
        pub fn get_ui_request(&self, start_time: &Instant, size: &winit::dpi::PhysicalSize<u32>) -> ControlsRequest {
            ControlsRequest {
                is_paused: self.is_paused,
                should_reset: false,
                should_clear_buffers: false,
                current_time: Some(self.get_time(start_time)),
                window_size: Some((size.width, size.height)),
            }
        }
    pub fn apply_ui_request(&mut self, request: ControlsRequest) {
        if request.should_reset {
            self.is_paused = false;
            self.pause_start = None;
            self.total_pause_duration = 0.0;
            self.current_frame = 0;
        } else if request.is_paused && !self.is_paused {
            self.pause_start = Some(Instant::now());
        } else if !request.is_paused && self.is_paused {
            if let Some(pause_start) = self.pause_start {
                self.total_pause_duration += pause_start.elapsed().as_secs_f32();
            }
            self.pause_start = None;
        }
        self.is_paused = request.is_paused;
    }

    pub fn render_controls_widget(ui: &mut egui::Ui, request: &mut ControlsRequest) {
        ui.vertical(|ui| { 
            ui.horizontal(|ui| {
                if ui.button(if request.is_paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                    request.is_paused = !request.is_paused;
                }
                if ui.button("↺ Reset").clicked() {
                    request.should_reset = true;
                    request.should_clear_buffers = true;
                }
                if let Some(time) = request.current_time { 
                    ui.label(format!("Time: {:.2}s", time));
                }
            });
            if let Some((width, height)) = request.window_size {
                ui.horizontal(|ui| {
                    ui.label(format!("Resolution: {}x{}", width, height));
                });
            }
        });
    }
}