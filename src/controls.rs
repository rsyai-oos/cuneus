use std::time::Instant;

#[derive(Clone)]
pub struct ControlsRequest {
    pub is_paused: bool,
    pub should_reset: bool,
}

pub struct ShaderControls {
    is_paused: bool,
    last_time: Option<f32>,
}

impl Default for ShaderControls {
    fn default() -> Self {
        Self {
            is_paused: false,
            last_time: None,
        }
    }
}

impl ShaderControls {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_ui_request(&self) -> ControlsRequest {
        ControlsRequest {
            is_paused: self.is_paused,
            should_reset: false,
        }
    }

    pub fn apply_ui_request(&mut self, request: ControlsRequest) {
        self.is_paused = request.is_paused;
        if request.should_reset {
            self.last_time = None; 
        }
    }

    pub fn get_time(&mut self, start_time: &Instant) -> f32 {
        if self.is_paused {
            self.last_time.unwrap_or(0.0) 
        } else {
            let current_time = start_time.elapsed().as_secs_f32();
            self.last_time = Some(current_time);
            current_time
        }
    }

    pub fn render_controls_widget(ui: &mut egui::Ui, request: &mut ControlsRequest) {
        ui.horizontal(|ui| {
            if ui.button(if request.is_paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                request.is_paused = !request.is_paused;
            }
            if ui.button("↺ Reset").clicked() {
                request.should_reset = true;
            }
        });
    }
}