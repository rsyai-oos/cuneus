use std::collections::VecDeque;
use std::time::Instant;

use tracing::info_span;
pub struct FpsTracker {
    last_frame_time: Instant,
    frame_times: VecDeque<f32>,
    current_fps: f32,
}

impl FpsTracker {
    pub fn new() -> Self {
        let span = info_span!("[FpsTracker]");
        let _guard = span.enter();
        log::info!("FpsTracker::new");
        Self {
            last_frame_time: Instant::now(),
            frame_times: VecDeque::with_capacity(60),
            current_fps: 0.0,
        }
    }

    pub fn update(&mut self) {
        let span = info_span!("[FpsTracker]");
        let _guard = span.enter();
        log::info!("FpsTracker::update");
        let now = Instant::now();
        let frame_time = now.duration_since(self.last_frame_time).as_secs_f32();
        self.last_frame_time = now;

        // lets filter out unreasonable frame times to avoid spikes
        if frame_time > 0.0 && frame_time < 1.0 {
            self.frame_times.push_back(frame_time);
            if self.frame_times.len() > 30 {
                self.frame_times.pop_front();
            }

            // shouldn't happen, but who knows anyway...:
            if !self.frame_times.is_empty() {
                let avg_frame_time: f32 =
                    self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32;
                self.current_fps = 1.0 / avg_frame_time;
            }
        }
    }

    pub fn fps(&self) -> f32 {
        self.current_fps
    }
}
