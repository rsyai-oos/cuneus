use std::path::PathBuf;
use std::sync::mpsc;
use image::ImageError;

#[derive(Debug)]
pub enum ExportError {
    IoError(std::io::Error),
    ImageError(ImageError),
}

impl From<std::io::Error> for ExportError {
    fn from(err: std::io::Error) -> Self {
        ExportError::IoError(err)
    }
}

impl From<ImageError> for ExportError {
    fn from(err: ImageError) -> Self {
        ExportError::ImageError(err)
    }
}
#[derive(Debug, Clone)]
pub struct ExportSettings {
    pub export_path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub start_time: f32,
    pub end_time: f32,
    pub fps: u32,
    pub is_exporting: bool,
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            export_path: PathBuf::from("./export"),
            width: 1920,
            height: 1080,
            start_time: 0.0,
            end_time: 5.0,
            fps: 60,
            is_exporting: false,
        }
    }
}

#[derive(Default)]
pub struct ExportUiState {
    pub show_window: bool,
    pub temp_width: u32,
    pub temp_height: u32,
    pub temp_start_time: f32,
    pub temp_end_time: f32,
    pub temp_fps: u32,
}

/// Manages the export process and UI state
pub struct ExportManager {
    settings: ExportSettings,
    export_channel: Option<mpsc::Receiver<(u32, f32)>>,
    ui_state: ExportUiState,
}

impl ExportManager {
    pub fn new() -> Self {
        let settings = ExportSettings::default();
        let ui_state = ExportUiState {
            temp_width: settings.width,
            temp_height: settings.height,
            temp_start_time: settings.start_time,
            temp_end_time: settings.end_time,
            temp_fps: settings.fps,
            show_window: false,
        };
        
        Self {
            settings,
            export_channel: None,
            ui_state,
        }
    }

    /// Returns a reference to the current export settings
    pub fn settings(&self) -> &ExportSettings {
        &self.settings
    }

    /// Returns whether an export is currently in progress
    pub fn is_exporting(&self) -> bool {
        self.settings.is_exporting
    }
    pub fn settings_mut(&mut self) -> &mut ExportSettings {
        &mut self.settings
    }
    /// Attempts to get the next frame for export
    pub fn try_get_next_frame(&mut self) -> Option<(u32, f32)> {
        self.export_channel.as_ref()?.try_recv().ok()
    }

    /// Starts the export process
    pub fn start_export(&mut self) {
        if self.settings.is_exporting {
            return;
        }

        self.settings.is_exporting = true;
        let settings = self.settings.clone();
        let (tx, rx) = mpsc::channel();

        std::thread::spawn(move || {
            let total_frames = ((settings.end_time - settings.start_time) * settings.fps as f32) as u32;
            
            for frame in 0..total_frames {
                let time = settings.start_time + (frame as f32 / settings.fps as f32);
                if tx.send((frame, time)).is_err() {
                    break; // Exit if the receiver is dropped
                }
            }
        });
        
        self.export_channel = Some(rx);
    }

    /// Completes the export process
    pub fn complete_export(&mut self) {
        self.settings.is_exporting = false;
        self.export_channel = None;
    }

    /// Returns references to both UI state and settings for the UI to use
    pub fn get_ui_elements(&mut self) -> (&mut ExportUiState, &mut ExportSettings) {
        (&mut self.ui_state, &mut self.settings)
    }

    /// Draws the export UI using egui
    pub fn draw_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut start_export = false;
        
        if !self.settings.is_exporting {
            ui.horizontal(|ui| {
                ui.label("Export Path:");
                if ui.button("Browse").clicked() {
                    if let Some(path) = rfd::FileDialog::new()
                        .set_directory(&self.settings.export_path)
                        .pick_folder() {
                        self.settings.export_path = path;
                    }
                }
            });
            
            ui.add(egui::DragValue::new(&mut self.ui_state.temp_width)
                .prefix("Width: ")
                .clamp_range(1..=7680));
                
            ui.add(egui::DragValue::new(&mut self.ui_state.temp_height)
                .prefix("Height: ")
                .clamp_range(1..=4320));
                
            ui.add(egui::DragValue::new(&mut self.ui_state.temp_start_time)
                .prefix("Start Time: ")
                .speed(0.1));
                
            ui.add(egui::DragValue::new(&mut self.ui_state.temp_end_time)
                .prefix("End Time: ")
                .speed(0.1));
                
            ui.add(egui::DragValue::new(&mut self.ui_state.temp_fps)
                .prefix("FPS: ")
                .clamp_range(1..=240));

            if ui.button("Start Export").clicked() {
                // Apply temporary values to actual settings
                self.settings.width = self.ui_state.temp_width;
                self.settings.height = self.ui_state.temp_height;
                self.settings.start_time = self.ui_state.temp_start_time;
                self.settings.end_time = self.ui_state.temp_end_time;
                self.settings.fps = self.ui_state.temp_fps;
                start_export = true;
            }
        } else {
            ui.label("Exporting...");
            // if ui.button("Cancel Export").clicked() {
            //     self.complete_export();
            // }
        }
        
        start_export
    }
    /// Updates the UI state from settings
    pub fn sync_ui_state(&mut self) {
        self.ui_state.temp_width = self.settings.width;
        self.ui_state.temp_height = self.settings.height;
        self.ui_state.temp_start_time = self.settings.start_time;
        self.ui_state.temp_end_time = self.settings.end_time;
        self.ui_state.temp_fps = self.settings.fps;
    }
}

