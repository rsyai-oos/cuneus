use winit::window::Window;
use winit::event::{ElementState, KeyEvent};
use winit::keyboard::Key;

pub struct KeyInputHandler {
    is_fullscreen: bool,
}
impl KeyInputHandler {
    pub fn new() -> Self {
        Self {
            is_fullscreen: false,
        }
    }
    pub fn handle_keyboard_input(&mut self, window: &Window, event: &KeyEvent) -> bool {
        if event.state == ElementState::Pressed && !event.repeat {
            if let Key::Character(ch) = &event.logical_key {
                if ch == "f" || ch == "F" {
                    self.toggle_fullscreen(window);
                    return true;
                }
            }
        }
        false
    }
    fn toggle_fullscreen(&mut self, window: &Window) {
        if !self.is_fullscreen {
            window.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
        } else {
            window.set_fullscreen(None);
        }
        self.is_fullscreen = !self.is_fullscreen;
    }
}