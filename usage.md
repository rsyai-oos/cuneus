# How To Use Cuneus

## Basic Usage
In fact you can simply copy a rust file in the “bin” folder and just go to the wgsl stage. But to set the parameters in egui you only need to change the parameters.

1. Copy the template below and change only the `ShaderParams` struct and WGSL shader code:

```rust
// src/bin/your_shader.rs
use cuneus::{Core, ShaderManager, UniformProvider, UniformBinding, BaseShader};
use winit::event::*;

// 1. Define your parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    // Add your parameters here
    param1: f32,
    param2: f32,
}

impl UniformProvider for ShaderParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct MyShader {
    base: BaseShader,
    params_uniform: UniformBinding<ShaderParams>,
}

// 2. Everything else remains the same, copy from spiral.rs or other shaders. You only need to change egui thingies based on your taste.
impl ShaderManager for MyShader {
    fn init(core: &Core) -> Self {
        // Copy from spiral.rs, just update initial param values
    }
    fn update(&mut self, core: &Core) {
        // Copy from spiral.rs
    }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        // Copy from spiral.rs, update only the UI sliders
    }
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool {
        // Copy from spiral.rs
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let (app, event_loop) = ShaderApp::new("Your Shader", 800, 600);
    let shader = MyShader::init(app.core());
    app.run(event_loop, shader)
}
```
2- Create your WGSL shader with these bind groups: (for two passes or textures please look at the examples)
```wgsl
// shaders/your_shader.wgsl
struct TimeUniform {
    time: f32,
};
@group(0) @binding(0)
var<uniform> u_time: TimeUniform;

struct Params {
    // Match your ShaderParams struct
    param1: f32,
    param2: f32,
};
@group(1) @binding(0)
var<uniform> params: Params;

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let res = vec2<f32>(1920.0, 1080.0);
    // Your pretty code
}
```
