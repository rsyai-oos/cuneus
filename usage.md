
# How To Use Cuneus

In fact you can simply copy a rust file in the “bin” folder and just go to the wgsl stage. But to set the parameters in egui you only need to change the parameters.

Note: Please don't be surprised to see that some Uniforms are common both in “baseshader.rs” (an important backend file in cuneus where most things are init) and in the final rust file we created for our shader (like mandelbrot.rs). The only purpose of this is related to hot reload. :-)

## Quick Start

1. Copy one of the template files from `src/bin/` that best matches your needs:
   - `mandelbrot.rs`: Minimal single-pass shader without GUI controls
   - `spiral.rs`: Simple single-pass shader with texture support
   - `feedback.rs`: Basic two-pass shader
   - `fluid.rs`: Multi-pass shader with texture support
   - `attractor.rs`: Three-pass rendering example
   - `xmas.rs`: Single pass with extensive parameter controls
   - `audiovis.rs` Audio visualizer example to show how you can use spectrum/bpm data from rust.
  
if you want 4 passes or more the logic is exactly the same. 

2. Rename and modify the copied file to create your shader
3. Focus on writing your WGSL shader code :-)

## GStreamer Requirement for Video Textures

Cuneus supports using Videos as textures, which requires GStreamer bins to be installed on your system.

### Installation

Install from the [GStreamer website](https://gstreamer.freedesktop.org/download/#macos), both the runtime and development libraries.

[build.rs](https://github.com/altunenes/cuneus/blob/main/build.rs) contains configuration for macOS frameworks and library paths. Adjust as needed for your system for development :-) 

> **Note**: GStreamer is only required if you need video texture functionality. Image-based textures should work without it.

> **Note2** Development libraries are required only for building the project.

Please see how I build the project in github actions, you can use it as a reference:

[release](https://github.com/altunenes/cuneus/blob/main/.github/workflows/release.yaml)

## Template Structure

### Basic Single Pass Shader (No GUI)

The simplest way to start is with a basic shader like `mandelbrot.rs`. This template includes:

1. Core imports and setup
2. Minimal shader parameters
3. Basic render pipeline
4. WGSL shader code

I created this by completely copying and pasting xmas.rs, and I could only focus on my shader.

```rust
// 1. Required imports
use cuneus::{Core, ShaderManager, BaseShader /* ... */};

// 2. Optional parameters if needed
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    // Your parameters here
}

// 3. Main shader structure
struct MyShader {
    base: BaseShader,
    // Add any additional fields needed
}

// 4. Implement required traits
impl ShaderManager for MyShader {
    fn init(core: &Core) -> Self { /* ... */ }
    fn update(&mut self, core: &Core) { /* ... */ }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> { /* ... */ }
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool { /* ... */ }
}
```

### Adding GUI Controls (optimal)

To add parameter controls through egui:

1. Define your parameters struct
2. Add UI controls in the render function

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    rotation_speed: f32,
    intensity: f32,
    // Add more parameters as needed
}

// In render function:
let full_output = if self.base.key_handler.show_ui {
    self.base.render_ui(core, |ctx| {
        egui::Window::new("Settings").show(ctx, |ui| {
            changed |= ui.add(egui::Slider::new(&mut params.rotation_speed, 0.0..=5.0)
                .text("Rotation Speed")).changed();
            // Add more controls
        });
    })
};
```

## WGSL Shader Patterns

### Standard Vertex Shader
All shaders use this common vertex shader (vertex.wgsl): If you are not doing anything special, this file can always remain fixed. If you look at my shader examples, they all use this same file.

```wgsl
struct VertexOutput {
    @location(0) tex_coords: vec2<f32>,
    @builtin(position) out_pos: vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VertexOutput {
    let tex_coords = vec2<f32>(pos.x * 0.5 + 0.5, 1.0 - (pos.y * 0.5 + 0.5));
    return VertexOutput(tex_coords, vec4<f32>(pos, 0.0, 1.0));
}
```

### Single Pass Fragment Shader
Basic structure for a fragment shader:
```wgsl
// Time uniform
@group(0) @binding(0)
var<uniform> u_time: TimeUniform;

// Optional EGUI parameters
@group(1) @binding(0)
var<uniform> params: Params;

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, 
           @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    // Your shader code here
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
```

### Multi-Pass Shader
For effects requiring multiple passes:
```wgsl
@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@fragment
fn fs_pass1(...) -> @location(0) vec4<f32> {
    // First pass processing
}

@fragment
fn fs_pass2(...) -> @location(0) vec4<f32> {
    // Second pass processing
}
```


### Hot Reloading
cuneus supports hot reloading of shaders. Simply modify your WGSL files and they will automatically reload.

### Export Support
Built-in support for exporting frames as images. Access through the UI when enabled. "Start time" is not working correctly currently.

### Texture Support
Load and use textures in your shaders:
```rust
if let Some(ref texture_manager) = self.base.texture_manager {
    render_pass.set_bind_group(0, &texture_manager.bind_group, &[]);
}
```

## Resolution Handling

cuneus handles both logical and physical resolution:

1. Initial window size is set in logical pixels:

```rust
   let (app, event_loop) = ShaderApp::new("My Shader", 800, 600);
 ```
2.  On high-DPI displays (like Retina), the physical resolution is automatically scaled:
    e.g., 800x600 logical becomes 1600x1200 physical on a 2x scaling display
    Your shader's UV coordinates (0.0 to 1.0) automatically adapt to any resolution
    Export resolution can be set independently through the UI

Your WGSL shaders can access actual dimensions when needed:
```wgsl
let dimensions = vec2<f32>(textureDimensions(my_texture));
```

### How to use Audio Data 

 On the Rust side, unlike other backends, make sure you only add this single line. 

https://github.com/altunenes/cuneus/blob/776434e2f5ac8797dd5a07ffe86659745a8e88a6/src/bin/audiovis.rs#L370 

I passed the data about spectrum and BPM to the resolution uniform. You can use them from here:

https://github.com/altunenes/cuneus/blob/776434e2f5ac8797dd5a07ffe86659745a8e88a6/shaders/audiovis.wgsl#L10-L15

then use it as you desired. 

note that, spectrum data is not raw. I process it on the rust side. If this is not suitable for you, you can fix it. Audio is not my specialty. If you have a better idea please open a PR.
https://github.com/altunenes/cuneus/blob/main/src/spectrum.rs#L47

### Adding Interactive Controls
1. Start with a template that includes GUI (e.g., `xmas.rs`)
2. Define your parameters in the ShaderParams struct
3. Add UI controls in the render function
4. Connect parameters to your shader
