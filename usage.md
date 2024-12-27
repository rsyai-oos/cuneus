# How To Use Cuneus

## Basic Usage

In fact you can simply copy a rust file in the “bin” folder and just go to the wgsl stage. But to set the parameters in egui you only need to change the parameters.

Those are base examples:

- spiral.rs: Simple single-pass shader with texture support.
- feedback.rs: very basic two-pass shader.
- fluid.rs:  multi-pass shader with texture support.
- attractor.rs: simple three-pass rendering
- xmas.rs: single pass, no texture, many parameters. 

Copy the chosen template to a new file in src/bin/
Modify only these key sections:

1- Shader parameters struct (optimal, if you want to use GUI sliders) & Parameter UI controls
2- Compute your WGSL 

thats it. :-)  

Common Patterns
All shaders follow the same basic structure:

Parameter definition is only need for egui:
```rust

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    // Add your parameters here
    param1: f32,
    param2: f32,
}
```
Shader implementation with common components:

- init(): Sets up bind groups and pipelines
- update(): Handles time updates and hot reloading
- render(): Manages render passes and UI
- handle_input(): Processes user input

WGSL Binding Patterns
Basic Vertex Shader (vertex.wgsl)
All shaders use this standard vertex shader:
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
Single Pass Shader
```wgsl
@group(0) @binding(0)
var<uniform> u_time: TimeUniform;

@group(1) @binding(0)
var<uniform> params: Params;

@fragment
fn fs_main(...) -> @location(0) vec4<f32> {
    // Your shader code
}
```

Basic Feedback Shader (feedback.wgsl)
```wgsl
@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

@group(1) @binding(0)
var<uniform> u_time: TimeUniform;

@group(2) @binding(0)
var<uniform> params: Params;

@fragment
fn fs_pass1(...) -> @location(0) vec4<f32> {
    let previous = textureSample(prev_frame, tex_sampler, tex_coords);
    // Blend with new frame
}

@fragment
fn fs_pass2(...) -> @location(0) vec4<f32> {
    // Post-processing or display
}
```
