
# How To Use Cuneus

In fact you can simply copy a rust file in the “bin” folder and just go to the wgsl stage. But to set the parameters in egui you only need to change the parameters.

Note: Please don't be surprised to see that some Uniforms are common both in “RenderKit.rs” (an important backend file in cuneus where most things are init) and in the final rust file we created for our shader. The only purpose of this is related to hot reload. :-)

## Quick Start

1. Copy one of the template files from `src/bin/` that best matches your needs:
    - compute.rs is simply debug screen, you can start with here.


2. Rename and modify the copied file to create your shader
3. Focus on writing your WGSL shader code :-)

## GStreamer Requirement for Video/Camera Textures

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

This template includes:

1. Core imports and setup
2. Minimal shader parameters
3. Basic render pipeline
4. WGSL shader code


```rust
// 1. Required imports
use cuneus::{Core, ShaderManager, RenderKit /* ... */};

// 2. Optional parameters if needed
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ShaderParams {
    // Your parameters here
}

// 3. Main shader structure
struct MyShader {
    base: RenderKit,
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
    // Add more parameters as needed, note please be careful about the GPU memory aligment! (I usally use paddings)
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

## Using Compute Shaders

Cuneus provides robust support for GPU compute shaders, enabling you to harness parallel processing power for various tasks such as image processing, simulations, and data manipulation.

### Quick Start

There are two main ways to use compute shaders in Cuneus:

1. **Simple integration with RenderKit** - Best for basic compute shaders please see: [compute.rs](https://github.com/altunenes/cuneus/blob/main/src/ bin/compute.rs)
2. **Custom implementation** - For more complex scenarios like ping-pong buffers or multi-pass compute shaders, see: [computecolors.rs](https://github.com/altunenes/cuneus/blob/main/src/bin/computecolors.rs)

#### Option 1: Using RenderKit (Simpler)

```rust
// In your ShaderManager implementation
fn init(core: &Core) -> Self {
    let mut base = RenderKit::new(
        core,
        include_str!("../../shaders/vertex.wgsl"),
        include_str!("../../shaders/blit.wgsl"),
        &[&texture_bind_group_layout],
        None,
    );

    // Create a compute shader with configuration
    let compute_config = cuneus::compute::ComputeShaderConfig {
        workgroup_size: [16, 16, 1],
        workgroup_count: None,  // Auto-determine from texture size
        dispatch_once: false,   // Run every frame
        storage_texture_format: cuneus::compute::COMPUTE_TEXTURE_FORMAT_RGBA16,
        enable_atomic_buffer: false,
        entry_points: vec!["main".to_string()],
        sampler_address_mode: wgpu::AddressMode::ClampToEdge,
        sampler_filter_mode: wgpu::FilterMode::Linear,
        label: "My Compute Shader".to_string(),
    };
    
    base.compute_shader = Some(cuneus::compute::ComputeShader::new_with_config(
        core,
        include_str!("../../shaders/my_compute.wgsl"),
        compute_config,
    ));
    
    Self { base }
}

// In your render function
fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
    // ...
    let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Render Encoder"),
    });
    
    // Run the compute shader
    self.base.dispatch_compute_shader(&mut encoder, core);
    
    // Render the compute shader output
    if let Some(compute_texture) = self.base.get_compute_output_texture() {
        render_pass.set_bind_group(0, &compute_texture.bind_group, &[]);
        // ...
    }
    // ...
}
```

#### WGSL Compute Shader Structure

```wgsl
// Time uniform
struct TimeUniform { 
    time: f32, 
    delta: f32,
    frame: u32,
    _padding: u32 
}
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

// Optional parameters
struct Params {
    intensity: f32,
    scale: f32,
    // ...
}
@group(1) @binding(0) var<uniform> params: Params;

// Input/output textures
@group(2) @binding(0) var input_texture: texture_2d<f32>;
@group(2) @binding(1) var tex_sampler: sampler;
@group(2) @binding(2) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Get dimensions
    let dims = textureDimensions(output);
    
    // Bounds check
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    
    // Process and write result
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(1.0, 0.0, 0.0, 1.0));
}
```

### Advanced: Multi-Pass Compute Shaders
For more complex effects, use multiple entry points:

```rust
// Configure with multiple entry points
let config = ComputeShaderConfig {
    entry_points: vec![
        "preprocess".to_string(),
        "process".to_string(),
        "postprocess".to_string()
    ],
    enable_atomic_buffer: true,
    // ...
};
```

With corresponding WGSL:
```wgsl
@compute @workgroup_size(16, 16, 1)
fn preprocess(@builtin(global_invocation_id) id: vec3<u32>) {
    // First pass
}

@compute @workgroup_size(16, 16, 1)
fn process(@builtin(global_invocation_id) id: vec3<u32>) {
    // Second pass
}

@compute @workgroup_size(16, 16, 1)
fn postprocess(@builtin(global_invocation_id) id: vec3<u32>) {
    // Third pass
}
```

### Ping-Pong Buffers
For effects that need to read their previous output:

```rust
// Create two textures and swap between them
let source_bind_group = if self.source_is_a {
    &self.texture_pair.bind_group_a
} else {
    &self.texture_pair.bind_group_b
};

// After rendering, swap buffers
self.source_is_a = !self.source_is_a;
```


Utility Functions

```rust
// Create two textures and swap between them
let source_bind_group = if self.source_is_a {
    &self.texture_pair.bind_group_a
} else {
    &self.texture_pair.bind_group_b
};

// After rendering, swap buffers
self.source_is_a = !self.source_is_a;
```

For complete examples, see `src/bin/clifford_compute.rs` and `src/bin/computecolors.rs`.


### How to use Audio Data 

 On the Rust side, unlike other backends, make sure you only add this single line. 

https://github.com/altunenes/cuneus/blob/776434e2f5ac8797dd5a07ffe86659745a8e88a6/src/bin/audiovis.rs#L370 

I passed the data about spectrum and BPM to the resolution uniform. You can use them from here:

https://github.com/altunenes/cuneus/blob/776434e2f5ac8797dd5a07ffe86659745a8e88a6/shaders/audiovis.wgsl#L10-L15

then use it as you desired. 

note that, spectrum data is not raw. I process it on the rust side. If this is not suitable for you, you can fix it. Audio is not my specialty. If you have a better idea please open a PR.
https://github.com/altunenes/cuneus/blob/main/src/spectrum.rs#L47

## Font Rendering

Cuneus provides built-in font rendering for text overlays, scoring systems, and creative text effects in shaders.

### Basic Font Usage

Enable fonts in your compute shader configuration:

```rust
let compute_config = ComputeShaderConfig {
    enable_fonts: true,  // Enable font system
    // ... other config
};
```

### WGSL Font Functions

Use these functions in your compute shaders:

```wgsl
// Render single character at any size
let alpha = render_char_sized(pixel_pos, position, 'A', 64.0);

// Render numbers (integers)
let alpha = render_number(pixel_pos, position, 123u, 32.0);

// Render text strings (use word arrays)
let word: array<u32, 8> = array<u32, 8>(72u, 101u, 108u, 108u, 111u, 0u, 0u, 0u); // "Hello"
let alpha = render_word(pixel_pos, position, word, 5u, 48.0);

// Render floating point numbers
let alpha = render_float(pixel_pos, position, 3.14, 24.0);
```

### Font System Details

- **Font:** Courier Prime Bold
- **Atlas:** 1024×1024 texture with 64×64 character cells
- **Characters:** Full ASCII printable set (32-126)
- **Performance:** GPU-optimized, real-time rendering
- **Quality:** High-resolution with gamma correction

The font system is fully scalable and can be positioned anywhere in your shader effects.

### Adding Interactive Controls
1. Start with a template that includes GUI (e.g., `xmas.rs`)
2. Define your parameters in the ShaderParams struct
3. Add UI controls in the render function
4. Connect parameters to your shader

## Real-time Audio Synthesis

Cuneus supports generating audio directly from GPU shaders. Your compute shader can calculate frequencies and write audio data that gets played in real-time.

### Basic Setup
```wgsl
// Write audio data from GPU to CPU
if global_id.x == 0u && global_id.y == 0u {
    audio_buffer[0] = frequency;    // Hz
    audio_buffer[1] = amplitude;    // 0.0-1.0  
    audio_buffer[2] = waveform;     // 0=sine, 1=square, etc
}
```

For complete implementation details, see `src/bin/synth.rs`
