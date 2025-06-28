# Cuneus Usage Guide

Cuneus is a Rust-based GPU shader engine that simplifies creating interactive visual effects. It supports both traditional fragment shaders and modern compute shaders with built-in UI controls, hot-reloading, and export capabilities.

## Quick Start

1. **Copy a template** from `src/bin/` that matches your needs:
   - `debugscreen.rs` - Simple compute shader with text rendering
   - `sinh.rs` - Fragment shader with interactive parameters  
   - `audiovis.rs` - Media processing with audio spectrum analysis
   - `fft.rs`, `spiral.rs` - Media kit usage (input as video/texture/webcam/hdri)

2.  **Customize the template** for your effect.
3.  **Write your WGSL shader** and iterate with hot-reloading.

## Core Structure

Every shader follows this pattern, using the `RenderKit` helper for common tasks.

```rust
use cuneus::prelude::*; // Use the prelude for easy access to core types

struct MyShader {
    base: RenderKit,
    // Your custom uniforms and state
    params_uniform: UniformBinding<ShaderParams>,
}

impl ShaderManager for MyShader {
    fn init(core: &Core) -> Self { /* ... Setup RenderKit, uniforms, etc. ... */ }
    fn update(&mut self, core: &Core) { /* ... Handle hot-reloading, FPS, etc. ... */ }
    fn render(&mut self, core: &Core) -> Result<(), SurfaceError> { /* ... Render logic ... */ }
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool { /* ... Input logic ... */ }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (app, event_loop) = ShaderApp::new("My Shader", 800, 600);
    app.run(event_loop, MyShader::init)
}
```

## Fragment Shaders

For traditional frag shaders:

**Rust setup:**

```rust
let base = RenderKit::new(
    core,
    include_str!("../../shaders/vertex.wgsl"),    // Standard fullscreen quad
    include_str!("../../shaders/my_effect.wgsl"), // Your fragment shader
    &[&time_layout, &params_layout],               // Bind group layouts
    None,
);
```

**WGSL structure:**
```wgsl
// Bind groups match the order provided in Rust
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: ShaderParams;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy / u_resolution.dimensions;
    // Your effect here
    return vec4<f32>(uv, 0.0, 1.0);
}
```

## Compute Shaders


**Rust (`debugscreen.rs`):**
```rust
// In init(), create a compute shader config
let compute_config = ComputeShaderConfig {
    workgroup_size: [16, 16, 1],
    enable_fonts: true,  // Optionally enable the font system
    ..Default::default()
};

// Add the compute shader to your RenderKit instance
base.compute_shader = Some(cuneus::compute::ComputeShader::new_with_config(
    core,
    include_str!("../../shaders/my_compute.wgsl"),
    compute_config,
));
```

**WGSL (`debugscreen.wgsl`):**
```wgsl
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    // Your compute logic here
    textureStore(output, id.xy, vec4<f32>(1.0, 0.0, 0.0, 1.0));
}
```
*For complex effects like particle systems, you can also manage compute pipelines manually instead of using `base.compute_shader`. See `cliffordcompute.rs` for an example.*

## Media Support

Cuneus provides comprehensive media integration through GStreamer.

**Loading & Updating:**
```rust
// Load any supported media file
self.base.load_media(core, path)?;

// For video/webcam, update the texture each frame in update()
if self.base.using_video_texture {
    self.base.update_video_texture(core, &core.queue);
}
```

**UI Controls:**
```rust
// In your render() UI block
ShaderControls::render_media_panel(
    ui,
    &mut controls_request,
    self.base.using_video_texture,
    self.base.get_video_info(),
    self.base.using_hdri_texture,
    self.base.get_hdri_info(),
    self.base.using_webcam_texture,
    self.base.get_webcam_info(),
);
```

**Supported Formats:**
- **Images:** PNG, JPG, JPEG, BMP, TIFF, WebP
- **Videos:** MP4, AVI, MKV, WebM, MOV (with audio)
- **HDRI:** HDR, EXR (with exposure/gamma controls)
- **Webcam:** Live camera feed

## Advanced Patterns

### Multi-Pass Rendering (Ping-Pong)
For effects that need previous frame data (e.g., feedback loops).
```rust
// Rust: In init(), create texture pairs
let texture_pair = create_feedback_texture_pair(core, ...);
// Rust: In render(), swap textures each frame
let (source_tex, target_tex) = if frame % 2 == 0 {
    (&pair.0, &pair.1)
} else {
    (&pair.1, &pair.0)
};

// WGSL: A pass takes the previous frame's result as input
@group(0) @binding(0) var prev_frame: texture_2d<f32>;
```

### Compute Shaders with Atomic Buffers
For GPU accumulation and complex algorithms.
```rust
// Rust: Enable in config and create buffer manually
let config = ComputeShaderConfig { enable_atomic_buffer: true, .. };
let atomic_buffer = cuneus::AtomicBuffer::new(core.device, size, &layout);

// WGSL: Access the buffer in your shader
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

// ... later in compute shader ...
atomicAdd(&atomic_buffer[index], 1u);
```

### Audio Spectrum Analysis
Access real-time audio data from media files.
```rust
// Rust: In update() or render(), call this to populate the uniform
self.base.update_audio_spectrum(&core.queue);

// WGSL: Access data in the ResolutionUniform
@group(3) @binding(0) var<uniform> u_resolution: ResolutionUniform;
let spectrum_value = u_resolution.audio_data[frequency_bin][component];
let bpm = u_resolution.bpm;
```

## Essential Uniforms

```wgsl
struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 }
struct ResolutionUniform { dimensions: vec2<f32>, _padding: vec2<f32>, audio_data: [[f32; 4]; 32], bpm: f32, _bpm_padding: [f32; 3] }
struct MouseUniform { position: vec2<f32>, click_position: vec2<f32>, wheel: vec2<f32>, buttons: vec2<u32> }
```

## Built-in Features

- **Hot Reload**: Modify WGSL files and see changes instantly
- **Built-in UI**: Press `H` to toggle controls, `F` for fullscreen  
- **Export**: Built-in frame capture for creating videos/images
- **Text Rendering**: GPU-accelerated font system for overlays
- **Drag & Drop**: Load media files by dropping them on the window

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
