# Cuneus Usage Guide

Cuneus is a Rust-based GPU shader engine that simplifies creating interactive visual effect. It supports both traditional fragment shaders and modern compute shaders with built-in UI controls, hot reload, and export capabilities.

## Quick Start

1. **Copy a template** from `src/bin/` that matches your needs:
   - `debugscreen.rs` - Simple compute shader with text rendering
   - `sinh.rs` - Fragment shader with interactive parameters  
   - `audiovis.rs` - Media processing with audio spectrum analysis

2. **Customize the template** for your effect
3. **Write your WGSL shader** and iterate with hot reload

## Core Structure

Every shader follows this pattern:

```rust
use cuneus::{Core, ShaderApp, ShaderManager, RenderKit};

struct MyShader {
    base: RenderKit,
    // Custom uniforms/parameters
}

impl ShaderManager for MyShader {
    fn init(core: &Core) -> Self { /* Setup */ }
    fn update(&mut self, core: &Core) { /* Per-frame updates */ }
    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> { /* Rendering */ }
    fn handle_input(&mut self, core: &Core, event: &WindowEvent) -> bool { /* Input */ }
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
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> params: MyParams;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag_coord.xy / u_resolution.dimensions;
    // Your effect here
    return vec4<f32>(uv, 0.0, 1.0);
}
```

## Compute Shaders

**Rust setup:**

```rust
let config = ComputeShaderConfig {
    workgroup_size: [16, 16, 1],
    enable_fonts: true,  // For text rendering
    ..Default::default()
};

base.compute_shader = Some(ComputeShader::new_with_config(
    core,
    include_str!("../../shaders/my_compute.wgsl"),
    config,
));
```

**WGSL structure:**

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

## Interactive Controls

Add real-time parameter controls with egui:

```rust
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MyParams {
    intensity: f32,
    scale: f32,
    _padding: [f32; 2],  // GPU alignment
}

// In render function:
let full_output = if self.base.key_handler.show_ui {
    self.base.render_ui(core, |ctx| {
        egui::Window::new("Controls").show(ctx, |ui| {
            ui.add(egui::Slider::new(&mut params.intensity, 0.0..=2.0));
            ui.add(egui::Slider::new(&mut params.scale, 0.1..=10.0));
        });
    })
} else {
    self.base.render_ui(core, |_| {})
};
```

## Media Support

Cuneus provides comprehensive media integration through GStreamer:

### Loading Media
```rust
self.base.load_media(core, path)?;
```

### Media UI Controls
Add media panel to your UI:
```rust
ShaderControls::render_media_panel(
    ui,
    &mut controls_request,
    using_video_texture,
    video_info,
    using_hdri_texture, 
    hdri_info,
    using_webcam_texture,
    webcam_info
);
```

### Supported Formats
- **Images**: PNG, JPG, JPEG, BMP, TIFF, WebP
- **Videos**: MP4, AVI, MKV, WebM, MOV (with audio support)
- **HDRI**: HDR, EXR (with exposure/gamma controls)
- **Webcam**: Live camera feed

### Using in Shaders
```wgsl
@group(0) @binding(0) var input_texture: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

// Access current media frame
let media_color = textureSample(input_texture, tex_sampler, uv);
```

## Advanced Patterns

### Multi-Pass Rendering (Ping-Pong)
For effects that need previous frame data:
```rust
// Create texture pairs for ping-pong
let texture_pair = create_feedback_texture_pair(core, width, height, layout);

// Swap between frames
let source_tex = if frame_count % 2 == 0 { &pair.0 } else { &pair.1 };
let target_tex = if frame_count % 2 == 0 { &pair.1 } else { &pair.0 };
```

### Compute Shaders with Atomic Buffers
For GPU accumulation and complex algorithms:
```rust
// Enable atomic buffer in config
let config = ComputeShaderConfig {
    enable_atomic_buffer: true,
    atomic_buffer_multiples: 4,
    ..Default::default()
};
```

### Audio Spectrum Analysis
```rust
// Update audio spectrum (video/webcam sources only)
self.base.update_audio_spectrum(&core.queue);
```

```wgsl
// Access in shader
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

