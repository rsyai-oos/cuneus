# Cuneus Usage Guide

Cuneus is a Rust-based GPU shader engine that simplifies creating interactive visual effects. It features a unified backend system that seamlessly handles single-pass, multi-pass, and atomic compute shaders with built-in UI controls, hot-reloading, and media integration.


## Core Structure

Every shader follows this pattern, using the unified `ComputeShader` backend:

```rust
use cuneus::prelude::*;
use cuneus::compute::*;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MyParams {
    strength: f32,
    color: [f32; 3],
    _padding: f32,
}

impl UniformProvider for MyParams {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

struct MyShader {
    base: RenderKit,
    compute_shader: ComputeShader,
    current_params: MyParams,
}

impl ShaderManager for MyShader {
    fn init(core: &Core) -> Self {
        let base = RenderKit::new(core, /* ... */);
        
        let initial_params = MyParams {
            strength: 1.0,
            color: [1.0, 0.5, 0.2],
            _padding: 0.0,
        };

        // Unified builder pattern - works for all shader types
        let config = ComputeShader::builder()
            .with_entry_point("main")
            .with_custom_uniforms::<MyParams>()
            .with_workgroup_size([16, 16, 1])
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("My Shader")
            .build();

        let compute_shader = ComputeShader::from_builder(
            core,
            include_str!("../../shaders/my_shader.wgsl"),
            config,
        );

        compute_shader.set_custom_params(initial_params, &core.queue);

        Self { base, compute_shader, current_params: initial_params }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        let delta = 1.0/60.0;
        self.compute_shader.set_time(current_time, delta, &core.queue);
        self.base.fps_tracker.update();
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        // UI and parameter updates...
        
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("My Shader Encoder"),
        });

        // Single unified dispatch - works for all shader types
        self.compute_shader.dispatch(&mut encoder, core);
        
        // Render to screen...
        Ok(())
    }
}
```

## Backend

Backend uses a fluent builder pattern that handles all shader types through a single interface:

### Single-Pass Shaders
```rust
let config = ComputeShader::builder()
    .with_entry_point("main")                    // Single entry point
    .with_custom_uniforms::<MyParams>()          // Custom parameters
    .with_mouse()                                // Mouse input
    .with_workgroup_size([16, 16, 1])
    .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
    .with_label("Single Pass Shader")
    .build();

// Later: self.compute_shader.dispatch(&mut encoder, core);
```

### Multi-Pass Shaders
```rust
let passes = vec![
    PassDescription::new("buffer_a", &[]),
    PassDescription::new("buffer_b", &["buffer_a"]),
    PassDescription::new("main_image", &["buffer_a", "buffer_b"]),
];

let config = ComputeShader::builder()
    .with_multi_pass(&passes)                    // Multi-pass pipeline
    .with_custom_uniforms::<MyParams>()
    .with_workgroup_size([16, 16, 1])
    .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
    .with_label("Multi Pass Shader")
    .build();

// Automatic execution: self.compute_shader.dispatch(&mut encoder, core);
```

### Atomic Buffer Shaders
```rust
let config = ComputeShader::builder()
    .with_entry_point("main")
    .with_custom_uniforms::<MyParams>()
    .with_atomic_buffer()                        // Enable atomic operations
    .with_workgroup_size([16, 16, 1])
    .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
    .with_label("Atomic Shader")
    .build();

// Usage in WGSL: @group(2) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;
```

### Multi-Stage with Manual Control
```rust
// For complex shaders like buddhabrot that need manual control
let mut config = ComputeShader::builder()
    .with_entry_point("Splat")                   // Primary entry point
    .with_custom_uniforms::<MyParams>()
    .with_atomic_buffer()
    .with_workgroup_size([16, 16, 1])
    .build();

config.entry_points.push("main_image".to_string()); // Add secondary entry point

// Manual dispatch control:
// self.compute_shader.dispatch_stage_with_workgroups(&mut encoder, 0, [2048, 1, 1]);
// self.compute_shader.dispatch_stage(&mut encoder, core, 1);
```

### Advanced Features
```rust
let config = ComputeShader::builder()
    .with_entry_point("main")
    .with_custom_uniforms::<MyParams>()
    .with_fonts()                               // GPU text rendering (Group 2)
    .with_audio(4096)                           // Audio buffer (Group 2)  
    .with_mouse()                               // Mouse input (Group 2)
    .with_channels(4)                           // External textures (Group 2)
    .with_input_texture()                       // Input texture (Group 1)
    .with_storage_buffer(                       // Custom storage (Group 3)
        StorageBufferSpec::new("my_data", 1024)
    )
    .with_workgroup_size([16, 16, 1])
    .build();
```

## Unified Bind Group Layout

The unified backend uses a standardized 4-group layout:

### WGSL Bind Group Structure
```wgsl
// Group 0: Time and frame data (always present)
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

// Group 1: I/O textures and custom parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> u_params: MyParams;      // Custom uniforms
@group(1) @binding(2) var input_texture: texture_2d<f32>;   // Optional input texture
@group(1) @binding(3) var input_sampler: sampler;

// Group 2: Engine resources (fonts, audio, mouse, channels)
@group(2) @binding(0) var<uniform> u_font: FontUniforms;    // Font system
@group(2) @binding(1) var t_font_atlas: texture_2d<f32>;
@group(2) @binding(2) var s_font_atlas: sampler;
@group(2) @binding(3) var<storage, read_write> audio_buffer: array<f32>;
@group(2) @binding(4) var<uniform> u_mouse: MouseUniform;
@group(2) @binding(5) var channel0: texture_2d<f32>;        // External textures
@group(2) @binding(6) var channel0_sampler: sampler;

// Group 3: User storage buffers and atomic buffers
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;
@group(3) @binding(1) var<storage, read_write> my_data: array<f32>;
```

### Essential Uniforms
```wgsl
struct TimeUniform { 
    time: f32, 
    delta: f32, 
    frame: u32, 
    _padding: u32 
}

struct MouseUniform { 
    position: vec2<f32>, 
    click_position: vec2<f32>, 
    wheel: vec2<f32>, 
    buttons: vec2<u32> 
}

struct FontUniforms {
    atlas_size: vec2<f32>,
    char_size: vec2<f32>,
    screen_size: vec2<f32>,
    _padding: vec2<f32>,
}
```

## Dispatch Methods

The unified backend provides multiple dispatch options:

```rust
// Automatic dispatch - handles hot reload, frame counting, all passes
self.compute_shader.dispatch(&mut encoder, core);

// Manual stage control (for complex multi-stage shaders if you need)
self.compute_shader.dispatch_stage(&mut encoder, core, stage_index);
self.compute_shader.dispatch_stage_with_workgroups(&mut encoder, stage_index, [x, y, z]);

// Manual frame management (if needed)
self.compute_shader.increment_frame();
```

## Media Integration

Load and use various media formats seamlessly:

```rust
// In your shader setup - enable input texture support
let config = ComputeShader::builder()
    .with_input_texture()                       // Enables Group 1 input texture
    .with_channels(2)                           // Enables Group 2 channel textures
    .build();

// Loading media
self.base.load_media(core, "video.mp4")?;      // Loads to input_texture
self.compute_shader.set_channel_texture(core, 0, "image.png")?; // Loads to channel0

// In update() for video
if self.base.using_video_texture {
    self.base.update_video_texture(core, &core.queue);
}
```

## Audio Systems

### CPU Audio Synthesis
```rust
// For reading GPU-computed audio data
use cuneus::audio::SynthesisManager;

let audio_synthesis = SynthesisManager::new()?.start_gpu_synthesis()?;

// In update() - read GPU audio buffer
if let Ok(audio_data) = pollster::block_on(
    self.compute_shader.read_audio_buffer(&core.device, &core.queue)
) {
    let frequency = audio_data[0];
    let amplitude = audio_data[1];
    synth.set_voice(0, frequency, amplitude, amplitude > 0.01);
}
```

### GPU Audio Generation
```wgsl
// Write audio data from GPU compute shader
if global_id.x == 0u && global_id.y == 0u {
    audio_buffer[0] = frequency;               // Hz
    audio_buffer[1] = amplitude;               // 0.0-1.0  
    audio_buffer[2] = f32(waveform_type);      // 0=sine, 1=square, etc
    audio_buffer[3] = final_frequency;         // Processed frequency
    audio_buffer[4] = final_amplitude;         // Processed amplitude
}
```

## Text Rendering

Render text directly in compute shaders:

```rust
// Enable fonts in builder
.with_fonts()

// WGSL usage
fn draw_char(pos: vec2<f32>, char_code: u32, color: vec3<f32>) -> vec3<f32> {
    let char_uv = get_char_uv(char_code);
    let char_sample = textureSample(t_font_atlas, s_font_atlas, char_uv);
    return mix(color, vec3<f32>(1.0), char_sample.r);
}
```


## Built-in Features

- **Hot Reload**: Modify WGSL files and see changes instantly
- **Built-in UI**: Press `H` to toggle controls, `F` for fullscreen  
- **Export**: Built-in frame capture for creating videos/images
- **Media Support**: Drag & drop videos, images, webcam input
