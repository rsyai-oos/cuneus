# Cuneus Usage Guide

Cuneus is a Rust-based GPU shader engine with a unified backend that handles single-pass, multi-pass, and atomic compute shaders with built-in UI controls, hot-reloading, and media integration.

## Core Structure

Every shader follows this pattern:

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
        let initial_params = MyParams { strength: 1.0, color: [1.0, 0.5, 0.2], _padding: 0.0 };

        // Universal builder pattern - handles all shader types
        let config = ComputeShader::builder()
            .with_entry_point("main")                    // Single pass
            .with_multi_pass(&passes)                    // Multi-pass (alternative)
            .with_custom_uniforms::<MyParams>()         // Custom parameters
            .with_atomic_buffer()                        // Atomic operations (optional)
            .with_mouse()                                // Mouse input (optional)
            .with_fonts()                                // GPU text rendering (optional)
            .with_audio(4096)                            // Audio buffer (optional)
            .with_channels(2)                            // External textures (optional)
            .with_input_texture()                        // Input texture (optional)
            .with_workgroup_size([16, 16, 1])           // Default workgroup (WGSL overrides this)
            .with_texture_format(COMPUTE_TEXTURE_FORMAT_RGBA16)
            .with_label("My Shader")
            .build();

        let mut compute_shader = ComputeShader::from_builder(core, include_str!("../../shaders/my_shader.wgsl"), config);
        
        // Hot reload (recommended)
        compute_shader.enable_hot_reload(core.device.clone(), std::path::PathBuf::from("shaders/my_shader.wgsl"), /* ... */);
        
        compute_shader.set_custom_params(initial_params, &core.queue);
        Self { base, compute_shader, current_params: initial_params }
    }

    fn update(&mut self, core: &Core) {
        let current_time = self.base.controls.get_time(&self.base.start_time);
        self.compute_shader.set_time(current_time, 1.0/60.0, &core.queue);
        self.compute_shader.check_hot_reload(&core.device);  // Hot reload check
        self.base.fps_tracker.update();
    }

    fn render(&mut self, core: &Core) -> Result<(), wgpu::SurfaceError> {
        let mut encoder = core.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        // Universal dispatch - works for all shader types
        self.compute_shader.dispatch(&mut encoder, core);
        
        // For advanced manual control only:
        // self.compute_shader.dispatch_stage(&mut encoder, core, stage_index);
        // self.compute_shader.current_frame += 1;  // Manual frame increment required!

        // Render to screen...
        Ok(())
    }
}
```

## Key Concepts

### Workgroup Sizes
- **WGSL takes precedence**: `@workgroup_size(256, 1, 1)` in WGSL overrides backend settings
- **Backend is fallback**: `.with_workgroup_size([16, 16, 1])` used only if WGSL doesn't specify
- **Different entry points**: Each `@compute` function can have different workgroup sizes

### Frame Management
- **Automatic**: `.dispatch()` automatically increments frame counter (recommended)
- **Manual**: `.dispatch_stage()` requires manual `self.compute_shader.current_frame += 1`
- **WGSL access**: Frame counter available as `time_data.frame` for accumulation effects

### Multi-Pass Setup
```rust
let passes = vec![
    PassDescription::new("stage1", &[]).with_workgroup_size([256, 1, 1]),
    PassDescription::new("stage2", &["stage1"]).with_workgroup_size([16, 16, 1]),
];
let config = ComputeShader::builder().with_multi_pass(&passes).build();
```

### Manual Entry Points (Advanced)
```rust
let mut config = ComputeShader::builder().with_entry_point("main").build();
config.entry_points.push("secondary".to_string());
// Use dispatch_stage() for individual control + manual frame management
```

## Standard Bind Group Layout

Your WGSL shaders should follow this layout:

```wgsl
// Group 0: Time data (automatic)
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

// Group 1: I/O + Custom parameters (you provide these)
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: MyParams;

// Group 2: Engine resources (mouse, fonts, audio, channels - optional)
@group(2) @binding(0) var<uniform> mouse: MouseUniform;
@group(2) @binding(1) var<uniform> font_uniform: FontUniforms;
@group(2) @binding(2) var font_texture: texture_2d<f32>;
@group(2) @binding(3) var<storage, read_write> audio_buffer: array<f32>;
@group(2) @binding(4) var channel0: texture_2d<f32>;
@group(2) @binding(5) var channel0_sampler: sampler;

// Group 3: User storage buffers and atomic operations (optional)
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;
@group(3) @binding(1) var<storage, read_write> my_data: array<f32>;
```

## Audio & Media Integration

### GPU Audio Generation
```wgsl
// Write audio data from compute shader to CPU
if global_id.x == 0u && global_id.y == 0u {
    audio_buffer[0] = frequency;        // Hz
    audio_buffer[1] = amplitude;        // 0.0-1.0
    audio_buffer[2] = waveform_type;    // 0=sine, 1=square, etc.
}
```

### CPU Audio Reading
```rust
// Read GPU-computed audio data
if let Ok(audio_data) = self.compute_shader.read_audio_buffer(&core.device, &core.queue).await {
    let frequency = audio_data[0];
    let amplitude = audio_data[1];
    // Use with SynthesisManager for actual audio output
}
```

### GPU Text Rendering
```wgsl
// Draw text directly in compute shaders
fn draw_char(pos: vec2<f32>, char_code: u32, color: vec3<f32>) -> vec3<f32> {
    let char_uv = get_char_uv(char_code, font_uniform);
    let char_sample = textureSample(font_texture, font_sampler, char_uv);
    return mix(color, vec3<f32>(1.0), char_sample.r);
}
```



## Best Practices

1. **Use `.dispatch()`** for most cases - it handles everything automatically
2. **Specify workgroup sizes in WGSL** with `@workgroup_size()` decorators  
3. **Follow bind group layout** for consistency
4. **Only use manual dispatch** for complex conditional logic (accumulated rendering, etc.)
5. **Always increment frame manually** when using `dispatch_stage()`.
