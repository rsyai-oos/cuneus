// Scene Color Palette - Single-Pass Compute Shader
// MIT License, Enes Altun, 2025

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

// Group 1: Primary Pass I/O & Parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: SceneColorParams;
@group(1) @binding(2) var input_texture: texture_2d<f32>;
@group(1) @binding(3) var input_sampler: sampler;

struct SceneColorParams {
    num_segments: f32,
    palette_height: f32,
    samples_x: i32,
    samples_y: i32,
    
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

fn mix_f32(a: f32, b: f32, t: f32) -> f32 {
    return a * (1.0 - t) + b * t;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }

    let pixel_pos = vec2<i32>(id.xy);
    let dimensions = vec2<f32>(dims);
    let uv = vec2<f32>(id.xy) / dimensions;
    
    // Original texture in the main area
    if (uv.y <= (1.0 - params.palette_height)) {
        let color = textureLoad(input_texture, pixel_pos, 0);
        textureStore(output, pixel_pos, color);
        return;
    }
    
    // Create a color palette on the bottom
    let segmentWidth = dimensions.x / params.num_segments;
    let segmentIndex = i32(f32(id.x) / segmentWidth);
    
    let startX = f32(segmentIndex) / params.num_segments;
    let endX = f32(segmentIndex + 1) / params.num_segments;
    
    // Calculate average color for this segment
    var avgColor = vec3<f32>(0.0);
    var sampleCount = 0;
    
    // Sample evenly across the vertical dimension and within the segment horizontally
    // Use fixed loop bounds to avoid dynamic loops issue
    for (var y = 0; y < 32; y = y + 1) {
        if (y >= params.samples_y) { break; }
        let sampleY = f32(y) / f32(params.samples_y - 1);
        for (var x = 0; x < 32; x = x + 1) {
            if (x >= params.samples_x) { break; }
            // Sample within the segment's horizontal range
            let sampleX = mix_f32(startX, endX, f32(x) / f32(params.samples_x - 1));
            let sample_pixel = vec2<i32>(i32(sampleX * dimensions.x), i32(sampleY * dimensions.y));
            avgColor += textureLoad(input_texture, sample_pixel, 0).rgb;
            sampleCount = sampleCount + 1;
        }
    }
    
    // Calculate the average
    avgColor = avgColor / f32(sampleCount);
    
    let result = vec4<f32>(avgColor, 1.0);
    textureStore(output, pixel_pos, result);
}