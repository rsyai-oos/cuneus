// MIT License, Enes Altun, 2025
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(1) @binding(0) var<uniform> u_time: TimeUniform;
@group(2) @binding(0) var<uniform> params: Params;
@group(3) @binding(0) var<uniform> u_resolution: ResolutionUniform;

struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};

struct TimeUniform {
    time: f32,
};

struct Params {
    // Palette settings
    num_segments: f32,
    palette_height: f32,
    samples_x: i32,
    samples_y: i32,
    
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

fn mix(a: f32, b: f32, t: f32) -> f32 {
    return a * (1.0 - t) + b * t;
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let dimensions = u_resolution.dimensions;
    let uv = fragCoord.xy / dimensions;
    
    // original texture in the main area
    if (uv.y <= (1.0 - params.palette_height)) {
        return textureSample(tex, tex_sampler, uv);
    }
    
    // I m going to create a plate on bottom, this can be adjustable via egui
    let segmentWidth = dimensions.x / params.num_segments;
    let segmentIndex = i32(fragCoord.x / segmentWidth);
    

    let startX = f32(segmentIndex) / params.num_segments;
    let endX = f32(segmentIndex + 1) / params.num_segments;
    
    // average color for this segment
    var avgColor = vec3<f32>(0.0);
    var sampleCount = 0;
    
    // Sample evenly across the vertical dimension and within the segment horizontally
    for (var y = 0; y < params.samples_y; y = y + 1) {
        let sampleY = f32(y) / f32(params.samples_y - 1);
        for (var x = 0; x < params.samples_x; x = x + 1) {
            // Sample within the segment's horizontal range
            let sampleX = mix(startX, endX, f32(x) / f32(params.samples_x - 1));
            avgColor += textureSample(tex, tex_sampler, vec2<f32>(sampleX, sampleY)).rgb;
            sampleCount = sampleCount + 1;
        }
    }
    // Calculate the average
    avgColor = avgColor / f32(sampleCount);
    return vec4<f32>(avgColor, 1.0);
}