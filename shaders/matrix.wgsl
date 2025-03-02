// MIT License, altunenes, 2025, matrix formula inspired by the twitter: https://x.com/iquilezles/status/1440847977560494084
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
    red_power: f32,     // Default: 3/2
    green_power: f32,   // Default: 4/5
    blue_power: f32,    // Default: 3/2
    green_boost: f32,   // Default: 1.2
    contrast: f32,      // Default: 1.1
    gamma: f32,         // Default: 1.0
}
//matrix transformation
fn mt(color: vec3<f32>, params: Params) -> vec3<f32> {
    var transformed = vec3<f32>(
        pow(color.r, params.red_power),
        pow(color.g, params.green_power),
        pow(color.b, params.blue_power)
    );
    transformed.g *= params.green_boost;
    var adjusted = (transformed - 0.5) * params.contrast + 0.5;
    return clamp(adjusted, vec3<f32>(0.0), vec3<f32>(1.0));
}


fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}
@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let resolution = u_resolution.dimensions;
    let uv = tex_coords;
    let original_color = textureSample(tex, tex_sampler, uv).rgb;
    var matrix_color = mt(original_color, params);
    let scanline = 0.5 + 0.5 * sin(uv.y * resolution.y * 0.7);
    let scanline_intensity = 0.1;
    let scanline_mask = 1.0 - scanline_intensity * (1.0 - scanline);
    matrix_color *= scanline_mask;
    let luminance = dot(matrix_color, vec3<f32>(0.299, 0.587, 0.114));
    let glow_intensity = 0.3;
    let green_glow = vec3<f32>(0.0, luminance * glow_intensity, 0.0);
    matrix_color += green_glow;
    let alpha = textureSample(tex, tex_sampler, uv).a;
     matrix_color = gamma(matrix_color, params.gamma);
    return vec4<f32>(matrix_color, alpha);
}