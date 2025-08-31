// MIT License, altunenes, 2025, matrix formula inspired by the twitter: https://x.com/iquilezles/status/1440847977560494084
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: Params;
@group(1) @binding(2) var tex: texture_2d<f32>;
@group(1) @binding(3) var tex_sampler: sampler;

struct TimeUniform {
    time: f32,
};

struct Params {
    red_power: f32,
    green_power: f32,
    blue_power: f32,
    green_boost: f32,
    contrast: f32, 
    gamma: f32,
    glow: f32,
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
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let resolution = textureDimensions(output);
    let coord = vec2<i32>(global_id.xy);
    
    if (coord.x >= i32(resolution.x) || coord.y >= i32(resolution.y)) {
        return;
    }
    
    let uv = (vec2<f32>(coord) + 0.5) / vec2<f32>(resolution);
    
    // Get input texture dimensions and convert UV to input texture coordinates
    let tex_dims = textureDimensions(tex);
    let tex_coords = vec2<i32>(i32(uv.x * f32(tex_dims.x)), i32(uv.y * f32(tex_dims.y)));
    let clamped_coords = clamp(tex_coords, vec2<i32>(0), vec2<i32>(tex_dims) - vec2<i32>(1));
    
    let original_color = textureLoad(tex, clamped_coords, 0).rgb;
    
    var matrix_color = mt(original_color, params);
    let scanline = 0.5 + 0.5 * sin(uv.y * f32(resolution.y) * 0.7);
    let scanline_intensity = 0.1;
    let scanline_mask = 1.0 - scanline_intensity * (1.0 - scanline);
    matrix_color *= scanline_mask;
    
    let luminance = dot(matrix_color, vec3<f32>(0.299, 0.587, 0.114));
    let glow_intensity = params.glow;
    let green_glow = vec3<f32>(0.0, luminance * glow_intensity, 0.0);
    matrix_color += green_glow;
    
    let alpha = textureLoad(tex, clamped_coords, 0).a;
    matrix_color = gamma(matrix_color, params.gamma);
    
    textureStore(output, coord, vec4<f32>(matrix_color, alpha));
}