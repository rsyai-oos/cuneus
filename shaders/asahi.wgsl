struct TimeUniform {
    time: f32,
};
struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};
struct Params {
    // Colors
    color_petal_start_a: vec3<f32>,
    _pad1: f32,
    color_petal_end_a: vec3<f32>,
    _pad2: f32,
    color_petal_start_b: vec3<f32>,
    _pad3: f32,
    color_petal_end_b: vec3<f32>,
    _pad4: f32,
    bg_color: vec3<f32>,
    _pad5: f32,
    
    // Animation parameters
    petal_size: f32,
    space_factor: f32,
    animation_speed: f32,
    _pad6: f32,
};

@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const LAMBDA: f32 = 0.8; 

fn rot(a: f32) -> mat2x2<f32> {
    return mat2x2<f32>(cos(a), -sin(a), sin(a), cos(a));
}

fn draw_petal_polar(uv: vec2<f32>, pos: vec2<f32>, size: f32, dir: vec2<f32>, color_direction: f32) -> vec4<f32> {
    var dist = uv - pos;
    let angle = -atan2(dir.y, dir.x);
    dist = dist * (rot(angle) * LAMBDA);
    dist.x = dist.x - size * 0.25;
    
    let r = length(dist) * 1.5;
    let a = atan2(dist.y, dist.x);
    
    var f = -1.0;
    if(a > PI * 0.5 || a < -PI * 0.5) {
        f = size * cos(a * 2.0);
    }
    
    let petalMask = smoothstep(0.0, -1.0, (r - f) / fwidth(r - f));
    
    let color = select(
        mix(params.color_petal_start_b, params.color_petal_end_b, r / (size * 1.0)),
        mix(params.color_petal_start_a, params.color_petal_end_a, r / (size * 1.0)),
        color_direction > 0.0
    );
    
    return vec4<f32>(color, petalMask);
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = u_resolution.dimensions;
    let uv = 1.3 * (FragCoord.xy - 0.5 * dimensions) / dimensions.y;
    var fragColor = vec4<f32>(params.bg_color, 1.0);
    
    let petalSize = params.petal_size;
    let space = (2.5 * PI / 15.0) * params.space_factor;
    let phase = sin(u_time.time / params.animation_speed) * PI;
    
    let left = vec2<f32>(-0.5, 0.0);
    let right = vec2<f32>(0.5, 0.0);
    
    // Left side
    for(var i = 0.0; i < 2.0 * PI; i += space) {
        let pos = left + vec2<f32>(cos(i), sin(i)) * 0.25;
        let angle = i + phase;
        let dir = vec2<f32>(cos(angle), sin(angle));
        let leftcolor = draw_petal_polar(uv, pos, petalSize, dir, 1.0);
        fragColor = mix(fragColor, leftcolor, leftcolor.a);
    }
    
    // Right side
    for(var i = 0.0; i < 2.0 * PI; i += space) {
        let pos = right + vec2<f32>(cos(i), sin(i)) * 0.25;
        let angle = i - phase;
        let dir = vec2<f32>(cos(angle), sin(angle));
        let colors = draw_petal_polar(uv, pos, petalSize, dir, -1.0);
        fragColor = mix(fragColor, colors, colors.a);
    }
    
    return fragColor;
}