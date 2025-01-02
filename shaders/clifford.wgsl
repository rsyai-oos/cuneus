@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

struct TimeUniform {
    time: f32,
};
@group(1) @binding(0)
var<uniform> time_data: TimeUniform;

struct Params {
    decay: f32,
    speed: f32,
    intensity: f32,
    scale: f32,
};
@group(2) @binding(0)
var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;

fn clifford_attractor(p: vec2<f32>, t: f32) -> vec2<f32> {
    let a = 1.7 + sin(t * 0.1) * 0.1;
    let b = 1.7 + cos(t * 0.15) * 0.1;
    let c = 0.6 + sin(t * 0.2) * 0.1;
    let d = 1.2 + cos(t * 0.25) * 0.1;
    
    return vec2<f32>(
        sin(a * p.y) + c * cos(a * p.x),
        sin(b * p.x) + d * cos(b * p.y)
    );
}

fn generate_point(uv: vec2<f32>, t: f32) -> vec4<f32> {
    var p = vec2<f32>(2.0 * (uv.x - 0.5), 2.0 * (uv.y - 0.5));
    var color = vec3<f32>(
        0.5 + 0.5 * sin(t + uv.x * TWO_PI),
        0.5 + 0.5 * cos(t + uv.y * TWO_PI),
        0.5 + 0.5 * sin(t + (uv.x + uv.y) * PI)
    );

    var accumColor = vec3<f32>(0.0);
    
    for(var i = 0; i < 25; i++) {
        let next = clifford_attractor(p, t);
        p = next;
        
        if(i > 20) {
            let dist = length(uv - (p * 0.2 + 0.5));
            accumColor += color * smoothstep(0.02, 0.0, dist) * params.intensity;
        }
    }
    
    return vec4<f32>(accumColor, 1.0);
}

@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(prev_frame));
    let uv = tex_coords;
    
    let t = time_data.time * params.speed;
    let current = generate_point(uv, t);
    
    let previous = textureSample(prev_frame, tex_sampler, tex_coords);
    
    return mix(current, previous, params.decay);
}

@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(prev_frame, tex_sampler, tex_coords);
    return vec4<f32>(pow(color.rgb, vec3<f32>(1.2)), color.a);
}