@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(1) @binding(0) var input_texture: texture_2d<f32>;
@group(1) @binding(1) var input_sampler: sampler;

struct TimeUniform {
    time: f32,
    frame: u32,
};
@group(2) @binding(0)
var<uniform> time_data: TimeUniform;

struct Params {
    rotation_speed: f32,
    motor_strength: f32,
    distortion: f32,
    feedback: f32,
};
@group(3) @binding(0)
var<uniform> params: Params;

const rut = 5u;
const PI = 3.14159265359;
const ANG = 4.0 * PI / f32(rut);

fn r2d() -> mat2x2<f32> {
    return mat2x2<f32>(
        vec2<f32>(cos(ANG), sin(ANG)),
        vec2<f32>(-sin(ANG), cos(ANG))
    );
}

fn get_rot(pos: vec2<f32>, b: vec2<f32>, dims: vec2<f32>) -> f32 {
    var p = b;
    var rot = 0.0;
    let m = r2d();
    
    for(var i = 0u; i < rut; i = i + 1u) {
        let sample = textureSample(prev_frame, tex_sampler, fract((pos + p) / dims)).xy;
        rot += dot(sample - vec2<f32>(0.5), p.yx * vec2<f32>(1.0, -1.0));
        p = m * p;
    }
    return rot / f32(rut) / dot(b, b);
}

fn get_val(uv: vec2<f32>) -> f32 {
    return length(textureSample(prev_frame, tex_sampler, uv).xyz);
}

fn get_grad(uv: vec2<f32>, delta: f32) -> vec2<f32> {
    let d = vec2<f32>(delta, 0.0);
    return vec2<f32>(
        get_val(uv + d.xy) - get_val(uv - d.xy),
        get_val(uv + d.yx) - get_val(uv - d.yx)
    ) / delta;
}

@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let pos = FragCoord.xy;
    let dims = vec2<f32>(textureDimensions(prev_frame));
    var b = vec2<f32>(cos(ANG), sin(ANG));
    var v = vec2<f32>(0.0);
    let bb_max = 0.7 * dims.y;
    let bb_max_squared = bb_max * bb_max;
    let m = r2d();

    for(var l = 0u; l < 20u; l = l + 1u) {
        if(dot(b, b) > bb_max_squared) { break; }
        
        var p = b;
        for(var i = 0u; i < rut; i = i + 1u) {
            v += p.yx * get_rot(pos + p, b, dims);
            p = m * p;
        }
        b *= 2.0;
    }
    
    var color: vec4<f32>;
    if (time_data.frame <= 4u) {
        color = textureSample(input_texture, input_sampler, tex_coords);
    } else {
        let distorted_uv = fract((pos + v * vec2<f32>(-1.0, 1.0) * 2.0) / dims);
        let fluid_color = textureSample(prev_frame, tex_sampler, distorted_uv);
        let texture_color = textureSample(input_texture, input_sampler, distorted_uv);
        color = mix(texture_color, fluid_color, params.feedback);
    }
    let scr = tex_coords * 2.0 - 1.0;
    let timee = 1.0 + 0.2 * sin(time_data.time);
    let motor_force = params.motor_strength * timee * scr / (dot(scr, scr) / 0.1 + 0.3);
    let motor_pos = pos + motor_force * dims * params.distortion;
    let motor_color = textureSample(input_texture, input_sampler, fract(motor_pos / dims));
    color = mix(color, motor_color, length(motor_force) * 2.0);
    
    return color;
}
@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let delta = 1.0 / f32(textureDimensions(prev_frame).y);
    let n = normalize(vec3<f32>(
        get_grad(tex_coords, delta).x,
        get_grad(tex_coords, delta).y,
        150.0
    ));
    let light = normalize(vec3<f32>(
        1.0 + 0.2 * sin(time_data.time * 0.5),
        1.0 + 0.2 * cos(time_data.time * 0.5),
        2.0
    ));
    let diff = clamp(dot(n, light), 0.5, 1.0);
    let spec = pow(clamp(dot(reflect(light, n), vec3<f32>(0.0, 0.0, -1.0)), 0.0, 1.0), 36.0) * 2.5;
    let cl = textureSample(prev_frame, tex_sampler, tex_coords);
    return cl * vec4<f32>(diff) + vec4<f32>(spec);
}