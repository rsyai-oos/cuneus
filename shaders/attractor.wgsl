@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

struct TimeUniform {
    time: f32,
};
@group(1) @binding(0)
var<uniform> time_data: TimeUniform;

struct Params {
    min_radius: f32,
    max_radius: f32,
    size: f32,
    decay: f32,
};
@group(2) @binding(0)
var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const S: f32 = 1.5 * PI;
fn gamma_correction(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
}
fn oscillate(minn: f32, maxxi: f32, interval: f32, now: f32) -> f32 {
    return minn + (maxxi - minn) * 0.5 * (sin(2.0 * PI * now / interval) + 1.0);
}

fn cl(p: vec2<f32>, a: f32, b: f32, c: f32, d: f32, t: f32) -> vec2<f32> {
    let sint = sin(t);
    let cost = cos(t);
    let x = sin(a * p.y + t) + c * cos(a * p.x + t) * sin(b * p.x + t);
    let y = sin(b * p.x + t) + d * cos(b * p.y + t) * sin(a * p.y + t);
    var result = vec2<f32>(x, y);
    result *= 1.5;
    result = vec2<f32>(
        result.x * cost - result.y * sint,
        result.x * sint + result.y * cost
    );
    return result;
}

@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(prev_frame));
    let uv = 12.0 * (2.0 * (FragCoord.xy / dimensions - 0.5)) * vec2<f32>(dimensions.x/dimensions.y, 1.0);
    let dasds = oscillate(12.0, 4.0, 20.0, time_data.time);
    let radii = oscillate(params.min_radius, params.max_radius, 5.0, time_data.time) / dasds;
    let sizw = oscillate(0.07, 0.04, 12.0, time_data.time);
    let t = time_data.time / 4.0;
    let scale = 3.0;
    let a = 1.7;
    let b = 1.7;
    let c = 0.6;
    let d = 1.2;
    
    var color = vec3<f32>(0.0);
    for(var i: f32 = 0.0; i < 155.0; i += 1.0) {
        let Cindex = floor(i / 4.0);
        let Pindex = i;
        let initialT = Pindex / 44.0 * S + time_data.time * 0.3;
        let Cscale = (Cindex + 1.0) * radii * 0.1;
        let initialPoint = vec2<f32>(
            cos(initialT) * Cscale,
            sin(initialT) * Cscale
        );
        let attPo = cl(initialPoint, a, b, c, d, t);
        let circlePoint = attPo * scale;
        let chang = oscillate(0.1, 0.5, 12.0, time_data.time);
        let pointColor = 0.5 + chang * cos(vec3<f32>(1.0, S/3.0, S*2.0/3.0) + Cindex * 0.99);
        let dist = length(uv - circlePoint);
        
        color += pointColor * sizw / (dist + 0.0001);
    }
    let des = oscillate(1.5, 2.0, 12.0, time_data.time);
    color = sqrt(color) * des - 1.5;
    let feedback = textureSample(prev_frame, tex_sampler, tex_coords);
    color = mix(color, feedback.rgb, 0.96);
    color = gamma_correction(color, 0.9);
    return vec4<f32>(color, 1.0);
}
@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let currentFrame = textureSample(prev_frame, tex_sampler, tex_coords); // Source texture from Pass 1
    return currentFrame * params.decay;
}
@fragment
fn fs_pass3(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(prev_frame, tex_sampler, tex_coords);
}