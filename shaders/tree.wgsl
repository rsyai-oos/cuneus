@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

struct TimeUniform {
    time: f32,
    frame: u32,
};
@group(1) @binding(0)
var<uniform> time_data: TimeUniform;

struct Params {
    pixel_offset: f32,
    pixel_offset2: f32,
    lights: f32,
    exp: f32,
    frame: f32,
    col1:f32,
    col2:f32,
    decay: f32,
};
@group(2) @binding(0)
var<uniform> params: Params;

const PI: f32 = 3.14159265;
const ITER: i32 = 200;
const ZZ: f32 = 77.0;
const H: f32 = 0.1;

fn hash(p4: vec4<f32>) -> vec4<f32> {
    var p = fract(p4 * vec4<f32>(0.1031, 0.1030, 0.0973, 0.1099));
    p += dot(p, p.wzxy + 33.33);
    return fract((p.xxyz + p.yzzw) * p.zywx);
}

fn complex_power(z: vec2<f32>, x: f32) -> vec2<f32> {
    let r = length(z);
    let theta = atan2(z.y, z.x);
    let r_pow = pow(r, x);
    let x_theta = x * theta;
    return r_pow * vec2<f32>(cos(x_theta), sin(x_theta));
}

fn f(z: vec2<f32>) -> vec2<f32> {
    return complex_power(z, 1.5) - vec2<f32>(0.2, 0.0);
}

fn implicit(z_in: vec2<f32>) -> vec2<f32> {
    var z = z_in;
    var dz = vec2<f32>(H, 0.0);
    var i = 0;
    
    for (i = 0; i < ITER; i++) {
        dz = 1.5 * pow(length(z), 0.5) * dz;
        z = f(z);
        if (dot(z, z) > ZZ) { break; }
    }
    
    return vec2<f32>(f32(i), dot(z, z) / dot(dz, dz));
}

@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let R = vec2<f32>(textureDimensions(prev_frame));
    let U = FragCoord.xy;
    let U0 = U;
    let zoom = 0.3;
    let pan = vec2<f32>(0.8, 1.7);
    
    let uv = ((vec2<f32>(U.x, R.y - U.y) - 0.5 * R) / min(R.y, R.x) + pan) * zoom;
    let z_and_i = implicit(uv);
    let iter_ratio = z_and_i.x / f32(ITER);
    let sharpness = z_and_i.y;
    
    let col1 = 0.5 + 0.5 * cos(params.col1 + vec3<f32>(0.0, 0.5, 1.0) + PI * vec3<f32>(2.0 * sharpness));
    let col2 = 0.5 + 0.5 * cos(params.col2 + PI * vec3<f32>(sharpness));
    let col = mix(col1, col2, iter_ratio);
    let col_sqrt = sqrt(col);
    
    var Q = vec4<f32>(0.0);
    Q.x = (col_sqrt.r + col_sqrt.g + col_sqrt.b) / 3.0;
    return Q;
}

@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let R = vec2<f32>(textureDimensions(prev_frame));
    let U = FragCoord.xy;
    let pixel_offset = params.pixel_offset;
    let pixel_offset_2 = params.pixel_offset2;
    let n = textureSample(prev_frame, tex_sampler, (U - vec2<f32>(pixel_offset_2, -pixel_offset)) / R);
    let e = textureSample(prev_frame, tex_sampler, (U - vec2<f32>(pixel_offset, pixel_offset_2)) / R);
    let s = textureSample(prev_frame, tex_sampler, (U - vec2<f32>(pixel_offset_2, pixel_offset)) / R);
    let w = textureSample(prev_frame, tex_sampler, (U - vec2<f32>(-pixel_offset, pixel_offset_2)) / R);
    
    var Q = vec4<f32>(0.0);
    Q.x = 0.5 * (e.x - w.x);
    Q.y = 0.5 * (s.x - n.x);
    Q.z = textureSample(prev_frame, tex_sampler, U / R).x;
    
    return Q;
}

@group(0) @binding(0) var texBufferC: texture_2d<f32>; // = BufferC
@group(0) @binding(1) var samplerC: sampler;
@group(0) @binding(2) var texBufferB: texture_2d<f32>; //  = BufferB
@group(0) @binding(3) var samplerB: sampler;

@fragment
fn fs_pass3(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let R = vec2<f32>(textureDimensions(texBufferC));
    var U = FragCoord.xy;
    var accumulated = textureSample(texBufferC, samplerC, U / R);
    let frameCount = max(f32(time_data.frame), 1.0);
    let frameWeight = 1.0 / frameCount;
    let seed = vec4<f32>(U, f32(time_data.frame) * 0.1, 1.0);
    let h = hash(seed);
    var d = vec2<f32>(cos(2.0 * PI * h.x), sin(2.0 * PI * h.x));
    let amplitude = min(0.4 * params.frame, 0.1);
    var currentFrameContribution = vec4<f32>(0.0);
    var iter = params.col1;
    for(var i: f32 = 0.0; i < iter; i += 1.0) {
        U += d;
        
        let b = textureSample(texBufferB, samplerB, U / R);
        
        d += (1.0 + h.z) * 30.0 * b.xy;
        d = normalize(d);
        
        currentFrameContribution += amplitude * exp(-params.exp * length(d - vec2<f32>(0.0, 1.0))) * 
                                  max(sin(-2.0 + 6.0 * h.z + vec4<f32>(1.0, 2.0, 3.0, 4.0)), vec4<f32>(0.0));
        
        currentFrameContribution -= vec4<f32>(1.0, 2.0, 3.0, 4.0) * 0.0005 * b.z * params.frame;
    }
    let stabilizationFrames = 30.0;
    let blendFactor = min(frameCount, stabilizationFrames) / stabilizationFrames;
    let effectiveDecay = mix(params.decay, 1.0, blendFactor);
    var Q = accumulated * effectiveDecay * (1.0 - frameWeight) + currentFrameContribution * frameWeight;
    
    return Q;
}

@fragment
fn fs_pass4(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let R = vec2<f32>(textureDimensions(prev_frame));
    let U = FragCoord.xy;
    
    let raw_result = textureSample(prev_frame, tex_sampler, U / R);
    
    let result = pow(raw_result, vec4<f32>(params.lights));
    
    let gamma = 1.4;
    return pow(result, vec4<f32>(gamma));
}