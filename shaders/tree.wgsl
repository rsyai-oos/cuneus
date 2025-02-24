@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

struct TimeUniform {
    time: f32,
    frame: u32,
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
    
    let uv = ((U - 0.5 * R) / min(R.y, R.x) + pan) * zoom;
    let z_and_i = implicit(uv);
    let iter_ratio = z_and_i.x / f32(ITER);
    let sharpness = z_and_i.y;
    
    let col1 = 0.5 + 0.5 * cos(1.0 + vec3<f32>(0.0, 0.5, 1.0) + PI * vec3<f32>(2.0 * sharpness));
    let col2 = 0.5 + 0.5 * cos(4.1 + PI * vec3<f32>(sharpness));
    let col = mix(col1, col2, iter_ratio);
    let col_sqrt = sqrt(col);
    
    var Q = vec4<f32>(0.0);
    Q.x = (col_sqrt.r + col_sqrt.g + col_sqrt.b) / 3.0;
    
    let grad_n = textureSample(prev_frame, tex_sampler, (U0 + vec2<f32>(0.0, 1.0)) / R);
    let grad_e = textureSample(prev_frame, tex_sampler, (U0 + vec2<f32>(1.0, 0.0)) / R);
    let grad_s = textureSample(prev_frame, tex_sampler, (U0 - vec2<f32>(0.0, 1.0)) / R);
    let grad_w = textureSample(prev_frame, tex_sampler, (U0 - vec2<f32>(1.0, 0.0)) / R);
    
    Q.y = -(grad_e.w - grad_w.w);
    Q.z = -(grad_n.w - grad_s.w);
    Q.w = Q.x;
    
    return Q;
}

@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let R = vec2<f32>(textureDimensions(prev_frame));
    let U = FragCoord.xy;
    
    // Sample from BufferA
    let n = textureSample(prev_frame, tex_sampler, (U + vec2<f32>(0.0, 1.0)) / R);
    let e = textureSample(prev_frame, tex_sampler, (U + vec2<f32>(1.0, 0.0)) / R);
    let s = textureSample(prev_frame, tex_sampler, (U - vec2<f32>(0.0, 1.0)) / R);
    let w = textureSample(prev_frame, tex_sampler, (U - vec2<f32>(1.0, 0.0)) / R);
    
    var Q = vec4<f32>(0.0);
    Q.x = 0.5 * (e.x - w.x);
    Q.y = 0.5 * (n.x - s.x);
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
    
    // Start with the current buffer value (feedback from itself)
    // Stronger retention of previous frame for smoother accumulation
    var Q = textureSample(texBufferC, samplerC, U / R) * 0.995;
    
    // Use a slowed-down frame counter to make particles build up more gradually
    // We'll use real frame # divided by 4 to slow down the appearance of new patterns
    let frame = time_data.frame / 4u;
    let h = hash(vec4<f32>(U, f32(frame), 1.0));
    var d = vec2<f32>(cos(2.0 * PI * h.x), sin(2.0 * PI * h.x));
    
    for(var i: f32 = 0.0; i < 40.0; i += 1.0) {
        U += d;
        
        // Sample vector field from BufferB
        let b = textureSample(texBufferB, samplerB, U / R);
        
        d += (1.0 + h.z) * 30.0 * b.xy;
        d = normalize(d);
        
        Q += 0.02 * exp(-10.0 * length(d - vec2<f32>(0.0, -1.0))) * 
             max(sin(-2.0 + 6.0 * h.z + vec4<f32>(1.0, 2.0, 3.0, 4.0)), vec4<f32>(0.0));
        
        Q -= vec4<f32>(1.0, 2.0, 3.0, 4.0) * 0.0006 * b.z;
    }
    
    return Q;
}

@fragment
fn fs_pass4(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let R = vec2<f32>(textureDimensions(prev_frame));
    let U = FragCoord.xy;
    
    let raw_result = textureSample(prev_frame, tex_sampler, U / R);
    
    let result = pow(raw_result, vec4<f32>(0.55));
    
    return result;
}