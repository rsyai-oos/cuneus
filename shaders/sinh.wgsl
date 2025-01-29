// MIT License Enes Altun, 2025
//Sinh forumla: https://paulbourke.net/fractals/sinh/
struct TimeUniform {
    time: f32,
};

struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: Params;

struct Params {
    color1: vec3<f32>,
    pad1: f32,
    gradient_color: vec3<f32>,
    _pad2: f32,
    c_value_max: f32,
    iterations: i32,
    aa_level: i32,
    _pad3: f32,
};
fn gamma_correction(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
}
const PI: f32 = 3.14159265;
const ZZ: f32 = 3.5;

fn c_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}

fn c_sinh(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        sinh(z.x) * cos(z.y),
        cosh(z.x) * sin(z.y + 0.01)
    );
}

fn c_abs(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(abs(z.x), abs(z.y));
}

fn c_sinh_pow4(z: vec2<f32>) -> vec2<f32> {
    let sinh_z = c_sinh(z);
    return c_mul(c_mul(sinh_z, sinh_z), c_mul(sinh_z, sinh_z));
}

fn implicit(z: vec2<f32>, c: vec2<f32>) -> vec2<f32> {
    var z_curr = z;
    var i = 0;
    
    for(; i < params.iterations; i = i + 1) {
        z_curr = c_abs(c_sinh_pow4(z_curr)) + c;
        if (dot(z_curr, z_curr) > ZZ * ZZ) { break; }
    }
    
    return vec2<f32>(f32(i), dot(z_curr, z_curr));
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let dimensions = u_resolution.dimensions;
    var col = vec3<f32>(0.0);
    let pixel_size = vec2<f32>(1.0) / dimensions;
    let c_value = mix(2.197, params.c_value_max, 0.01 + 0.01 * sin(0.1 * params.pad1));
    let oscillation = 0.00004 + 0.02040101 * (sin(0.1 * params.pad1) + 0.1);
    let frequency = 1.0;
    let oscillating_frequency = 1.0;
    let phase = 45.2;
    let A = 5.25 * sin(oscillating_frequency * u_time.time + phase) + 5.75;
    for(var m = 0; m < params.aa_level; m = m + 1) {
        for(var n = 0; n < params.aa_level; n = n + 1) {
            let offset = vec2<f32>(f32(m), f32(n)) / f32(params.aa_level);
            let pos = FragCoord.xy + offset;
            let uv = ((pos - 0.5 * dimensions) / min(dimensions.y, dimensions.x) * 2.0) * 0.5;
            let c = vec2<f32>(oscillation, c_value);
            let z_and_i = implicit(uv, c);
            let iter_ratio = z_and_i.x / f32(params.iterations);
            let lenSq = z_and_i.y;
            let col1 = 0.5 + 0.5 * cos(3.0 + u_time.time + params.color1 + PI * vec3<f32>(2.0 * lenSq));
            let col2 = 0.5 + 0.5 * cos(4.1 + u_time.time + PI * vec3<f32>(lenSq));
            let col3 = 4.5 + 0.5 * cos(3.0 + u_time.time + vec3<f32>(1.0, 0.5, 0.0) + PI * vec3<f32>(2.0 * sin(lenSq)));
            let gradientIndex = fract(iter_ratio * 24.0);
            let blend = fract(gradientIndex);
            let col4 = params.gradient_color;
            
            col = col + sqrt(col1 * col2 * col3) * col4;
        }
    }
    
    col = sqrt(col / f32(params.aa_level * params.aa_level));
    col = gamma_correction(col, 0.412);
    return vec4<f32>(col, 1.0);
}