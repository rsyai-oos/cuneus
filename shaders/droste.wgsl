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
    branches: f32,
    scale: f32,
    time_scale: f32,
    rotation: f32, 
    zoom: f32, 
    offset_x: f32,
    offset_y: f32,
    iterations: f32,
    smoothing: f32, 
    use_animation: f32, 
};


const PI: f32 = 3.141592653589793;
const TWO_PI: f32 = 6.283185307179586;

fn complex_exp(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(exp(z.x) * cos(z.y), exp(z.x) * sin(z.y));
}

fn complex_log(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(log(length(z)), atan2(z.y, z.x));
}

fn complex_mult(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_mag(z: vec2<f32>) -> f32 {
    return pow(length(z), 2.0);
}

fn complex_reciprocal(z: vec2<f32>) -> vec2<f32> {
    let mag = complex_mag(z);
    return vec2<f32>(z.x / mag, -z.y / mag);
}

fn complex_div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return complex_mult(a, complex_reciprocal(b));
}

fn complex_power(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return complex_exp(complex_mult(b, complex_log(a)));
}

fn f(x: f32, n: f32) -> f32 {
    return pow(n, -floor(log(x) / log(n)));
}

fn droste(z: vec2<f32>, zoom: f32, twists: f32, scale: f32, iterations: f32) -> vec2<f32> {
    let ratio = 1.0 / scale;
    let angle = atan(log(ratio) * twists / TWO_PI);
    
    var result = z;
    let iter_count = abs(iterations);
    
    for(var i = 0.0; i < iter_count; i += 1.0) {
        result = complex_exp(
            complex_div(
                complex_log(result), 
                complex_exp(vec2<f32>(0.0, angle)) * cos(angle)
            )
        );
        result *= zoom;
        let a_z = abs(result);
        result *= f(max(a_z.x, a_z.y) * 2.0, ratio);
    }
    if (iterations < 0.0) {
        result = complex_div(result, vec2<f32>(ratio, 0.0));
    } else {
        result = result / ratio;
    }
    
    return result;
}
fn apply_smoothing(uv: vec2<f32>, amount: f32) -> vec2<f32> {
    if (amount == 0.0) { return uv; }
    let smooth_factor = clamp(abs(amount), 0.0, 1.0);
    let direction = sign(amount);
    if (direction > 0.0) {
        return mix(uv, sin(uv * PI), smooth_factor);
    } else {
        return mix(uv, tanh(uv), smooth_factor);
    }
}
@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let resolution = u_resolution.dimensions;
    var uv = (FragCoord.xy / resolution.xy) * 2.0 - 1.0;
    uv.x *= resolution.x / resolution.y;
    let offset = vec2<f32>(params.offset_x, params.offset_y) * (1.0 + abs(params.smoothing) * 0.5);
    uv += offset;
    var current_zoom = params.zoom;
    if (params.use_animation > 0.5) {
        let time = u_time.time * params.time_scale * (1.0 + abs(params.iterations) * 0.1);
        let ft = fract(time);
        current_zoom = 1.0 + ft * (params.scale - 1.0);
    }
    uv = droste(uv, current_zoom, params.branches, params.scale, params.iterations);
    let rotation = params.rotation * (1.0 + abs(params.iterations) * 0.1);
    let rot = mat2x2<f32>(
        cos(rotation), -sin(rotation),
        sin(rotation), cos(rotation)
    );
    uv = rot * uv;
    
    uv = apply_smoothing(uv, params.smoothing);
    
    uv = (uv + params.zoom) / 2.0;
    
    return textureSample(tex, tex_sampler, uv);
}