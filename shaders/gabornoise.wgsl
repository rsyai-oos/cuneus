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
    width: f32,
    height: f32,
    steps: f32,
    _pad1: f32,
    
    kernel_size: f32,
    num_kernels: f32,
    frequency: f32,
    frequency_var: f32,
    
    seed: f32,
    animation_speed: f32,
    gamma: f32,
    _pad2: f32,
};

const PI: f32 = 3.14159265358979;

fn hash21(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));    
    return fract(sin(h) * 43758.5453123);
}

// Hash function to get a random 2D vector in the range [-1, 1]
fn hash22(p: vec2<f32>) -> vec2<f32> {
    let h = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(h) * 43758.5453123);
}
fn gabor(pos: vec2<f32>, frequency: f32, direction: vec2<f32>, phase: f32) -> f32 {
    let g = exp(-PI * dot(pos, pos) / (params.kernel_size * params.kernel_size));
    let wave = cos(2.0 * PI * frequency * dot(pos, direction) + phase);
    return g * wave;
}
fn gaborNoise(uv: vec2<f32>) -> f32 {
    var noise = 0.0;
    let freq = params.frequency;
    let seed = params.seed;
    // kernels across the whole screen
    for (var i = 0.0; i < params.num_kernels; i += 1.0) {
        let randVal = hash21(vec2<f32>(i, seed));
        let kernelPos = vec2<f32>(
            hash21(vec2<f32>(i * 0.123, seed)),
            hash21(vec2<f32>(i * 0.456, seed * 2.0))
        );
        let phase = 2.0 * PI * hash21(vec2<f32>(seed, i));
        let kernelFreq = freq * (1.0 - params.frequency_var + 2.0 * params.frequency_var * hash21(vec2<f32>(i * 1.23, seed)));
        let angle = 2.0 * PI * hash21(vec2<f32>(i * 0.67, seed));
        let kernelDir = vec2<f32>(cos(angle), sin(angle));
        let kernelContribution = gabor(uv - kernelPos, kernelFreq, kernelDir, phase);
        noise += kernelContribution;
    }
    // Normalize and map to [0, 1] range
    return 0.5 + 0.5 * (noise / sqrt(params.num_kernels));
}
fn r(v: vec2<f32>) -> vec2<f32> { 
    return vec2<f32>(floor(v.x * params.width) / params.width, floor(v.y * params.height) / params.height); 
}
fn gamma(color: vec3<f32>, gamma_value: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma_value));
}
@fragment
fn fs_main(@builtin(position) fragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let dimensions = u_resolution.dimensions;
    let ndc = fragCoord.xy / dimensions;
    let uv = r(ndc);
    let col = textureSample(tex, tex_sampler, uv).rgb;
    let c = round((col.r + col.g + col.b) / 3.0 * params.steps) / params.steps;
    let val = gaborNoise(r(vec2<f32>((uv.x + c * u_time.time * params.animation_speed) % 1.0, uv.y)));
    var color: vec3<f32>;
    if (params.gamma == 1.0) {
        color = vec3<f32>(val);
    } else {
        color = gamma(vec3<f32>(val), params.gamma);
    }
    return vec4<f32>(color, 1.0);
}