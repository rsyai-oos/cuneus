// inspired by https://www.shadertoy.com/view/3sl3WH
@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

struct TimeUniform {
    time: f32,
};
@group(1) @binding(0)
var<uniform> time_data: TimeUniform;

struct Params {
    cloud_density: f32,
    lightning_intensity: f32,
    branch_count: f32,
    feedback_decay: f32,
};
@group(2) @binding(0)
var<uniform> params: Params;

// Utility functions
fn IHash(a: i32) -> i32 {
    var x = a;
    x = (x ^ 61) ^ (x >> 16);
    x = x + (x << 3);
    x = x ^ (x >> 4);
    x = x * 0x27d4eb;
    x = x ^ (x >> 15);
    return x;
}

fn Hash(a: i32) -> f32 {
    return f32(IHash(a)) / f32(0x7FFFFFFF);
}

fn rand2(seed: i32) -> vec2<f32> {
    return vec2<f32>(
        Hash(seed ^ 348593),
        Hash(seed ^ 859375)
    );
}

fn randn(randuniform: vec2<f32>) -> vec2<f32> {
    var r = randuniform;
    r.x = sqrt(-2.0 * log(1e-9 + abs(r.x)));
    r.y = r.y * 6.28318;
    return r.x * vec2<f32>(cos(r.y), sin(r.y));
}

fn line_dist(a: vec2<f32>, b: vec2<f32>, uv: vec2<f32>, thickness: f32) -> f32 {
    let dist = length(uv-(a+normalize(b-a)*min(length(b-a),max(0.0,dot(normalize(b-a),(uv-a))))));
    return dist / thickness;
}

// Pass 1: Lightning Generation
@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(prev_frame));
    let uv = (FragCoord.xy * 2.0 - dimensions.xy) / dimensions.y;
    
    var ds = 1e4;
    let anim_frame = i32(time_data.time * 10.0);
    let lightning_color = vec3<f32>(0.7, 0.9, 1.0);
    let branch_count = i32(4.0 + params.branch_count * 3.0);
    
    for(var q = 0; q < branch_count; q = q + 1) {
        var seed = anim_frame * (q + 1);
        
        var a = vec2<f32>(0.0, 1.0);
        var b = vec2<f32>(0.0, 0.7) + 0.4 * randn(rand2(seed)) / 8.0;
        
        for(var k = 0; k < 30; k = k + 1) {
            let l = length(b - a);
            let r = randn(rand2(seed));
            
            let c = (a + b) * 0.5 + l * r / 8.0;
            let d = b * 1.9 - a * 0.9 + l * randn(rand2(seed + 1)) / 4.0;
            let e = b * 1.9 - a * 0.9 + l * randn(rand2(seed + 2)) / 4.0;
            
            let thickness = 2.0;
            
            let d0 = line_dist(a, c, uv, thickness);
            let d1 = line_dist(c, b, uv, thickness);
            let d2 = line_dist(b, d, uv, thickness);
            let d3 = line_dist(b, e, uv, thickness);
            
            if(d0 < min(d1, min(d2, d3))) {
                b = c;
                seed = seed + 1;
            } else if(d1 < min(d2, d3)) {
                a = c;
                seed = seed + 2;
            } else if(d2 < d3) {
                a = b;
                b = d;
                seed = seed + 3;
            } else {
                a = b;
                b = e;
                seed = seed + 4;
            }
        }
        
        ds = min(ds, line_dist(a, b, uv, 2.0));
    }
    
    let lightning_intensity = max(0.0, 1.0 - ds * dimensions.y / 2.0) * params.lightning_intensity;
    let lightning = lightning_color * lightning_intensity;
    
    let prev = textureSample(prev_frame, tex_sampler, tex_coords);
    let decay = params.feedback_decay * 0.95;
    
    return max(vec4<f32>(lightning, lightning_intensity), prev * decay);
}

// Pass 2: Just pass with additional decay
@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let previous = textureSample(prev_frame, tex_sampler, tex_coords);
    return previous * 0.98;
}

// Pass 3: Final with enhanced glow
@fragment
fn fs_pass3(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let lightning = textureSample(prev_frame, tex_sampler, tex_coords);
    let background_color = vec3<f32>(0.0, 0.0, 0.0);
    var final_color = lightning.rgb;
    
    // Enhanced glow effect
    let glow = length(lightning.rgb) * 0.3;
    final_color = final_color + vec3<f32>(0.3, 0.4, 0.5) * glow;
    
    return vec4<f32>(final_color, 1.0);
}