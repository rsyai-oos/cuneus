@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<i32>>;

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
    rotation_x: f32,
    rotation_y: f32,
    rotation_z: f32,
    rotation_speed: f32,
    attractor_a: f32,
    attractor_b: f32,
    attractor_c: f32,
    attractor_d: f32,
    attractor_animate_amount: f32,
}
@group(2) @binding(0)
var<uniform> params: Params;


const M: array<mat3x3<f32>, 5> = array<mat3x3<f32>, 5>(
    mat3x3<f32>(
        0.8, 0.0, 0.0,
        0.0, 0.5, 0.0,
        0.0, 0.0, 1.0
    ),
    mat3x3<f32>(
        0.4, -0.2, 0.0,
        0.3, 0.3, 0.0,
        -0.4, 0.2, 1.0
    ),
    mat3x3<f32>(
        0.4, 0.2, 0.0,
        -0.3, 0.3, 0.0,
        0.4, 0.2, 1.0
    ),
    mat3x3<f32>(
        0.3, 0.0, 0.0,
        0.0, 0.3, 0.0,
        0.0, -0.2, 1.0
    ),
    mat3x3<f32>(
        0.2, 0.0, 0.0,
        0.0, 0.2, 0.0,
        0.0, 0.4, 1.0
    )
);

fn hash11(p: f32) -> f32 {
    var p_mut = fract(p * 0.1031);
    p_mut *= p_mut + 33.33;
    p_mut *= p_mut + p_mut;
    return fract(p_mut);
}

fn gamma_correction(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
}

@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(prev_frame));
    let aspect = dimensions.x / dimensions.y;
    var u = tex_coords * 2.0 - 1.0;
    u.x *= aspect;
    u.y = -u.y;

    let prev = textureSample(prev_frame, tex_sampler, tex_coords);
    var output = prev;
    var p = vec3<f32>(1.0);
    var d = 9.0;
    
    for(var i: f32 = 0.0; i < 200.0; i += 1.0) {
        let r = hash11(i + time_data.time * 0.1);
        var j = i32(r * 4.0) & 3;
        if (r > 0.95) {
            j += 1;
        }
        
        p = M[j] * p;
        
        let d1 = length(p.xy - u);
        let d2 = length(vec2<f32>(-p.x, p.y) - u);
        d = min(d, min(d1, d2));
    }
    
    output.x += exp(-800.0 * d) * 0.5;
    output.y += exp(-900.0 * d) * 0.4;
    output.z += exp(-700.0 * d) * 0.3;
    
    return mix(output, prev, 0.9);
}

@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(prev_frame, tex_sampler, tex_coords);
    var final_color = vec3<f32>(0.7 - log(1.0 + color.xxx) * 0.3);
    final_color = gamma_correction(final_color, 0.412);
    return vec4<f32>(final_color, 1.0);
}