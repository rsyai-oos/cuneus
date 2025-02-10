//Enes Altun 2025; MIT License
@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<i32>>;

struct TimeUniform {
    time: f32,
};
@group(1) @binding(0)
var<uniform> time_data: TimeUniform;

struct Params {
    // Matrix 1
    m1_scale: f32,
    m1_y_scale: f32,
    // Matrix 2
    m2_scale: f32,
    m2_shear: f32,
    m2_shift: f32,
    // Matrix 3
    m3_scale: f32,
    m3_shear: f32,
    m3_shift: f32,
    // Matrix 4
    m4_scale: f32,
    m4_shift: f32,
    // Matrix 5
    m5_scale: f32,
    m5_shift: f32,
    time_scale: f32,
    decay: f32,
    intensity: f32,
}
@group(2) @binding(0)
var<uniform> params: Params;

fn create_matrices() -> array<mat3x3<f32>, 5> {
    return array<mat3x3<f32>, 5>(
        mat3x3<f32>(
            params.m1_scale, 0.0, 0.0,
            0.0, params.m1_y_scale, 0.0,
            0.0, 0.0, 1.0
        ),
        mat3x3<f32>(
            params.m2_scale, -params.m2_shear, 0.0,
            params.m2_shift, params.m2_scale, 0.0,
            -0.4, params.m2_shear, 1.0 
        ),
        mat3x3<f32>(
            params.m3_scale, params.m3_shear, 0.0,
            -params.m3_shift, 0.7, 0.0,
            0.4, params.m3_shear, 1.0 
        ),
        mat3x3<f32>(
            params.m4_scale, 0.0, 0.0,
            0.0, 0.7, 0.0,
            0.0, -params.m4_shift, 1.0
        ),
        mat3x3<f32>(
            params.m5_scale, 0.0, 0.0,
            0.0, params.m5_scale, 0.0,
            0.0, params.m5_shift, 1.0
        )
    );
}

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
    let M = create_matrices();
    
    for(var i: f32 = 0.0; i < 200.0; i += 1.0) {
        let r = hash11(i + time_data.time * params.time_scale);
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
    let fade_factor = pow(1.0 - params.intensity, 3.0);
    output = mix(vec4<f32>(0.0), output, fade_factor);
    return mix(output, prev * fade_factor, params.decay);
}
@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(prev_frame, tex_sampler, tex_coords);
    var final_color = vec3<f32>(0.7 - log(1.0 + color.xxx) * 1.3);
    final_color = gamma_correction(final_color, 0.2);
    return vec4<f32>(final_color, 1.0);
}