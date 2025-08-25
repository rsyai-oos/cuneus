//Enes Altun 2025; MIT License
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct RorschachParams {
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
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    brightness: f32,
    exposure: f32,
    gamma: f32,
    particle_count: f32,
    scale: f32,
    dof_amount: f32,
    dof_focal_dist: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

@group(1) @binding(1) var<uniform> params: RorschachParams;

@group(2) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m2 = mat2x2<f32>;
alias m3 = mat3x3<f32>;
alias m4 = mat4x4<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;

var<private> R: v2;
var<private> seed: u32;

fn rot(a: f32) -> m2 { 
    return m2(cos(a), -sin(a), sin(a), cos(a)); 
}

fn rotX(a: f32) -> m3 {
    let r = rot(a); 
    return m3(1.0, 0.0, 0.0, 0.0, r[0][0], r[0][1], 0.0, r[1][0], r[1][1]);
}

fn rotY(a: f32) -> m3 {
    let r = rot(a); 
    return m3(r[0][0], 0.0, r[0][1], 0.0, 1.0, 0.0, r[1][0], 0.0, r[1][1]);
}

fn rotZ(a: f32) -> m3 {
    let r = rot(a); 
    return m3(r[0][0], r[0][1], 0.0, r[1][0], r[1][1], 0.0, 0.0, 0.0, 1.0);
}

fn hash_u(_a: u32) -> u32 { 
    var a = _a; 
    a ^= a >> 16;
    a *= 0x7feb352du;
    a ^= a >> 15;
    a *= 0x846ca68bu;
    a ^= a >> 16;
    return a; 
}

fn hash_f() -> f32 { 
    var s = hash_u(seed); 
    seed = s;
    return (f32(s) / f32(0xffffffffu)); 
}

fn hash_v2() -> v2 { 
    return v2(hash_f(), hash_f()); 
}

fn hash_v3() -> v3 { 
    return v3(hash_f(), hash_f(), hash_f()); 
}

fn sin_add(a: f32) -> f32 {
    return sin(a) * 0.5 + 0.5;
}

fn hash11(p: f32) -> f32 {
    var p_mut = fract(p * 0.1031);
    p_mut *= p_mut + 33.33;
    p_mut *= p_mut + p_mut;
    return fract(p_mut);
}

fn sample_disk() -> v2 {
    let r = hash_v2();
    return v2(sin(r.x * tau), cos(r.x * tau)) * sqrt(r.y);
}

fn create_matrices() -> array<m3, 5> {
    return array<m3, 5>(
        m3(
            params.m1_scale, 0.0, 0.0,
            0.0, params.m1_y_scale, 0.0,
            0.0, 0.0, 1.0
        ),
        m3(
            params.m2_scale, -params.m2_shear, 0.0,
            params.m2_shift, params.m2_scale, 0.0,
            -0.4, params.m2_shear, 1.0 
        ),
        m3(
            params.m3_scale, params.m3_shear, 0.0,
            -params.m3_shift, 0.7, 0.0,
            0.4, params.m3_shear, 1.0 
        ),
        m3(
            params.m4_scale, 0.0, 0.0,
            0.0, 0.7, 0.0,
            0.0, -params.m4_shift, 1.0
        ),
        m3(
            params.m5_scale, 0.0, 0.0,
            0.0, params.m5_scale, 0.0,
            0.0, params.m5_shift, 1.0
        )
    );
}

fn aces_tonemap(color: v3) -> v3 {
    const m1 = mat3x3<f32>(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    const m2 = mat3x3<f32>(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );

    var v = m1 * color;    
    var a = v * (v + 0.0245786) - 0.000090537;
    var b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return m2 * (a / b);
}

fn gamma_correction(color: v3, gamma: f32) -> v3 {
    return pow(max(color, v3(0.0)), v3(1.0 / gamma));
}

@compute @workgroup_size(256, 1, 1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let Ru = vec2<u32>(textureDimensions(output));
    R = v2(Ru);
    
    let muv = v2(params.rotation_x, params.rotation_y);
    seed = id.x + hash_u(time_data.frame);
    
    let particle_count = u32(params.particle_count);
    if (id.x >= particle_count) {
        return;
    }
    
    let M = create_matrices();
    let iterations = 200 + i32(sin_add(time_data.time * params.time_scale * 0.2) * 100.0);
    
    var p = v3(1.0 + hash_f() * 0.1, 1.0 + hash_f() * 0.1, 1.0);
    
    let particle_offset = hash_f() * 100.0;
    let base_time = time_data.time * params.time_scale * 2.0 + particle_offset;
    
    var focusDist = params.dof_focal_dist * 2.0 - 1.0;
    let dofFac = 1.0 / v2(R.x/R.y, 1.0) * params.dof_amount;
    
    for(var i = 0; i < iterations; i++) {
        let r = hash11(f32(i) + base_time);
        var j = i32(r * 4.0 * base_time) & 3;
        if (r > 0.95) {
            j += 1;
        }
        
        p = M[j] * p;
        
        if (i < 50) {
            continue;
        }
        
        var view_p = p * params.scale;
        view_p = rotX(muv.y * pi) * view_p;
        view_p = rotY(muv.x * tau) * view_p;
        view_p = rotZ(sin(time_data.time * params.time_scale * 0.3) * 0.2) * view_p;
        
        let aspect = R.x / R.y;
        var screen_pos = view_p.xy;
        screen_pos.x /= aspect;
        
        let d = view_p.z - focusDist;
        screen_pos += sample_disk() * abs(d) * 0.05 * dofFac;
        
        var uv = screen_pos * 0.5 + 0.5;
        uv.y = 1.0 - uv.y;
        
        if (uv.x >= 0.0 && uv.x < 1.0 && uv.y >= 0.0 && uv.y < 1.0) {
            let coords = vec2<u32>(uv * R);
            let idx = coords.x + Ru.x * coords.y;
            
            if (idx < Ru.x * Ru.y) {
                let intensity = u32(params.exposure * 100.0);
                let phase = f32(i) / f32(iterations);
                let color_shift = sin(phase * pi * 2.0 + time_data.time * 0.5) * 0.5 + 0.5;
                
                atomicAdd(&atomic_buffer[idx], u32((1.0 - color_shift) * f32(intensity)));
                atomicAdd(&atomic_buffer[idx + Ru.x * Ru.y], u32(color_shift * f32(intensity)));
                
                let mirror_uv = v2(1.0 - uv.x, uv.y);
                if (mirror_uv.x >= 0.0 && mirror_uv.x < 1.0 && mirror_uv.y >= 0.0 && mirror_uv.y < 1.0) {
                    let mirror_coords = vec2<u32>(mirror_uv * R);
                    let mirror_idx = mirror_coords.x + Ru.x * mirror_coords.y;
                    
                    if (mirror_idx < Ru.x * Ru.y) {
                        atomicAdd(&atomic_buffer[mirror_idx], u32((1.0 - color_shift) * f32(intensity)));
                        atomicAdd(&atomic_buffer[mirror_idx + Ru.x * Ru.y], u32(color_shift * f32(intensity)));
                    }
                }
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }
    
    let hist_id = id.x + u32(res.x) * id.y;
    let col1 = f32(atomicLoad(&atomic_buffer[hist_id])) * v3(params.color1_r, params.color1_g, params.color1_b);
    let col2 = f32(atomicLoad(&atomic_buffer[hist_id + res.x * res.y])) * v3(params.color2_r, params.color2_g, params.color2_b);
    var col = (col1 + col2) * params.brightness;
    
    let uv = v2(f32(id.x), f32(id.y)) / v2(f32(res.x), f32(res.y));
    let center_dist = length(uv - 0.5);
    
    col *= (1.0 + center_dist * 0.2);
    
    let time_color = sin(time_data.time * params.time_scale * 0.5 + center_dist * pi) * 0.1;
    col += v3(time_color * 0.1, time_color * 0.05, -time_color * 0.1);
    
    col = aces_tonemap(col);
    
    col = gamma_correction(col, params.gamma);
    
    textureStore(output, vec2<i32>(id.xy), v4(col, 1.0));
    
    atomicStore(&atomic_buffer[hist_id], 0u);
    atomicStore(&atomic_buffer[hist_id + res.x * res.y], 0u);
}