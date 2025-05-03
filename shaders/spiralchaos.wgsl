// Spiral Chaos Game
// Inspirations: https://compute.toys/view/1512 sintel, 2024; https://compute.toys/view/252 wrighter, 2024 

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

// Parameters uniform
struct SpiralParams {
    a: f32,              
    b: f32,              
    c: f32,              
    dof_amount: f32,     
    dof_focal_dist: f32,
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    brightness: f32,     
    color1_r: f32,       
    color1_g: f32,       
    color1_b: f32,       
    color2_r: f32,       
    color2_g: f32,       
    color2_b: f32,       
    _padding: u32,
}
@group(1) @binding(0) var<uniform> params: SpiralParams;

// Storage textures
@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

// Aliases for common types
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

fn sample_disk() -> v2 {
    let r = hash_v2();
    return v2(sin(r.x * tau), cos(r.x * tau)) * sqrt(r.y);
}

const golden_angle = 2.39996;
fn spiral_point(t: f32, radius_factor: f32, z_factor: f32) -> v3 {
    let a = 0.1;
    let b = 0.1 + params.a * 0.2;
    
    let radius = a * exp(b * t);
    let x = radius * cos(t);
    let y = radius * sin(t);
    let z = z_factor * sin(t * 0.5);
    
    return v3(x, y, z) * radius_factor;
}

fn projParticle(_p: v3) -> v3 {
    var p = _p;
    p.x /= R.x/R.y;
    return p;
}

// ACES tonemapping function for better color reproduction
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

// First compute pass: Generate and splat particles
@compute @workgroup_size(256, 1, 1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let Ru = vec2<u32>(textureDimensions(output));
    R = v2(Ru);
    
    // Virtual mouse positions
    let muv = v2(params.rotation_x, params.rotation_y);
    seed = id.x + hash_u(time_data.frame);
    // Number of iterations per particle
    let iters = 120 + i32(sin_add(time_data.time * 0.2) * 80.0);
    // Base time value with some variation per particle
    let base_time = time_data.time * (0.2 + params.b * 0.3) + hash_f() * 0.1;
    // DOF parameters
    var focusDist = params.dof_focal_dist * 2.0 - 1.0;
    if (params.click_state > 0) {
        focusDist = muv.y * 2.0;
    }
    let dofFac = 1.0 / v2(R.x/R.y, 1.0) * params.dof_amount;
    let num_arms = i32(3.0 + params.c * 5.0);
    let arm_idx = i32(hash_f() * f32(num_arms));
    let arm_offset = f32(arm_idx) * tau / f32(num_arms);
    // Initial point position
    var p = v3(0.0, 0.0, 0.0);
    for(var i = 0; i < iters; i++) {
        let r = hash_f();
        let t = base_time + f32(i) * 0.1;
        let angle = t * 2.0 + arm_offset;
        let spiral_p = spiral_point(angle, 0.8, 0.3);
        var next_p = spiral_p;
        next_p = rotZ(sin(time_data.time * 0.3) * 0.2) * next_p;
        next_p = rotY(time_data.time * 0.1) * next_p;
        
        // Mix with previous point with some randomness
        p = mix(p, next_p, 0.2 + 0.1 * sin(time_data.time * 0.5));
        
        // Add some chaos sometimes
        if (i % 5 == 0 && r < 0.3) {
            p += v3(sin(t), cos(t), sin(t * 0.7)) * 0.1;
            p = rotX(angle * 0.05) * p;
        }
        
        // Prepare for projection
        var q = p;
        

        q = rotY(muv.x * 2.0) * q;
        q = rotX(muv.y * 1.0) * q;
        
        q = rotZ(sin(time_data.time * 0.1) * 0.5) * q;
        
        q = projParticle(q);
        
        // depth of field effect
        var k = q.xy;
        let d = q.z - focusDist;
        k += sample_disk() * abs(d) * 0.08 * dofFac;
        
        // Map to screen coordinates
        let uv = k.xy/2.0 + 0.5;
        let cc = vec2<u32>(uv.xy * R.xy);
        let idx = cc.x + Ru.x * cc.y;
        
        // Store point if within screen bounds and accumulate point with color data
        if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0 && idx < u32(Ru.x*Ru.y)) {
            atomicAdd(&atomic_buffer[idx], u32((1.0 - r) * 100.0));
            atomicAdd(&atomic_buffer[idx + Ru.x*Ru.y], u32(r * 150.0 + f32(arm_idx) * 20.0));
        }
    }
}

// Second compute pass: Render accumulated points to screen
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }
    let hist_id = id.x + u32(res.x) * id.y;
    var col1 = f32(atomicLoad(&atomic_buffer[hist_id])) * vec3<f32>(params.color1_r, params.color1_g, params.color1_b);
    var col2 = f32(atomicLoad(&atomic_buffer[hist_id + res.x*res.y])) * vec3<f32>(params.color2_r, params.color2_g, params.color2_b);
    var col = (col1 + col2) * params.brightness;
    let color_shift = sin(time_data.time * 0.2) * 0.2;
    col = mix(
        col,
        col * vec3<f32>(1.0 + color_shift, 1.0, 1.0 - color_shift),
        0.3
    );
    col = aces_tonemap(col);
    
    col += vec3<f32>(0.001, 0.001, 0.003);
    
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
    
    // Clear accumulation buffer for next frame
    atomicStore(&atomic_buffer[hist_id], 0u);
    atomicStore(&atomic_buffer[hist_id + res.x*res.y], 0u);
}