struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct LorenzParams {
    sigma: f32,          
    rho: f32,            
    beta: f32,           
    step_size: f32,      
    motion_speed: f32,  
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
    scale: f32,          
    dof_amount: f32,     
    dof_focal_dist: f32,
    gamma: f32,
    exposure: f32,
    particle_count: f32,
    decay_speed: f32,
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

@group(2) @binding(0) var<uniform> params: LorenzParams;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

alias iv4 = vec4<i32>;
alias iv3 = vec3<i32>;
alias iv2 = vec2<i32>;
alias uv4 = vec4<u32>;
alias uv3 = vec3<u32>;
alias uv2 = vec2<u32>;
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
var<private> muv: v2;

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

const COL_CNT: i32 = 4;
var<private> kCols = array<v3, 4>( 
     vec3(1.0, 0.6, 0.2), vec3(0.2, 0.6, 1.0),
     vec3(1.0, 1.0, 1.0) * 1.2, vec3(1.0, 0.4, 0.8) * 1.1
);

fn mix_cols(_idx: f32) -> v3 {
    let idx = _idx % 1.0;
    var cols_idx = i32(idx * f32(COL_CNT));
    var fract_idx = fract(idx * f32(COL_CNT));
    fract_idx = smoothstep(0.0, 1.0, fract_idx);
    return mix(kCols[cols_idx], kCols[(cols_idx + 1) % COL_CNT], fract_idx);
}

fn get_cyclic_params(t: f32) -> vec3<f32> {
    let speed = params.motion_speed * 0.1;
    
    let sigma = params.sigma * (1.0 + 0.1 * sin(t * 0.27 * speed));
    let rho = params.rho * (1.0 + 0.05 * sin(t * 0.31 * speed));
    let beta = params.beta * (1.0 + 0.05 * sin(t * 0.41 * speed));
    
    return vec3<f32>(sigma, rho, beta);
}

fn lorenz_step(p: v3, lorenz_params: vec3<f32>, dt: f32) -> v3 {
    let sigma = lorenz_params.x;
    let rho = lorenz_params.y;
    let beta = lorenz_params.z;
    
    let dx = sigma * (p.y - p.x);
    let dy = p.x * (rho - p.z) - p.y;
    let dz = p.x * p.y - beta * p.z;
    
    return p + vec3<f32>(dx, dy, dz) * dt;
}

fn projParticle(_p: v3) -> v3 {
    var p = _p;
    
    p = rotY(muv.x * 2.0) * p;
    p = rotX(muv.y * 2.0) * p;
    
    p = rotZ(sin(time_data.time * 0.05) * 0.2) * p;
    
    // perspective projection
    p.z += 4.0;
    p /= p.z * 0.4;
    p.z = _p.z;
    
    p.x /= R.x / R.y;
    
    return p;
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


@compute @workgroup_size(256, 1, 1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let Ru = vec2<u32>(textureDimensions(output));
    R = v2(Ru);
    muv = v2(params.rotation_x, params.rotation_y);
    seed = hash_u(id.x + hash_u(Ru.x * id.y * 200u) * 20u + hash_u(id.x) * 250u + hash_u(time_data.frame));
    seed = hash_u(seed);
    
    let particleIdx = id.x;
    
    let iters = 200 + i32(sin_add(time_data.time * 0.1) * 100.0);
    
    var t = time_data.time * 0.3 - hash_f() * 1.0/30.0;
    
    // DOF setup
    var focusDist = params.dof_focal_dist * 2.0 - 1.0;
    if (params.click_state > 0) {
        focusDist = muv.y * 2.0;
    }
    let dofFac = 1.0 / v2(R.x/R.y, 1.0) * params.dof_amount;
    
    let lorenz_params = get_cyclic_params(t);
    
    // Proper Lorenz starting positions with small variations
    var p = v3(
        0.1 + (hash_f() * 2.0 - 1.0) * 0.1,
        0.001 + (hash_f() * 2.0 - 1.0) * 0.1,
        (hash_f() * 2.0 - 1.0) * 0.1
    );
    
    let dt = params.step_size * 0.5;
    
    // Burn-in to get onto the attractor
    for(var i = 0; i < 50; i++) {
        p = lorenz_step(p, lorenz_params, dt);
    }
    
    // Main splatting loop
    for(var i = 0; i < iters; i++) {
        p = lorenz_step(p, lorenz_params, dt);
        
        // Skip more iterations for sparser distribution
        if (i % 5 != 0) {
            continue;
        }
        
        // Transform and scale
        var view_p = p * params.scale;
        
        var q = projParticle(view_p);
        
        // DOF effect
        var k = q.xy;
        let d = q.z - focusDist;
        k += sample_disk() * abs(d) * 0.05 * dofFac;
        
        // Map to screen coordinates
        let uv = k.xy/2.0 + 0.5;
        let cc = uv2(uv.xy * R.xy);
        let idx = cc.x + Ru.x * cc.y;
        
        // Simple splatting - just accumulate hits
        if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0 && idx < u32(Ru.x*Ru.y)) {
            atomicAdd(&atomic_buffer[idx], 1u);
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }
    
    R = v2(res);
    let hist_id = id.x + u32(res.x) * id.y;
    
    // Get particle density
    var col = f32(atomicLoad(&atomic_buffer[hist_id])) * vec3<f32>(1.0);
    
    // logarithmic tone mapping
    let sc = 124452.7;
    col = log(col * params.brightness * 50.0 + 1.0) / log(sc);
    
    col = smoothstep(v3(0.0), v3(1.0), col * mix_cols(col.x * 0.8));
    
    let user_color1 = vec3<f32>(params.color1_r, params.color1_g, params.color1_b);
    let user_color2 = vec3<f32>(params.color2_r, params.color2_g, params.color2_b);
    let color_blend = sin(time_data.time * 0.1) * 0.5 + 0.5;
    let user_color = mix(user_color1, user_color2, color_blend);
    col = mix(col, col * user_color, 0.8);
    
    col *= params.exposure * 0.3;
    
    col = pow(col, vec3<f32>(1.0 / params.gamma));
    
    let time_shift = sin(time_data.time * 0.3) * 0.1;
    col = mix(
        col,
        col * vec3<f32>(1.0 + time_shift, 1.0, 1.0 - time_shift),
        0.2
    );
    
    col += vec3<f32>(0.001, 0.001, 0.003);
    
    col = clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
    
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
    
    let current_val = atomicLoad(&atomic_buffer[hist_id]);
    if (current_val > 0u) {
        let decay_rate = max(1u, current_val / u32(params.decay_speed));
        let decayed = max(0u, current_val - decay_rate);
        atomicStore(&atomic_buffer[hist_id], decayed);
    }
}