// Inspired by neural wave patterns
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct NeuralParams {
    detail: f32,             
    animation_speed: f32,    
    pattern: f32,         
    structure_smoothness: f32,
    saturation: f32,         
    base_rotation: f32,      
    rot_variation: f32,      
    rotation_x: f32,         
    rotation_y: f32,         
    click_state: i32,        
    brightness_mult: f32,    
    color1_r: f32,           
    color1_g: f32,           
    color1_b: f32,           
    color2_r: f32,           
    color2_g: f32,           
    color2_b: f32,           
    dof_amount: f32,         
    dof_focal_dist: f32,     
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

@group(2) @binding(0) var<uniform> params: NeuralParams;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

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

// Rodrigues
fn erot(p: v3, ax: v3, ro: f32) -> v3 {
    return mix(dot(p, ax) * ax, p, cos(ro)) + sin(ro) * cross(ax, p);
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

fn projParticle(p: v3) -> v3 {
    var projected = p;
    projected.x /= R.x/R.y;
    return projected;
}

fn neural_wave_point(t: f32, seed_offset: f32) -> v3 {
    var p = v2(hash_f() * 2.0 - 1.0, hash_f() * 2.0 - 1.0) * 1.5;
    let dist_squared = dot(p, p);
    
    var S = 15.0;
    var a = 0.0;
    var n = v2(0.0);
    var q = v2(0.0);
    
    let time_factor = t * params.animation_speed;
    let detail = i32(params.detail);
    
    let axis = normalize(v3(0.0, 0.0, 1.0));
    
    for (var j = 1; j < detail; j++) {

        let rot_angle = params.base_rotation + sin(t) * params.rot_variation;
        var rotatedP = erot(v3(p, 0.0), axis, rot_angle);
        p = rotatedP.xy;
        
        n = erot(v3(n, 0.0), axis, rot_angle).xy;
        
        q = p * S + time_factor + 
            sin(time_factor - dist_squared * 0.0) * 2.0 + 
            f32(j) + n;
        
        a += dot(cos(q) / S, v2(params.saturation));
        
        n -= sin(q*2.0);
        
        S *= params.structure_smoothness;
    }
    
    let result = params.pattern * ((a + 0.3) + a + a);
    
    let x = p.x * 0.7;
    let y = p.y * 0.71;
    let z = result * 0.2;
    
    let wave = cos(5.0 * t + (p.y * p.x * pi)) * 0.3;
    
    return v3(x, y, z + wave);
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

//Generate and splat particles
@compute @workgroup_size(256, 1, 1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let Ru = vec2<u32>(textureDimensions(output));
    R = v2(Ru);
    let muv = v2(params.rotation_x, params.rotation_y);
    seed = id.x + hash_u(time_data.frame);
    
    let n_particles = 15 + i32(sin_add(time_data.time * 0.2) * 15.0);
    
    var focusDist = 0.48 * 2.0 - 1.0;
    if (params.click_state > 0) {
        focusDist = muv.y * 2.0;
    }
    let dofFac = 1.0 / v2(R.x/R.y, 1.0) * params.dof_amount;
    
    for(var i = 0; i < n_particles; i++) {
        let seed_offset = f32(i) / f32(n_particles);
        var p = neural_wave_point(time_data.time, seed_offset);
        p = rotY(muv.x * 2.0) * p;
        p = rotX(muv.y * 1.0) * p;
        p = rotZ(sin(time_data.time * 0.1) * 0.5) * p;
        var q = projParticle(p);
        
        var k = q.xy;
        let d = q.z - focusDist;
        k += sample_disk() * abs(d) * 0.08 * dofFac;
        
        let uv = k.xy/params.dof_focal_dist + 0.5;
        let cc = vec2<u32>(uv.xy * R.xy);
        let idx = cc.x + Ru.x * cc.y;
        
        let r = hash_f() * 10.8 + 10.2;
        
        let wave = cos(5.0 * time_data.time + (p.y * p.x * pi));
        let color_shift = 0.5 + 0.3 * sin(time_data.time + wave * 2.0);
        
        if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0 && idx < u32(Ru.x*Ru.y)) {
            atomicAdd(&atomic_buffer[idx], u32((1.0 - color_shift) * 100.0));
            atomicAdd(&atomic_buffer[idx + Ru.x*Ru.y], u32(color_shift * 150.0 + f32(i % 5) * 20.0));
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }
    let hist_id = id.x + u32(res.x) * id.y;
    
    var col1 = f32(atomicLoad(&atomic_buffer[hist_id])) * vec3<f32>(params.color1_r, params.color1_g, params.color1_b);
    var col2 = f32(atomicLoad(&atomic_buffer[hist_id + res.x*res.y])) * vec3<f32>(params.color2_r, params.color2_g, params.color2_b);
    var col = (col1 + col2) * params.brightness_mult;
    
    let t = time_data.time;
    let wave = sin(t * 0.5) * 0.5 + 0.5;
    
    col = mix(
        col,
        col * vec3<f32>(
            1.3 + 0.5 * sin(2.0 * t + wave),
            1.3 + 0.5 * sin(2.0 * t + 2.0 * pi / 3.0 + 2.0 * wave),
            1.3 + 0.5 * sin(2.0 * t + 4.0 * pi / 3.0 + 2.0 * wave)
        ),
        0.3
    );
    
    col = aces_tonemap(col);
    
    col += vec3<f32>(0.0, 0.001, 0.003);
    
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
    

    atomicStore(&atomic_buffer[hist_id], 0u);
    atomicStore(&atomic_buffer[hist_id + res.x*res.y], 0u);
}