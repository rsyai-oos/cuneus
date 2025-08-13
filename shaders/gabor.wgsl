struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct GaborParams {
    frequency: f32,        
    orientation: f32,
    phase: f32, 
    speed:f32,
    sigma_x: f32,         
    sigma_y: f32,  
    amplitude: f32,        
    aspect_ratio: f32,     
    z_scale: f32,          
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
    dof_amount: f32,       
    dof_focal_dist: f32,   
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

@group(2) @binding(0) var<uniform> params: GaborParams;
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

fn gabor(p: v2) -> f32 {
    let rotated = m2(
        cos(params.orientation), sin(params.orientation),
        -sin(params.orientation), cos(params.orientation)
    ) * p;
    
    let gaussian = exp(-(
        pow(rotated.x / params.sigma_x, 2.0) + 
        pow(rotated.y / params.sigma_y, 2.0)
    ) / 2.0);
    
    let animated_phase = params.phase + time_data.time * params.speed;
    
    let sinusoidal = cos(params.frequency * rotated.x * tau + animated_phase);
    
    return params.amplitude * gaussian * sinusoidal;
}

fn projParticle(p: v3) -> v3 {
    var projected = p;
    projected.x /= R.x/R.y;
    return projected;
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
    let muv = v2(params.rotation_x, params.rotation_y);
    seed = id.x + hash_u(time_data.frame);
    
    let iters = 200 + i32(sin_add(time_data.time * 0.2) * 100.0);
    let base_time = time_data.time * 0.5 + hash_f() * 0.1;
    
    var focusDist = params.dof_focal_dist * 2.0 - 1.0;
    if (params.click_state > 0) {
        focusDist = muv.y * 2.0;
    }
    let dofFac = 1.0 / v2(R.x/R.y, 1.0) * params.dof_amount;
    
    let range = 3.0; 
    
    for(var i = 0; i < iters; i++) {
        let x = (hash_f() * 2.0 - 1.0) * range;
        let y = (hash_f() * 2.0 - 1.0) * range * params.aspect_ratio;
        let g = gabor(v2(x, y));
        let z = g * params.z_scale;
        if (abs(g) < 0.05) {
            continue;
        }
        var p = v3(x, y, z);
        p = rotY(muv.x * 2.0) * p;
        p = rotX(muv.y * 1.0) * p;
        p = rotZ(sin(time_data.time * 0.1) * 0.3) * p;
        var q = projParticle(p);
        var k = q.xy;
        let d = q.z - focusDist;
        k += sample_disk() * abs(d) * 0.08 * dofFac;
        let uv = k.xy/2.0 + 0.5;
        let cc = vec2<u32>(uv.xy * R.xy);
        let idx = cc.x + Ru.x * cc.y;
        let intensity = abs(g);
        let color_shift = (g + 1.0) * 0.5;
        if (uv.x > 0.0 && uv.x < 1.0 && uv.y > 0.0 && uv.y < 1.0 && idx < u32(Ru.x*Ru.y)) {
            if (g > 0.0) {
                atomicAdd(&atomic_buffer[idx], u32(intensity * 100.0));
            } else {
                atomicAdd(&atomic_buffer[idx + Ru.x*Ru.y], u32(intensity * 100.0));
            }
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
    
    atomicStore(&atomic_buffer[hist_id], 0u);
    atomicStore(&atomic_buffer[hist_id + res.x*res.y], 0u);
}