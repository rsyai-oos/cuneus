// Experimental Buddhabrot Compute Shader, Enes Altun, 2025
// A special rendering of the Mandelbrot set tracking escape trajectories
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct BuddhabrotParams {
    max_iterations: u32,     
    escape_radius: f32,      
    zoom: f32,               
    offset_x: f32,           
    offset_y: f32,           
    rotation: f32,           
    exposure: f32,           
    low_iterations: u32, 
    high_iterations: u32, 
    motion_speed: f32,       
    color1_r: f32,           
    color1_g: f32,           
    color1_b: f32,           
    color2_r: f32,           
    color2_g: f32,           
    color2_b: f32,           
    sample_density: f32,     
    dithering: f32,          
}
@group(1) @binding(0) var<uniform> params: BuddhabrotParams;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;


alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m2 = mat2x2<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;

var<private> R: v2;
var<private> seed: u32;

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

fn rot(a: f32) -> m2 { 
    return m2(cos(a), -sin(a), sin(a), cos(a)); 
}

fn screen_to_complex(uv: v2) -> v2 {
    var p = (uv - 0.5) * 2.0;
    p.x *= R.x / R.y;
    p = rot(params.rotation) * p;
    p = p / params.zoom + v2(params.offset_x, params.offset_y);
    
    return p;
}

fn cmul(a: v2, b: v2) -> v2 {
    return v2(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x
    );
}


fn sin_wave(value: f32, speed: f32, amplitude: f32, offset: f32) -> f32 {
    return sin(time_data.time * speed + offset) * amplitude + value;
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

fn escapes_within_range(c: v2, max_iters: u32) -> bool {
    var z = v2(0.0, 0.0);
    var n: u32 = 0;
    
    for (; n < max_iters; n++) {
        z = cmul(z, z) + c;
        if (dot(z, z) > params.escape_radius) {
            return n >= 20u && n < max_iters;
        }
    }
    return false;
}

fn complex_to_screen(p: v2) -> v2 {
    var uv = (p - v2(params.offset_x, params.offset_y)) * params.zoom;
    
    uv = rot(-params.rotation) * uv;
    
    uv.x /= R.x / R.y;
    return uv * 0.5 + 0.5;
}

@compute @workgroup_size(64, 1, 1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let Ru = vec2<u32>(textureDimensions(output));
    R = v2(Ru);
    seed = id.x + hash_u(time_data.frame);
    
    let actual_max_iters = params.max_iterations + u32(sin(time_data.time * 0.1) * 50.0);
    let samples_per_thread = 8u + u32(params.sample_density * 12.0);
    
    for (var s: u32 = 0u; s < samples_per_thread; s++) {
        var c: v2;
        let sample_strategy = (s + time_data.frame) % 3u;
        
        if (sample_strategy == 0u) {
            let angle = hash_f() * tau;
            let radius = 0.1 + hash_f() * 0.35;
            c = v2(
                cos(angle) * radius - 0.25,
                sin(angle) * radius
            );
        } else if (sample_strategy == 1u) {
            c = v2(
                hash_f() * 3.0 - 2.0,
                hash_f() * 2.5 - 1.25
            );
        } else {
            let angle = hash_f() * tau;
            let base_radius = 0.75 + hash_f() * 0.15;
            let distortion = 0.15 * (1.0 + cos(angle * 3.0));
            c = v2(
                cos(angle) * (base_radius + distortion) - 0.5,
                sin(angle) * base_radius
            );
        }
        if (!escapes_within_range(c, actual_max_iters)) {
            continue;
        }
        var z = v2(0.0, 0.0);
        var n: u32 = 0u;
        for (; n < actual_max_iters; n++) {
            z = cmul(z, z) + c;

            if (dot(z, z) > params.escape_radius) {
                break;
            }

            if (n < 15u || n % 2u != 0u) {
                continue;
            }

            if (abs(z.x) > 3.0 || abs(z.y) > 3.0) {
                continue;
            }

            let uv = complex_to_screen(z);

            if (uv.x >= 0.0 && uv.x < 1.0 && uv.y >= 0.0 && uv.y < 1.0) {
                let pixel_x = u32(uv.x * f32(Ru.x));
                let pixel_y = u32(uv.y * f32(Ru.y));
                let pixel_idx = pixel_x + Ru.x * pixel_y;

                if (pixel_idx < Ru.x * Ru.y) {
                    var buffer_offset: u32;
                    if (n >= params.high_iterations) {
                        buffer_offset = 0u;
                    } else if (n >= params.low_iterations) {
                        buffer_offset = Ru.x * Ru.y;
                    } else {
                        buffer_offset = 2u * Ru.x * Ru.y;
                    }
                    atomicAdd(&atomic_buffer[pixel_idx + buffer_offset], 1u);
                }
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }
    let idx = id.x + id.y * res.x;

    let count_r = f32(atomicLoad(&atomic_buffer[idx]));
    let count_g = f32(atomicLoad(&atomic_buffer[idx + res.x * res.y]));
    let count_b = f32(atomicLoad(&atomic_buffer[idx + 2u * res.x * res.y]));

    var col = v3(
        count_r * params.exposure * params.color1_r,
        count_g * params.exposure * params.color1_g,
        count_b * params.exposure * params.color1_b
    );
    

    let uv = v2(f32(id.x) / f32(res.x), f32(id.y) / f32(res.y));
    let dist_from_center = length(uv - 0.5) * 2.0;
    

    col = mix(
        col,
        v3(
            count_r * params.exposure * params.color2_r,
            count_g * params.exposure * params.color2_g,
            count_b * params.exposure * params.color2_b
        ),
        dist_from_center * 0.5
    );
    

    if (params.dithering > 0.0) {
        seed = idx + hash_u(time_data.frame);
        let noise = (hash_f() * 2.0 - 1.0) * params.dithering * 0.01;
        col += v3(noise);
    }
    
    col = aces_tonemap(col);
    
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(col, 1.0));
    
    if (params.motion_speed > 0.0) {
        atomicStore(&atomic_buffer[idx], 0u);
        atomicStore(&atomic_buffer[idx + res.x * res.y], 0u);
        atomicStore(&atomic_buffer[idx + 2u * res.x * res.y], 0u);
    }
}