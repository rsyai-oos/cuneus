struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct ParticleParams {
    a: f32,              
    b: f32,              
    c: f32,              
    d: f32,
    num_circles: f32,     
    num_points: f32,     
    particle_intensity: f32,
    gamma: f32,
    feedback_mix: f32,
    feedback_decay: f32,
    scale: f32,
    rotation: f32,
    bloom_scale: f32,
    animation_speed: f32,
    color_shift_speed: f32,
    color_scale: f32,
}
@group(1) @binding(0) var output_texture: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: ParticleParams;

@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

const PI: f32 = 3.14159265359;
const TAU: f32 = 6.28318530718;

alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;

fn oscillate(min_value: f32, max_value: f32, interval: f32, t: f32) -> f32 {
    return min_value + (max_value - min_value) * 0.5 * (sin(TAU * t / interval) + 1.0);
}

fn clifford_attractor(p: v2, a: f32, b: f32, c: f32, d: f32) -> v2 {
    let x = sin(a * p.y) + c * cos(a * p.x);
    let y = sin(b * p.x) + d * cos(b * p.y);
    return v2(x, y);
}

fn apply_gamma(color: v3, gamma: f32) -> v3 {
    return pow(color, v3(1.0 / gamma));
}

fn calculate_point_color(i: i32, t: f32, color_shift: f32, color_scale: f32) -> v3 {
    return 0.5 + 0.5 * sin(v3(1.0, TAU/3.0, TAU*2.0/3.0) + f32(i) * 0.87 + t * color_shift) * color_scale;
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

@compute @workgroup_size(16, 16)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = textureDimensions(output_texture);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { 
        return; 
    }

    let frag_coord = v2(f32(id.x) + 0.5, f32(screen_size.y - id.y) - 0.5);
    let uv = params.bloom_scale * (2.0 * frag_coord - v2(f32(screen_size.x), f32(screen_size.y))) / f32(screen_size.y);
    
    let num_circles = i32(params.num_circles);
    let num_points = i32(params.num_points);
    let circle_radius = oscillate(4.5, 4.5, 5.0, time_data.time) / f32(num_circles);
    let intensity = oscillate(0.02, 0.01, 12.0, time_data.time);
    
    let attr_a = params.a;
    let attr_b = params.b;
    let attr_c = params.c;
    let attr_d = params.d;
    let scale = params.scale;
    var color = v3(0.0);
    let max_contribution_radius = 20.0 / intensity;
    for(var i = 0; i < min(num_circles, 10); i++) {
        let circle_factor = f32(i+1) * circle_radius * 0.2;
        let point_color_base = calculate_point_color(i, time_data.time, params.color_shift_speed, 1.0);
        for(var j = 0; j < min(num_points, 10); j++) {
            let t = f32(j) / f32(min(num_points, 10)) * TAU + time_data.time * params.animation_speed;
            let initial_point = v2(cos(t), sin(t)) * circle_factor;
            var attractor_point = initial_point;
            for(var k = 0; k < 7; k++) {
                attractor_point = clifford_attractor(attractor_point, attr_a, attr_b, attr_c, attr_d);
            }
            let circle_point = attractor_point * scale;
            let rough_dist = distance(uv, circle_point);
            if (rough_dist > max_contribution_radius) {
                continue;
            }
            let dist = length(uv - circle_point);
            color += point_color_base * intensity / dist * params.particle_intensity;
        }
    }

    color = clamp(color, v3(0.0), v3(1.0));
    color = sqrt(color) * 2.0 - 1.0;

    let idx = id.x + screen_size.x * id.y;
    let r = f32(atomicLoad(&atomic_buffer[idx])) / 255.0;
    let g = f32(atomicLoad(&atomic_buffer[idx + screen_size.x * screen_size.y])) / 255.0;
    let blue = f32(atomicLoad(&atomic_buffer[idx + 2 * screen_size.x * screen_size.y])) / 255.0;
    let feedback_color = v3(r, g, blue);
    
    color = mix(color, feedback_color * params.feedback_decay, params.feedback_mix);
    
    color = apply_gamma(color, params.gamma);
    
    color = aces_tonemap(color * params.color_scale);
    
    textureStore(output_texture, vec2<i32>(id.xy), v4(color, 1.0));
    
    atomicStore(&atomic_buffer[idx], u32(clamp(color.r * 255.0, 0.0, 255.0)));
    atomicStore(&atomic_buffer[idx + screen_size.x * screen_size.y], u32(clamp(color.g * 255.0, 0.0, 255.0)));
    atomicStore(&atomic_buffer[idx + 2 * screen_size.x * screen_size.y], u32(clamp(color.b * 255.0, 0.0, 255.0)));
}