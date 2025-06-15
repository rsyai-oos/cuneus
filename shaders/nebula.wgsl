// MIT License, by Pablo Roman Andrioli aka "Kali", 2013
// Shadertoy: https://www.shadertoy.com/view/XlfGRj 
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct NebulaParams {
    iterations: i32,
    formuparam: f32,
    volsteps: i32,
    stepsize: f32,
    zoom: f32,
    tile: f32,
    speed: f32,
    brightness: f32,
    darkmatter: f32,
    distfading: f32,
    saturation: f32,
    n_boxes: f32,
    rotation: i32,
    depth: f32,
    color_mode: i32,
    _padding1: f32,
    
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    scale: f32,
    
    dof_amount: f32,
    dof_focal_dist: f32,
    exposure: f32,
    gamma: f32,
    
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
    
    time_scale: f32,
    
    spiral_mode: i32,
    spiral_strength: f32,
    spiral_speed: f32,
    visual_mode: i32,
    _padding2: f32,
    _padding3: f32,
}
@group(1) @binding(0) var<uniform> params: NebulaParams;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

const pi = 3.14159265359;

fn rot(a: f32) -> mat2x2<f32> {
    let s = sin(a);
    let c = cos(a);
    return mat2x2<f32>(c, -s, s, c);
}

fn sdBox(p: vec2<f32>, b: vec2<f32>) -> f32 {
    let d = abs(p) - b;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

fn periphery(p: vec2<f32>, s: f32, time: f32) -> f32 {
    var per = 0.0;
    
    for (var i = 1.0; i < params.n_boxes + 1.0 * cos(time); i += 1.0) {
        var p0 = p;
        
        let a = radians(31.0 * cos(time) / params.n_boxes * cos(time)) * i;
        let mean = (params.n_boxes + params.depth) * 0.4;
        p0 = rot(a) * p0;
        
        if (params.rotation != 0) {
            p0 = rot(time * i * mean * 1e-2 / (params.n_boxes * 0.08) * time) * p0;
        }
        
        let box = sdBox(p0, vec2<f32>(s));
        let gamma = mean * 1e-4 * 0.7;
        let box_result = gamma / abs(box);
        
        per += box_result;
    }
    return per;
}

fn mainVR(fragCoord: vec2<f32>, res: vec2<f32>, ro: vec3<f32>, rd: vec3<f32>, time: f32) -> vec4<f32> {
    var uv = fragCoord.xy / res.xy - 0.5;
    uv.y *= res.y / res.x;
    
    if (params.spiral_mode == 1) {
        let radius = length(uv);
        let angle = atan2(uv.y, uv.x);
        let spiral_angle = angle + log(radius + 0.1) * params.spiral_strength + time * params.spiral_speed;
        let spiral_radius = radius * (1.0 + sin(spiral_angle * 3.0) * 0.3);
        uv = vec2<f32>(cos(spiral_angle) * spiral_radius, sin(spiral_angle) * spiral_radius);
    } else if (params.spiral_mode == 2) {
        let radius = length(uv);
        let hole_factor = smoothstep(0.0, 0.5, radius);
        uv *= hole_factor * params.spiral_strength * 0.5;
    } else if (params.spiral_mode == 3) {
        let radius = length(uv);
        let angle = atan2(uv.y, uv.x);
        let tunnel_depth = 1.0 / (radius * params.spiral_strength + 0.1) + time * params.spiral_speed;
        let tunnel_radius = radius * tunnel_depth * 0.3;
        uv = vec2<f32>(cos(angle) * tunnel_radius, sin(angle) * tunnel_radius);
    }
    
    var dir = vec3<f32>(uv * params.zoom, 1.0);
    let time_scaled = time * params.speed + 0.25;
    
    let a1 = 0.5 + params.rotation_x / 800.0 * 2.0;
    let a2 = 0.8 + params.rotation_y / 600.0 * 2.0;
    let rot1 = rot(a1);
    let rot2 = rot(a2);
    dir = vec3<f32>(rot1 * dir.xy, dir.z);
    dir = vec3<f32>(rot2 * dir.xy, dir.z);
    
    var ray_origin = vec3<f32>(1.0, 0.5, 0.5);
    ray_origin += vec3<f32>(time_scaled * 2.0, time_scaled, -2.0);
    ray_origin = vec3<f32>(rot1 * ray_origin.xy, ray_origin.z);
    ray_origin = vec3<f32>(rot2 * ray_origin.xy, ray_origin.z);
    
    let depth_from_center = length(dir.xy);
    let focal_distance = params.dof_focal_dist * 3.0;
    let depth_diff = abs(depth_from_center - focal_distance);
    let dof_blur = 1.0 - smoothstep(0.0, params.dof_amount * 2.0, depth_diff);
    
    var s = 0.1;
    var fade = 1.0;
    var v = vec3<f32>(0.0);
    
    for (var r = 0; r < params.volsteps; r++) {
        var p = ray_origin + s * dir * 0.5;
        
        if (params.spiral_mode == 1) {
            let p_radius = length(p.xy);
            let p_angle = atan2(p.y, p.x);
            let spiral_p_angle = p_angle + log(p_radius + 0.1) * params.spiral_strength * 0.5 + time * params.spiral_speed * 0.3;
            let spiral_p_radius = p_radius * (1.0 + sin(spiral_p_angle * 2.0 + p.z * 0.5) * 0.4);
            p.x = cos(spiral_p_angle) * spiral_p_radius;
            p.y = sin(spiral_p_angle) * spiral_p_radius;
        } else if (params.spiral_mode == 2) {
            let p_radius = length(p.xy);
            let hole_factor = smoothstep(0.0, 0.6, p_radius);
            p.z *= hole_factor * params.spiral_strength * 0.5;
        } else if (params.spiral_mode == 3) {
            let p_radius = length(p.xy);
            let tunnel_factor = 1.0 / (p_radius * params.spiral_strength * 0.5 + 0.1);
            p.z *= tunnel_factor * 0.5;
        }
        
        p = abs(vec3<f32>(params.tile) - (p % (vec3<f32>(params.tile) * 2.0)));
        
        var pa = 0.0;
        var a = 0.0;
        
        for (var i = 0; i < params.iterations; i++) {
            p = abs(p) / dot(p, p) - params.formuparam;
            a += abs(length(p) - pa);
            pa = length(p);
        }
        
        let dm = max(0.0, params.darkmatter - a * a * 0.001);
        a *= a * a;
        
        a *= mix(0.3, 1.0, dof_blur);
        fade *= mix(0.8, 1.0, dof_blur);
        
        if (r > 6) {
            fade *= 1.3 - dm;
        }
        
        v += vec3<f32>(fade);
        v += vec3<f32>(s, s * s, s * s * s) * a * params.brightness * fade;
        fade *= params.distfading;
        s += params.stepsize;
    }
    
    v = mix(vec3<f32>(length(v)), v, params.saturation);
    
    return vec4<f32>(v * 0.03, 1.0);
}

fn H(h: f32) -> vec3<f32> {
    return (cos(h * 6.3 + vec3<f32>(0.0, 23.0, 21.0)) * 0.5 + 0.5);
}

fn aces_tonemap(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn gamma_correction(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
}

@compute @workgroup_size(16, 16, 1)
fn volumetric_render(@builtin(global_invocation_id) id: vec3<u32>) {
    return;
}

fn hash_u32(a: u32) -> u32 {
    var x = a;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<f32>(textureDimensions(output));
    if (f32(id.x) >= res.x || f32(id.y) >= res.y) { return; }
    
    let pixel_idx = id.x + u32(res.x) * id.y;
    let color_offset = u32(res.x * res.y);
    
    let base_intensity = f32(atomicLoad(&atomic_buffer[pixel_idx])) * 0.001;
    let red_extra = f32(atomicLoad(&atomic_buffer[pixel_idx + color_offset])) * 0.01;
    let blue_extra = f32(atomicLoad(&atomic_buffer[pixel_idx + color_offset * 2u])) * 0.01;
    
    var color = vec3<f32>(
        base_intensity,
        base_intensity * 0.7,
        base_intensity
    );
    
    let fragCoord = vec2<f32>(f32(id.x), f32(id.y));
    var uv = fragCoord.xy / res.xy - 0.5;
    uv.y *= res.y / res.x;
    var dir = vec3<f32>(uv * params.zoom, 1.0);
    let ray_origin = vec3<f32>(1.0, 0.5, 0.5);
    let nebula_color = mainVR(fragCoord, res, ray_origin, dir, time_data.time * params.time_scale);
    
    var final_color = nebula_color.rgb + color * 0.1;
    
    final_color *= params.exposure;
    final_color = aces_tonemap(final_color);
    final_color = gamma_correction(final_color, params.gamma);
    
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(final_color, 1.0));
}