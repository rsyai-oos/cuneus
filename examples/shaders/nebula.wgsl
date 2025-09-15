// nebula: MIT License, Enes Altun. 2025
// Based on the kali's shader:
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
    dust_intensity: f32,
    distfading: f32,
    color_variation: f32,
    n_boxes: f32,
    rotation: i32,
    depth: f32,
    color_mode: i32,
    _padding1: f32,
    
    rotation_x: f32,
    rotation_y: f32,
    click_state: i32,
    scale: f32,
    
    exposure: f32,
    gamma: f32,

    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
    _padding8: f32,
    _padding9: f32,
    _padding10: f32,

    time_scale: f32,
    visual_mode: i32,
    _padding2: f32,
    _padding3: f32,
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: NebulaParams;

@group(2) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

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



fn mainVR(fragCoord: vec2<f32>, res: vec2<f32>, ro: vec3<f32>, rd: vec3<f32>, time: f32) -> vec4<f32> {
    var uv = fragCoord.xy / res.xy - 0.5;
    uv.y *= res.y / res.x;
    
    
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
    let focal_distance = 0.0 * 2.0;
    let depth_diff = abs(depth_from_center - focal_distance);
    let dof_blur = 1.0 - smoothstep(0.0, 0.0 * 1.5, depth_diff * depth_diff);

    var s = 0.1;
    var fade = 1.0;
    var v = vec3<f32>(0.0);
    
    for (var r = 0; r < params.volsteps; r++) {
        var p = ray_origin + s * dir * 0.5;
        
        
        p = abs(vec3<f32>(params.tile) - (p* (vec3<f32>(params.tile) * 1.0)));
        
        var pa = 0.0;
        var a = 0.0;
        
        for (var i = 0; i < params.iterations; i++) {
            let power_mod = 2.0 + sin(length(p) * 0.5) * 0.3;
            p = abs(p) / pow(dot(p, p), power_mod * 0.5) - params.formuparam;
            a += abs(length(p) - pa);
            pa = length(p);
        }
        
        a *= a * a;
        
        let dof_factor = smoothstep(0.3, 1.0, dof_blur);
        a *= 2.0*mix(0.2, 1.0, dof_factor);
        fade *= mix(0.7, 1.0, dof_factor);
        
        let dust_noise = sin(p.x * 0.5 + time * 0.1) * cos(p.y * 0.3) * sin(p.z * 0.4);
        let dust_factor = max(0.0, params.dust_intensity);
        let dust_color = vec3<f32>(0.3, 0.7, 0.7);
        
        let color_phase = a * params.color_variation * 2.0 + time * 0.5;
        let enhanced_color = vec3<f32>(
            0.8 + sin(color_phase) * 0.4,
            0.6 + sin(color_phase + 2.0) * 0.1, 
            0.9 + sin(color_phase + 4.0) * 0.3
        );
        let h_blur = sin(p.x * 0.5 + 0.3) * 0.2 + 1.0;
        let v_blur = cos(p.y * 0.4 +  0.25) * 0.2 + 1.0;
        let blur_factor = vec3<f32>(h_blur, v_blur, mix(h_blur, v_blur, 0.5));
        let dust_layer = vec3<f32>(fade) * dust_color * dust_factor;
        let main_layer = vec3<f32>(s, s * s, s * s * s) * a * params.brightness * fade * enhanced_color;
        let layered_blur = blur_factor * vec3<f32>(0.6, 0.6, 0.4);
        v += dust_layer * layered_blur.x;
        v += main_layer * layered_blur;
        fade *= params.distfading * mix(0.95, 1.05, blur_factor.z);
        s += params.stepsize;
    }
    
    v = mix(vec3<f32>(length(v)), v, 1.0);
    
    return vec4<f32>(v * 0.03, 1.0);
}



fn color_grade(color: vec3<f32>) -> vec3<f32> {
    var graded = color;
    
    graded = pow(graded, vec3<f32>(0.9));
    graded *= vec3<f32>(1.1, 1.05, 0.95);
    graded = mix(graded, graded * graded, 0.3);
    
    let luminance = dot(graded, vec3<f32>(0.299, 0.587, 0.114));
    graded = mix(vec3<f32>(luminance), graded, 1.2);
    
    return graded;
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
    
    var final_color = nebula_color.rgb + color * 0.05;
    
    final_color *= params.exposure;
    final_color = color_grade(final_color);
    final_color = aces_tonemap(final_color);
    final_color = gamma_correction(final_color, params.gamma);
    
    textureStore(output, vec2<i32>(id.xy), vec4<f32>(final_color, 1.0));
}