@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

struct TimeUniform {
    time: f32,
};
@group(1) @binding(0)
var<uniform> time_data: TimeUniform;

struct Params {
    palette_a: vec3<f32>,
    _pad1: f32,
    palette_b: vec3<f32>,
    _pad2: f32,
    palette_c: vec3<f32>,
    _pad3: f32,
    palette_d: vec3<f32>,
    _pad4: f32,
    highlight_color: vec3<f32>,
    _pad5: f32,
    octaves: f32,
    num_vortices: f32,
    vortex_scale: f32,
    flow_influence: f32,
    min_radius: f32,
    max_radius: f32,
    palette_time: f32,
    decay: f32,
};
@group(2) @binding(0)
var<uniform> params: Params;

fn rand(co: vec2<f32>) -> f32 {
    return fract(sin(dot(co.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn rand2(co: vec2<f32>) -> f32 {
    return fract(cos(dot(co.xy, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}
fn fluid_vortex(uv: vec2<f32>, center: vec2<f32>, strength: f32, radius: f32, t: f32, flow_dir: vec2<f32>) -> vec2<f32> {
    let delta = uv - center;
    let dist = length(delta);
    let angle = atan2(delta.y, delta.x);
    let rots = strength * (1.0 - smoothstep(0.0, radius, dist));
    let rot_vec = vec2<f32>(cos(angle + t * 2.0), sin(angle + t * 2.0));
    let flow_vec = flow_dir * params.flow_influence;
    let spiral = exp(-dist / (radius * 0.5));
    return mix(rot_vec, flow_vec, dist/radius) * spiral * rots;
}
fn vort_pos(index: f32, t: f32, base_flow: vec2<f32>) -> vec2<f32> {
    let angle = (index / params.num_vortices) * 6.28318 + t * 0.2;
    let radius = 0.2 + 0.05 * sin(t * 0.5 + index);
    
    return vec2<f32>(
        0.5 + cos(angle) * radius,
        0.5 + sin(angle) * radius
    );
}
fn simplex(vl: vec2<f32>) -> f32 {
    let min_step = 1.0;
    let grid = floor(vl);
    let grid_pnt1 = grid;
    let grid_pnt2 = vec2<f32>(grid.x, grid.y + min_step);
    let grid_pnt3 = vec2<f32>(grid.x + min_step, grid.y);
    let grid_pnt4 = vec2<f32>(grid_pnt3.x, grid_pnt2.y);
    
    let s = rand2(grid_pnt1);
    let t = rand2(grid_pnt3);
    let u = rand2(grid_pnt2);
    let v = rand2(grid_pnt4);
    
    let x1 = smoothstep(0.0, 1.0, fract(vl.x));
    let interp_x1 = mix(s, t, x1);
    let interp_x2 = mix(u, v, x1);
    
    let y = smoothstep(0.0, 1.0, fract(vl.y));
    return mix(interp_x1, interp_x2, y);
}

fn fb(vl: vec2<f32>) -> f32 {
    let persistence = 2.1;
    var amplitude = 0.55;
    var rez = 0.0;
    var p_local = vl;
    
    for(var i = 0.0; i < params.octaves; i += 1.0) {
        rez += amplitude * simplex(p_local);
        amplitude /= persistence;
        p_local *= persistence;
    }
    return rez;
}

fn complex_fbm(p_in: vec2<f32>, t: f32) -> f32 {
    let base_flow = vec2<f32>(cos(t * 0.3), sin(t * 0.2)) * 1.25;
    var total_vortex = vec2<f32>(0.0);
    for(var i = 0.0; i < params.num_vortices; i += 1.0) {
        let center = vort_pos(i, t, base_flow);
        let ang_s = sin(t * 2.0 + i * 6.28318 / params.num_vortices);
        let strength = 14.0 + 0.4 * ang_s;
        
        total_vortex += fluid_vortex(p_in, center, strength, params.vortex_scale, t + i, base_flow);
    }
    
    var p = p_in + total_vortex * 0.25;
    
    return fb(
        p + base_flow + 1.2 * fb(
            p + 1.1 * fb(
                p + 2.2 * fb(p)
            )
        )
    );
}

@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(prev_frame));
    let uv = FragCoord.xy / dimensions;
    let fluid = complex_fbm(uv, time_data.time);
    return vec4<f32>(fluid, 0.0, 0.0, 1.0);
}

@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let uv = tex_coords;
    let DELTA = 0.01;
    
    let left = textureSample(prev_frame, tex_sampler, uv - vec2<f32>(DELTA, 0.0)).x;
    let right = textureSample(prev_frame, tex_sampler, uv + vec2<f32>(DELTA, 0.0)).x;
    let up = textureSample(prev_frame, tex_sampler, uv - vec2<f32>(0.0, DELTA)).x;
    let down = textureSample(prev_frame, tex_sampler, uv + vec2<f32>(0.0, DELTA)).x;
    
    var velocity = vec2<f32>(right - left, down - up) * 4.0;
    let dx = right - left;
    let dy = down - up;
    let curl_value = (dy - dx) / (2.0 * DELTA);
    velocity += vec2<f32>(-curl_value, curl_value) * 0.15;
    velocity = velocity + vec2<f32>(-curl_value, curl_value) * 0.1;
    return vec4<f32>(velocity, curl_value * 0.5 + 0.5, 1.0);
}

fn get_palette(t: f32) -> vec3<f32> {
    return params.palette_a + 
           params.palette_b * 
           cos(3.28318 * (params.palette_c * t + params.palette_d));
}

fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}

@fragment
fn fs_pass3(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let uv = tex_coords;
    
    let fluid = textureSample(prev_frame, tex_sampler, uv).x;
    let velocity = textureSample(prev_frame, tex_sampler, uv).xy;
    
    let col1 = get_palette(fluid * 0.5 + params.palette_time * 1.35);
    let col2 = get_palette(fluid * 0.6 + params.palette_time * 1.5);
    
    let dis_uv = uv + velocity * 0.12;
    let dis_fluid = textureSample(prev_frame, tex_sampler, dis_uv).x;
    
    var fc = mix(col1, col2, dis_fluid);
    
    let velocity_mag = length(velocity);
    fc += params.highlight_color * smoothstep(0.45, 0.85, velocity_mag) * 0.25;
    
    let center = uv - 0.5;
    let vignette = 1.5 - dot(center, center) * 1.8;
    fc *= vignette;
    
    fc = gamma(fc, 0.45);
    
    return vec4<f32>(fc, 1.0);
}