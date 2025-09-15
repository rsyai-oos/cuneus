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
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

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

const PI=3.14159265359;

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
    let t = time * params.speed + 0.25;
    
    let a1 = 0.5 + params.rotation_x / 800.0 * 2.0;
    let a2 = 0.8 + params.rotation_y / 600.0 * 2.0;
    let rot1 = rot(a1);
    let rot2 = rot(a2);
    dir = vec3<f32>(rot1 * dir.xy, dir.z);
    dir = vec3<f32>(rot2 * dir.xy, dir.z);
    //ray origin and depth calc
    var ray = vec3<f32>(1.0, 0.5, 0.5);
    ray += vec3<f32>(t * 2.0, t, -2.0);
    ray = vec3<f32>(rot1 * ray.xy, ray.z);
    ray = vec3<f32>(rot2 * ray.xy, ray.z);
    
    let dc = length(dir.xy);
    let fd = 0.0 * 2.0;
    let dd = abs(dc - fd);
    let dof = 1.0 - smoothstep(0.0, 0.0 * 1.5, dd * dd);

    var s = 0.1;
    var fade = 1.0;
    var v = vec3<f32>(0.0);
    
    for (var r = 0; r < params.volsteps; r++) {
        var p = ray + s * dir * 0.5;
        
        
        p = abs(vec3<f32>(params.tile) - (p* (vec3<f32>(params.tile) * 1.0)));
        
        var pa = 0.0;
        var a = 0.0;
        
        for (var i = 0; i < params.iterations; i++) {
            let pm = 2.0 + sin(length(p) * 0.5) * 0.3;
            p = abs(p) / pow(dot(p, p), pm * 0.5) - params.formuparam;
            a += abs(length(p) - pa);
            pa = length(p);
        }
        
        a *= a * a;
        
        let df = smoothstep(0.3, 1.0, dof);
        a *= 2.0*mix(0.2, 1.0, df);
        fade *= mix(0.7, 1.0, df);
        //dust noise dust color and facrot: df
        let dn = sin(p.x * 0.5 + time * 0.1) * cos(p.y * 0.3) * sin(p.z * 0.4);
        let df2 = max(0.0, params.dust_intensity);
        let dc2 = vec3<f32>(0.3, 0.7, 0.7);
        //color phase etc
        let cp = a * params.color_variation * 2.0 + time * 0.5;
        let col = vec3<f32>(
            0.8 + sin(cp) * 0.4,
            0.6 + sin(cp + 2.0) * 0.1,
            0.9 + sin(cp + 4.0) * 0.3
        );
        //blurss: layered main etc
        let hb = sin(p.x * 0.5 + 0.3) * 0.2 + 1.0;
        let vb = cos(p.y * 0.4 +  0.25) * 0.2 + 1.0;
        let bf = vec3<f32>(hb, vb, mix(hb, vb, 0.5));
        let dl = vec3<f32>(fade) * dc2 * df2;
        let ml = vec3<f32>(s, s * s, s * s * s) * a * params.brightness * fade * col;
        let lb = bf * vec3<f32>(0.6, 0.6, 0.4);
        v += dl * lb.x;
        v += ml * lb;
        fade *= params.distfading * mix(0.95, 1.05, bf.z);
        s += params.stepsize;
    }
    
    v = mix(vec3<f32>(length(v)), v, 1.0);
    
    return vec4<f32>(v * 0.03, 1.0);
}



fn grade(color: vec3<f32>) -> vec3<f32> {
    var graded = color;
    
    graded = pow(graded, vec3<f32>(0.9));
    graded *= vec3<f32>(1.1, 1.05, 0.95);
    graded = mix(graded, graded * graded, 0.3);
    //luminance
    let lum = dot(graded, vec3<f32>(0.299, 0.587, 0.114));
    graded = mix(vec3<f32>(lum), graded, 1.2);
    
    return graded;
}

fn H(h: f32) -> vec3<f32> {
    return (cos(h * 6.3 + vec3<f32>(0.0, 23.0, 21.0)) * 0.5 + 0.5);
}

fn aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn gamma(color: vec3<f32>, g: f32) -> vec3<f32> {
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / g));
}

@compute @workgroup_size(16, 16, 1)
fn volumetric_render(@builtin(global_invocation_id) id: vec3<u32>) {
    return;
}

fn hash(a: u32) -> u32 {
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
    //pixel idx, color offset
    let pi = id.x + u32(res.x) * id.y;
    let co = u32(res.x * res.y);
    // intensity, some extra red blue 
    let bi = f32(atomicLoad(&atomic_buffer[pi])) * 0.001;
    let re = f32(atomicLoad(&atomic_buffer[pi + co])) * 0.01;
    let be = f32(atomicLoad(&atomic_buffer[pi + co * 2u])) * 0.01;

    var color = vec3<f32>(
        bi,
        bi * 0.7,
        bi
    );
    
    let fc = vec2<f32>(f32(id.x), f32(id.y));
    var uv = fc.xy / res.xy - 0.5;
    uv.y *= res.y / res.x;

    var dir = vec3<f32>(uv * params.zoom, 1.0);
    let ro2 = vec3<f32>(1.0, 0.5, 0.5);
    let neb = mainVR(fc, res, ro2, dir, u_time.time * params.time_scale);
    
    var fc2 = neb.rgb + color * 0.05;

    fc2 *= params.exposure;
    fc2 = grade(fc2);
    fc2 = aces(fc2);
    fc2 = gamma(fc2, params.gamma);

    textureStore(output, vec2<i32>(id.xy), vec4<f32>(fc2, 1.0));
}