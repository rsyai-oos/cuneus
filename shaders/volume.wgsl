//Enes Altun, 2025 
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct VolumeParams {
    speed: f32,
    intensity: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
    color3_r: f32,
    color3_g: f32,
    color3_b: f32,
    contrast: f32,
};
@group(1) @binding(0) var<uniform> params: VolumeParams;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias v4 = vec4<f32>;
alias m2 = mat2x2<f32>;
const pi = 3.14159265;
const tau = 6.28318531;

var<private> R: v2;

fn n(p: v2) -> f32 {
    return sin(p.x * 3.0 + sin(p.y * 2.7)) * cos(p.y * 1.1 + cos(p.x * 2.3));
}

fn f(p: v3) -> f32 {
    var v = 0.0;
    var a = 1.0;
    var pos = p;
    
    for (var i = 0; i < 7; i++) {
        v += n(pos.xy + pos.z * 0.5) * a;
        pos *= 2.0;
        a *= 0.5;
    }
    
    return v;
}

fn mm2(a: f32) -> m2 {
    let c = cos(a);
    let s = sin(a);
    return m2(c, -s, s, c);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let C = v2(f32(gid.x), f32(gid.y));
    let r = R;
    var o = v3(0.0);
    var i = 0.0;
    var d = 0.0;
    var z = 0.0;
    let t = time_data.time * params.speed;
    var N = 0.0;
    
    for (var j = 0; j < 50; j++) {
        i += 1.0;
        z += d * 0.6;
        
        var p = z * normalize(v3(C - 0.5 * r.xy, r.y));
        p.z += t;
        
        let R_mat = mm2(p.z * 1.1);
        let rotated_xy = R_mat * p.xy;
        p.x = rotated_xy.x;
        p.y = rotated_xy.y;
        
        N = f(p + t * 0.1);
        d = length(p.xy) - 1.0 + N * 0.3;
        p.z = (p.z % 4.0) - 2.0;
        d = abs(d) + 0.01;
        
        var c = v3(params.color1_r, params.color1_g, params.color1_b) / (length(p.xy + N) * 0.8);
        c += v3(params.color2_r, params.color2_g, params.color2_b) / (length(p.xz + N) * 0.8);
        c += v3(params.color3_r, params.color3_g, params.color3_b) * (0.5 + 0.5 * sin(N * 1.0 + t));
        
        o += c / d * 0.12;
    }
    
    let result = tanh(o * params.intensity * params.contrast);
    textureStore(output, gid.xy, v4(result, 1.0));
}