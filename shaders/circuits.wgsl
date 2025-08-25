// Enes Altun, 2025 
// Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct CircuitParams {
    rotation_speed: f32,
    distance_offset: f32,
    gamma: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
    intensity: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
};
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: CircuitParams;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m2 = mat2x2<f32>;
const pi = 3.14159265;
const tau = 6.28318531;

var<private> R: v2;

fn n(p: v2) -> f32 {
    return sin(p.x * 4.0 + sin(p.y * 3.1)) * cos(p.y * 1.3 + cos(p.x * 2.7));
}

fn mm2(a: f32) -> m2 {
    let c = cos(a);
    let s = sin(a);
    return m2(c, -s, s, c);
}

fn fract_v4(v: v4) -> v4 {
    return v - floor(v);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let C = v2(f32(gid.x), f32(gid.y));
    let r = R;
    var o = v4(0.0);
    var i = 0.0;
    var d = 0.0;
    var z = 0.0;
    let t = time_data.time;
    
    for (var j = 0; j < 75; j++) {
        i += 1.0;
        z += d * 0.6;
        
        let rd = normalize(v3((C - 0.5 * r.xy) / r.y, 1.0));
        var p = v4(z * rd, t);
        p.z += t;
        
        let r_m = mm2(p.z * params.rotation_speed);
        let r_xy = r_m * p.xy;
        p.x = r_xy.x;
        p.y = r_xy.y;

        let N = n(p.xy + t * 0.2);
        p = abs(fract_v4(p) - 0.5);
        
        let d1 = length(p.xy) - 0.08 + N * 0.04;
        let d2 = length(p.xz) - 0.12 + N * 0.03;
        let d3 = max(p.x, p.y) - 0.15 + sin(t + N) * 0.1;
        
        d = min(d1, min(d2, d3));
        d = abs(d) + params.distance_offset;
        
        var c = v3(params.color1_r, params.color1_g, params.color1_b) / (length(p.xy + N));
        c += v3(params.color2_r, params.color2_g, params.color2_b) / (length(p.xz + N));
        
        o += v4(c, 1.0) / d;
    }
    
    var result = tanh(o * params.intensity);
    result = pow(result, v4(1.0 / params.gamma));
    textureStore(output, gid.xy, result);
}