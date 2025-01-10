// MIT License Enes Altun, 2025
// https://genuary.art/
// Make a landscape using only primitive shapes. (Day 6)

struct TimeUniform {
    time: f32,
};

@group(0) @binding(0) var<uniform> u_time: TimeUniform;

struct Params {
    color1: vec3<f32>,
    _pad1: f32,
    gradient_color: vec3<f32>,
    _pad2: f32,
    c_value_max: f32,
    iterations: f32,
    aa_level: i32,
    _pad3: f32,
};
@group(1) @binding(0) var<uniform> params: Params;

const PI: f32 = 3.14159265359;
fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}
fn pal(t: f32, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    return a + b * cos(2.0 * PI * (c * t + d));
}

fn wav(x: f32, p: f32) -> f32 {
    let x1 = x * params.c_value_max;
    let t = u_time.time * 0.5;
    return (sin(x1 * 0.25 + t + p) * 5.0 +
            sin(x1 * 4.5 + t * 2.0 + p) * 0.2 +
            sin(x1 + t * 1.5 + p) * 2.3 +
            sin(x1 * 0.8 + t + p) * 2.5) * 0.15;
}

fn sdc(p: vec2<f32>, r: f32) -> f32 {
    return length(p) - r;
}

@fragment
fn fs_main(@builtin(position) f: vec4<f32>) -> @location(0) vec4<f32> {
    let d = vec2<f32>(1920.0, 1080.0);
    var st = (f.xy - d * 0.5) / min(d.x, d.y) * 3.0;
    st.y = -st.y;
    st += 1.0;
    let th = params.iterations;
    let sf = 0.015 + 0.1 * length(smoothstep(vec2(0.1, 0.2), vec2(2.0, 0.7), abs(st)));

    let t = u_time.time * 0.25;
    let ws = 0.2;
    let n = floor((st.y + t) / ws);
    let y = fract((st.y + t) / ws);

    let sg = clamp((st.y + 0.8) * 0.6, 0.0, 1.0);
    let sc = mix(vec3(0.0, 0.0, 0.0), params.color1, pow(sg, 12.2));

    let mp = vec2(0.0, 1.7);
    let mr = 0.5;
    let md = sdc(st - mp, mr);
    let mg = exp(-md * 1.2) * 0.8;
    let mm = smoothstep(0.0, 0.01, -md);
    var cc = 0.0;
    var wc = vec3(0.0);

    for (var i: f32 = -8.0; i < 8.0; i = i + 1.0) {
        let ff = wav(st.x, (n - i) * 1.5) - y - i;
        let bf = smoothstep(-0.4, abs(st.y) + 0.1, ff);
        cc = mix(cc, 0.0, bf);
        let wi = smoothstep(th + sf, th - sf, abs(ff)) * (1.0 - abs(st.y) * 0.3) + smoothstep(4.0 - abs(ff * 1.5), 0.0, ff) * 0.3;
        cc = cc + wi;
        let wb = pal(sin((n - i) * 0.15), vec3(0.6), vec3(0.4), vec3(0.27), vec3(0.0, 0.05, 0.15));
        wc = mix(wc, wb * (cc + 0.2), bf);
    }
    var fc = sc;
    fc = mix(fc, vec3(1.0, 0.98, 0.9), mm);
    fc = fc + params.gradient_color * mg;

    let wv = smoothstep(0.3, -0.6, st.y);
    fc = mix(fc, mix(sc * 0.1, wc, 0.8), wv);

    let rs = 1.1;
    let rm = exp(-abs(st.x - mp.x) * 3.0) * smoothstep(0.1, -0.8, st.y) * (rs + 0.2 * sin(st.y * 30.0 + t));
    fc = fc + vec3(0.9, 0.9, 0.8) * rm;
    fc = gamma(fc, 0.4);
    return vec4(fc, 1.0);
}