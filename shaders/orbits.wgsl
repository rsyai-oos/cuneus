// MIT License Enes Altun, 2025
// Trap technique: https://iquilezles.org/articles/ftrapsgeometric/
struct TimeUniform {
    time: f32,
};
const PI = 3.14159;
struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: Params;
struct Params {
    base_color: vec3<f32>,
    x: f32,
    rim_color: vec3<f32>,
    y: f32,
    accent_color: vec3<f32>,
    _pad3: f32,
    _pad4: f32,
    iteration: i32,
    col_ext: f32,
    zoom: f32,
    trap_pow: f32,
    trap_x: f32,
    trap_y: f32,
    trap_c1: f32,
    aa: i32,
    trap_s1: f32,
    wave_speed: f32,
    fold_intensity: f32,
};
// nt: normalize trap
fn nt(d: f32, s: f32) -> f32 {
    return .5 * (1. + tanh(s * d));
}

// op: oscillate with pause
fn op(mn: f32, mx: f32, i: f32, p: f32, t: f32) -> f32 {
    let c = 2. * i + p;
    let m = t % c;
    if(m < i) {
        return mix(mx, mn, .5 - .5 * cos(PI * m / i));
    } else if(m < i + p) {
        return mn;
    } else {
        return mix(mn, mx, .5 - .5 * cos(PI * (m - i - p) / i));
    }
}
// im: implicit fractal calculation
fn im(c: vec2<f32>, t1: vec2<f32>, t2: vec2<f32>, t: f32) -> vec4<f32> {
    var z = vec2(0.);      // position
    var dz = vec2(1., 0.); // derivative
    var t1m = 1e20;        // trap1 min
    var t2m = 1e20;        // trap2 min
    var t1s = 0.;          // trap1 sum
    var t1c = 0.;          // trap1 count
    let mi = params.iteration;
    let dt = t * .001;
    var i = 0;
    for(; i < mi; i++) {
        dz = 2. * vec2(z.x * dz.x - z.y * dz.y, z.x * dz.y + z.y * dz.x) + vec2(1., 0.);
        let x = z.x * z.x - z.y * z.y + c.x;
        z.y = 2. * z.x * z.y + c.y;
        z.x = x;
        z += .1 * vec2(sin(.001 * dt), cos(.001 * dt));
        let d1 = length(z - t1);
        let f = 1. - smoothstep(.6, 1.4, d1);
        t1c += f;
        t1s += f * d1;
        t1m = min(t1m, d1);
        t2m = min(t2m, dot(z - t2, z - t2));
        if(dot(z, z) > 12.5) { break; }
    }
    let d = sqrt(dot(z, z) / dot(dz, dz)) * log(dot(z, z));
    return vec4(f32(i), d, t1s / max(t1c, 1.), t2m);
}
// g: gamma correction
fn g(c: vec3<f32>, g: f32) -> vec3<f32> {
    return pow(c, vec3(1. / g));
}
@fragment
fn fs_main(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    let ss = u_resolution.dimensions;
    let frag = vec2(fc.x, ss.y - fc.y);
    let AA = params.aa;
    let t = u_time.time;
    let t01 = t * .1;
    // cp: camera path
    let cp = vec2(sin(.0002 * t / 10.), cos(.0002 * t / 10.));
    // p: pan position
    var p = vec2(.8085, .2607);
    if(t > 17.) { p.y += .00001 * (t - 45.); }
    // zl: zoom level
    let zl = op(.0005, .0005, 10., 5., t01);
    // t1, t2: traps
    let t1 = vec2(0., params.fold_intensity);
    let t2 = vec2(params.trap_x, params.trap_y) + params.wave_speed * vec2(cos(.3 * t), sin(.3 * t));
    var col = vec3(0.);
    for(var m = 0; m < AA; m++) {
        for(var n = 0; n < AA; n++) {
            let so = vec2(f32(m), f32(n)) / f32(AA);
            let mr = min(ss.x, ss.y);
            let uv = ((frag + so - .5 * ss) / mr * params.zoom + p + cp) * 2.033 - vec2(params.x, params.y);
            let zd = im(uv, t1, t2, t01);
            let ir = zd.x / f32(params.iteration);
            if(ir < 1.1) {
                let c1 = pow(clamp(2. * zd.y / zl, 0., 1.), .5);
                let c2 = pow(clamp(1.5 * zd.z, 0., 1.), 2.);
                let c3 = pow(clamp(.4 * zd.w, 0., 1.), .25);
                // cl1, cl2: colors based on params
                let cl1 = .5 + .5 * sin(vec3(3.) + 4. * c2 + params.rim_color);
                let cl2 = .5 + .5 * sin(vec3(4.1) + 2. * c3 + params.accent_color);
                // bc: base color
                let bc = 2. * sqrt(c1 * cl1 * cl2);
                // te: time effect
                let te = op(params.trap_pow, params.trap_pow, 6., 0., t01);
                // ec: exterior color
                let ec = .5 + .5 * cos(params.col_ext * zd.w + zd.z + params.base_color + PI * 6. * ir + te);
                // bf: blend factor
                let bf = smoothstep(.2, params.trap_s1, ir);
                let pc = mix(bc, ec, bf);
                let tce = op(1., 1., 10., 5., t01);
                // vc: variation color
                let vc = pc * (.5 + .5 * sin(PI * vec3(.5, .7, .9) * ir + tce));
                col += mix(pc, vc, params.trap_c1);
            }
        }
    }
    //gamma and vignette
    col = g(col / f32(AA * AA), .4);
    let q = frag.xy / ss;
    col *= .7 + .3 * pow(16. * q.x * q.y * (1. - q.x) * (1. - q.y), .15);
    return vec4(col, 1.);
}