// Enes Altun, 2025, MIT License
// heavily inspired from https://www.shadertoy.com/view/MslGWN  

struct TimeUniform {
    time: f32,
};

struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};

struct Params {
    zoom_base: f32,      
    space_distort_x: f32,
    space_distort_y: f32,
    space_distort_z: f32,
    
    zoom_delay: f32,
    zoom_speed: f32,
    max_zoom: f32,
    min_zoom: f32,
    
    noise_scale: f32,
    time_scale: f32,
    _pad1: vec2<f32>,
    
    disk_color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: Params;

fn mod289_f32(x: f32) -> f32 {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn mod289_vec4(x: vec4<f32>) -> vec4<f32> {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

fn perm(x: vec4<f32>) -> vec4<f32> {
    return mod289_vec4(((x * 34.0) + 1.0) * x);
}

fn noise(p: vec3<f32>) -> f32 {
    let a = floor(p);
    let d = p - a;
    let d2 = d * d * (3.0 - 2.0 * d);
    
    let b = a.xxyy + vec4<f32>(0.0, 1.0, 0.0, 1.0);
    let k1 = perm(b.xyxy);
    let k2 = perm(k1.xyxy + b.zzww);
    
    let c = k2 + a.zzzz;
    let k3 = perm(c);
    let k4 = perm(c + 1.0);
    
    let o1 = fract(k3 * (1.0 / 41.0));
    let o2 = fract(k4 * (1.0 / 41.0));
    
    let o3 = o2 * d2.z + o1 * (1.0 - d2.z);
    let o4 = o3.yw * d2.x + o3.xz * (1.0 - d2.x);
    
    return o4.y * d2.y + o4.x * (1.0 - d2.y);
}

fn rnd3(co: vec2<f32>) -> vec3<f32> {
    let a = fract(cos(co.x * 8.3e-3 + co.y) * vec3<f32>(1.3e5, 4.7e5, 2.9e5));
    let b = fract(sin(co.x * 0.3e-3 + co.y) * vec3<f32>(8.1e5, 1.0e5, 0.1e5));
    return mix(a, b, 0.5);
}

fn getZoom(t: f32) -> f32 {
    let at = max(0.0, t - params.zoom_delay);
    let rz = params.min_zoom + (exp(at * params.zoom_speed) - 1.0);
    return min(rz, params.max_zoom);
}

fn bhLens(uv: vec2<f32>, hp: vec2<f32>, m: f32, zf: f32) -> vec2<f32> {
    let dir = uv - hp;
    let d = length(dir);
    let rs = m * 0.1 * (1.0 + log(zf) * params.zoom_base);
    let def = rs / (d * d) * (1.0 - exp(-d * 5.0));
    let dp = uv - normalize(dir) * def * (1.0 + log(zf) * 0.05);
    let dE = exp(-d * 10.0) * sin(d * 50.0 - u_time.time * 2.0) * 0.02;
    return dp + vec2<f32>(dE * dir.y, -dE * dir.x);
}

fn eF(pin: vec3<f32>, s: f32, cx: f32, spd: f32, zf: f32) -> f32 {
    var p = pin * (1.0 + log(zf) * params.zoom_base);
    let str = 7.0 + 0.03 * log(1.0e-6 + fract(sin(u_time.time * spd) * 4373.11));
    var acc = s / 4.0;
    var prv = 0.0;
    var tw = 0.0;
    
    for(var i = 0.0; i < cx; i += 1.0) {
        let mg = dot(p, p);
        p = abs(p) / mg + vec3<f32>(params.space_distort_x, params.space_distort_y, params.space_distort_z);
        let w = exp(-i / 7.0);
        acc += w * exp(-str * pow(abs(mg - prv), 2.2));
        tw += w;
        prv = mg;
    }
    return max(0.0, 5.0 * acc / tw - 0.7);
}

fn genStar(sd: vec2<f32>, br: f32, t: f32, zf: f32) -> vec4<f32> {
    let r = rnd3(sd);
    let off = vec2<f32>(
        sin(t * (r.x * 0.5 + 0.1) + r.y * 6.28) * 0.01,
        cos(t * (r.y * 0.5 + 0.1) + r.x * 6.28) * 0.01
    ) / zf;
    
    let sz = mix(0.0, 1.5, pow(r.x, 2.0)) * (1.0 + log(zf) * 0.1);
    var sC = mix(vec3<f32>(1.0, 0.8, 0.6), vec3<f32>(0.6, 0.8, 1.0), r.z);
    var fSz = sz;
    
    if(r.x > 0.99) {
        sC = vec3<f32>(1.0, 0.4, 0.2);
        fSz *= 2.0;
    } else if(r.y > 0.995) {
        sC = vec3<f32>(0.6, 0.8, 1.0);
        fSz *= 1.5;
    }
    
    let pSpd = r.y * 2.0 + 0.5;
    let pF = 1.0 + 0.2 * sin(t * pSpd);
    var sBr = pow(r.y, 20.0) * br * pF;
    sBr *= 1.0 + 0.3 * sin(t * 5.0 + r.x * 100.0);
    
    return vec4<f32>(sC * sBr * fSz, sBr) * (1.0 - smoothstep(1.0, 2.0, zf));
}

@fragment
fn fs_main(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = 2.0 * vec2<f32>(fc.x, u_resolution.dimensions.y - fc.y) / u_resolution.dimensions - 1.0;
    let uvs = uv * u_resolution.dimensions / max(u_resolution.dimensions.x, u_resolution.dimensions.y);
    
    let zf = getZoom(u_time.time);
    let sUv = uvs / zf;
    let bhp = vec2<f32>(0.0);
    let lUv = bhLens(sUv, bhp, 0.15, zf);
    
    var p = vec3<f32>(lUv / 4.0, 0.0) + vec3<f32>(1.0, -1.3, 0.0);
    p += 0.2 * vec3<f32>(
        sin(u_time.time / 16.0),
        sin(u_time.time / 12.0),
        sin(u_time.time / 128.0)
    );
    
    let zs = 1.0 + log(zf) * params.zoom_base;
    var fq: array<f32, 4>;
    fq[0] = noise(vec3<f32>(0.01 * params.noise_scale * zs, 0.25, u_time.time / 10.0));
    fq[1] = noise(vec3<f32>(0.07 * params.noise_scale * zs, 0.25, u_time.time / 10.0));
    fq[2] = noise(vec3<f32>(0.15 * params.noise_scale * zs, 0.25, u_time.time / 10.0));
    fq[3] = noise(vec3<f32>(0.30 * params.noise_scale * zs, 0.25, u_time.time / 10.0));
    
    let t1 = eF(p, fq[2], 26.0, 1.0, zf);
    let t2 = eF(p * 1.5, fq[3], 18.0, 1.0 * 0.7, zf);
    let t3 = eF(p * 0.5, fq[1], 22.0, 1.0 * 1.3, zf);
    
    let v = (1.0 - exp((abs(uv.x) - 1.0) * 6.0)) * (1.0 - exp((abs(uv.y) - 1.0) * 6.0));
    
    let nC1 = mix(fq[3] - 0.3, 1.0, v) * vec4<f32>(1.5 * fq[2] * t1 * t1 * t1, 2.2 * fq[1] * t1 * t1, fq[3] * t1, 1.0);
    let nC2 = mix(fq[2] - 0.2, 1.0, v) * vec4<f32>(1.3 * fq[1] * t2 * t2 * t2, 1.8 * fq[3] * t2 * t2, fq[2] * t2, 0.8);
    let nC3 = mix(fq[1] - 0.1, 1.0, v) * vec4<f32>(4.2 * fq[3] * t3 * t3, 1.4 * fq[2] * t3 * t3, fq[1] * t3, 0.6);
    
    var sc = vec4<f32>(0.0);
    for(var i = 0.0; i < 3.0; i += 1.0) {
        let sd = (p.xy + i * 0.5) * 2.0;
        let ps = floor(sd * u_resolution.dimensions.x);
        sc += genStar(ps, 1.2 - i * 0.3, u_time.time, zf);
    }
    
    let th = lUv - bhp;
    let dh = length(th);
    let dg = exp(-dh * 10.0) * (2.0 + log(zf) * 0.5);
    let ad = vec4<f32>(params.disk_color[0], params.disk_color[1], params.disk_color[2], params.disk_color[3]) * dg;
    
    var col = nC1 + nC2 * 0.8 + nC3 * 0.6 + sc + ad;
    
    let zci = 1.0 + log(zf) * 0.1;
    col = vec4<f32>(pow(col.rgb, vec3<f32>(0.9 / zci)), col.a);
    
    let sCol = vec4<f32>(
        smoothstep(0.0, 1.0, col.r),
        smoothstep(0.0, 1.0, col.g),
        smoothstep(0.0, 1.0, col.b),
        col.a
    );
    col = mix(col, sCol, 0.2);
    
    if(zf > 1.0) {
        let rb = normalize(uv - bhp) * 0.001 * log(zf);
        let bUv = uv + rb;
        let bCol = col;
        col = mix(col, bCol, 0.5);
    }
    
    col = vec4<f32>(pow(col.rgb, vec3<f32>(params.time_scale)), col.a);
    return col;
}