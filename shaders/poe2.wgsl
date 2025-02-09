
//MIT License, Enes Altun, 2025

struct TimeUniform {
    time: f32,
};
struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};
struct Params {
    bg_color: vec3<f32>,
    _pad1: f32,
    hi_color: vec3<f32>,
    _pad2: f32,
    sh_color: vec3<f32>,
    _pad3: f32,
    
    r_base: vec3<f32>,
    _pad4: f32,
    r_dark: vec3<f32>,
    _pad5: f32,
    m_spec: vec3<f32>,
    _pad6: f32,
    
    light_pos: vec2<f32>,
    ao_strength: f32,
    noise_scale: f32,
    
    wear_amount: f32,
    wear_scale: f32,
    speed: f32,
    _pad7: f32,
};

@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: Params;

fn hash(p: vec2<f32>) -> f32 {
    var p2 = 50.0 * fract(p * 0.31831 + vec2<f32>(0.71, 0.113));
    return -1.0 + 2.0 * fract(p2.x * p2.y * (p2.x + p2.y));
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    
    let n00 = hash(i);
    let n01 = hash(i + vec2<f32>(0.0, 1.0));
    let n10 = hash(i + vec2<f32>(1.0, 0.0));
    let n11 = hash(i + vec2<f32>(1.0, 1.0));
    
    return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}

fn getLit(n: vec2<f32>, ao_: f32, p: vec2<f32>) -> vec3<f32> {
    let ld = normalize(params.light_pos);
    let li = dot(n, ld) * 0.5 + 0.5;
    let e = pow(1.0 - abs(dot(n, normalize(p))), 5.0);
    
    let dLit = mix(params.sh_color, params.hi_color, li);
    let eLit = params.hi_color * e * 0.5;
    
    return (dLit + eLit) * (1.0 - ao_ * params.ao_strength);
}

fn getEnv(n: vec2<f32>) -> vec3<f32> {
    let tl = n.y * 0.5 + 0.5;
    return mix(params.sh_color, params.hi_color, tl) * 0.3;
}

fn getMetal(p: vec2<f32>, n: vec2<f32>, ao_: f32, sh: f32) -> vec3<f32> {
    let e = pow(1.0 - abs(dot(n, normalize(p))), 3.0);
    let bCol = mix(params.r_dark, params.r_base, sh);
    let baseCol = bCol * (1.0 - ao_ * params.ao_strength);
    
    let spec = pow(max(0.3, dot(n, normalize(params.light_pos))), 4.0);
    
    return mix(baseCol, params.m_spec, spec * 0.4 + e * 0.3);
}

fn getBg(uv: vec2<f32>, n: vec2<f32>) -> vec3<f32> {
    let g = (uv.y + 0.5) * 0.5;
    let p = noise(uv) * params.noise_scale;
    return mix(params.bg_color, params.hi_color * 0.2, g + p);
}

fn sdGear(p: vec2<f32>, r: f32, th: f32, tc: f32, t: f32) -> f32 {
    let a = atan2(p.y, p.x);
    let len = length(p);
    
    let ta = a * tc;
    let mz = smoothstep(-0.3, 0.3, cos(a + t));
    let cf = 1.0 - mz * 0.15;
    
    let teeth = th * cf * 
                sign(cos(ta)) * 
                smoothstep(0.0, 0.4, abs(cos(ta))) *
                (1.0 + 0.15 * sin(ta * 2.0 + t));
    
    let tr = 0.002 * smoothstep(0.8, 2.0, abs(cos(ta)));
    let gear = len - (r + teeth - tr);
    let hollow = len - (r * 0.6);
    
    let wear = params.wear_amount * noise(p * params.wear_scale + t * 1.1);
    
    return max(gear + wear, -hollow);
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (FragCoord.xy - 0.5 * u_resolution.dimensions) / u_resolution.dimensions.y;
    
    let r = 0.25;
    let th = 0.04;
    let gs = 2.0 * (r + th) - 0.1;
    
    let lgc = vec2<f32>(-gs/2.0, 0.0);
    var lgp = uv - lgc;
    let la = u_time.time * params.speed + (3.14159 / 8.0);
    lgp = vec2<f32>(
        lgp.x * cos(la) - lgp.y * sin(la),
        lgp.x * sin(la) + lgp.y * cos(la)
    );
    
    let rgc = vec2<f32>(gs/2.0, 0.0);
    var rgp = uv - rgc;
    let ra = -u_time.time * params.speed;
    rgp = vec2<f32>(
        rgp.x * cos(ra) - rgp.y * sin(ra),
        rgp.x * sin(ra) + rgp.y * cos(ra)
    );
    
    let lg = sdGear(lgp, r, th, 8.0, u_time.time * params.speed);
    let rg = sdGear(rgp, r, th, 8.0, -u_time.time * params.speed);
    let g = min(lg, rg);
    
    let n = normalize(vec2<f32>(dpdx(g), dpdy(g)));
    let sh = dot(n, normalize(params.light_pos)) * 0.5 + 0.5;
    
    var ao_ = 0.0;
    for(var i = 1; i <= 4; i = i + 1) {
        let d = f32(i) * 0.015;
        let sp = uv + n * d;
        let aos = min(
            sdGear(sp - lgc, r, th, 8.0, u_time.time * params.speed),
            sdGear(sp - rgc, r, th, 8.0, -u_time.time * params.speed)
        );
        ao_ += smoothstep(-0.005, 0.005, aos);
    }
    ao_ = ao_ / 4.0;
    
    let lit = getLit(n, ao_, uv);
    let mCol = getMetal(uv, n, ao_, sh);
    let bg = getBg(uv, n);
    
    let sm = smoothstep(-0.005, 0.005, g);
    var col = mix(mCol * lit, bg, sm);
    let vig = 1.5 - length(uv) * 0.3;
    col = col * vig;
    col = gamma(col, 0.41);
    return vec4<f32>(col, 1.0);
}
fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}