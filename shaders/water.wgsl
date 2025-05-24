// Arctic Night Ocean 2025, Enes Altun; MIT License
// REFERENCES and Inspirations:
// But please:
// still if you want to port/use this shader please give credits to these authors, my code is under MIT License.
// Waves: procedural ocean: https://www.shadertoy.com/view/MdXyzX;  afl_ext (2017) : MIT License
// Aurora: https://www.shadertoy.com/view/XtGGRt nimitz, (2017): CC 3.0 BY-NC
// Cheap Stars: https://www.shadertoy.com/view/lsfGWH urraka (2013): ShaderToy Default License
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct WaterParams {
    camera_pos_x: f32,
    camera_pos_y: f32,
    camera_pos_z: f32,
    camera_yaw: f32,
    camera_pitch: f32,
    
    water_depth: f32,
    drag_mult: f32,
    camera_height: f32,
    
    wave_iterations_raymarch: u32,
    wave_iterations_normal: u32,
    
    time_speed: f32,
    sun_speed: f32,
    
    mouse_x: f32,
    mouse_y: f32,
    
    atmosphere_intensity: f32,
    water_color_r: f32,
    water_color_g: f32,
    water_color_b: f32,
    
    sun_color_r: f32, 
    sun_color_g: f32,
    sun_color_b: f32,
    
    cloud_coverage: f32,
    cloud_speed: f32,
    cloud_height: f32,
    
    night_sky_r: f32,
    night_sky_g: f32,
    night_sky_b: f32,
    
    exposure: f32,
    gamma: f32,
    
    fresnel_strength: f32,
    reflection_strength: f32,
};
@group(1) @binding(0) var<uniform> params: WaterParams;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias v4 = vec4<f32>;
alias m3 = mat3x3<f32>;
alias m2 = mat2x2<f32>;
const pi = 3.14159265;
const tau = 6.28318531;

var<private> R: v2;

//hash
fn h21(p: v2) -> f32 {
    return fract(sin(dot(p, v2(12.99, 78.23))) * 43758.55);
}

fn h23(p: v2) -> v3 {
    let p3 = fract(v3(p.xyx) * v3(.103, .103, .097));
    return fract((p3 + dot(p3, p3.yzx + 33.33)).xxy * (p3 + dot(p3, p3.yzx + 33.33)).yzz);
}

//rotate matrix
fn mm2(a: f32) -> m2 {
    let c = cos(a); let s = sin(a);
    return m2(c, s, -s, c);
}

//triangle wave
fn tri(x: f32) -> f32 { return clamp(abs(fract(x) - .5), .01, .49); }
fn tri2(p: v2) -> v2 { return v2(tri(p.x) + tri(p.y), tri(p.y + tri(p.x))); }

//noise for aurora
fn tnoise(p: v2, spd: f32) -> f32 {
    let t = time_data.time * spd;
    var z = 1.8; var z2 = 2.5; var rz = 0.0;
    var pp = p * mm2(p.x * .06);
    var bp = pp;
    
    for (var i = 0; i < 3; i++) {
        let dg = tri2(bp * 1.85) * .75;
        pp -= dg * mm2(t) / z2;
        bp *= 1.3; z2 *= .45; z *= .42;
        pp *= 1.21 + (rz - 1.0) * .02;
        rz += tri(pp.x + tri(pp.y)) * z;
        pp *= mm2(2.4);
    }
    return clamp(1.0 / pow(rz * 29.0, 1.3), 0.0, .55);
}

//rotation
fn rot(axis: v3, a: f32) -> m3 {
    let s = sin(a); let c = cos(a); let oc = 1.0 - c;
    return m3(
        oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
        oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
        oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c
    );
}

//wave dx
fn wave(p: v2, d: v2, f: f32, ts: f32) -> v2 {
    let x = dot(d, p) * f + ts;
    let w = exp(sin(x) - 1.0);
    return v2(w, -w * cos(x));
}

//waves
fn waves(p: v2, iter: i32) -> f32 {
    let phase = length(p) * .1;
    var i = 0.0; var f = 1.0; var tm = 2.0; var w = 1.0;
    var sv = 0.0; var sw = 0.0; var pos = p;
    
    let dist = length(p);
    let maxIter = min(iter, i32(mix(f32(iter), 4.0, clamp(dist / 50.0, 0.0, 1.0))));
    
    for (var j = 0; j < maxIter; j++) {
        let d = v2(sin(i), cos(i));
        let res = wave(pos, d, f, time_data.time * params.time_speed * tm + phase);
        
        pos += d * res.y * w * params.drag_mult;
        sv += res.x * w; sw += w;
        
        w = mix(w, 0.0, .2);
        f *= 1.18; tm *= 1.07; i += 1232.4;
    }
    
    return sv / sw;
}

//raymarch water
fn rmw(cam: v3, start: v3, end: v3, depth: f32) -> f32 {
    var pos = start;
    let d = normalize(end - start);
    
    for (var i = 0; i < 64; i++) {
        let h = waves(pos.xz, i32(params.wave_iterations_raymarch)) * depth - depth;
        if (h + .01 > pos.y) { return distance(pos, cam); }
        pos += d * (pos.y - h);
    }
    
    return distance(start, cam);
}

//normal
fn norm(p: v2, e: f32, depth: f32) -> v3 {
    let ex = v2(e, 0.0);
    let H = waves(p, i32(params.wave_iterations_normal)) * depth;
    let a = v3(p.x, H, p.y);
    
    return normalize(cross(
        a - v3(p.x - e, waves(p - ex.xy, i32(params.wave_iterations_normal)) * depth, p.y),
        a - v3(p.x, waves(p + ex.yx, i32(params.wave_iterations_normal)) * depth, p.y + e)
    ));
}

//moon dir
fn moon(t: f32) -> v3 {
    let mt = t * params.sun_speed;
    let ma = mt * pi * .5;
    
    let my = sin(ma) * .6 + .2;
    let mx = cos(ma) * .4;
    let mz = cos(ma * .7) * .8;
    
    return normalize(v3(mx, my, mz));
}

//stars
fn stars(fc: v2, lod: f32) -> f32 {
    if (lod > .5) { return 0.0; }
    
    let sz = 30.0; let prob = 1.95;
    let pos = floor(fc / sz);
    
    var col = 0.0;
    let sv = h21(pos);
    
    if (sv > prob) {
        let c = sz * pos + sz * .5;
        let t = .9 + .2 * sin(time_data.time + (sv - prob) / (1.0 - prob) * 45.0);
        let d = distance(fc, c);
        col = 1.0 - d / (sz * .5);
        
        let dx = abs(fc.x - c.x);
        let dy = abs(fc.y - c.y);
        
        if (dx > .1 && dy > .1) {
            col = col * t / max(dy, .1) * t / max(dx, .1);
        } else {
            col *= t * 2.0;
        }
    } else if (h21(fc / R) > .996) {
        let r = h21(fc);
        col = r * (.25 * sin(time_data.time * (r * 5.0) + 720.0 * r) + .75);
    }
    
    return clamp(col, 0.0, 1.0);
}

//aurora
fn aurora(ro: v3, rd: v3, q: f32) -> v4 {
    var col = v4(0.0); var avg = v4(0.0);
    
    let ai = params.cloud_coverage;
    let asp = params.cloud_speed;
    
    if (ai < .05 || rd.y <= 0.0) { return col; }
    
    let iter = i32(mix(10.0, 20.0, q));
    
    for (var i = 0; i < iter; i++) {
        let fi = f32(i);
        let off = .006 * params.cloud_height * h21(ro.xz + rd.xz * fi) * smoothstep(0.0, 15.0, fi);
        let pt = ((.8 + pow(fi, 1.4) * .002 * params.cloud_height) - ro.y) / (rd.y * 2.0 + .4);
        let bp = ro + (pt - off) * rd;
        let rzt = tnoise(bp.xz * .8, asp * .8);
        var c2 = v4(0.0, 0.0, 0.0, rzt);
        
        let cp = sin(1.0 - v3(2.15, -.5, 1.2) + fi * .043) * .5 + .5;
        let cp2 = sin(v3(1.5, 2.8, .8) + fi * .067 + time_data.time * .1) * .5 + .5;
        
        c2 = v4((cp.x + cp2.z * .3) * rzt * .9, (cp.y + cp2.x * .4) * rzt * 1.4, 
                 (cp.z + cp2.y * .2) * rzt * 1.1, rzt);
        
        avg = mix(avg, c2, .5);
        col += avg * exp2(-fi * .065 - 2.0) * smoothstep(0.0, 8.0, fi);
    }
    
    return col * clamp(rd.y * 18.0 + .3, 0.0, 1.0) * ai * 2.5;
}

//sky atmosphere
fn sky(rd: v3, md: v3, t: f32, q: f32) -> v3 {
    let mdot = dot(rd, md);
    let mb = max(0.0, md.y);
    
    let dark = v3(params.night_sky_r, params.night_sky_g, params.night_sky_b) * .3;
    let horiz = v3(params.night_sky_r, params.night_sky_g, params.night_sky_b);
    let mlit = v3(.15, .18, .25);
    
    let grad = smoothstep(-.2, 1.0, rd.y);
    var sc = mix(horiz, dark, grad);
    sc = mix(sc, mlit, mb * .3);
    
    let mglow = pow(max(0.0, mdot), 128.0) * mb;
    sc += v3(params.sun_color_r, params.sun_color_g, params.sun_color_b) * mglow;
    
    let scatt = pow(max(0.0, 1.0 - rd.y), 3.0);
    let mscatt = pow(max(0.0, mdot), 8.0) * mb * scatt;
    sc += v3(.8, .9, 1.0) * mscatt * .03;
    
    if (rd.y > 0.0 && q > .5) {
        let fc = (rd.xz / (rd.y + .1)) * R.x * .5 + R * .5;
        let sf = stars(fc, 1.0 - q);
        let scol = mix(v3(.8, .9, 1.0), v3(1.0, .9, .8), h21(fc * .01));
        sc += sf * scol * (1.0 - mb * .1);
    }
    
    if (rd.y > 0.0 && params.cloud_coverage > .05) {
        let ar = aurora(v3(0.0), rd, q);
        sc = sc * (1.0 - ar.a * .8) + ar.rgb;
    }
    
    return sc * params.atmosphere_intensity;
}

//moon disk
fn mdisk(rd: v3, md: v3) -> f32 {
    let mdot = dot(rd, md);
    let mb = max(0.0, md.y);
    let phase = sin(time_data.time * .01) * .2 + .8;
    
    return pow(max(0.0, mdot), 3500.0) * 25.0 * mb * phase;
}

//plane intersect
fn plane(o: v3, d: v3, p: v3, n: v3) -> f32 {
    return clamp(dot(p - o, n) / dot(d, n), -1.0, 9991999.0);
}

//camera ray
fn ray(fc: v2) -> v3 {
    let uv = ((fc / R) * 2.0 - 1.0) * v2(R.x / R.y, 1.0);
    var proj = normalize(v3(uv, 1.5));
    
    let ym = rot(v3(0.0, -1.0, 0.0), params.camera_yaw);
    let pm = rot(v3(1.0, 0.0, 0.0), params.camera_pitch);
    
    return ym * (pm * proj);
}

//tonemap
fn tonemap(c: v3) -> v3 {
    let m1 = m3(.597, .076, .028, .355, .908, .134, .048, .016, .838);
    let m2 = m3(1.605, -.102, -.003, -.531, 1.108, -.073, -.074, -.006, 1.076);
    
    let v = m1 * c;
    let a = v * (v + .0246) - .000091;
    let b = v * (.9837 * v + .433) + .238;
    return pow(clamp(m2 * (a / b), v3(0.0), v3(1.0)), v3(1.0 / params.gamma));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let fc = v2(f32(gid.x), f32(dims.y - gid.y));
    let rd = ray(fc);
    
    let ct = time_data.time * params.time_speed;
    let md = moon(ct);
    
    let o = v3(params.camera_pos_x + ct * .02, params.camera_pos_y, params.camera_pos_z);
    
    //underwater
    if (o.y < 0.0) {
        let depth = -o.y;
        let df = exp(-depth * .15);
        var uw = v3(.01, .04, .08) * df;
        
        let mb = max(0.0, md.y);
        uw += max(0.0, rd.y) * .05 * df * mb * v3(.01, .02, .04);
        
        textureStore(output, gid.xy, vec4<f32>(tonemap(uw * params.exposure), 1.0));
        return;
    }
    
    //sky
    if (rd.y >= 0.0) {
        var sc = sky(rd, md, ct, 1.0);
        let mc = v3(params.sun_color_r, params.sun_color_g, params.sun_color_b);
        sc += mdisk(rd, md) * mc;
        
        textureStore(output, gid.xy, vec4<f32>(tonemap(sc * params.exposure), 1.0));
        return;
    }
    
    //water
    let wh = v3(0.0, 0.0, 0.0);
    let wl = v3(0.0, -params.water_depth, 0.0);
    
    let hh = plane(o, rd, wh, v3(0.0, 1.0, 0.0));
    let lh = plane(o, rd, wl, v3(0.0, 1.0, 0.0));
    let hp = o + rd * hh;
    let lp = o + rd * lh;
    
    let dist = rmw(o, hp, lp, params.water_depth);
    let whp = o + rd * dist;
    
    let dq = 1.0 - clamp(dist / 50.0, 0.0, 1.0);
    
    var N = norm(whp.xz, .01, params.water_depth);
    N = mix(N, v3(0.0, 1.0, 0.0), .8 * min(1.0, sqrt(dist * .01) * 1.1));
    
    let fresnel = (.04 + .96 * pow(1.0 - max(0.0, dot(-N, rd)), 5.0)) * params.fresnel_strength;
    
    var R = normalize(reflect(rd, N));
    R.y = abs(R.y);
    
    var refl = sky(R, md, ct, dq * .5);
    let mc = v3(params.sun_color_r, params.sun_color_g, params.sun_color_b);
    refl = (refl + mdisk(R, md) * mc) * params.reflection_strength;
    
    let mb = max(0.0, md.y);
    var wc = v3(params.water_color_r, params.water_color_g, params.water_color_b) * .4;
    wc = mix(wc, wc * v3(.7, .8, 1.3), .6);
    
    let mci = v3(params.sun_color_r, params.sun_color_g, params.sun_color_b);
    wc = mix(wc, wc * mci, mb * .02);
    
    let scatt = wc * .6 * (.4 + (whp.y + params.water_depth) / params.water_depth);
    
    let result = tonemap((fresnel * refl + scatt) * params.exposure);
    textureStore(output, gid.xy, vec4<f32>(result, 1.0));
}