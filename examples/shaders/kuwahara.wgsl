// Kuwahara Filter, Enes Altun, 2025, MIT License
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: KuwaharaParams;

@group(2) @binding(0) var channel0: texture_2d<f32>;
@group(2) @binding(1) var channel0_sampler: sampler;

@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;
@group(3) @binding(4) var input_texture2: texture_2d<f32>;
@group(3) @binding(5) var input_sampler2: sampler;

struct KuwaharaParams {
    radius: f32,
    q: f32,
    alpha: f32,
    filter_strength: f32,
    sigma_d: f32,
    sigma_r: f32,
    edge_threshold: f32,
    color_enhance: f32,
    filter_mode: i32,
    show_tensors: i32,
    _pad1: u32,
    _pad2: u32, 
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

const PI: f32 = 3.14159265359;

const BLUR_SAMPLES: i32 = 35;
const BLUR_LOD: i32 = 2; 
const BLUR_SLOD: i32 = 4;

fn gaussian_weight(i: vec2f, sigma: f32) -> f32 {
    let si = i / sigma;
    return exp(-0.5 * dot(si, si)) / (6.28 * sigma * sigma);
}

fn blur_tensor(uv: vec2f, ts: vec2f) -> vec3f {
    var result = vec3f(0.0);
    var tw = 0.0;
    let s = BLUR_SAMPLES / BLUR_SLOD;
    let sig = params.sigma_r * 2.5;
    let lod = max(0.0, params.sigma_d - 0.5);
    
    for (var i = 0; i < s * s; i++) {
        let d = vec2f(f32(i % s), f32(i / s)) * f32(BLUR_SLOD) - f32(BLUR_SAMPLES) / 2.0;
        let w = gaussian_weight(d, sig);
        let suv = clamp(uv + ts * d, vec2f(0.0), vec2f(1.0));
        let td = textureSampleLevel(input_texture0, input_sampler0, suv, lod);
        
        result += td.xyz * w;
        tw += w;
    }
    
    return result / tw;
}

fn calc_region_stats(uv: vec2f, lower: vec2i, upper: vec2i, ts: vec2f) -> vec2f {
    var csum = vec3f(0.0);
    var cvar = vec3f(0.0);
    var cnt = 0;
    
    for (var j = lower.y; j <= upper.y; j++) {
        for (var i = lower.x; i <= upper.x; i++) {
            let off = vec2f(f32(i), f32(j)) * ts;
            let suv = clamp(uv + off, vec2f(0.0), vec2f(1.0));
            let sc = get_input_color(suv);
            
            csum += sc;
            cvar += sc * sc;
            cnt++;
        }
    }
    
    if (cnt > 0) {
        let mc = csum / f32(cnt);
        let rv = cvar / f32(cnt) - (mc * mc);
        let tv = rv.r + rv.g + rv.b;
        let lum = dot(mc, vec3f(0.299, 0.587, 0.114));
        let cv = tv * 0.7 + dot(rv, vec3f(0.299, 0.587, 0.114)) * 0.3;
        
        return vec2f(lum, cv);
    }
    return vec2f(0.0, 999999.0);
}

fn ACESFilm(color: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
}

fn saturate(rgb: vec3f, adj: f32) -> vec3f {
    let W = vec3f(0.2125, 0.7154, 0.0721);
    let intensity = vec3f(dot(rgb, W));
    return mix(intensity, rgb, adj);
}

fn get_input_color(uv: vec2f) -> vec3f {
    let dims = textureDimensions(channel0);
    if (dims.x > 1 && dims.y > 1) {
        return textureSampleLevel(channel0, channel0_sampler, uv, 0.0).rgb;
    }
    let center = vec2f(0.5);
    let dist = distance(uv, center);
    let circle = smoothstep(0.2, 0.21, dist);
    return mix(vec3f(0.8, 0.4, 0.2), vec3f(0.1, 0.1, 0.2), circle);
}

// structure tensor pass
@compute @workgroup_size(16, 16, 1)
fn structure_tensor(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let ts = 1.0 / vec2f(dims);
    let d = ts * params.sigma_d;
    
    // sobel kernels
    let sx = (
        -1.0 * get_input_color(clamp(uv + vec2f(-d.x, -d.y), vec2f(0.0), vec2f(1.0))) +
        -2.0 * get_input_color(clamp(uv + vec2f(-d.x,  0.0), vec2f(0.0), vec2f(1.0))) + 
        -1.0 * get_input_color(clamp(uv + vec2f(-d.x,  d.y), vec2f(0.0), vec2f(1.0))) +
        1.0 * get_input_color(clamp(uv + vec2f( d.x, -d.y), vec2f(0.0), vec2f(1.0))) +
        2.0 * get_input_color(clamp(uv + vec2f( d.x,  0.0), vec2f(0.0), vec2f(1.0))) + 
        1.0 * get_input_color(clamp(uv + vec2f( d.x,  d.y), vec2f(0.0), vec2f(1.0)))
    ) / (4.0);

    let sy = (
        -1.0 * get_input_color(clamp(uv + vec2f(-d.x, -d.y), vec2f(0.0), vec2f(1.0))) + 
        -2.0 * get_input_color(clamp(uv + vec2f( 0.0, -d.y), vec2f(0.0), vec2f(1.0))) + 
        -1.0 * get_input_color(clamp(uv + vec2f( d.x, -d.y), vec2f(0.0), vec2f(1.0))) +
        1.0 * get_input_color(clamp(uv + vec2f(-d.x,  d.y), vec2f(0.0), vec2f(1.0))) +
        2.0 * get_input_color(clamp(uv + vec2f( 0.0,  d.y), vec2f(0.0), vec2f(1.0))) + 
        1.0 * get_input_color(clamp(uv + vec2f( d.x,  d.y), vec2f(0.0), vec2f(1.0)))
    ) / 4.0;
    
    // rgb gradients
    let gr = length(vec2f(sx.r, sy.r));
    let gg = length(vec2f(sx.g, sy.g));
    let gb = length(vec2f(sx.b, sy.b));
    
    let cw = vec3f(0.299, 0.587, 0.114);
    let wg = gr * cw.r + gg * cw.g + gb * cw.b;
    
    let gx = dot(sx, cw) + (gr + gg + gb) * 0.1;
    let gy = dot(sy, cw) + (gr + gg + gb) * 0.1;
    
    // tensor components
    let Jxx = gx * gx + wg * 0.05;
    let Jyy = gy * gy + wg * 0.05;
    let Jxy = gx * gy;
    
    textureStore(output, id.xy, vec4f(Jxx, Jyy, Jxy, wg));
}

// tensor field pass
@compute @workgroup_size(16, 16, 1)
fn tensor_field(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let ts = 1.0 / vec2f(dims);

    let st = blur_tensor(uv, ts);
    
    let Jxx = st.x;
    let Jyy = st.y;
    let Jxy = st.z;

    // eigenvalues
    let l1 = 0.5 * (Jyy + Jxx + sqrt(Jyy * Jyy - 2.0 * Jxx * Jyy + Jxx * Jxx + 4.0 * Jxy * Jxy));
    let l2 = 0.5 * (Jyy + Jxx - sqrt(Jyy * Jyy - 2.0 * Jxx * Jyy + Jxx * Jxx + 4.0 * Jxy * Jxy));

    // eigenvector
    var v = vec2f(l1 - Jxx, -Jxy);
    var ori: vec2f;
    if (length(v) > 0.0) { 
        ori = normalize(v);
    } else {
        ori = vec2f(0.0, 1.0);
    }

    let phi = atan2(ori.y, ori.x);
    
    // anisotropy
    var anis = 0.0;
    if (l1 + l2 > 0.0) {
        anis = (l1 - l2) / (l1 + l2);
    }
    
    textureStore(output, id.xy, vec4f(ori.x, ori.y, phi, anis));
}

// kuwahara filter pass
@compute @workgroup_size(16, 16, 1) 
fn kuwahara_filter(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let ts = 1.0 / vec2f(dims);
    
    let orig = get_input_color(uv);
    var result = vec4f(orig, 1.0);
    
    if (params.filter_mode == 0) {
        // classic mode
        let r = i32(params.radius);
        
        var qmean: array<vec4f, 4>;
        var qvar: array<f32, 4>;
        
        for (var dy = -r; dy <= r; dy++) {
            for (var dx = -r; dx <= r; dx++) {
                let off = vec2f(f32(dx), f32(dy)) * ts;
                let suv = clamp(uv + off, vec2f(0.0), vec2f(1.0));
                let sc = get_input_color(suv);
                
                var q = 0;
                if (dx >= 0 && dy < 0) { q = 1; }   
                else if (dx < 0 && dy >= 0) { q = 2; }   
                else if (dx >= 0 && dy >= 0) { q = 3; }   
                
                qmean[q] += vec4f(sc, 1.0);
                let ri = length(sc);
                qvar[q] += ri * ri;
            }
        }
        
        var minvar = 999999.0;
        var selq = 0;
        
        for (var q = 0; q < 4; q++) {
            if (qmean[q].w > 0.0) {
                let mc = qmean[q].rgb / qmean[q].w;
                let mi = length(mc);
                let variance = (qvar[q] / qmean[q].w) - (mi * mi);
                let avar = variance / (params.q * params.q);
                
                if (avar < minvar) {
                    minvar = avar;
                    selq = q;
                }
            }
        }
        
        if (qmean[selq].w > 0.0) {
            let sc = qmean[selq].rgb / qmean[selq].w;
            result = vec4f(mix(orig, sc, params.filter_strength), 1.0);
        }
    } else {
        // anisotropic mode
        let td = textureSampleLevel(input_texture1, input_sampler1, uv, 0.0);
        let ori = td.xy;
        let anis = td.w;
        
        let alpha = params.alpha;
        let radius = params.radius;
        
        let eff_anis = select(0.0, anis, anis > params.edge_threshold);
        
        let a = radius * (1.0 + eff_anis * alpha * 0.8);
        let b = radius * max(0.3, 1.0 - eff_anis * alpha * 0.6);
        
        var qmeans: array<vec3f, 4>;
        var qvars: array<f32, 4>;
        var qcnts: array<f32, 4>;
        
        for (var k = 0; k < 4; k++) {
            qmeans[k] = vec3f(0.0);
            qvars[k] = 0.0;
            qcnts[k] = 0.0;
        }
        
        let maxr = i32(min(radius + 2.0, 10.0));
        for (var j = -maxr; j <= maxr; j++) {
            for (var i = -maxr; i <= maxr; i++) {
                let off = vec2f(f32(i), f32(j));
                
                let ex = off.x * ori.x + off.y * ori.y;
                let ey = -off.x * ori.y + off.y * ori.x;
                let ed = (ex * ex) / (a * a) + (ey * ey) / (b * b);
                
                if (ed <= 1.0) {
                    let suv = clamp(uv + off * ts, vec2f(0.0), vec2f(1.0));
                    let sc = get_input_color(suv);
                    let ri = length(sc);
                    
                    if (i <= 0 && j <= 0) { 
                        qmeans[0] += sc;
                        qvars[0] += ri * ri;
                        qcnts[0] += 1.0;
                    }
                    if (i >= 0 && j <= 0) {  
                        qmeans[1] += sc;
                        qvars[1] += ri * ri;
                        qcnts[1] += 1.0;
                    }
                    if (i <= 0 && j >= 0) { 
                        qmeans[2] += sc;
                        qvars[2] += ri * ri;
                        qcnts[2] += 1.0;
                    }
                    if (i >= 0 && j >= 0) { 
                        qmeans[3] += sc;
                        qvars[3] += ri * ri;  
                        qcnts[3] += 1.0;
                    }
                }
            }
        }
        
        var minvar = 999999.0;
        var best = orig;
        
        for (var q = 0; q < 4; q++) {
            if (qcnts[q] > 0.0) {
                let mc = qmeans[q] / qcnts[q];
                let mi = length(mc);
                let variance = (qvars[q] / qcnts[q]) - (mi * mi);
                let avar = variance / (params.q * params.q);
                
                if (avar < minvar) {
                    minvar = avar;
                    best = mc;
                }
            }
        }
        
        result = vec4f(mix(orig, best, params.filter_strength), 1.0);
    }
    
    // color enhance
    var fc = result.rgb;
    
    if (abs(params.color_enhance - 1.0) > 0.01) {
        let enh = params.color_enhance;
        let lum = dot(fc, vec3f(0.299, 0.587, 0.114));
        let sat_factor = mix(1.0, enh * 1.2, 0.5);
        fc = mix(vec3f(lum), fc, sat_factor);
        let contrast_factor = 0.9 + (enh - 1.0) * 0.1;
        fc = (fc - 0.5) * contrast_factor + 0.5;
        fc = clamp(fc, vec3f(0.0), vec3f(1.0));
    }
    
    result = vec4f(fc, result.a);
    
    textureStore(output, id.xy, result);
}

// main image pass
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let result = textureSampleLevel(input_texture2, input_sampler2, uv, 0.0);
    
    textureStore(output, id.xy, result);
}