//This is an example shader that uses audio data to create a visualizer effect to show how to use audio data in a shader.
//MIT Licese, Enes Altun, 2025
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(1) @binding(0) var<uniform> u_time: TimeUniform;
@group(2) @binding(0) var<uniform> params: Params;
@group(3) @binding(0) var<uniform> u_resolution: ResolutionUniform;

struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
    audio_data: array<vec4<f32>, 32>,  // 128 processed bands
    bpm: f32,
};
struct TimeUniform {
    time: f32,
};
// These are unrelated, you can change them to match your needs (rust side)
struct Params { 
    red_power: f32,
    green_power: f32,
    blue_power: f32,
    green_boost: f32,
    contrast: f32, 
    gamma: f32,
    glow: f32,
}

// gAV: get audio value at any frequency (0-1 range), notice how we use audio_data array to get audio data from rust side
fn gAV(f: f32) -> f32 {
    let idx = f * 128.0;
    let i = u32(idx);
    let fp = idx - f32(i);
    
    if (i >= 128u) { return 0.0; }
    
    let vi = i / 4u;
    let vc = i % 4u;
    
    var v1 = 0.0;
    if (vc == 0u) { v1 = u_resolution.audio_data[vi].x; }
    else if (vc == 1u) { v1 = u_resolution.audio_data[vi].y; }
    else if (vc == 2u) { v1 = u_resolution.audio_data[vi].z; }
    else { v1 = u_resolution.audio_data[vi].w; }
    
    if (i >= 127u) { return v1; }
    
    let ni = i + 1u;
    let nvi = ni / 4u;
    let nvc = ni % 4u;
    
    var v2 = 0.0;
    if (nvc == 0u) { v2 = u_resolution.audio_data[nvi].x; }
    else if (nvc == 1u) { v2 = u_resolution.audio_data[nvi].y; }
    else if (nvc == 2u) { v2 = u_resolution.audio_data[nvi].z; }
    else { v2 = u_resolution.audio_data[nvi].w; }
    
    return mix(v1, v2, fp);
}

// gAR: get audio range energy
fn gAR(l: f32, h: f32) -> f32 {
    var s = 0.0;
    var c = 0.0;
    for(var i = l; i <= h; i += 0.02) {
        s += gAV(i);
        c += 1.0;
    }
    return s / c;
}

// nG: neon glow
fn nG(c: vec3<f32>, i: f32) -> vec3<f32> {
    return c * (1.0 + i * i * 4.0);
}

fn h21(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    return fract(p3.x * p3.y * 43758.5453);
}


fn sN(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(h21(i), h21(i + vec2<f32>(1.0, 0.0)), u.x),
        mix(h21(i + vec2<f32>(0.0, 1.0)), h21(i + vec2<f32>(1.0, 1.0)), u.x),
        u.y
    );
}


fn fbm(p: vec2<f32>, o: i32) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var f = 1.0;
    var p2 = p;
    
    for (var i = 0; i < o; i++) {
        v += a * sN(p2);
        p2 *= 2.0;
        a *= 0.5;
    }
    
    return v;
}

// sE: smoke effect
fn sE(uv: vec2<f32>, t: f32, i: f32) -> vec3<f32> {
    let s1 = fbm(uv * 4.0 + vec2<f32>(t * 0.05, 0.0), 3);
    let s2 = fbm(uv * 2.0 - vec2<f32>(0.0, t * 0.035), 2);
    let cs = mix(s1, s2, 0.5 + 0.5 * sin(t * 0.1));
    return vec3<f32>(0.01, 0.01, 0.03) * cs * i;
}

// hsv2rgb: convert HSV to RGB
fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = h * 12.0;
    let x = c * (1.0 - abs(fract(hp / 2.0) * 2.0 - 1.0));
    
    var rgb: vec3<f32>;
    if(hp < 1.0) { rgb = vec3<f32>(c, x, 0.0); }
    else if(hp < 2.0) { rgb = vec3<f32>(x, c, 0.0); }
    else if(hp < 3.0) { rgb = vec3<f32>(0.0, c, x); }
    else if(hp < 4.0) { rgb = vec3<f32>(0.0, x, c); }
    else if(hp < 5.0) { rgb = vec3<f32>(x, 0.0, c); }
    else { rgb = vec3<f32>(c, 0.0, x); }
    
    return rgb + vec3<f32>(v - c);
}

// bC: beam color cycling
fn bC(x: f32, t: f32) -> vec3<f32> {
    let xc = (x - (t / 8.0)) * 3.0;
    let mc = fract(xc) * 3.0;
    var c = vec3<f32>(0.25);
    
    if (mc < 1.0) {
        c.r += 1.0 - mc;
        c.g += mc;
    } else if (mc < 2.0) {
        let ac = mc - 1.0;
        c.g += 1.0 - ac;
        c.b += ac;
    } else {
        let ac = mc - 2.0;
        c.b += 1.0 - ac;
        c.r += ac;
    }
    
    return c;
}

// bE: bloom effect
fn bE(c: vec3<f32>) -> vec3<f32> {
    let bp = max(vec3<f32>(0.0), c - vec3<f32>(0.7));
    return c + bp * bp * 0.5;
}

@fragment
fn fs_main(@builtin(position) p: vec4<f32>, @location(0) tc: vec2<f32>) -> @location(0) vec4<f32> {
    let t = u_time.time;
    
    // Audio energy in different ranges
    let bE = gAR(0.0, 0.2);  // bass energy
    let mE = gAR(0.2, 0.6);  // mid energy
    let hE = gAR(0.6, 1.0);  // high energy
    let tE = (bE * 1.5 + mE + hE) / 3.5;  // total energy
    
    // fc: final color - start with dark background
    var fc = vec3<f32>(0.005, 0.005, 0.01);
    fc += sE(tc, t, 0.5 + tE * 0.5); 
    
    // Bar visualization parameters
    let eB = 0.15;  // eq bottom
    let eH = 0.7;   // eq height
    let bW = 1.0 / 130.0;  // band width
    let bS = bW * 0.3;  // band spacing
    

    let rF = 1.0 - eB;  // reflection floor
    let rD = 0.1;  // reflection depth
    
    // Draw frequency bars
    for (var i = 0; i < 128; i++) {
        let bX = (f32(i) + 1.0) * bW;  // band X position
        let bA = gAV(f32(i) / 128.0);  // band audio value
        
        let bH = bA * eH;  // bar height
        let bB = 1.0 - eB - bH;  // bar bottom
        let bT = 1.0 - eB;  // bar top
        
        let inX = tc.x >= bX && tc.x < bX + bW - bS;  // in X bounds
        let inY = tc.y <= bT && tc.y >= bB;  // in Y bounds
        
        let dX = min(abs(tc.x - bX), abs(tc.x - (bX + bW - bS)));  // distance to X edge
        let dY = min(abs(tc.y - bT), abs(tc.y - bB));  // distance to Y edge
        let eD = min(dX, dY);  // edge distance
        
        // Draw bar
        if (inX && inY) {
            let fT = f32(i) / 128.0;  // frequency factor
            let hT = (bT - tc.y) / max(bH, 0.001);  // height factor
            
            let tE = smoothstep(0.5, 0.0, length(vec2<f32>(
                (tc.x - (bX + (bW - bS) * 0.5)) / (bW - bS),
                (tc.y - (bB + bH * 0.5)) / bH
            )) * 2.0 - 0.5);
            
            //neon
            let nC = hsv2rgb(fT, 0.9, 0.7 + 0.3 * (1.0 - hT)) * (0.7 + 0.3 * tE);
            let fl = 0.95 + 0.05 * sin(t * (1.0 + fT * 2.0) + f32(i));  // flicker
            
            fc = nG(nC, bA) * fl;
        }
        // Outer glow
        else if (eD < 0.01 && ((tc.x >= bX - 0.01 && tc.x < bX + bW - bS + 0.01) && 
                (tc.y <= bT + 0.01 && tc.y >= bB - 0.01))) {
            let fT = f32(i) / 128.0;
            let gS = smoothstep(0.01, 0.0, eD) * bA;
            fc += hsv2rgb(fT, 0.9, 0.9) * gS * 0.5;
        }
        // Bar reflection
        if (inX && tc.y > rF && tc.y < rF + rD) {
            let rDist = (tc.y - rF) / rD;
            let rInt = (1.0 - rDist) * 3.2 * bA;
            let rCol = hsv2rgb(f32(i) / 128.0, 0.9, 0.8) * rInt;
            fc = mix(fc, rCol, smoothstep(rD, 0.0, tc.y - rF) * 0.5);
        }
    }
    // waveform vis
    let wY = 0.1;  // waveform Y position
    let wH = 0.05 + tE * 0.03;  // waveform height
    let wX = tc.x;  // waveform X
    let wV = gAV(wX) * 0.8;  // waveform value
    let wP = wY + sin(wX * 100.0) * wV * wH;  // wave position
    let dW = abs(tc.y - wP);  // distance to wave
    if (tc.y < wY + wH * 2.0 && tc.y > 0.0) {
        let bI = abs(1.0 / (30.0 * dW * (1.0 + mE * 2.0)));
        let bCol = bC(wX, t);
        fc += bCol * bI * (0.2 + tE * 0.8);
        if (dW < 0.0025) {
            fc = mix(fc, bC(wX - t * 0.05, t) * 1.5, 0.7);
        }
    }
    fc *= 1.0 - 0.04 + 0.04 * sin(tc.y * 100.0 + t);
    //color adjustments contrast and gamma etc
    fc.r = pow(fc.r, 1.0 / params.red_power);
    fc.g = pow(fc.g, 1.0 / params.green_power) * (1.0 + params.green_boost);
    fc.b = pow(fc.b, 1.0 / params.blue_power);
    fc = pow(max(fc, vec3<f32>(0.0)), vec3<f32>(params.gamma));
    fc = (fc - 0.5) * params.contrast + 0.5;
    fc = bE(fc) * (1.0 + params.glow * 0.7);
    fc *= mix(1.0, smoothstep(0.5, 0.0, length(tc - 0.5) - 0.2), 0.85);
    return vec4<f32>(fc, textureSample(tex, tex_sampler, tc).a);
}