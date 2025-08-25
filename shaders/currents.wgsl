// Photon tracing: currents
// Very complex example demonstrating multi-buffer ping-pong computation
// I hope this example is useful for those who came from the Shadertoy, I tried to use same terminology (bufferA, ichannels etc)
// I used the all buffers (buffera,b,c,d,mainimage) and complex ping-pong logic 
// This photon tracing technique is from Wyatt's https://www.shadertoy.com/view/tfB3Rw code, "fractal with photon tracking", 2025.
// (my pattern is different but the rendering method is directly coming from this code)
// Be aware though, If you do anything commercial with this rendering technique, you should definitely ask Wyatt about licensing (wyatthf@gmail.com). The goal here is to reproduce a complex but meaningful shadertoy code in cuneus.

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct CurrentsParams {
    sphere_radius: f32,
    sphere_pos_x: f32,
    sphere_pos_y: f32,
    critic2_interval: f32,
    critic2_pause: f32,
    critic3_interval: f32,
    metallic_reflection: f32,
    line_intensity: f32,
    pattern_scale: f32,
    noise_strength: f32,
    gradient_r: f32,
    gradient_g: f32,
    gradient_b: f32,
    gradient_w: f32,
    line_color_r: f32,
    line_color_g: f32,
    line_color_b: f32,
    line_color_w: f32,
    gradient_intensity: f32,
    line_intensity_final: f32,
    c2_min: f32,
    c2_max: f32,
    c3_min: f32,
    c3_max: f32,
    fbm_scale: f32,
    fbm_offset: f32,
    pattern_mode: f32,
    mandel_zoom_min: f32,
    mandel_zoom_max: f32,
    mandel_pan_x: f32,
    mandel_pan_y: f32,
    mandel_trap1_x: f32,
    mandel_trap1_y: f32,
    mandel_trap2_x: f32,
    mandel_trap2_y: f32,
    gamma: f32,
}

// Group 1: Primary Pass I/O & Parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: CurrentsParams;

// Group 3: User-Defined Data Buffers (multi-pass input textures)
@group(3) @binding(0) var input_texture0: texture_2d<f32>; // Primary input (self or main)
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>; // Secondary input
@group(3) @binding(3) var input_sampler1: sampler;
@group(3) @binding(4) var input_texture2: texture_2d<f32>; // Tertiary input  
@group(3) @binding(5) var input_sampler2: sampler;

// Type aliases
alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;
alias m2 = mat2x2<f32>;

const TAU: f32 = 6.28318530718;
const PI: f32 = 3.14159265;

var<private> R: v2;

fn hash(p4: v4) -> v4 {
    var p = fract(p4 * v4(0.1031, 0.1030, 0.0973, 0.1099));
    p += dot(p, p.wzxy + 33.33);
    return fract((p.xxyz + p.yzzw) * p.zywx);
}

// utility for smooth oscillation
fn op(min_val: f32, max_val: f32, interval: f32, p_d: f32, current_time: f32) -> f32 {
    let cycle_time = 2.0 * interval + p_d;
    let t = current_time % cycle_time;
    var p: f32;
    
    if (t < interval) {
        p = t / interval;
        p = 0.5 - 0.5 * cos(PI * p);
        return mix(max_val, min_val, p);
    } else if (t < interval + p_d) {
        return min_val;
    } else {
        p = (t - interval - p_d) / interval;
        p = 0.5 - 0.5 * cos(PI * p);
        return mix(min_val, max_val, p);
    }
}

fn hash21(p: v2) -> f32 {
    return fract(cos(sin(dot(p, v2(0.009123898, 0.00231233))) * 48.512353) * 11111.5452313);
}

fn noise(p: v2) -> f32 {
    let i = floor(p);
    let f = fract(p);
    
    let a = hash21(i);
    let b = hash21(i + v2(1.0, 0.0));
    let c = hash21(i + v2(0.0, 1.0));
    let d = hash21(i + v2(1.0, 1.0));
    
    let u = f * f * (3.0 - 2.0 * f);
    
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p: v2) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;

    for (var i = 0; i < 3; i++) {
        value += amplitude * noise((p + v2(-1.0, 1.0) * 3.0 / 1.0) * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value / (1.0 - amplitude * 2.0);
}
//create metalic shep
fn c_m_p(local_uv: v2) -> v3 {
    let len = length(local_uv);
    if (len > 1.0) { return v3(0.0); }
    
    let normal = normalize(v3(local_uv, sqrt(1.0 - len * len)));
    
    var h1 = smoothstep(0.5, 1.4, distance(local_uv, v2(-0.1, 0.1))) * 0.5;
    h1 += smoothstep(0.1, 0.9, 1.3 - distance(local_uv, v2(-0.3, 0.3))) * 0.5;
    h1 += smoothstep(0.1, 0.5, 0.5 - distance(local_uv, v2(-0.4, 0.4)));
    h1 += smoothstep(0.1, 0.5, 0.4 - distance(local_uv, v2(0.2, 0.6)));
    
    let metallic = h1 * (1.0 - smoothstep(0.95, 1.0, len));
    return v3(params.metallic_reflection) * metallic;
}
// mandelbrot
const MAX_ITER: i32 = 333;
const BOUND: f32 = 77.0;

fn implicit_mandelbrot(c: v2, trap1: v2, trap2: v2) -> v4 {
    var z = v2(0.0);
    var dz = v2(1.0, 0.0);
    var trap1_min = 1e20;
    var trap2_min = 1e20;
    var i: i32 = 0;
    
    for (i = 0; i < MAX_ITER; i++) {
        dz = 2.0 * v2(z.x * dz.x - z.y * dz.y, z.x * dz.y + z.y * dz.x) + v2(1.0, 0.0);
        let xnew = z.x * z.x - z.y * z.y + c.x;
        z.y = 2.0 * z.x * z.y + c.y;
        z.x = xnew;
        z += 0.1 * v2(sin(0.001), cos(0.001));
        trap1_min = min(trap1_min, length(z - trap1));
        trap2_min = min(trap2_min, dot(z - trap2, z - trap2));
        if (dot(z, z) > BOUND) { break; }
    }
    
    let d = sqrt(dot(z, z) / dot(dz, dz)) * log(dot(z, z));
    return v4(f32(i), d, trap1_min, trap2_min);
}

// Texture sampling functions for compute shader
fn sample_input0(uv: v2) -> v4 {
    let coord = vec2<i32>((uv / R) * vec2<f32>(textureDimensions(input_texture0, 0)));
    let clamped_coord = clamp(coord, vec2<i32>(0), vec2<i32>(textureDimensions(input_texture0, 0)) - vec2<i32>(1));
    return textureLoad(input_texture0, clamped_coord, 0);
}

fn sample_input1(uv: v2) -> v4 {
    let coord = vec2<i32>((uv / R) * vec2<f32>(textureDimensions(input_texture1, 0)));
    let clamped_coord = clamp(coord, vec2<i32>(0), vec2<i32>(textureDimensions(input_texture1, 0)) - vec2<i32>(1));
    return textureLoad(input_texture1, clamped_coord, 0);
}

fn sample_input2(uv: v2) -> v4 {
    let coord = vec2<i32>((uv / R) * vec2<f32>(textureDimensions(input_texture2, 0)));
    let clamped_coord = clamp(coord, vec2<i32>(0), vec2<i32>(textureDimensions(input_texture2, 0)) - vec2<i32>(1));
    return textureLoad(input_texture2, clamped_coord, 0);
}

// Backward compatibility 
fn sample_input(uv: v2) -> v4 {
    return sample_input0(uv);
}

// Buffer A - simple pattern (ichannel0=BufferA)
// This is mosty about our pattern generation
@compute @workgroup_size(16, 16, 1)
fn buffer_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let U = v2(f32(gid.x), f32(gid.y));
    let U0 = U;
    
    var uv = U / R * 2.0 - 1.0;
    uv.y = -uv.y;
    uv.x *= R.x / R.y;


    let ball_pos = v2(params.sphere_pos_x, params.sphere_pos_y);
    let ball_radius = params.sphere_radius;
     // sphere distance
    let s_dist = distance(uv, ball_pos);
   // pattern coords  
    var duv = uv * v2(params.pattern_scale, params.pattern_scale * 0.87) + v2(0.0, 20.0);
     // oscillator 3
    let c3 = op(params.c3_min, params.c3_max, params.critic3_interval, 0.0, time_data.time);
    // warp pattern
    duv *= distance(uv, v2(1.5, -2.0)) / c3;
     // sphere influence
    duv.x += smoothstep(1.0 - ball_radius * 1.5, 1.0, 1.0 - distance(uv, ball_pos - v2(0.1, 0.0))) * 15.0;
     // distortion mask
    let dw = smoothstep(0.0, 2.0, 1.0 - distance(uv * 0.5, v2(0.4, -0.85)));
     // oscillator 2
    let c2 = op(params.c2_min, params.c2_max, params.critic2_interval, params.critic2_pause, time_data.time);
     // add noise distortion
    duv += (fbm(uv * params.fbm_scale) - params.fbm_offset) * dw * c2;
    // line pattern
    let lp = sin(duv.x + duv.y);
     // diagonal metric
    let dm = (duv.x + duv.y) / TAU;
     // width adjust
    let wa = smoothstep(ball_radius * 3.0, ball_radius * 0.8, s_dist) * 0.5;
    // brightness
    let br = clamp(3.0 - distance(uv * v2(1.0, 2.0), v2(-1.0, -1.0)), 0.0, 1.0); 
    
    var color: v3;
    var pattern = 0.0;
    
    if (params.pattern_mode < 0.5) {
        // CURRENTS MODE
        if (abs(uv.x) > 1.3 || abs(uv.y) > 1.0) {
            color = v3(0.0);
            pattern = 0.0;
        } 
        else if (dm < (0.5 + wa) && dm > (-1.0 - wa)) {
            let gi = min(1.0, (0.75 - abs(dm + 0.25)) * 5.0);
            color = mix(v3(params.gradient_r, params.gradient_g, params.gradient_b), 
                       v3(0.93, 0.64, 0.17), -uv.y) * gi * params.line_intensity;
            pattern = 3.8 * gi;
        } 
        else {
            color = v3(params.line_color_r, params.line_color_g, params.line_color_b) * lp * br * params.line_intensity;
            pattern = 0.4 * abs(lp) * br;
        }
        
        let sc = c_m_p((uv - ball_pos) / ball_radius);
        let sm = 1.0 - smoothstep(ball_radius - 0.002, ball_radius + 0.01, s_dist);
        let fsm = sm * smoothstep(-1.1, -0.4, dm);
        color = mix(color, sc, fsm);
        pattern = mix(pattern, length(sc) * 0.5, fsm);
    } else {
        // MANDELBROT
        let cam_path = v2(sin(0.0002), cos(0.0002));
        let pan = v2(params.mandel_pan_x, params.mandel_pan_y);
        let zoom_level = op(params.mandel_zoom_min, params.mandel_zoom_max, 10.0, 5.0, time_data.time);
        
        let trap1 = v2(params.mandel_trap1_x, params.mandel_trap1_y);
        let trap2 = v2(params.mandel_trap2_x, params.mandel_trap2_y) + 0.1 * v2(cos(0.13), sin(0.13));
        
        let fractal_uv = ((U - 0.5 * R) / min(R.y, R.x) * zoom_level + pan + cam_path) * 2.033 - v2(2.14278);
        
        let z_data = implicit_mandelbrot(fractal_uv, trap1, trap2);
        let iter_ratio = z_data.x / f32(MAX_ITER);
        let d = z_data.y;
        let trap1_dist = z_data.z;
        let trap2_dist = z_data.w;
        
        color = v3(0.0);
        
        if (iter_ratio < 1.0) {
            let c1 = pow(clamp(2.00 * d / zoom_level, 0.0, 1.0), 0.5);
            let c2 = pow(clamp(1.5 * trap1_dist, 0.0, 1.0), 2.0);
            let c3 = pow(clamp(0.4 * trap2_dist, 0.0, 1.0), 0.25);
            
            let col1 = 0.5 + 0.5 * sin(3.0 + 2.0 * c2 + v3(0.0, 0.5, 1.0));
            let col2 = 0.5 + 0.5 * sin(4.1 + 2.0 * c3 + v3(1.0, 0.5, 0.0));
            
            let exterior_color = 0.5 + 0.5 * sin(2.0 * trap1_dist + 
                                         v3(0.0, 0.5, 1.0) + 
                                         PI * v3(3.0 * iter_ratio) + 
                                         op(12.0, 12.0, 10.0, 5.0, time_data.time));
            
            color = col1 + exterior_color;
            color *= 1.0 - pow(iter_ratio, 0.8);
            color += 0.2 * sin(v3(0.1, 0.2, 0.3) + trap1_dist * 2.0);
        }
        
        pattern = (color.r + color.g + color.b) / 3.0;
        pattern = smoothstep(0.0, 0.3, pattern);
    }
    // add grain
    color -= noise(uv * 300.0 + fract(4.0) * 1.0) / params.noise_strength; 
    
    var Q = v4(0.0);
    
    Q.x = color.x;
    Q.y = color.y; 
    Q.z = color.z;
    Q.w = pattern;
    
    let grad_n = sample_input0(U0 + v2(0.0, 1.0)); // north neighbor
    let grad_e = sample_input0(U0 + v2(1.0, 0.0)); // east neighbor  
    let grad_s = sample_input0(U0 - v2(0.0, 1.0)); // south neighbor
    let grad_w = sample_input0(U0 - v2(1.0, 0.0)); // west neighbor
    
    Q.y = -(grad_e.w - grad_w.w); // gradient X for path tracing
    Q.z = -(grad_n.w - grad_s.w); // gradient Y for path tracing
    
    textureStore(output, gid.xy, Q);
}

// Buffer B - (ichannel0=BufferB, ichannel1=BufferA)
@compute @workgroup_size(16, 16, 1)
fn buffer_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let U = v2(f32(gid.x), f32(gid.y));
    
    // Buffer B self-reference
    var Q = sample_input0(U); // Previous frame Buffer B
    Q.x += 2.0 * Q.z;
    Q.y += 2.0 * Q.w;
    Q.z += 3.5 * sample_input1(Q.xy).y; // Read from BufferA
    Q.w += 3.5 * sample_input1(Q.xy).z;
    
    if (length(Q.zw) > 0.0) {
        let norm = normalize(Q.zw);
        Q.z = norm.x;
        Q.w = norm.y;
    }
    
    for (var x = -2.0; x <= 2.0; x += 1.0) {
        for (var y = -2.0; y <= 2.0; y += 1.0) {
            var q = sample_input0(U + v2(x, y)); 
            q.x += 3.0 * q.z;
            q.y += 3.0 * q.w;
            q.z += 3.5 * sample_input1(q.xy).y; 
            q.w += 3.5 * sample_input1(q.xy).z;
            
            if (length(q.zw) > 0.0) {
                let norm = normalize(q.zw);
                q.z = norm.x;
                q.w = norm.y;
            }
            
            if (length(U - q.xy) < length(U - Q.xy)) {
                Q = q;
            }
        }
    }
    
    if (length(U - 0.5 * R) < 10.0) {
        let h = hash(v4(U, f32(time_data.frame), 1.0));
        Q = v4(U, sin(2.0 * PI * h.x), cos(2.0 * PI * h.x));
    }
    
    textureStore(output, gid.xy, Q);
}

// Buffer C - (ichannel0=BufferC, iChannel1=BufferA)
@compute @workgroup_size(16, 16, 1)
fn buffer_c(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let U = v2(f32(gid.x), f32(gid.y));
    
    // Buffer C self-reference
    var Q = sample_input0(U); // Previous frame Buffer C
    Q.x += 6.0 * Q.z;
    Q.y += 6.0 * Q.w;
    Q.z += sample_input1(Q.xy).y; // Read from BufferA
    Q.w += sample_input1(Q.xy).z;
    
    if (length(Q.zw) > 0.0) {
        let norm = normalize(Q.zw);
        Q.z = norm.x;
        Q.w = norm.y;
    }
    
    for (var x = -2.0; x <= 2.0; x += 1.0) {
        for (var y = -2.0; y <= 2.0; y += 1.0) {
            var q = sample_input0(U + v2(x, y)); 
            q.x += 3.0 * q.z;
            q.y += 3.0 * q.w;
            q.z += 3.5 * sample_input1(q.xy).y; 
            q.w += 3.5 * sample_input1(q.xy).z;
            
            if (length(q.zw) > 0.0) {
                let norm = normalize(q.zw);
                q.z = norm.x;
                q.w = norm.y;
            }
            
            if (length(U - q.xy) < length(U - Q.xy)) {
                Q = q;
            }
        }
    }
    
    if (length(U - 0.5 * R) < 10.0) {
        let h = hash(v4(U, f32(time_data.frame), 1.0));
        Q = v4(U, sin(2.0 * PI * h.x), cos(2.0 * PI * h.x));
    }
    
    textureStore(output, gid.xy, Q);
}

// Buffer D (ichannel0=BufferD, iChannel1=BufferC, iChannel2=BufferB)
@compute @workgroup_size(16, 16, 1)
fn buffer_d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let U = v2(f32(gid.x), f32(gid.y));
    
    var Q = sample_input0(U); // BufferD (ichannel0)
    let buffer_c = sample_input1(U); // BufferC (ichannel1)
    let buffer_b = sample_input2(U); // BufferB (ichannel2)
    

    let grad_col = v4(params.gradient_r, params.gradient_g, params.gradient_b, params.gradient_w);
    let line_col = v4(params.line_color_r, params.line_color_g, params.line_color_b, params.line_color_w);
    
    Q += params.gradient_intensity * (4.0 - grad_col) * exp(-length(U - buffer_c.xy));
    Q += params.line_intensity_final * line_col * exp(-length(U - buffer_b.xy));
    
    textureStore(output, gid.xy, Q);
}

// Main (ichannel0=bufferD)
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(output);
    R = v2(dims);
    
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    
    let U = v2(f32(gid.x), f32(gid.y));
    
    let buffer_d = sample_input0(U); // BufferD
    var Q = 0.8 * atan(1.5 * buffer_d / f32(time_data.frame + 1u));
    
    Q = pow(Q, vec4<f32>(params.gamma));
    
    textureStore(output, gid.xy, Q);
}