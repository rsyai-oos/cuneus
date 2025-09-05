// Enes Altun, MIT License

struct TimeUniform { time: f32, delta: f32, frame: u32, _padding: u32 };
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: SplattingParams;
@group(1) @binding(2) var input_texture: texture_2d<f32>;
@group(1) @binding(3) var input_sampler: sampler;

struct SplattingParams {
    animation_speed: f32,
    splat_size: f32,
    particle_spread: f32,
    intensity: f32,
    particle_density: f32,
    brightness: f32,
    physics_strength: f32,
    _padding: u32,
}

@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<i32>>;

fn hash(p: vec4<f32>) -> vec4<f32> {
    var pm = p;
    pm = fract(pm * vec4<f32>(0.1031, 0.1030, 0.0973, 0.1099));
    pm += dot(pm, pm.wzxy + 33.33);
    return fract((pm.xxyz + pm.yzzw) * pm.zywx);
}

// scramble particles everywhere, then slowly bring them back to form the image
fn getPhysPos(op: vec2<f32>, col: vec3<f32>, sz: vec2<f32>) -> vec2<f32> {
    let t = time_data.time * params.animation_speed;
    let hi = vec4<f32>(op, col.r, col.g);
    let rnd = hash(hi);
    
    // 8 second cycle: 3s scramble -> 3s settle -> 2s breathe
    let ct = 8.0;
    let pt = fract(t / ct) * ct;
    
    var si: f32;
    if (pt < 3.0) {
        si = 5.0 - (pt / 3.0) * 4.0; // massive chaos at start
    } else if (pt < 6.0) {
        let up = (pt - 3.0) / 3.0;
        si = 1.0 * (1.0 - up); // gradually settle back
    } else {
        let lp = (pt - 6.0) / 2.0;
        si = sin(lp * 3.14159) * 0.1; // gentle breathing
    }
    
    var p = vec3<f32>(
        (op.x / sz.x - 0.5) * 2.0,
        (op.y / sz.y - 0.5) * 2.0,
        (rnd.z - 0.5) * 0.5
    );
    
    let env = t + sin(t * 0.4);
    var ch = 1.0;
    if (length(col) > 0.5) {
        ch = 12.0;
    } else {
        ch = -1.0;
    }
    
    // during heavy scramble, particles can teleport anywhere on screen
    var fp = op;
    if (si > 1.0) {
        let rsp = vec2<f32>(rnd.x * sz.x, rnd.y * sz.y);
        let rm = (si - 1.0) / 4.0;
        fp = mix(op, rsp, rm);
    }
    
    let its = i32(4.0 + min(si, 3.0) * 8.0);
    for (var i = 0; i < its && i < 30; i++) {
        let r = hi.x * f32(i) * 0.3;
        let ms = 0.08 + si * 0.8;
        let cm = vec3<f32>(
            sin(env + p.x * 3.0) * ms,
            cos(env + p.y * 3.0) * ms,
            sin(env * 0.4) * ms * 0.5
        ) * params.particle_spread;
        
        p += cm;
        
        if (r < 0.6) {
            let fs = 0.03 + si * 0.3;
            let fld = p / (dot(p, p) * 2.0 + 0.5) * fs;
            p += fld * ch * (0.15 + si * 0.2);
        }
        
        p += (col - 0.5) * 0.015 * params.particle_spread * (1.0 + si * 2.0);
    }
    
    let ds = 0.03 + si * 0.5;
    let disp = vec2<f32>(p.x, p.y) * sz * ds * params.particle_spread;
    let pp = fp + disp;
    
    let ats = 1.0 - params.physics_strength * (0.1 + min(si, 2.0) * 0.4);
    let att = (op - pp) * ats;
    
    return pp + att;
}

@compute @workgroup_size(16, 16, 1)
fn clear_buffer(@builtin(global_invocation_id) id: vec3<u32>) {
    let scr = textureDimensions(output);
    if (id.x >= scr.x || id.y >= scr.y) { return; }
    
    let i = i32(id.y * scr.x + id.x);
    atomicStore(&atomic_buffer[i*4],   0);
    atomicStore(&atomic_buffer[i*4+1], 0);
    atomicStore(&atomic_buffer[i*4+2], 0);
    atomicStore(&atomic_buffer[i*4+3], 0);
}

@compute @workgroup_size(16, 16, 1)
fn project_colors(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(input_texture);
    let scr = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let op = vec2<f32>(id.xy);
    let uv = (op + 0.5) / vec2<f32>(dims);
    let col = textureSampleLevel(input_texture, input_sampler, uv, 0.).xyz;
    
    if (length(col) < 0.02) { return; }
    
    let hi = vec4<f32>(op, col.r, col.g);
    let ph = hash(hi);
    
    // skip some pixels to reduce particle count, brighter pixels more likely to survive
    let bt = 1.0 - params.particle_density;
    let cb = length(col) * 0.3;
    let pt = bt - cb;
    if (ph.x > pt) { return; }
    
    let sp = op * vec2<f32>(scr) / vec2<f32>(dims);
    let pp = getPhysPos(sp, col, vec2<f32>(scr));
    
    let sr = max(1, i32(params.splat_size * 5.0));
    
    for (var dx = -sr; dx <= sr; dx++) {
        for (var dy = -sr; dy <= sr; dy++) {
            let x = min(max(0, i32(pp.x) + dx), i32(scr.x) - 1);
            let y = min(max(0, i32(pp.y) + dy), i32(scr.y) - 1);
            
            let d = sqrt(f32(dx*dx + dy*dy));
            let fo = max(0.0, 1.0 - d / f32(sr));
            
            if (fo > 0.02) {
                let idx = y * i32(scr.x) + x;
                let w = i32(1024.0 * fo);
                let bc = col * 5.0;
                atomicAdd(&atomic_buffer[idx*4],   i32(f32(w) * bc.x * params.intensity));
                atomicAdd(&atomic_buffer[idx*4+1], i32(f32(w) * bc.y * params.intensity));
                atomicAdd(&atomic_buffer[idx*4+2], i32(f32(w) * bc.z * params.intensity));
                atomicAdd(&atomic_buffer[idx*4+3], w);
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn generate_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let scr = textureDimensions(output);
    if (id.x >= scr.x || id.y >= scr.y) { return; }
    
    let idx = i32(id.y * scr.x + id.x);
    let cnt = atomicLoad(&atomic_buffer[idx*4+3]);
    
    if (cnt > 0) {
        let r = f32(atomicLoad(&atomic_buffer[idx*4])) / (f32(cnt) * 1024.0);
        let g = f32(atomicLoad(&atomic_buffer[idx*4+1])) / (f32(cnt) * 1024.0);
        let b = f32(atomicLoad(&atomic_buffer[idx*4+2])) / (f32(cnt) * 1024.0);
        
        let fc = vec3<f32>(r, g, b) * params.brightness;
        textureStore(output, vec2<i32>(id.xy), vec4<f32>(fc, 1.0));
    } else {
        textureStore(output, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
    }
}