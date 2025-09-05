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
    trail_length: f32,
    trail_decay: f32,
    flow_strength: f32,
    _padding1: f32,
    _padding2: u32,
}

@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<i32>>;

fn hash(p: vec4<f32>) -> vec4<f32> {
    var pm = p;
    pm = fract(pm * vec4<f32>(0.1031, 0.1030, 0.0973, 0.1099));
    pm += dot(pm, pm.wzxy + 33.33);
    return fract((pm.xxyz + pm.yzzw) * pm.zywx);
}

// time-aware physics for trails
fn getPhysPosAtTime(op: vec2<f32>, col: vec3<f32>, sz: vec2<f32>, custom_time: f32) -> vec2<f32> {
    let t = custom_time;
    let hi = vec4<f32>(op, col.r, col.g);
    let rnd = hash(hi);
    
    let wave = sin(t * 0.785398) * 0.5 + 0.5;
    let si = wave * 1.5; 
    
    // create 3D position from 2D image coordinates  
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
    
    let rsp = vec2<f32>(
        sin(custom_time * 0.3 + rnd.x * 6.28) * rnd.y,
        cos(custom_time * 0.4 + rnd.z * 6.28) * rnd.w
    ) * sz * 0.3;
    let fp = mix(op, op + rsp * si, smoothstep(0.0, 1.0, si));
    
    let its = i32(3.0 + si * 4.0);
    for (var i = 0; i < its && i < 15; i++) {
        let ms = 0.05 + si * 0.3;
        let cm = vec3<f32>(
            sin(env + p.x * 2.0) * ms,
            cos(env + p.y * 2.0) * ms,
            sin(env * 0.3) * ms * 0.3
        ) * params.particle_spread;
        
        p += cm;
        
        let fs = 0.02 + si * 0.15;
        let fld = p / (dot(p, p) * 3.0 + 1.0) * fs;
        p += fld * ch * (0.1 + si * 0.15);
        
        p += (col - 0.5) * 0.01 * params.particle_spread * (0.8 + si);
    }
    
    let ds = 0.02 + si * 0.25;
    let disp = vec2<f32>(p.x, p.y) * sz * ds * params.particle_spread;
    let pp = fp + disp;
    
    let ats = 1.0 - params.physics_strength * (0.1 + min(si, 1.5) * 0.3);
    let att = (op - pp) * ats;
    
    return pp + att;
}

// scramble particles everywhere, then slowly bring them back to form the image
fn getPhysPos(op: vec2<f32>, col: vec3<f32>, sz: vec2<f32>) -> vec2<f32> {
    let t = time_data.time * params.animation_speed;
    let hi = vec4<f32>(op, col.r, col.g);
    let rnd = hash(hi);
    
    let wave = sin(t * 0.785398) * 0.5 + 0.5; 
    let si = wave * 1.5;
    
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
    
    let rsp = vec2<f32>(
        sin(t * 0.3 + rnd.x * 6.28) * rnd.y,
        cos(t * 0.4 + rnd.z * 6.28) * rnd.w
    ) * sz * 0.3;
    let fp = mix(op, op + rsp * si, smoothstep(0.0, 1.0, si));
    
    let its = i32(3.0 + si * 4.0);
    for (var i = 0; i < its && i < 15; i++) {
        let ms = 0.05 + si * 0.3;
        let cm = vec3<f32>(
            sin(env + p.x * 2.0) * ms,
            cos(env + p.y * 2.0) * ms,
            sin(env * 0.3) * ms * 0.3
        ) * params.particle_spread;
        
        p += cm;
        
        let fs = 0.02 + si * 0.15;
        let fld = p / (dot(p, p) * 3.0 + 1.0) * fs;
        p += fld * ch * (0.1 + si * 0.15);
        
        p += (col - 0.5) * 0.01 * params.particle_spread * (0.8 + si);
    }
    
    let ds = 0.02 + si * 0.25;
    let disp = vec2<f32>(p.x, p.y) * sz * ds * params.particle_spread;
    let pp = fp + disp;
    
    let ats = 1.0 - params.physics_strength * (0.1 + min(si, 1.5) * 0.3);
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
    
    let trail_steps = i32(params.trail_length * 6.0) + 1;
    let sr = max(1, i32(params.splat_size * 5.0));
    
    for (var trail = 0; trail < trail_steps && trail < 8; trail++) {
        let trail_time_offset = f32(trail) * 0.15 * params.flow_strength;
        let historical_time = time_data.time * params.animation_speed - trail_time_offset;
        
        let historical_pos = getPhysPosAtTime(sp, col, vec2<f32>(scr), historical_time);
        
        // Add flowing trail effect on top of scrambling
        let trail_flow = vec2<f32>(
            sin(historical_time * 2.0 + sp.x * 0.01) * params.flow_strength,
            cos(historical_time * 1.7 + sp.y * 0.01) * params.flow_strength
        ) * 15.0 * params.trail_length; // scale with trail length
        
        let trail_pos = historical_pos + trail_flow;
        let trail_intensity = pow(params.trail_decay, f32(trail));
        
        // Only draw if trail position is on screen
        if (trail_pos.x >= 0.0 && trail_pos.x < f32(scr.x) && 
            trail_pos.y >= 0.0 && trail_pos.y < f32(scr.y)) {
            
            // Draw splat at trail position
            for (var dx = -sr; dx <= sr; dx++) {
                for (var dy = -sr; dy <= sr; dy++) {
                    let x = min(max(0, i32(trail_pos.x) + dx), i32(scr.x) - 1);
                    let y = min(max(0, i32(trail_pos.y) + dy), i32(scr.y) - 1);
                    
                    let d = sqrt(f32(dx*dx + dy*dy));
                    let fo = max(0.0, 1.0 - d / f32(sr)) * trail_intensity;
                    
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