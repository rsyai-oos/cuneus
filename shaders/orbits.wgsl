struct TimeUniform {
    time: f32,
};
const PI: f32 = 3.141592653589793;

@group(0) @binding(0) var<uniform> u_time: TimeUniform;
fn osc(minValue: f32, maxValue: f32, interval: f32, currentTime: f32) -> f32 {
    return minValue + (maxValue - minValue) * 0.5 * (sin(2.0 * PI * currentTime / interval) + 1.0);
}
struct Params {
    base_color: vec3<f32>,
    x: f32,
    rim_color: vec3<f32>,
    y: f32,
    accent_color: vec3<f32>,
    _pad3: f32,
    
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
    _pad4: f32,
};

@group(1) @binding(0) var<uniform> params: Params;

fn normalize_trap(trap_dist: f32, scale: f32) -> f32 {
    return 0.5 + 0.5 * tanh(scale * trap_dist);
}

fn implicit(c: vec2<f32>, trap1: vec2<f32>, trap2: vec2<f32>, currentTime: f32) -> vec4<f32> {
    var z: vec2<f32> = vec2<f32>(0.0, 0.0);
    var dz: vec2<f32> = vec2<f32>(1.0, 0.0);
    var trap1_min: f32 = 1e20;
    var trap2_min: f32 = 1e20;
    var MAX_ITER: i32 = params.iteration;
    var BOUND: f32 = 3.0;

    var i: i32 = 0;
    for (i = 0; i < MAX_ITER; i = i + 1) {
        dz = 2.0 * vec2<f32>(z.x * dz.x - z.y * dz.y, z.x * dz.y + z.y * dz.x) + vec2<f32>(1.0, 0.0);
        let xnew: f32 = z.x * z.x - z.y * z.y + c.x;
        z.y = 2.0 * z.x * z.y + c.y;
        z.x = xnew;
        let dampenedTime: f32 = currentTime * 0.001;
        z = z + 0.1 * vec2<f32>(sin(0.001 * dampenedTime), cos(0.001 * dampenedTime));
        
        trap1_min = min(trap1_min, length(z - trap1));
        trap2_min = min(trap2_min, dot(z - trap2, z - trap2));

        if (dot(z, z) > BOUND) {
            break;
        }
    }
    let d: f32 = sqrt(dot(z, z) / dot(dz, dz)) * log(dot(z, z));
    return vec4<f32>(f32(i), d, trap1_min, trap2_min);
}

fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let screen_size = vec2<f32>(1920.0, 1080.0);
    let fragCoord = vec2<f32>(FragCoord.x, screen_size.y - FragCoord.y);
    let uv_base = 0.4 * (fragCoord - 0.5 * screen_size) / screen_size.y;
    let AA: i32 = params.aa;
    var MAX_ITER: i32 = params.iteration;
    var BOUND: f32 = 3.0; 
    let camSpeed = vec2<f32>(0.0002, 0.0002);
    let camPath = vec2<f32>(
        sin(camSpeed.x * u_time.time / 10.0),
        cos(camSpeed.y * u_time.time / 10.0)
    );
    
    var pan: vec2<f32> = vec2<f32>(0.8030, 0.2585);

    if (u_time.time > 2.0) {
        let timeSince14 = u_time.time - 45.0;
        pan.y = pan.y + 0.00002 * timeSince14;
    }
    
    let zoomLevel: f32 = osc(0.0004, 0.0004, 20.0, u_time.time * 0.1);
    let trap1 = vec2<f32>(0.0, 1.0);
    let trap2 = vec2<f32>(params.trap_x, params.trap_y) + 0.5* vec2<f32>(
        cos(params.trap_c1 * u_time.time),
        sin(params.trap_c1* u_time.time)
    );
    
    var col = vec3<f32>(0.0,0.0,0.0);
    
    for (var m: i32 = 0; m < AA; m = m + 1) {
        for (var n: i32 = 0; n < AA; n = n + 1) {
            let sample_offset = vec2<f32>(f32(m), f32(n)) / f32(AA);
            let min_res = min(screen_size.x, screen_size.y);
            let uv_sample = ((fragCoord + sample_offset - 0.5 * screen_size) / min_res * params.zoom + pan + camPath) * 2.033 - vec2<f32>(params.x, params.y);
            
            let z_data = implicit(uv_sample, trap1, trap2, u_time.time* 0.1);
            let iter_ratio = smoothstep(0.0, 1.0, z_data.x / f32(MAX_ITER));
            let d = z_data.y;
            let trap1_dist = z_data.z;
            let trap2_dist = z_data.w;
            
            if (iter_ratio < 1.0) {
                let c1 = pow(clamp(normalize_trap(2.00 * d / zoomLevel, 1.0), 0.0, 1.0), 0.5);
                let c2 = pow(clamp(normalize_trap(1.5 * trap1_dist, 2.0), 0.0, 1.0), 2.0);
                let c3 = pow(clamp(normalize_trap(0.4 * trap2_dist, 0.25), 0.0, 1.0), 0.25);
                
                let phase1 = 2.0 * PI * (c2 + c3);
                let phase2 = 2.0 * PI * c3;
                let phase3 = 2.0 * PI * iter_ratio;
                
                let col1 = 0.5 + 0.5 * sin(phase1 + params.rim_color);
                let col2 = 0.5 + 0.5 * sin(phase2 + params.accent_color);
                let osc_val = osc(0.0, 1.0, 10.0, u_time.time);
                let exteriorColor = 0.5 + 0.5 * sin(2.0 * PI * normalize_trap(trap2_dist, params.trap_pow) + params.base_color + phase3 + osc_val);

                col = col + mix(col1 + col2, exteriorColor, iter_ratio);
            }
        }
    }

    col = gamma(col / f32(AA * AA), 0.4);
    return vec4<f32>(col, 1.0);
}