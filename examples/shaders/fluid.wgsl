//adapted from https://www.shadertoy.com/view/MsGSRd by flockaroo, 2016-06-15 (CC BY-NC-SA 3.0)
// But this time we respect the original texture colors with different rotations

struct TimeUniform { 
    time: f32, 
    delta: f32,
    frame: u32, 
    _padding: u32
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct FluidParams {
    rotation_speed: f32,
    motor_strength: f32,
    distortion: f32,
    feedback: f32,
    particle_size: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
    _padding4: f32,
    _padding5: f32,
    _padding6: f32,
    _padding7: f32,
};
// Group 1: Primary Pass I/O & Parameters (Cuneus Way)
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: FluidParams;

// Group 2: Engine Resources (Channels - accessible from all passes!)
@group(2) @binding(0) var channel0: texture_2d<f32>;
@group(2) @binding(1) var channel0_sampler: sampler;

// Group 3: Multipass feedback
@group(3) @binding(0) var pass_in: texture_2d<f32>;
@group(3) @binding(1) var bilinear: sampler;

const ROT_NUM = 5u;
const PI = 3.14159265359;
const ANG = 4.0 * PI / f32(ROT_NUM);

fn get_rot_matrix() -> mat2x2<f32> {
    return mat2x2<f32>(
        vec2<f32>(cos(ANG), sin(ANG)),
        vec2<f32>(-sin(ANG), cos(ANG))
    );
}

// bilinear sampling 
fn sample_level(tex: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(tex));
    
    if (params.particle_size < 0.5) {
        let coords = vec2<i32>(uv * dims);
        let clamped_coords = clamp(coords, vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
        return textureLoad(tex, clamped_coords, 0);
    } else if (params.particle_size < 1.5) {

        let pixel_uv = uv * dims;
        let base_coords = floor(pixel_uv);
        let frac_coords = fract(pixel_uv);
        
        let c00 = textureLoad(tex, clamp(vec2<i32>(base_coords), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1)), 0);
        let c10 = textureLoad(tex, clamp(vec2<i32>(base_coords + vec2<f32>(1.0, 0.0)), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1)), 0);
        let c01 = textureLoad(tex, clamp(vec2<i32>(base_coords + vec2<f32>(0.0, 1.0)), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1)), 0);
        let c11 = textureLoad(tex, clamp(vec2<i32>(base_coords + vec2<f32>(1.0, 1.0)), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1)), 0);
        
        let c0 = mix(c00, c10, frac_coords.x);
        let c1 = mix(c01, c11, frac_coords.x);
        return mix(c0, c1, frac_coords.y);
    } else {
        let pixel_uv = uv * dims;
        let center_coords = vec2<i32>(pixel_uv);
        
        var total_color = vec4<f32>(0.0);
        var total_weight = 0.0;
        
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            for (var dy = -1; dy <= 1; dy = dy + 1) {
                let sample_coords = clamp(
                    center_coords + vec2<i32>(dx, dy), 
                    vec2<i32>(0), 
                    vec2<i32>(dims) - vec2<i32>(1)
                );
                
                let weight = 1.0 / (1.0 + f32(abs(dx) + abs(dy)));
                total_color += textureLoad(tex, sample_coords, 0) * weight;
                total_weight += weight;
            }
        }
        
        return total_color / total_weight;
    }
}

fn get_rot(pos: vec2<f32>, b: vec2<f32>) -> f32 {
    var p = b;
    var rot = 0.0;
    let m = get_rot_matrix();
    let dims = vec2<f32>(textureDimensions(output));
    
    for(var i = 0u; i < ROT_NUM; i = i + 1u) {
        let sample = sample_level(pass_in, fract((pos + p) / dims)).xy;
        rot += dot(sample - vec2<f32>(0.5), p.yx * vec2<f32>(1.0, -1.0));
        p = m * p;
    }
    return rot / f32(ROT_NUM) / dot(b, b);
}

fn get_val(uv: vec2<f32>) -> f32 {
    return length(sample_level(pass_in, uv).xyz);
}

fn get_grad(uv: vec2<f32>, delta: f32) -> vec2<f32> {
    let d = vec2<f32>(delta, 0.0);
    return vec2<f32>(
        get_val(uv + d.xy) - get_val(uv - d.xy),
        get_val(uv + d.yx) - get_val(uv - d.yx)
    ) / delta;
}

@compute @workgroup_size(16, 16, 1)
fn buffer_a(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = textureDimensions(output);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let pos = vec2<f32>(id.xy);
    let dims = vec2<f32>(screen_size);
    
    var b = vec2<f32>(cos(ANG), sin(ANG));
    var v = vec2<f32>(0.0);
    
    let bb_max = 0.7 * dims.y;
    let bb_max_squared = bb_max * bb_max;
    let m = get_rot_matrix();
    
    for(var l = 0u; l < 20u; l = l + 1u) {
        if(dot(b, b) > bb_max_squared) { break; }
        
        var p = b;
        for(var i = 0u; i < ROT_NUM; i = i + 1u) {
            v += p.yx * get_rot(pos + p, b);
            p = m * p;
        }
        b *= 2.0;
    }
    
    var color: vec4<f32>;
    if (time_data.frame <= 4u) {
        // Initialize with channel0 texture for first several frames
        color = sample_level(channel0, fract(pos / dims));
    } else {
        // After frame 4, mix channel0 (external) with pass_in (previous buffer_a output)
        let distorted_uv = fract((pos + v * vec2<f32>(-1.0, 1.0) * params.rotation_speed) / dims);
        
        let fluid_color = sample_level(pass_in, distorted_uv);       // Previous frame from buffer_a
        let texture_color = sample_level(channel0, distorted_uv);    // External texture (channel0)
        
        color = mix(texture_color, fluid_color, params.feedback);
        
        if (params.feedback > 1.005) {
            let offset = 1.0 / dims;
            let neighbor_avg = (
                sample_level(pass_in, fract(distorted_uv + vec2<f32>(offset.x, 0.0))) +
                sample_level(pass_in, fract(distorted_uv + vec2<f32>(-offset.x, 0.0))) +
                sample_level(pass_in, fract(distorted_uv + vec2<f32>(0.0, offset.y))) +
                sample_level(pass_in, fract(distorted_uv + vec2<f32>(0.0, -offset.y)))
            ) * 0.25;
            
            let diff = length(color.rgb - neighbor_avg.rgb);
            if (diff > 0.3) {
                color = mix(color, neighbor_avg, 0.1);
            }
        }
    }
    
    let scr = (vec2<f32>(id.xy) / dims) * 2.0 - vec2<f32>(1.0);
    let time_factor = 1.0 + 0.2 * sin(time_data.time);
    let motor_force = params.motor_strength * time_factor * scr.xy / (dot(scr, scr) / 0.1 + 0.3);
    
    let motor_pos = pos + motor_force * dims * params.distortion;
    let motor_color = sample_level(channel0, fract(motor_pos / dims));
    
    color = mix(color, motor_color, length(motor_force) * 2.0);
    
    textureStore(output, id.xy, color);
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let screen_size = textureDimensions(output);
    if (id.x >= screen_size.x || id.y >= screen_size.y) { return; }

    let fragCoord = vec2<f32>(id.xy) + 0.5;
    let uv = fragCoord / vec2<f32>(screen_size);
    
    let delta = 1.0 / f32(screen_size.y);
    let grad = get_grad(uv, delta);
    let n = normalize(vec3<f32>(grad.x, grad.y, 150.0));
    
    let light = normalize(vec3<f32>(
        1.0 + 0.2 * sin(time_data.time * 0.5),
        1.0 + 0.2 * cos(time_data.time * 0.5),
        2.0
    ));
    let diff = clamp(dot(n, light), 0.5, 1.0);
    let spec = pow(clamp(dot(reflect(light, n), vec3<f32>(0.0, 0.0, -1.0)), 0.0, 1.0), 36.0) * 2.5;
    
    let base_color = sample_level(pass_in, uv);
    let final_color = base_color * vec4<f32>(diff) + vec4<f32>(spec);
    textureStore(output, id.xy, final_color);
}