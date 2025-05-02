// Compute shader version of the Clifford attractor (see clifford.wgsl)
// Uses a multi-pass approach with ping-pong textures

// Time uniform for animation
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

// Attractors parameter uniform
struct Params {
    decay: f32,
    speed: f32,
    intensity: f32,
    scale: f32,
    rotation_x: f32,
    rotation_y: f32,
    rotation_z: f32,
    rotation_speed: f32,
    attractor_a: f32,
    attractor_b: f32,
    attractor_c: f32,
    attractor_d: f32,
    attractor_animate_amount: f32,
    num_points: u32,
    iterations_per_point: u32,
    clear_buffer: u32,
    _padding: u32,
}
@group(1) @binding(0) var<uniform> params: Params;

// Textures and storage buffers
@group(2) @binding(0) var prev_frame: texture_2d<f32>;
@group(2) @binding(1) var tex_sampler: sampler;
@group(2) @binding(2) var output: texture_storage_2d<rgba16float, write>;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<i32>>;

const PI: f32 = 3.14159265359;
const TWO_PI: f32 = 6.28318530718;
const ATOMIC_SCALE: f32 = 24.0;
const FOV: f32 = 0.6;

struct Camera {
    pos: vec3<f32>,
    cam: mat3x3<f32>,
    fov: f32,
    size: vec2<f32>
}

var<private> rng_state: vec4<u32>;

fn pcg4d(a: vec4<u32>) -> vec4<u32> {
    var v = a * 1664525u + 1013904223u;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v = v ^ (v >> vec4<u32>(16u));
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

fn rand4() -> vec4<f32> {
    rng_state = pcg4d(rng_state);
    return vec4<f32>(rng_state) / f32(0xffffffffu);
}

fn get_camera_matrix(ang: vec2<f32>) -> mat3x3<f32> {
    let x_dir = vec3<f32>(cos(ang.x)*sin(ang.y), cos(ang.y), sin(ang.x)*sin(ang.y));
    let y_dir = normalize(cross(x_dir, vec3<f32>(0.0, 1.0, 0.0)));
    let z_dir = normalize(cross(y_dir, x_dir));
    return mat3x3<f32>(-x_dir, y_dir, z_dir);
}

fn set_camera(dimensions: vec2<f32>, t: f32) -> Camera {
    var camera: Camera;
    
    let vertical_speed = params.rotation_speed;
    let base_rotation_speed = params.rotation_speed * 0.3;
    
    let ang = vec2<f32>(
        params.rotation_x + PI * 0.5 + sin(t * base_rotation_speed) * 0.1,
        params.rotation_y + PI * 0.45 + sin(t * vertical_speed) * 0.15
    );
    
    let z_rot = mat3x3<f32>(
        cos(params.rotation_z), -sin(params.rotation_z), 0.0,
        sin(params.rotation_z), cos(params.rotation_z), 0.0,
        0.0, 0.0, 1.0
    );

    camera.fov = FOV;
    camera.cam = z_rot * get_camera_matrix(ang);
    
    camera.pos = -(camera.cam * vec3<f32>(8.0, 0.0, 0.0));
    camera.size = dimensions;
    
    return camera;
}

fn project(cam: Camera, p: vec3<f32>) -> vec3<f32> {
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos) / td;
    let screen = dir * cam.cam;
    
    let screen_pos = screen.yz * cam.size.y / (cam.fov * screen.x) + 0.5 * cam.size;
    
    return vec3<f32>(screen_pos, screen.x * td);
}

fn additive_blend(color: vec3<f32>, depth: f32, index: i32) {
    let scaled_color = vec3<i32>(floor(ATOMIC_SCALE * 10.0 * color / (depth * depth + 0.2)));
    
    if(scaled_color.x > 0) { atomicAdd(&atomic_buffer[index * 4], scaled_color.x); }
    if(scaled_color.y > 0) { atomicAdd(&atomic_buffer[index * 4 + 1], scaled_color.y); }
    if(scaled_color.z > 0) { atomicAdd(&atomic_buffer[index * 4 + 2], scaled_color.z); }
}

fn rasterize_point(camera: Camera, pos: vec3<f32>, color: vec3<f32>, dims: vec2<u32>) -> bool {
    let screen_size = vec2<i32>(dims);
    let projected_pos = project(camera, pos);
    
    let jitter = rand4().xy * 0.5;
    let screen_coord = vec2<i32>(projected_pos.xy + jitter);
    
    if(screen_coord.x < 0 || screen_coord.x >= screen_size.x || 
       screen_coord.y < 0 || screen_coord.y >= screen_size.y || 
       projected_pos.z < 0.0) {
        return false;
    }

    let idx = screen_coord.x + screen_size.x * screen_coord.y;
    additive_blend(color, projected_pos.z, idx);
    return true;
}


fn clifford_attractor(p: vec2<f32>, t: f32) -> vec2<f32> {
    let anim = sin(t * 0.1) * params.attractor_animate_amount;
    let a = params.attractor_a + anim * 0.1;
    let b = params.attractor_b + cos(t * 0.15) * params.attractor_animate_amount * 0.1;
    let c = params.attractor_c + sin(t * 0.2) * params.attractor_animate_amount * 0.1;
    let d = params.attractor_d + cos(t * 0.25) * params.attractor_animate_amount * 0.1;
    
    return vec2<f32>(
        sin(a * p.y) + c * cos(a * p.x),
        -(sin(b * p.x) + d * cos(b * p.y))
    );
}

// First compute pass: Generate points and accumulate to atomic buffer
@compute @workgroup_size(16, 16, 1)
fn compute_points(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(prev_frame);
    
    // Skip if outside texture dimensions
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }
    
    // Clear the atomic buffer more efficiently (each workgroup clears its own region)
    if (params.clear_buffer != 0u) {
        let pixel_index = global_id.y * dimensions.x + global_id.x;
        atomicStore(&atomic_buffer[pixel_index * 4u], 0);
        atomicStore(&atomic_buffer[pixel_index * 4u + 1u], 0);
        atomicStore(&atomic_buffer[pixel_index * 4u + 2u], 0);
    }
    
    // Initialize RNG for this pixel
    rng_state = vec4<u32>(
        global_id.x ^ u32(time_data.time * 1000.0),
        global_id.y ^ u32(time_data.frame),
        u32(time_data.time * 1000.0),
        u32(time_data.frame * 13)
    );
    
    // Determine how many points this invocation should handle
    // Increased points per invocation significantly
    let points_per_invocation = max(5u, params.num_points / 256u);
    
    let t = time_data.time * params.speed;
    let camera = set_camera(vec2<f32>(dimensions), t);
    
    // Generate and process points
    for (var point_idx = 0u; point_idx < points_per_invocation; point_idx += 1u) {
        let r4 = rand4();
        var p = vec3<f32>(6.0 * (r4.x - 0.5), 6.0 * (r4.y - 0.5), 0.0);
        for (var i = 0u; i < params.iterations_per_point; i += 1u) {
            let next = clifford_attractor(p.xy, t);
            p = vec3<f32>(next.x, next.y, 0.0);
            
            if (i > 10u && i % 4u == 0u) {
            let dist = length(p.xy);
            let intensity = clamp(1.0 - dist * 0.05, 0.5, 1.0);
            let color = vec3<f32>(intensity, intensity, intensity);
                rasterize_point(camera, p * params.scale, color * params.intensity, dimensions);
            }
        }
    }
}

// Second compute pass: Apply post-processing
@compute @workgroup_size(16, 16, 1)
fn compute_feedback(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(prev_frame);
    
    // Skip if outside texture dimensions
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }
    
    let pixel_pos = vec2<i32>(global_id.xy);
    let pixel_index = pixel_pos.y * i32(dimensions.x) + pixel_pos.x;
    
    // Get the accumulated color from atomic buffer only (no previous frame)
    let accumulated = vec4<f32>(
        f32(atomicLoad(&atomic_buffer[pixel_index * 4])),
        f32(atomicLoad(&atomic_buffer[pixel_index * 4 + 1])),
        f32(atomicLoad(&atomic_buffer[pixel_index * 4 + 2])),
        ATOMIC_SCALE
    ) / ATOMIC_SCALE;
    
    let background_color = vec3<f32>(0.15, 0.15, 0.15); // Dark gray background
    let processed_color = params.intensity * pow(accumulated.rgb / max(accumulated.a, 0.001), vec3<f32>(0.6));

    // Blend with background - if no particles are at this pixel, use background color
    let has_particles = accumulated.r > 0.0 || accumulated.g > 0.0 || accumulated.b > 0.0;
    let final_color = select(background_color, processed_color, has_particles);

    textureStore(output, global_id.xy, vec4<f32>(final_color, 1.0));
}