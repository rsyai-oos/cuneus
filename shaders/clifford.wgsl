// atomic inspiration: https://compute.toys/view/1581; Draradech, 2024.
@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<i32>>;

struct TimeUniform {
    time: f32,
};
@group(1) @binding(0)
var<uniform> time_data: TimeUniform;

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
}
@group(2) @binding(0)
var<uniform> params: Params;

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

var<private> camera: Camera;
var<private> state: vec4<u32>;

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
    state = pcg4d(state);
    return vec4<f32>(state) / f32(0xffffffffu);
}

fn get_camera_matrix(ang: vec2<f32>) -> mat3x3<f32> {
    let x_dir = vec3<f32>(cos(ang.x)*sin(ang.y), cos(ang.y), sin(ang.x)*sin(ang.y));
    let y_dir = normalize(cross(x_dir, vec3<f32>(0.0, 1.0, 0.0)));
    let z_dir = normalize(cross(y_dir, x_dir));
    return mat3x3<f32>(-x_dir, y_dir, z_dir);
}

fn set_camera() {
    let dimensions = textureDimensions(prev_frame);
    let screen_size_f = vec2<f32>(dimensions);
    
    let t = time_data.time;
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
    camera.pos = -(camera.cam * vec3<f32>(12.0, 0.0, 0.0));
    camera.size = screen_size_f;
}


fn project(cam: Camera, p: vec3<f32>) -> vec3<f32> {
    let td = distance(cam.pos, p);
    let dir = (p - cam.pos) / td;
    let screen = dir * cam.cam;
    return vec3<f32>(screen.yz * cam.size.y / (cam.fov * screen.x) + 0.5 * cam.size, screen.x * td);
}

fn additive_blend(color: vec3<f32>, depth: f32, index: i32) {
    let scaled_color = vec3<i32>(floor(ATOMIC_SCALE * color / (depth * depth + 0.2) + rand4().xyz));
    
    if(scaled_color.x > 0) { atomicAdd(&atomic_buffer[index * 4], scaled_color.x); }
    if(scaled_color.y > 0) { atomicAdd(&atomic_buffer[index * 4 + 1], scaled_color.y); }
    if(scaled_color.z > 0) { atomicAdd(&atomic_buffer[index * 4 + 2], scaled_color.z); }
}

fn rasterize_point(pos: vec3<f32>, color: vec3<f32>, dims: vec2<f32>) -> vec4<f32> {
    let screen_size = vec2<i32>(dims);
    let projected_pos = project(camera, pos);
    let screen_coord = vec2<i32>(projected_pos.xy + 0.5 * rand4().xy);
    
    if(screen_coord.x < 0 || screen_coord.x >= screen_size.x || 
       screen_coord.y < 0 || screen_coord.y >= screen_size.y || 
       projected_pos.z < 0.0) {
        return vec4<f32>(0.0);
    }

    let idx = screen_coord.x + screen_size.x * screen_coord.y;
    additive_blend(color, projected_pos.z, idx);
    return vec4<f32>(color / (projected_pos.z * projected_pos.z + 0.2), 1.0);
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - abs(fract(h_prime / 2.0) * 2.0 - 1.0));
    let m = v - c;
    
    var r: f32 = 0.0;
    var g: f32 = 0.0;
    var b: f32 = 0.0;
    
    if (h_prime < 1.0) {
        r = c; g = x; b = 0.0;
    } else if (h_prime < 2.0) {
        r = x; g = c; b = 0.0;
    } else if (h_prime < 3.0) {
        r = 0.0; g = c; b = x;
    } else if (h_prime < 4.0) {
        r = 0.0; g = x; b = c;
    } else if (h_prime < 5.0) {
        r = x; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x;
    }
    
    return vec3<f32>(r + m, g + m, b + m);
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

fn point_gen(tex_coords: vec2<f32>, dims: vec2<f32>) -> vec4<f32> {
    let r4 = rand4();
    var p = vec3<f32>(2.0 * (r4.x - 0.5), 2.0 * (r4.y - 0.5), 0.0);
    
    let t = time_data.time * params.speed;
    var result = vec4<f32>(0.0);
    
    for(var i = 0; i < 25; i++) {
        let next = clifford_attractor(p.xy, t);
        p = vec3<f32>(next.x, next.y, 0.0);
        
        if(i > 20) {
            let angle = atan2(p.y, p.x) + t * 0.1;
            let hue = (degrees(angle) + 360.0) % 360.0;
            let dist = length(p.xy);
            let saturation = clamp(0.7 + sin(t * 0.2) * 0.3, 0.0, 1.0);
            let value = clamp(1.0 - dist * 0.2, 0.3, 1.0);
            
            let color = hsv_to_rgb(hue, saturation, value);
            result += rasterize_point(p * params.scale, color * params.intensity, dims);
        }
    }
    return result;
}
@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(prev_frame));
    let pixel_pos = vec2<i32>(FragCoord.xy);
    let pixel_index = pixel_pos.y * i32(dimensions.x) + pixel_pos.x;
    
    atomicStore(&atomic_buffer[pixel_index * 4], 0);
    atomicStore(&atomic_buffer[pixel_index * 4 + 1], 0);
    atomicStore(&atomic_buffer[pixel_index * 4 + 2], 0);
    state = vec4<u32>(
        u32(pixel_pos.x),
        u32(pixel_pos.y),
        u32(time_data.time * 1000.0),
        u32(time_data.time * 100.0)
    );
    
    set_camera();
    
    var accumulated_color = vec4<f32>(0.0);
    for(var i = 0; i < 5; i++) {
        accumulated_color += point_gen(tex_coords, dimensions);
    }
    let current = vec4<f32>(
        f32(atomicLoad(&atomic_buffer[pixel_index * 4])),
        f32(atomicLoad(&atomic_buffer[pixel_index * 4 + 1])),
        f32(atomicLoad(&atomic_buffer[pixel_index * 4 + 2])),
        ATOMIC_SCALE
    ) / ATOMIC_SCALE;
    
    let previous = textureSample(prev_frame, tex_sampler, tex_coords);
    return mix(current, previous, params.decay);
}

@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(prev_frame, tex_sampler, tex_coords);
    let exposed = params.intensity * color.rgb / max(color.a, 0.001);
    return vec4<f32>(exposed, 1.0);
}