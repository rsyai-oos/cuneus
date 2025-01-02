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
};
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
    let vertical_speed = 0.15;
    let rotation_speed = 0.05;
    
    let ang = vec2<f32>(
        PI * 0.5 + sin(t * rotation_speed) * 0.1,
        PI * 0.45 + sin(t * vertical_speed) * 0.15
    );

    camera.fov = FOV;
    camera.cam = get_camera_matrix(ang);
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

fn clifford_attractor(p: vec2<f32>, t: f32) -> vec2<f32> {
    let a = 1.7 + sin(t * 0.1) * 0.1;
    let b = 1.7 + cos(t * 0.15) * 0.1;
    let c = 0.6 + sin(t * 0.2) * 0.1;
    let d = 1.2 + cos(t * 0.25) * 0.1;
    
    return vec2<f32>(
        sin(a * p.y) + c * cos(a * p.x),
        sin(b * p.x) + d * cos(b * p.y)
    );
}

fn point_gen(tex_coords: vec2<f32>, dims: vec2<f32>) -> vec4<f32> {
    let r4 = rand4();
    var p = vec3<f32>(2.0 * (r4.x - 0.5), 2.0 * (r4.y - 0.5), 0.0);
    
    let t = time_data.time * params.speed;
    let color = vec3<f32>(1.0, 0.0, 0.0);
    var result = vec4<f32>(0.0);
    for(var i = 0; i < 25; i++) {
        let next = clifford_attractor(p.xy, t);
        p = vec3<f32>(next.x, next.y, 0.0);
        
        if(i > 20) {
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