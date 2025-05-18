struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct PathTracingParams {
    camera_pos_x: f32,
    camera_pos_y: f32,
    camera_pos_z: f32,
    camera_target_x: f32,
    camera_target_y: f32,
    camera_target_z: f32,
    fov: f32,
    aperture: f32,

    max_bounces: u32,
    samples_per_pixel: u32,
    accumulate: u32,

    num_spheres: u32,
    mouse_x: f32,
    mouse_y: f32,

    rotation_speed: f32,

    exposure: f32,
}
@group(1) @binding(0) var<uniform> params: PathTracingParams;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m3 = mat3x3<f32>;
const pi = 3.14159265359;

var<private> R: v2;
var<private> seed: u32;

struct Ray {
    origin: v3,
    direction: v3,
}

struct HitRecord {
    p: v3,
    normal: v3, 
    t: f32, 
    front_face: bool, 
    material: u32,
}

// Material definition
struct Material {
    albedo: v3,     
    emissive: v3,
    metallic: f32,
    roughness: f32,
    ior: f32,
    subsurface: f32,
    glow: f32,
}

struct Sphere {
    center: v3,
    radius: f32,
    material: u32,
}

fn hash_u(_a: u32) -> u32 { 
    var a = _a; 
    a ^= a >> 16;
    a *= 0x7feb352du;
    a ^= a >> 15;
    a *= 0x846ca68bu;
    a ^= a >> 16;
    return a; 
}

fn hash_f() -> f32 { 
    var s = hash_u(seed); 
    seed = s;
    return (f32(s) / f32(0xffffffffu)); 
}

fn hash_v2() -> v2 { 
    return v2(hash_f(), hash_f()); 
}

fn hash_v3() -> v3 { 
    return v3(hash_f(), hash_f(), hash_f()); 
}

fn random_unit_vector() -> v3 {
    let a = hash_f() * 2.0 * pi;
    let z = hash_f() * 2.0 - 1.0;
    let r = sqrt(1.0 - z*z);
    return v3(r * cos(a), r * sin(a), z);
}

fn random_in_unit_disk() -> v2 {
    let a = hash_f() * 2.0 * pi;
    let r = sqrt(hash_f());
    return v2(r * cos(a), r * sin(a));
}

fn reflect(v: v3, n: v3) -> v3 {
    return v - 2.0 * dot(v, n) * n;
}

fn hit_sphere(sphere: Sphere, ray: Ray, t_min: f32, t_max: f32, rec: ptr<function, HitRecord>) -> bool {
    let oc = ray.origin - sphere.center;
    let a = dot(ray.direction, ray.direction);
    let half_b = dot(oc, ray.direction);
    let c = dot(oc, oc) - sphere.radius * sphere.radius;
    let discriminant = half_b * half_b - a * c;
    
    if (discriminant < 0.0) {
        return false;
    }
    let sqrtd = sqrt(discriminant);
    var root = (-half_b - sqrtd) / a;
    // If this root is not in the acceptable range, try the other root
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root) {
            return false;
        }
    }
    // Record the hit with the closest root
    (*rec).t = root;
    (*rec).p = ray.origin + root * ray.direction;
    let outward_normal = ((*rec).p - sphere.center) / sphere.radius;
    // Determine if we're hitting from inside or outside and set normal accordingly
    let front_face = dot(ray.direction, outward_normal) < 0.0;
    (*rec).normal = select(-outward_normal, outward_normal, front_face);
    (*rec).front_face = front_face;
    (*rec).material = sphere.material;
    
    return true;
}

const scene_offset_x: f32 = 12.5;

fn create_scene(mouse_x: f32, mouse_y: f32, time: f32) -> array<Sphere, 12> {
    var spheres: array<Sphere, 12>;
    
    spheres[0] = Sphere(
        v3(0.0 + scene_offset_x, -100.5, -1.0),
        100.0,
        0u
    );
    spheres[1] = Sphere(
        v3(-1.7 + scene_offset_x, 0.5, -1.3),
        0.5,
        6u
    );
    spheres[2] = Sphere(
        v3(-0.2 + scene_offset_x, 0.25, -0.9),
        0.25,
        2u
    );
    
    spheres[3] = Sphere(
        v3(mouse_x * 2.0 - 1.0 + scene_offset_x, mouse_y * 0.5 + 0.2, -0.5),
        0.2,
        1u
    );
    
    let red_pulse = 2.0 + 0.15 * sin(time * 0.7);
    spheres[4] = Sphere(
        v3(2.0 + scene_offset_x, 1.7, -2.0),
        0.4 * red_pulse,
        4u
    );
    
    spheres[5] = Sphere(
        v3(0.8 + scene_offset_x, 0.15, -0.8),
        0.15,
        7u
    );
    
    let orb_time = time * 0.2;
    let orb_x = sin(orb_time) * 1.5;
    let orb_y = 0.7 + 0.3 * sin(orb_time * 0.7);
    let orb_z = cos(orb_time) * 1.5 - 1.0;
    
    spheres[6] = Sphere(
        v3(orb_x + scene_offset_x, orb_y, orb_z),
        0.25,
        14u
    );
    
    spheres[7] = Sphere(
        v3(-1.0 + scene_offset_x, 1.5, -1.5),
        0.3,
        5u
    );
    spheres[8] = Sphere(
        v3(1.5 + scene_offset_x, 0.4, -1.2),
        0.4,
        8u
    );
    
    spheres[9] = Sphere(
        v3(-0.3 + scene_offset_x, 0.15, -0.7),
        0.15,
        9u
    );
    
    spheres[10] = Sphere(
        v3(0.3 + scene_offset_x, 0.6, -1.5),
        0.2,
        10u
    );
    
    let pulse_rate = 0.3;
    let pulse_phase = smoothstep(0.0, 1.0, fract(time * pulse_rate));
    let pulse_size = 0.2 + 0.05 * pulse_phase;
    
    spheres[11] = Sphere(
        v3(-0.8 + scene_offset_x, 0.12, -0.3),
        pulse_size,
        12u
    );
    
    return spheres;
}

fn get_material(id: u32, rec: HitRecord) -> Material {
    var mat: Material;
    
    mat.ior = 1.5;
    mat.subsurface = 0.0;
    mat.glow = 0.0;
    
    switch(id) {
        case 0u: {
            let scale = 1.0;
            let pattern_x = floor(rec.p.x * scale);
            let pattern_z = floor(rec.p.z * scale);
            let is_dark = fract((pattern_x + pattern_z) * 0.5) < 0.5;
            
            if (is_dark) {
                mat.albedo = v3(0.02, 0.02, 0.02); 
            } else {
                mat.albedo = v3(0.7, 0.7, 0.7);
            }
            mat.emissive = v3(0.0);
            mat.metallic = 0.0;
            mat.roughness = 0.9;
        }
        case 1u: {
            mat.albedo = v3(0.95, 0.7, 0.3);
            mat.emissive = v3(0.0);
            mat.metallic = 0.9;
            mat.roughness = 0.1;
            mat.glow = 0.1;
        }
        case 2u: {
            mat.albedo = v3(0.95, 0.95, 1.0);
            mat.emissive = v3(0.0);
            mat.metallic = 1.0;
            mat.roughness = 0.0;
            mat.ior = 1.52;
            mat.glow = 0.05;
        }
        case 3u: {
            mat.albedo = v3(1.0);
            mat.emissive = v3(4.0, 3.5, 2.5); 
            mat.metallic = 0.0;
            mat.roughness = 1.0;
            mat.glow = 1.0;
        }
        case 4u: {
            let pure_red = v3(1.0, 0.05, 0.02);
            
            mat.albedo = v3(1.0, 0.2, 0.2);
            mat.emissive = pure_red * 4.0; 
            mat.metallic = 0.0;
            mat.roughness = 1.0;
            mat.glow = 1.0;
        }
        case 5u: {
            let pure_blue = v3(1.1, 0.0, 1.1);
            
            mat.albedo = v3(0.3, 0.3, 1.0);
            mat.emissive = pure_blue * 4.0;
            mat.metallic = 0.0;
            mat.roughness = 1.0;
            mat.glow = 1.0;
        }
        case 6u: {
            mat.albedo = v3(0.95, 0.95, 0.95);
            mat.emissive = v3(0.0);
            mat.metallic = 1.0;
            mat.roughness = 0.0;
        }
        case 7u: {
            mat.albedo = v3(0.1, 0.8, 0.2);
            mat.emissive = v3(0.0);
            mat.metallic = 0.1;
            mat.roughness = 0.7;
        }
        case 8u: { // Copper
            mat.albedo = v3(0.95, 0.64, 0.54);
            mat.emissive = v3(0.0);
            mat.metallic = 0.85;
            mat.roughness = 0.2;
            mat.glow = 0.1;
        }
        case 9u: { // Ruby-like
            mat.albedo = v3(0.9, 0.1, 0.2);
            mat.emissive = v3(0.5, 0.0, 0.0);
            mat.metallic = 0.2;
            mat.roughness = 0.1;
            mat.ior = 1.77;
            mat.subsurface = 0.3;
            mat.glow = 0.3;
        }
        case 10u: {
            mat.albedo = v3(0.8, 0.8, 0.9);
            mat.emissive = v3(0.0);
            mat.metallic = 1.0;
            mat.roughness = 0.1;
            mat.ior = 1.4;
            mat.subsurface = 0.0;
        }
        case 11u: {
            let scale = 5.0;
            let turbulence = sin(scale * rec.p.x) * sin(scale * rec.p.y) * sin(scale * rec.p.z);
            let pattern = 0.5 * (1.0 + sin(scale * rec.p.x + 10.0 * turbulence));

            mat.albedo = mix(
                v3(0.8, 0.8, 0.8),
                v3(0.2, 0.2, 0.35),
                pattern
            );
            mat.emissive = v3(0.0);
            mat.metallic = 0.0;
            mat.roughness = 0.05;
            mat.subsurface = 0.3;
        }
        case 12u: {
            let t = time_data.time * 0.3;
            
            let color_phase = t + dot(rec.normal, v3(0.5, 0.3, 0.2));
            let r = 0.5 + 0.5 * sin(color_phase);
            let g = 0.5 + 0.5 * sin(color_phase + 2.1);
            let b = 0.5 + 0.5 * sin(color_phase + 4.2);
            
            mat.albedo = v3(r, g, b);
            mat.emissive = mat.albedo * 2.5;
            mat.metallic = 0.0;
            mat.roughness = 0.1;
            mat.subsurface = 0.5;
            mat.glow = 0.8;
        }
        case 14u: {
            let t = time_data.time * 0.3;

            let color1 = v3(1.0, 0.7, 0.0);
            let color2 = v3(0.0, 0.9, 1.0);
            let blend = (sin(t) * 0.5 + 0.5);

            mat.albedo = v3(1.0);
            mat.emissive = mix(color1, color2, blend) * 5.0;
            mat.metallic = 0.0;
            mat.roughness = 1.0;
            mat.glow = 1.0;
        }
        default: {
            mat.albedo = v3(1.0, 0.0, 1.0);
            mat.emissive = v3(0.0);
            mat.metallic = 0.0;
            mat.roughness = 1.0;
        }
    }
    
    return mat;
}

fn subsurface_scatter(ray: Ray, rec: HitRecord, material: Material) -> v3 {
    // COMPLETELY DISABLE subsurface scattering
    // no light passes through any object
    return v3(0.0);
}

fn scatter(ray: Ray, rec: HitRecord, attenuation_out: ptr<function, v3>, scattered_out: ptr<function, Ray>) -> bool {
    let material = get_material(rec.material, rec);
    
    if (length(material.emissive) > 0.0) {
        *attenuation_out = material.emissive;
        return false; // Don't scatter for light sources
    }

    let subsurface = subsurface_scatter(ray, rec, material);
    
    var scatter_direction: v3;
    let unit_direction = normalize(ray.direction);
    
    // ALL materials will either reflect (if metallic) or diffuse (if not)
    // No refraction will ever happen
    if (material.metallic > 0.3 || rec.material == 2u || rec.material == 10u) {
        // Metal reflection (or forced reflection for former glass materials)
        let reflected = reflect(unit_direction, rec.normal);
        scatter_direction = reflected + material.roughness * random_unit_vector();
        
        if (dot(scatter_direction, rec.normal) < 0.0) {
            scatter_direction = rec.normal;
        }
    } else {
        // Lambertian (diffuse) reflection
        scatter_direction = rec.normal + random_unit_vector();
        // Avoid zero vector
        if (length(scatter_direction) < 0.001) {
            scatter_direction = rec.normal;
        }
    }
    
    // IMPORTANT: Add significant offset to prevent rays from leaking through surfaces
    (*scattered_out).origin = rec.p + rec.normal * 0.002;
    (*scattered_out).direction = normalize(scatter_direction);
    
    // Set attenuation based on material properties
    if (rec.material == 2u || rec.material == 10u) {
        // Override for previously glass materials - make them mirror-like
        *attenuation_out = v3(0.9);
    } else {
        *attenuation_out = material.albedo + subsurface;
    }
    
    return true;
}

fn trace_ray(ray: Ray, max_bounces: u32) -> v3 {
    var current_ray = ray;
    var current_attenuation = v3(1.0);
    var final_color = v3(0.0);
    
    let spheres = create_scene(params.mouse_x, params.mouse_y, time_data.time);
    
    for (var bounce: u32 = 0; bounce < max_bounces; bounce++) {
        var rec: HitRecord;
        rec.t = 1000.0; 
        var hit_anything = false;
        var closest_so_far = rec.t;

        for (var i: u32 = 0; i < min(params.num_spheres, 12u); i++) {
            if (hit_sphere(spheres[i], current_ray, 0.001, closest_so_far, &rec)) {
                hit_anything = true;
                closest_so_far = rec.t;
            }
        }
        
        if (hit_anything) {
            var scattered: Ray;
            var attenuation: v3;
            
            let material = get_material(rec.material, rec);
            
            final_color += current_attenuation * material.emissive;
            
            // Always scatter (be opaque) for non-emissive materials
            var should_scatter = true;
            
            // For emissive materials, don't scatter if they're really bright
            if (length(material.emissive) > 0.0) {
                should_scatter = false;
            }
            
            if (should_scatter) {
                // Force scatter - NEVER allow refraction
                if (!scatter(current_ray, rec, &attenuation, &scattered)) {
                    break;
                }
                
                if (rec.material == 2u || rec.material == 10u) {
                    attenuation = v3(0.9, 0.9, 0.9);
                }
                
                current_attenuation *= attenuation;
                
                current_ray.origin = rec.p + rec.normal * 0.002;
                current_ray.direction = scattered.direction;
            } else {
                break;
            }
        } else {
            //bg
            final_color += current_attenuation * v3(0.0, 0.0, 0.0);
            break;
        }
        
        // Russian roulette for path termination
        if (bounce > 2) {
            let p_continue = min(0.95, max(current_attenuation.r, max(current_attenuation.g, current_attenuation.b)));
            if (hash_f() > p_continue) {
                break;
            }
            current_attenuation /= p_continue;
        }
    }
    
    return final_color;
}

// ACES
fn color_preserving_tonemap(input_color: v3) -> v3 {
    let intensity = max(input_color.r, max(input_color.g, input_color.b));
    
    if (intensity <= 0.0001) {
        return v3(0.0);
    }
    
    let normalized_color = input_color / intensity;
    
    const m1 = mat3x3<f32>(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    const m2 = mat3x3<f32>(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );


    var tone_mapped_intensity = intensity;
    
    tone_mapped_intensity = (tone_mapped_intensity * (0.15 * tone_mapped_intensity + 0.05)) / 
                          (tone_mapped_intensity * (0.15 * tone_mapped_intensity + 0.5) + 0.06);

    let saturation_preservation = 0.9;
    
    var full_aces = input_color;
    var v = m1 * full_aces;    
    var a = v * (v + 0.0245786) - 0.000090537;
    var b = v * (0.983729 * v + 0.4329510) + 0.238081;
    full_aces = m2 * (a / b);

    let color_preservation_threshold = 1.0;
    let blend_factor = min(1.0, intensity / color_preservation_threshold);
    
    let colorized = normalized_color * tone_mapped_intensity;

    return mix(full_aces, colorized, blend_factor * saturation_preservation);
}


fn aces_tonemap(input_color: v3) -> v3 {

    var color = input_color * 1.4;

    const m1 = mat3x3<f32>(
        0.59719, 0.07600, 0.02840,
        0.35458, 0.90834, 0.13383,
        0.04823, 0.01566, 0.83777
    );
    const m2 = mat3x3<f32>(
        1.60475, -0.10208, -0.00327,
        -0.53108,  1.10813, -0.07276,
        -0.07367, -0.00605,  1.07602
    );

    var v = m1 * color;    
    var a = v * (v + 0.0245786) - 0.000090537;
    var b = v * (0.983729 * v + 0.4329510) + 0.238081;
    
    var tonemapped = pow(max(v3(0.0), m2 * (a / b)), v3(0.92));
    
    return tonemapped;
}

fn get_camera_ray(uv: v2) -> Ray {
    let aspect_ratio = R.x / R.y;
    
    let lookfrom = v3(params.camera_pos_x, params.camera_pos_y, params.camera_pos_z);
    let lookat = v3(params.camera_target_x, params.camera_target_y, params.camera_target_z);
    let vup = v3(0.0, 1.0, 0.0);
    let aperture = params.aperture;
    let focus_dist = length(lookfrom - lookat);
    
    let w = normalize(lookfrom - lookat);
    let u = normalize(cross(vup, w));
    let v = cross(w, u);
    
    let theta = params.fov * pi / 180.0;
    let h = tan(theta / 2.0);
    let viewport_height = 2.0 * h;
    let viewport_width = aspect_ratio * viewport_height;

    let offset = v2(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    
    let lens_radius = aperture / 2.0;
    let rd = lens_radius * random_in_unit_disk();
    let offset_u = u * rd.x;
    let offset_v = v * rd.y;
    
    let ray_origin = lookfrom + offset_u + offset_v;
    let ray_direction = normalize(
        viewport_width * focus_dist * offset.x * u +
        viewport_height * focus_dist * offset.y * v -
        focus_dist * w -
        offset_u -
        offset_v
    );
    
    return Ray(ray_origin, ray_direction);
}

fn has_animation() -> bool {
    return params.rotation_speed > 0.01;
}
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output);
    R = vec2<f32>(dimensions);
    
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }
    
    let uv = vec2<f32>(
        f32(global_id.x) + 0.5,
        f32(global_id.y) + 0.5
    ) / R;
    
    seed = global_id.x + global_id.y * dimensions.x + time_data.frame * 719393;
    
    var pixel_color = v3(0.0);
    let samples = params.samples_per_pixel;
    
    for (var s: u32 = 0; s < samples; s++) {
        let jitter = (hash_v2() - 0.5) / R;
        let jittered_uv = uv + jitter;
        
        let ray = get_camera_ray(jittered_uv);
        
        pixel_color += trace_ray(ray, params.max_bounces);
    }
    
    pixel_color /= f32(samples);
    
    pixel_color *= params.exposure * 1.2;
    
    pixel_color = color_preserving_tonemap(pixel_color);
    
    pixel_color = pow(pixel_color, v3(1.0 / 2.2));
    
    let pixel_idx = global_id.x + dimensions.x * global_id.y;

    let should_accumulate = params.accumulate > 0 && time_data.frame > 0 && 
                           (!has_animation() || time_data.frame < 64);

    if (should_accumulate) {
        let old_r = f32(atomicLoad(&atomic_buffer[pixel_idx * 3])) / 1000.0;
        let old_g = f32(atomicLoad(&atomic_buffer[pixel_idx * 3 + 1])) / 1000.0;
        let old_b = f32(atomicLoad(&atomic_buffer[pixel_idx * 3 + 2])) / 1000.0;
        let old_color = v3(old_r, old_g, old_b);
        
        let max_blend_frames = 8.0;
        let effective_frame = min(f32(time_data.frame), max_blend_frames);
        let blend_factor = 1.0 / (effective_frame + 1.0);

        pixel_color = mix(old_color, pixel_color, max(blend_factor, 0.1));
    }
    
    atomicStore(&atomic_buffer[pixel_idx * 3], u32(pixel_color.r * 1000.0));
    atomicStore(&atomic_buffer[pixel_idx * 3 + 1], u32(pixel_color.g * 1000.0));
    atomicStore(&atomic_buffer[pixel_idx * 3 + 2], u32(pixel_color.b * 1000.0));
    
    textureStore(output, vec2<i32>(global_id.xy), vec4<f32>(pixel_color, 1.0));
}