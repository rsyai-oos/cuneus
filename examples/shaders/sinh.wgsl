// Sinh: fractal with 3D vis, Enes Altun, 2025 Licence: 
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Ported from Shadertoy: https://www.shadertoy.com/view/dt2cWR 
// Math for sinh http://paulbourke.net/fractals/sinh/

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct SinhParams {
    aa: i32,
    camera_x: f32,
    camera_y: f32,
    camera_z: f32,
    orbit_speed: f32,
    magic_number: f32,
    cv_min: f32,
    cv_max: f32,
    os_base: f32,
    os_scale: f32,
    base_color_r: f32,
    base_color_g: f32,
    base_color_b: f32,
    light_color_r: f32,
    light_color_g: f32,
    light_color_b: f32,
    ambient_r: f32,
    ambient_g: f32,
    ambient_b: f32,
    gamma: f32,
    iterations: i32,
    bound: f32,
    fractal_scale: f32,
    vignette_offset: f32,
};
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

@group(1) @binding(1) var<uniform> params: SinhParams;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;

var<private> R: v2;
var<private> iResolution: v2;

fn mul(a: v2, b: v2) -> v2 {
    return v2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn sh(z: v2) -> v2 {
    return v2(sinh(z.x) * cos(z.y), cosh(z.x) * sin(z.y + 0.01));
}

fn fractal_f(z_in: v2, c: v2) -> v2 {
    var z = z_in;
    for (var i = 0; i < params.iterations; i++) {
        let s = sh(z);
        z = abs(mul(mul(s, s), mul(s, s))) + c;
        if (dot(z, z) > params.bound) {
            return v2(f32(i), dot(z, z));
        }
    }
    return v2(f32(params.iterations), dot(z, z));
}

fn r2(p: v2, a: f32) -> v2 {
    return p * cos(a) + v2(-p.y, p.x) * sin(a);
}

fn distance_field(p: v3) -> v2 {
    let d1 = length(p - v3(0.0, 0.0, 3.25)) - 3.25;
    let d2 = p.z;
    return v2(min(d1, d2), select(1.0, 0.0, d1 < d2));
}

fn calculate_normal(p: v3) -> v3 {
    let eps = v2(0.001, 0.0);
    return normalize(v3(
        distance_field(p + eps.xyy).x - distance_field(p - eps.xyy).x,
        distance_field(p + eps.yxy).x - distance_field(p - eps.yxy).x,
        distance_field(p + eps.yyx).x - distance_field(p - eps.yyx).x
    ));
}

fn raycast(ro: v3, rd: v3) -> v2 {
    var t = 0.001;
    for (var i = 0; i < 500; i++) {
        let h = distance_field(ro + t * rd);
        if (h.x < 0.0001 * (t * 0.125 + 1.0)) {
            return v2(t, h.y);
        }
        if (t > 100.0) {
            break;
        }
        t += h.x;
    }
    return v2(-1.0, 0.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output);
    iResolution = v2(f32(dims.x), f32(dims.y));
    R = iResolution;
    
    if (id.x >= dims.x || id.y >= dims.y) {
        return;
    }
    
    var scene_color = v3(0.0);
    
    let d = mix(15.0, 10.0, smoothstep(0.2, 0.8, 0.5 + 0.5 * sin(time_data.time * 0.5)));
    var ro = v3(params.camera_x, params.camera_y, params.camera_z);
    
    let rotated_xy = r2(ro.xy, time_data.time * params.orbit_speed);
    ro = v3(rotated_xy.x, rotated_xy.y, ro.z);
    
    let fw = normalize(-ro);
    let rt = normalize(cross(fw, v3(0.0, 0.0, 1.0)));
    let up = cross(rt, fw);
    
    let aa_samples = params.aa;
    for (var i = 0; i < aa_samples; i++) {
        for (var j = 0; j < aa_samples; j++) {
            let aa_offset = v2(f32(i), f32(j)) / f32(aa_samples);
            let screen_coord = v2(f32(id.x), f32(dims.y - id.y)) + aa_offset;
            let uv = (2.0 * screen_coord - iResolution) / iResolution.y;
            let rd = normalize(uv.x * rt + uv.y * up + 1.5 * fw);
            
            let t = raycast(ro, rd);
            if (t.x < 0.0) {
                continue;
            }
            
            let p = ro + t.x * rd;
            let N = select(v3(0.0, 0.0, 1.0), normalize(p - v3(0.0, 0.0, 2.25)), t.y < 0.5);
            
            let fp = select(p.xy, 6.825 * p.xy / (6.825 - p.z), t.y < 0.5);
            
            let fractal_uv = fp * params.fractal_scale;
            let cv = mix(params.cv_min, params.cv_max, 0.01 + 0.01 * sin(0.1 * params.magic_number));
            let os = params.os_base + params.os_scale * (sin(0.1 * params.magic_number) + 0.1);
            let zi = fractal_f(fractal_uv, v2(os, cv));
            let ls = zi.y;
            
            let c1 = 0.5 + 0.5 * cos(3.0 + time_data.time + v3(0.0, 0.5, 1.0) + pi * v3(2.0 * ls));
            let c2 = 0.5 + 0.5 * cos(4.1 + time_data.time + pi * v3(ls));
            let c3 = 4.5 + 0.5 * cos(3.0 + time_data.time + v3(1.0, 0.5, 0.0) + pi * v3(2.0 * sin(ls)));
            let bc = sqrt(c1 * c2 * c3) * v3(params.base_color_r, params.base_color_g, params.base_color_b);
            
            let lp = v3(1.0, 0.0, 20.0);
            let ld = normalize(lp - p);
            let ln = max(0.001, length(lp - p));
            
            var ao = 0.0;
            var s = 1.0;
            for (var k = 0; k < 15; k++) {
                let h = 0.01 + 0.15 * f32(k) * 0.25;
                ao += (h - distance_field(p + h * N).x) * s;
                s *= 0.85;
            }
            ao = max(0.2, 1.0 - ao);
            
            var sd = 1.0;
            var ts = 0.01;
            for (var k = 0; k < 30; k++) {
                let h = distance_field(p + 0.001 * N + ld * ts).x;
                sd = min(sd, 32.0 * h / ts);
                ts += max(0.001, min(h, 0.1));
                if (h < 0.0001 || ts > ln) {
                    break;
                }
            }
            sd = min(max(0.0, sd) + ao * 0.2, 1.0);
            
            let df = max(0.0, dot(N, ld));
            let at = 1.0 / (1.0 + ln * 0.01 + ln * ln * 0.002);
            let sp = pow(max(dot(reflect(-ld, N), -rd), 0.0), 20.0);
            let fr = max(0.0, 1.0 + dot(rd, N));
            
            var color = bc * (df + 0.5 * ao);
            color += bc * v3(params.light_color_r, params.light_color_g, params.light_color_b) * sp * 8.0;
            color += bc * v3(params.ambient_r, params.ambient_g, params.ambient_b) * fr * fr * 6.0;
            color *= at * sd * ao;
            
            color *= max(params.vignette_offset, min(1.1, 55.0 / dot(p, p)) - 0.1);
            
            scene_color += color;
        }
    }
    
    let final_color = sqrt(max(v3(0.0), scene_color / f32(aa_samples * aa_samples)));
    let gamma_corrected = pow(final_color, v3(1.0 / params.gamma));
    
    textureStore(output, vec2<i32>(id.xy), v4(gamma_corrected, 1.0));
}