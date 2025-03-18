// MIT License, Enes Altun, 2025
// resources for Skilling's algorithm
// https://github.com/joshspeagle/dynesty
// https://doi.org/10.1063/1.1751381 and https://doi.org/10.1063/1.1751382
// http://www.inference.org.uk/bayesys/test/hilbert.c
// https://www.shadertoy.com/view/3tl3zl
// lighting technique from @fad, (2023): Analytic Direct Lighting: https://www.shadertoy.com/view/dlcXR4
struct TimeUniform {
    time: f32,
};
struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};
struct Params {
    iterations: u32,
    num_rays: f32,
    _pad1: vec2<f32>,      
    scale: f32,             
    time_scale: f32,        
    vignette_radius: f32,   
    vignette_softness: f32, 
    color_offset: vec3<f32>,
    flanc: f32,            
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const MAX_SEGMENTS: u32 = 64u;
const MAX_ANGLES: u32 = 128u;
fn atan2(y: f32, x: f32) -> f32 {
    if (x > 0.0) {
        return atan(y / x);
    } else if (x < 0.0) {
        return atan(y / x) + select(-PI, PI, y >= 0.0);
    } else {
        return select(-PI / 2.0, PI / 2.0, y >= 0.0);
    }
}
fn fmod(x: f32, y: f32) -> f32 {
    return x - y * floor(x / y);
}
fn hilbert(idx: u32) -> vec2<u32> {
    var res = vec2<u32>(0u, 0u);
    var i = idx;
    for (var k = 0u; k < params.iterations; k++) {
        let r = vec2<u32>((i >> 1u) & 1u, (i ^ (i >> 1u)) & 1u);
        if (r.y == 0u) { 
            if (r.x == 1u) { 
                res = ((1u << k) - 1u) - res; 
            } 
            let temp = res.x;
            res.x = res.y;
            res.y = temp;
        }
        res += vec2<u32>(r.x << k, r.y << k);
        i >>= 2u;
    }
    return res;
}
// Generate Hilbert curve point for a given index: chp: calculate hilbert point
fn CHP(i: u32) -> vec2<f32> {
    let hpos = hilbert(i);
    let size = 1u << params.iterations;
    let np = vec2<f32>(hpos) / f32(size);
    let scale = 0.7 * min(u_resolution.dimensions.x, u_resolution.dimensions.y);
    let screen_pos = np * scale + (u_resolution.dimensions - vec2<f32>(scale)) * 0.5;
    
    return screen_pos;
}
fn get_point(i: u32) -> vec2<f32> {
    let num_points = 1u << (2u * params.iterations);
    if (i >= num_points) {
        return vec2<f32>(0.0);
    }
    return CHP(i);
}
struct LineSegment {
    p0: vec2<f32>,
    p1: vec2<f32>,
    emissive_color: vec3<f32>,
};
fn sdf(l: LineSegment, p: vec2<f32>) -> f32 {
    let pa = p - l.p0;
    let ba = l.p1 - l.p0;
    let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}
fn blend_over(top: vec4<f32>, bottom: vec4<f32>) -> vec4<f32> {
    let a = top.a + bottom.a * (1.0 - top.a);
    if (a < 0.0001) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>((top.rgb * top.a + bottom.rgb * bottom.a * (1.0 - top.a)) / a, a);
}
fn draw_sdf(dst: vec4<f32>, src: vec4<f32>, dist: f32) -> vec4<f32> {
    return blend_over(
        vec4<f32>(src.rgb, src.a * clamp(1.5 - abs(dist), 0.0, 1.0)), 
        dst
    );
}
// Matrix inverse 2x2
fn inverse_2x2(m: mat2x2<f32>) -> mat2x2<f32> {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if (abs(det) < 0.0001) {
        return mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
    }
    let inv_det = 1.0 / det;
    return mat2x2<f32>(
        m[1][1] * inv_det, -m[0][1] * inv_det,
        -m[1][0] * inv_det, m[0][0] * inv_det
    );
}

// insertion sort
fn sort_angles(segments: ptr<function, array<LineSegment, MAX_SEGMENTS>>, angles: ptr<function, array<f32, MAX_ANGLES>>, num_segments: u32) {
    let num_angles = 2u * num_segments;
    for (var i = 0u; i < num_segments; i++) {
        for (var j = 0u; j < 2u; j++) {
            let k = 2u * i + j;
            let p = select((*segments)[i].p0, (*segments)[i].p1, j == 1u);
            let angle = fmod(atan2(p.y, p.x), 2.0 * PI);
            var l = i32(k) - 1;
            while (l >= 0 && angle < (*angles)[u32(l)]) {
                (*angles)[u32(l) + 1u] = (*angles)[u32(l)];
                l -= 1;
            }
            (*angles)[u32(l) + 1u] = angle;
        }
    }
}
// add radiance from a line segment
fn integrate_radiance(segment: LineSegment, angle: vec2<f32>) -> vec3<f32> {
    return (angle[1] - angle[0]) * segment.emissive_color;
}

// add sky radiance for a basic angle range
fn isb(angle: vec2<f32>) -> vec3<f32> {
    let sky_color = vec3<f32>(0.01, 0.02, 0.04);
    return sky_color * (angle[1] - angle[0]);
}

// Integrate sky radiance with wrap-around support
fn isr(angle: vec2<f32>) -> vec3<f32> {
    if (angle[1] < 2.0 * PI) {
        return isb(angle);
    }
    return isb(vec2<f32>(angle[0], 2.0 * PI)) + 
           isb(vec2<f32>(0.0, angle[1] - 2.0 * PI));
}

// Find which segment is intersected by a ray at a specific angle
fn find_index(segments: ptr<function, array<LineSegment, MAX_SEGMENTS>>, angle: f32, num_segments: u32) -> i32 {
    var m = mat2x2<f32>(0.0, 0.0, 0.0, 0.0);
    m[1] = vec2<f32>(cos(angle), sin(angle));
    var best_index = -1;
    var best_u = 1e10;
    for (var i = 0u; i < num_segments; i++) {
        m[0] = (*segments)[i].p0 - (*segments)[i].p1;
        // Calculate the inverse and multiply manually in case of singular matrices
        let inv_m = inverse_2x2(m);
        let tu = inv_m * (*segments)[i].p0;
        // Check if valid intersection (0 <= t <= 1 and 0 <= u <= best_u)
        if (tu.x >= 0.0 && tu.x <= 1.0 && tu.y >= 0.0 && tu.y <= best_u) {
            best_u = tu.y;
            best_index = i32(i);
        }
    }
    return best_index;
}
// total radiance from all directions
fn calculate_fluence(segments: ptr<function, array<LineSegment, MAX_SEGMENTS>>, angles: ptr<function, array<f32, MAX_ANGLES>>, num_segments: u32) -> vec3<f32> {
    var fluence = vec3<f32>(0.0);
    let num_angles = 2u * num_segments;
    for (var i = 0u; i < num_angles; i++) {
        var a = vec2<f32>((*angles)[i], 0.0);
        if (i + 1u < num_angles) {
            a[1] = (*angles)[i + 1u];
        } else {
            a[1] = (*angles)[0] + 2.0 * PI;
        }
        if (abs(a[0] - a[1]) < 0.0001) {
            continue;
        }
        let mid_angle = (a[0] + a[1]) / 2.0;
        let j = find_index(segments, mid_angle, num_segments);
        if (j == -1) {
            fluence += isr(a);
        } else {
            fluence += integrate_radiance((*segments)[u32(j)], a);
        }
    }
    return fluence;
}
@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let num_points = 1u << (2u * params.iterations);
    let num_segments = min(num_points - 1u, MAX_SEGMENTS);
    var segments: array<LineSegment, MAX_SEGMENTS>;
    var angles: array<f32, MAX_ANGLES>;
    for (var i = 0u; i < MAX_ANGLES; i++) {
        angles[i] = 0.0;
    }
    let pixel_pos = vec2<f32>(frag_coord.x, u_resolution.dimensions.y - frag_coord.y);
    for (var i = 0u; i < num_segments; i++) {
        segments[i].p0 = get_point(i) - pixel_pos;
        segments[i].p1 = get_point(i + 1u) - pixel_pos;
        segments[i].emissive_color = vec3<f32>(0.0);
    }
    let t = fmod(u_time.time * params.time_scale * 0.1, 1.0);
    let exact_pos = t * f32(num_segments);
    let light_length: f32 = params.num_rays;
    for (var i = 0u; i < num_segments; i++) {
        // Calculate distance along the curve from this segment to the light center
        let fi = f32(i);
        let dcp = min(
            abs(fi - exact_pos),
            min(
                abs(fi + f32(num_segments) - exact_pos),
                abs(fi - f32(num_segments) - exact_pos)
            )
        );
        // Only light up segments within the light's range,
        // dcp: distance to curve position
        if (dcp < light_length) {
            let intensity = exp(-dcp * dcp / (light_length * 0.5));
            let color_pos = dcp / light_length;
            let base_color1 = vec3<f32>(1.0, 0.3, 0.05) + params.color_offset;
            let base_color2 = vec3<f32>(0.1, 0.7, 1.0) + params.color_offset;
            let light_color = mix(
                base_color1,
                base_color2,
                smoothstep(0.0, 1.0, color_pos)
            );
            segments[i].emissive_color = light_color * intensity * 2.0;
        }
    }
    sort_angles(&segments, &angles, num_segments);
    let fluence = calculate_fluence(&segments, &angles, num_segments);
    var frag_color = vec4<f32>(1.0 - params.flanc / pow(1.0 + fluence, vec3<f32>(params.vignette_softness)), 1.0);
    for (var i = 0u; i < num_segments; i++) {
        let base_curve_color = vec3<f32>(0.15);
        let has_emission = length(segments[i].emissive_color) > 0.01;
        let segment_color = select(
            base_curve_color,
            2.0 * segments[i].emissive_color,
            has_emission
        );
        let thickness = select(1.0, params.scale, has_emission);
        frag_color = draw_sdf(
            frag_color, 
            vec4<f32>(segment_color, 1.0),
            sdf(segments[i], vec2<f32>(0.0)) / thickness
        );
    }
    frag_color = vec4<f32>(gamma(frag_color.rgb, params.vignette_radius), frag_color.a);
    return frag_color;
}
fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(max(color, vec3<f32>(0.0)), vec3<f32>(1.0 / gamma));
}