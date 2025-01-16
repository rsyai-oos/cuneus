//MIT License, altunenes, 2023
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};


struct TimeUniform {
    time: f32,
};
struct Params {
    lambda: f32,
    theta: f32,
    alpha: f32,
    sigma: f32,
    gamma: f32,
    blue: f32,
    use_texture_colors: f32,
};

@group(1) @binding(0) var<uniform> u_time: TimeUniform;
@group(2) @binding(0) var<uniform> params: Params;
@group(3) @binding(0) var<uniform> u_resolution: ResolutionUniform;
const PI: f32 = 3.14159265359;

fn trge(x: f32) -> f32 {
    let f = fract(x - 0.5);
    let df = fwidth(x) * 2.0;
    return mix(abs(f - 0.5) * 2.0, 0.5, smoothstep(0.0, 1.0, df));
}

struct spires {
    pattern: f32,
    coord: vec2<f32>,
}

fn spiral(pos: vec2<f32>, slope: f32, resolution: vec2<f32>) -> spires {
    let l = length(pos);
    let ang = atan2(pos.y, pos.x) + 5.0 * u_time.time;
    
    let r = trge(ang / (2.0*PI) + l / slope);
    
    let phase = ang / 6.28318531;
    let segment = floor(l / slope + fract(phase));
    let blend = fract(phase);
    
    var coord = normalize(pos) * (segment - blend + 0.5) * slope;
    coord = (coord + 0.5 * resolution) / resolution;
    coord = clamp(coord, vec2<f32>(0.0), vec2<f32>(1.0));
    
    return spires(r, coord);
}

fn sts(uv: vec2<f32>, blur_amount: f32) -> vec3<f32> {
    let pixel_size = 1.0 / u_resolution.dimensions;     
    var color = vec3<f32>(0.0);
    let samples = 5;
    
    for(var i = -samples; i <= samples; i++) {
        for(var j = -samples; j <= samples; j++) {
            let offset = vec2<f32>(f32(i), f32(j)) * pixel_size * blur_amount;
            color += textureSample(tex, tex_sampler, uv + offset).xyz;
        }
    }
    
    return color / f32((2 * samples + 1) * (2 * samples + 1));
}

fn lumi(c: vec3<f32>) -> vec3<f32> {
    let d = clamp(dot(c.xyz, vec3<f32>(-0.25, 0.5, -0.25)), 0.0, 1.0);
    let color = mix(c, vec3<f32>(1.5), params.theta * d * 0.7); 
    let luma = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    return clamp(mix(color, vec3<f32>(luma), 0.7), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn luminance(uv: vec2<f32>) -> vec3<f32> {
    let c = sts(uv, 2.0);
    return select(
        lumi(c),
        c * 1.2,
        params.use_texture_colors > 0.5
    );
}

fn gradyy(fc: vec2<f32>, eps: f32, resolution: vec2<f32>) -> vec2<f32> {
    let e = vec2<f32>(eps, 0.0);
    
    let col = sts(fc / resolution, 1.5);
    
    let grad_x = dot(
        sts((fc + e.xy) / resolution, 1.5) - 
        sts((fc - e.xy) / resolution, 1.5),
        vec3<f32>(0.299, 0.587, 0.114)
    );
    
    let grad_y = dot(
        sts((fc + e.yx) / resolution, 1.5) - 
        sts((fc - e.yx) / resolution, 1.5),
        vec3<f32>(0.299, 0.587, 0.114)
    );

    return vec2<f32>(grad_x, grad_y) / (3.0 * eps);
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let resolution = u_resolution.dimensions;
    let centered_coord = 1.0*FragCoord.xy - 0.5 * resolution;
    
    let sr = spiral(centered_coord, params.lambda, resolution);
    let pattern = vec3<f32>(sr.pattern);
    let uv = sr.coord;
    
    let col = luminance(uv);
    
    let lff = select(col.x, length(col) / 1.732051, params.use_texture_colors > 0.5);
    let b = params.alpha * (1.0 - lff) + 0.35;
    var c = clamp(pattern.x - 1.0 + b, 0.0, 1.0);
    c = b - (b - c) * (b - c) / b / b;
    c = smoothstep(0.0, 1.0, c);
    
    let base_color = select(
        vec3<f32>(1.0 - c),
        col * (1.0 - c),
        params.use_texture_colors > 0.5
    );
    let light = normalize(vec3<f32>(0.5, 0.5, 2.0));
    let grad = gradyy(FragCoord.xy, 1.8, resolution);
    let n = normalize(vec3<f32>(grad, 1.2));
    let spec = dot(reflect(vec3<f32>(0.0, 0.0, -1.0), n), light);
    let diff = clamp(dot(light, n), 0.0, 1.0);
    
    let fegg = select(
        smoothstep(0.5, 1.0, base_color.x),
        smoothstep(0.5, 1.0, length(base_color) / 1.732051),
        params.use_texture_colors > 0.5
    );
    
    let sf = pow(clamp(spec, 0.0, 1.0), mix(1.0, 150.0, 1.0 - fegg)) * mix(1.0, 150.0, 1.0 - fegg) / 120.0;
    
    let final_color = select(
        mix(
            vec3<f32>(params.sigma, params.gamma, params.blue),
            vec3<f32>(1.0, 0.97, 0.9) * params.theta,
            smoothstep(0.0, 1.0, fegg)
        ),
        mix(
            col,
            col * params.theta,
            smoothstep(0.0, 1.0, fegg)
        ),
        params.use_texture_colors > 0.5
    );
    let vg = cos(1.7 * length((FragCoord.xy - 0.5 * resolution) / resolution.x));
    let vgf = smoothstep(0.0, 1.0, vg);
    
    return vec4<f32>((final_color * diff + 0.7 * sf) * vgf, 1.0);
}