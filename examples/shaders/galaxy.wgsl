//inspiration from Dave Hoskins' famous galaxy shader: https://www.shadertoy.com/view/MdXSzS

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};

@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: Params;

struct Params {
    max_iterations: i32,
    max_sub_iterations: i32,
    point_intensity: f32,
    center_scale: f32,
    time_scale: f32,
    dist_offset: f32,
    _pad1: vec2<f32>,
};

const PI: f32 = 3.14159265359;
const LIGHT_DIR: vec3<f32> = vec3<f32>(0.577350269, 0.577350269, 0.577350269);
fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}
// h3: hash for 3D vectors
fn h3(p: vec3<f32>) -> vec3<f32> {
    var pp = fract(p * vec3<f32>(443.8975, 397.2973, 491.1871));
    pp = pp + dot(pp.zxy, pp.yxz + 19.19);
    return fract((pp.xxy + pp.yxx) * pp.zyx);
}

// h: hash function
fn h(n: f32) -> f32 {
    return fract(sin(n) * 43758.5453);
}

// n2: 2D noise
fn n2(x: vec2<f32>) -> f32 {
    let p = floor(x);
    let f = fract(x);
    let n = p.x + p.y * 57.0;

    return mix(
        mix(h(n + 0.0), h(n + 1.0), smoothstep(0.0, 1.0, f.x)),
        mix(h(n + 57.0), h(n + 58.0), smoothstep(0.0, 1.0, f.x)),
        smoothstep(0.0, 1.0, f.y)
    );
}

// n3: 3D noise
fn n3(x: vec3<f32>) -> f32 {
    let p = floor(x);
    let f = fract(x);
    let n = p.x + p.y * 57.0 + 113.0 * p.z;

    return mix(
        mix(
            mix(h(n + 0.0), h(n + 1.0), smoothstep(0.0, 1.0, f.x)),
            mix(h(n + 57.0), h(n + 58.0), smoothstep(0.0, 1.0, f.x)),
            smoothstep(0.0, 1.0, f.y)
        ),
        mix(
            mix(h(n + 113.0), h(n + 114.0), smoothstep(0.0, 1.0, f.x)),
            mix(h(n + 170.0), h(n + 171.0), smoothstep(0.0, 1.0, f.x)),
            smoothstep(0.0, 1.0, f.y)
        ),
        smoothstep(0.0, 1.0, f.z)
    );
}

// Rotation matrix for FBM
const ROT_MATRIX: mat3x3<f32> = mat3x3<f32>(
    0.00,  1.60,  1.20,
    -1.60,  0.72, -0.96,
    -1.20, -0.96,  1.28
);

// fbs: fbm slow
fn fbs(p: vec3<f32>) -> f32 {
    var pp = p;
    var f = 0.5000 * n3(pp); pp = ROT_MATRIX * pp * 1.2;
    f += 0.2500 * n3(pp); pp = ROT_MATRIX * pp * 1.3;
    f += 0.1666 * n3(pp); pp = ROT_MATRIX * pp * 1.4;
    f += 0.0834 * n3(pp);
    return f;
}

// fb: fbm fast
fn fb(p: vec3<f32>) -> f32 {
    var pp = p;
    var f = 0.0;
    var a = 1.0;
    var s = 0.0;

    f += a * n3(pp); pp = ROT_MATRIX * pp * 1.149; s += a; a *= 0.75;
    f += a * n3(pp); pp = ROT_MATRIX * pp * 1.41; s += a; a *= 0.75;
    f += a * n3(pp); pp = ROT_MATRIX * pp * 1.51; s += a; a *= 0.65;
    f += a * n3(pp); pp = ROT_MATRIX * pp * 1.21; s += a; a *= 0.35;
    f += a * n3(pp); pp = ROT_MATRIX * pp * 1.41; s += a; a *= 0.75;
    f += a * n3(pp);

    return f / s;
}

// v3: voronoi 3D
fn v3(x: vec3<f32>) -> vec3<f32> {
    let p = floor(x);
    let f = fract(x);

    var id = 0.0;
    var res = vec2<f32>(100.0);

    for(var k: i32 = -1; k <= 1; k++) {
        for(var j: i32 = -1; j <= 1; j++) {
            for(var i: i32 = -1; i <= 1; i++) {
                let b = vec3<f32>(f32(i), f32(j), f32(k));
                let r = b - f + h3(p + b);
                let d = dot(r, r);

                let cond = max(sign(res.x - d), 0.0);
                let nCond = 1.0 - cond;

                let cond2 = nCond * max(sign(res.y - d), 0.0);
                let nCond2 = 1.0 - cond2;

                id = (dot(p + b, vec3<f32>(1.0, 57.0, 113.0)) * cond) + (id * nCond);
                res = vec2<f32>(d, res.x) * cond + res * nCond;

                res.y = cond2 * d + nCond2 * res.y;
            }
        }
    }

    return vec3<f32>(sqrt(res), abs(id));
}

// gs: create stars
fn gs(fragCoord: vec2<f32>) -> f32 {
    let R = vec2<f32>(textureDimensions(output));
    let dimensions = R;
    let gridSize = 1.0;
    let grid = floor(fragCoord / (dimensions / gridSize));

    let seed = grid.x + grid.y * gridSize;
    let starPos = fract(vec2<f32>(h(seed), h(seed + 1.0))) * (dimensions / gridSize);

    let dist = distance(fragCoord, starPos);
    let starRadius = 1.5;
    let bloom = 1.0 - smoothstep(starRadius, starRadius + 1.0, dist);

    return bloom * 1.5;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let R = vec2<f32>(textureDimensions(output));
    let coords = vec2<u32>(global_id.xy);
    
    if (coords.x >= u32(R.x) || coords.y >= u32(R.y)) {
        return;
    }
    
    let FragCoord = vec2<f32>(f32(coords.x), R.y - f32(coords.y));
    let dim = R;
    var uv = (FragCoord.xy - 0.5 * dim) / dim.y;

    let dfc = length(uv) * params.center_scale;
    let tf = params.time_scale * u_time.time;
    let at = tf + (5.0 + sin(tf)) * 0.1 / (dfc + params.dist_offset);

    let st = sin(at);
    let ct = cos(at);

    uv *= mat2x2<f32>(ct, st, -st, ct);

    let rotAngle = u_time.time * 0.05;
    uv *= mat2x2<f32>(cos(rotAngle), -sin(rotAngle),
                      sin(rotAngle), cos(rotAngle)) * 25.662;

    var bc = 0.0;
    var c1 = 0.0;
    var c2 = 0.0;
    var c3 = 0.0;
    var point: vec3<f32>;

    for(var i = 0; i < params.max_iterations; i++) {
        point = 0.09 * f32(i) * vec3<f32>(uv, 1.0);
        point += vec3<f32>(0.1, 0.01, -3.5 - sin(tf * 0.1) * 0.01);
        point = point * 0.7;
        point = f32(i) * 0.1 + vec3<f32>(0.0, 0.0, tf * 0.2) + point;
        
        for(var j = 0; j < params.max_sub_iterations; j++) {
            point = abs(point) / dot(point, point) - 0.44;
        }

        let pi = dot(point, point) * params.point_intensity;
        c1 += pi * (3.8 + sin(dfc * 1.0 + 3.5 - tf * 2.0));
        c2 += pi * (1.5 + sin(dfc * 13.5 + 12.2 - tf * 3.0));
        c3 += pi * (2.4 + sin(dfc * 14.5 + 1.5 - tf * 2.5));
    }

    let vp = v3(point + vec3<f32>(u_time.time * 0.2));
    let vi = vp.x * 3.5;

    c1 += vi * 0.5;
    c2 += vi * 0.5;
    c3 += vi * 0.4;
    var cf = vec3<f32>(sin(vp.x), sin(vp.y), sin(vp.z + u_time.time * 0.5));
    cf *= vec3<f32>(1.0, 0.5, 0.4);
    cf = abs(cf);
    c1 *= cf.r;
    c2 *= cf.g;
    c3 *= cf.b;

    bc = .1 * length(point.xy) * 0.12;
    c1 *= 0.2;
    c2 *= 0.5;
    c3 = smoothstep(0.1, 0.0, dfc) * 0.3;
    

    let dir = normalize(vec3<f32>(uv, 0.0));
    let sundot = dot(LIGHT_DIR, dir);

    var fc = vec3<f32>(bc, (c1 + bc) * 0.4, c2);
    fc += c3 * 1.9;
    fc.g += c3 * 0.45;

    let star = gs(FragCoord.xy);
    let sb = smoothstep(0.0, 1.0, star) * 0.8;
    fc += vec3<f32>(sb);

    fc.r += c1 * 0.5;
    fc.g += c2 * 0.6;
    fc.b += c3 * 0.4;

    fc += vec3<f32>(1.0, 0.1, 0.5) * smoothstep(0.2, 0.0, dfc) * 0.5;
    fc = clamp(fc, vec3<f32>(0.0), vec3<f32>(1.0));

    let xy2 = FragCoord.xy / dim;
    let vignette = 0.5 + 0.25 * pow(100.0 * xy2.x * xy2.y * (1.0 - xy2.x) * (1.0 - xy2.y), 0.5);
    fc *= vec3<f32>(vignette);
    fc = gamma(fc, 0.41);

    textureStore(output, vec2<i32>(i32(coords.x), i32(coords.y)), vec4<f32>(fc, 1.0));
}