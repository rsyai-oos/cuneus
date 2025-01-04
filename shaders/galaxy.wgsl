//inspiration from Dave Hoskins' famous galaxy shader: https://www.shadertoy.com/view/MdXSzS
struct TimeUniform {
    time: f32,
};

struct Params {
    max_iterations: i32,
    max_sub_iterations: i32,
    point_intensity: f32,
    center_scale: f32,
    time_scale: f32,
    dist_offset: f32,
    _pad1: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> params: Params;
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
    let dimensions = vec2<f32>(1920.0, 1080.0);
    let gridSize = 1.0;
    let grid = floor(fragCoord / (dimensions / gridSize));
    
    let seed = grid.x + grid.y * gridSize;
    let starPos = fract(vec2<f32>(h(seed), h(seed + 1.0))) * (dimensions / gridSize);
    
    let dist = distance(fragCoord, starPos);
    let starRadius = 1.5;
    let bloom = 1.0 - smoothstep(starRadius, starRadius + 1.0, dist);
    
    return bloom * 1.5;
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(1920.0, 1080.0);
    var uv = (FragCoord.xy - 0.5 * dimensions) / dimensions.y;
    
    let distFromCenter = length(uv) * params.center_scale;
    let timeFactor = params.time_scale * u_time.time;
    let adjTime = timeFactor + (5.0 + sin(timeFactor)) * 0.1 / (distFromCenter + params.dist_offset);
    
    let st = sin(adjTime);
    let ct = cos(adjTime);
    
    uv *= mat2x2<f32>(ct, st, -st, ct);
    
    let rotAngle = u_time.time * 0.05;
    uv *= mat2x2<f32>(cos(rotAngle), -sin(rotAngle), 
                      sin(rotAngle), cos(rotAngle)) * 25.662;
    
    var baseColor = 0.0;
    var color1 = 0.0;
    var color2 = 0.0;
    var color3 = 0.0;
    var point: vec3<f32>;
    
    for(var i = 0; i < params.max_iterations; i++) {
        point = 0.09 * f32(i) * vec3<f32>(uv, 1.0);
        point += vec3<f32>(0.1, 0.01, -3.5 - sin(timeFactor * 0.1) * 0.01);
        
        for(var j = 0; j < params.max_sub_iterations; j++) {
            point = abs(point) / dot(point, point) - 0.52;
        }
        
        let pointIntensity = dot(point, point) * params.point_intensity;
        color1 += pointIntensity * (3.8 + sin(distFromCenter * 13.0 + 3.5 - timeFactor * 2.0));
        color2 += pointIntensity * (1.5 + sin(distFromCenter * 13.5 + 2.2 - timeFactor * 3.0));
        color3 += pointIntensity * (2.4 + sin(distFromCenter * 14.5 + 1.5 - timeFactor * 2.5));
    }
    
    let vPoint = v3(point + vec3<f32>(u_time.time * 0.2));
    let vIntensity = vPoint.x * 1.5;
    
    color1 += vIntensity * 2.2;
    color2 += vIntensity * 0.8;
    color3 += vIntensity * 1.8;
    
    let colorFlow = vec3<f32>(sin(vPoint.x), sin(vPoint.y), sin(vPoint.z + u_time.time * 0.5));
    color1 *= colorFlow.r;
    color2 *= colorFlow.g;
    color3 *= colorFlow.b;
    
    baseColor = 3.1 * length(point.xy) * 0.12;
    color1 *= 0.5;
    color2 *= 0.5;
    color3 = smoothstep(0.1, 0.0, distFromCenter) * 0.3;
    
    let direction = normalize(vec3<f32>(uv, 0.0));
    let sundot = dot(LIGHT_DIR, direction);
    
    var finalColor = vec3<f32>(baseColor, (color1 + baseColor) * 0.25, color2);
    finalColor += color3 * 1.9;
    finalColor.g += color3 * 0.45;
    
    let star = gs(FragCoord.xy);
    let starBloom = smoothstep(0.0, 1.0, star) * 0.8;
    finalColor += vec3<f32>(starBloom);
    
    finalColor.r += color1 * 0.5;
    finalColor.g += color2 * 0.6;
    finalColor.b += color3 * 0.4;
    
    finalColor += vec3<f32>(1.0, 0.1, 0.5) * smoothstep(0.2, 0.0, distFromCenter) * 0.5;
    finalColor = clamp(finalColor, vec3<f32>(0.0), vec3<f32>(1.0));
    
    let xy2 = FragCoord.xy / dimensions;
    let vignette = 0.5 + 0.25 * pow(100.0 * xy2.x * xy2.y * (1.0 - xy2.x) * (1.0 - xy2.y), 0.5);
    finalColor *= vec3<f32>(vignette);
    finalColor = gamma(finalColor, 0.41);
    
    return vec4<f32>(finalColor, 1.0);
}