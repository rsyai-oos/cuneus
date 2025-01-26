// MIT License, Enes Altun, 2025
// resources for Skilling's algorithm
//  https://github.com/joshspeagle/dynesty
// https://doi.org/10.1063/1.1751381 and https://doi.org/10.1063/1.1751382
// http://www.inference.org.uk/bayesys/test/hilbert.c
// https://www.shadertoy.com/view/3tl3zl
struct TimeUniform {
    time: f32,
};

struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};

struct Params {
    iterations: u32,
    num_rays: u32,
    _pad1: vec2<f32>,      
    scale: f32,             
    time_scale: f32,        
    vignette_radius: f32,   
    vignette_softness: f32, 
    color_offset: vec3<f32>,
    _pad2: f32,            
};

@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: Params;

const PI: f32 = 3.14159265359;
const FLT_MAX: f32 = 33333.0;
//adapted from: https://www.shadertoy.com/view/3tl3zl tpfto, 2019.
fn hilbert(k: u32, s: u32) -> vec2<f32> {
    let bb = 1u << s;
    var b = bb;
    let t = vec2<u32>(k ^ (k >> 1));
    var hp = vec2<u32>(0u);
    
    for(var j: i32 = i32(s) - 1; j >= 0; j--) {
        b = b >> 1u;
        hp += (t >> vec2<u32>(u32(j + 1), u32(j))) & vec2<u32>(b);
    }
    for(var p = 2u; p < bb; p = p << 1u) {
        let q = p - 1u;
        if((hp.y & p) != 0u) {
            hp.x = hp.x ^ q;
        } else {
            let temp = (hp.x ^ hp.y) & q;
            hp.x = hp.x ^ temp;
            hp.y = hp.y ^ temp;
        }
        if((hp.x & p) != 0u) {
            hp.x = hp.x ^ q;
        }
    }
    
    return 2.0 * (vec2<f32>(hp) / f32(bb - 1u)) - 1.0;
}
fn intersect_line(ro: vec2<f32>, rd: vec2<f32>, line: vec4<f32>, t: ptr<function, f32>) -> bool {
    let A = ro;
    let B = ro + rd;
    let C = line.xy;
    let D = line.zw;
    
    let AmC = A - C;
    let DmC = D - C;
    let BmA = B - A;
    
    let denom = (BmA.x * DmC.y) - (BmA.y * DmC.x);
    
    if(abs(denom) > 0.0001) {
        let r = ((AmC.y * DmC.x) - (AmC.x * DmC.y)) / denom;
        let s = ((AmC.y * BmA.x) - (AmC.x * BmA.y)) / denom;
        
        if((r > 0.0 && r < *t) && (s > 0.0 && s < 1.0)) {
            *t = r;
            return true;
        }
    }
    return false;
}

fn intersect_scene(ro: vec2<f32>, rd: vec2<f32>, t: ptr<function, f32>, colour: ptr<function, vec3<f32>>) -> bool {
    var intersect = false;
    var minDist = *t;

    let s = params.iterations;
    let NUM = (1u << (2u * s)) - 1u;
    
    var a = hilbert(0u, s);
    var b: vec2<f32>;
    
    let scale = params.scale;
    let offset = vec2<f32>(0.0);
    let scaletime = u_time.time * params.time_scale;
    for(var i = 0u; i < NUM; i++) { 
        b = hilbert(i + 1u, s);
        let line = vec4<f32>((a * scale) + offset, (b * scale) + offset);
        let lineDir = normalize(line.zw - line.xy);
        let normal = vec2<f32>(-lineDir.y, lineDir.x);
        if(intersect_line(ro, rd, line, &minDist)) {
            let hue = f32(i) / f32(NUM) + scaletime;
            *colour = 0.5 + 0.5 * cos(3.28318 * (hue + params.color_offset));
            let lighting = abs(dot(-rd, normal));
            *colour = *colour * lighting;
            intersect = true;
        }
        
        a = b;
    }
    
    *t = minDist;
    return intersect;
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
let ro = -1.0 + 2.0 * (vec2<f32>(FragCoord.x, u_resolution.dimensions.y - FragCoord.y) / u_resolution.dimensions);
    
    var total = vec3<f32>(0.0);
    
    for(var i = 0u; i < params.num_rays; i++) {
        let angle = PI * (f32(i) / f32(params.num_rays));
        let rd = vec2<f32>(cos(angle), sin(angle));
        
        var t = FLT_MAX;
        var colour = vec3<f32>(1.0);
        
        if(intersect_scene(ro, rd, &t, &colour)) {
            total += colour;
        }
    }
    
   total = total / f32(params.num_rays);
    
    let dist = length(ro); 
    let radius = params.vignette_radius;
    let softness = params.vignette_softness;
    let vignette = smoothstep(radius, radius + softness, dist);
    total *= 1.0 - vignette * 0.95;
    
    let exposure = 1.5;
    return vec4<f32>(pow(total * exposure, vec3<f32>(1.0/1.1)), 1.0);
}