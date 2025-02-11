@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(1) @binding(0) var<uniform> u_time: TimeUniform;
@group(2) @binding(0) var<uniform> params: Params;
@group(3) @binding(0) var<uniform> u_resolution: ResolutionUniform;

struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};

struct TimeUniform {
    time: f32,
};

struct Params {
    scale: f32, 
    offset_value: f32,
    cell_index: f32,
    edge_width: f32,
    highlight: f32,
};
//functions written by, FabriceNeyret, 2025: https://www.shadertoy.com/view/3flGD7
//L/S functions, also for corner calculations belongs to him, you can find them in the same shader. I just adapted them to WGSL to use with Texture.
fn L(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 { 
    var p_local = p - a;
    var b_local = b - a;
    return length(p_local - b_local * clamp(dot(p_local, b_local) / dot(b_local, b_local), 0.0, 1.0));
}

fn S(P: vec2<f32>) -> vec2<f32> {
    var R = vec2<f32>(1.0, 87.0);
    return P + 0.5 * fract(1e4 * sin((P) * mat2x2<f32>(R.x, -R.x, R.y, -R.y))) 
         + 0.25 + 0.25 * cos(u_time.time + 6.3 * fract(1e4 * sin(dot(P, R - 37.0))) + vec2<f32>(0.0, 11.0));
}
fn lumi(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let R = u_resolution.dimensions;
    let uv = vec2<f32>(FragCoord.x, R.y - FragCoord.y) / R;
    
    let U = params.scale * FragCoord.xy / R.y;
    var P: vec2<f32>;
    
    var d: f32;
    var l = vec4<f32>(9.0);
    var A: array<vec2<f32>, 4>;
    
    for(var k = 0; k < 9; k++) {
        P = S(floor(U) + vec2<f32>(f32(k % 3), f32(k / 3)) + params.offset_value);
        d = length(P - U);
        
        if(d < l.x) {
            l = vec4<f32>(d, l.x, l.y, l.z);
            A[3] = A[2];
            A[2] = A[1];
            A[1] = A[0];
            A[0] = P;
        } else if(d < l.y) {
            l = vec4<f32>(l.x, d, l.y, l.z);
            A[3] = A[2];
            A[2] = A[1];
            A[1] = P;
        } else if(d < l.z) {
            l = vec4<f32>(l.x, l.y, d, l.z);
            A[3] = A[2];
            A[2] = P;
        } else if(d < l.w) {
            l.w = d;
            A[3] = P;
        }
    }
    
    var cUV = (A[i32(params.cell_index)] * R.y/params.scale) / R;
    cUV = vec2<f32>(cUV.x, 1.0 - cUV.y);
    let cCol = textureSample(tex, tex_sampler, vec2<f32>(cUV.x, 1.0 - cUV.y));
    
    P = A[1] - A[0];
    d = length(P)/2.0 - dot(U-A[0], P)/length(P);
    P = A[2] - A[0];
    d = min(d, length(P)/2.0 - dot(U-A[0], P)/length(P));
    P = A[3] - A[0];
    d = min(d, length(P)/2.0 - dot(U-A[0], P)/length(P));
    
    let eF = smoothstep(-params.edge_width, params.edge_width, d);
    
    let a = dot(A[0], A[0]);
    let b = dot(A[1], A[1]);
    let c = dot(A[2], A[2]);
    let mat = mat2x2<f32>(
        A[2].x - A[1].x, A[2].y - A[1].y,
        A[2].x - A[0].x, A[2].y - A[0].y
    );
    P = vec2<f32>(c-b, c-a) / 2.0 * mat;
    
    let lE = smoothstep(15.0/R.y, 0.0, L(U, A[0], P));
    
    let eCol = vec4<f32>(0.0);
    var vCol = mix(cCol, eCol, smoothstep(0.0, 0.08, 1.0 - eF));
    
    let eH = smoothstep(0.1, 0.0, abs(d)) * params.highlight;
    vCol += cCol * eH;
    
    let cP = smoothstep(15.0/R.y, 0.0, length(P - U) - 0.02);
    let cI = 0.2;
    vCol = mix(vCol, cCol * (0.0 + cI), cP);
    
    return clamp(vCol, vec4<f32>(0.0), vec4<f32>(1.0));
}