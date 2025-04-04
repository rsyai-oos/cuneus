struct TimeUniform {
    time: f32,
};

struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};

@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: Params;

struct Params {
    square_size: f32,
    circle_radius: f32,
    edge_thickness: f32,
    animation_speed: f32,
    
    background_color: vec3<f32>,
    edge_color_intensity: f32,
    
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

const PI: f32 = 3.14159265358979323846;
fn luminance_modulation(t: f32, phase: f32) -> f32 {
    return 0.5 + 0.3 * sin(t + phase);
}
@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    var uv = (FragCoord.xy - 0.5 * u_resolution.dimensions) / u_resolution.dimensions.y;
    var fragColor = vec4<f32>(params.background_color, 1.0);
    
    let t = u_time.time * params.animation_speed;
    let S = params.square_size;
    let R = params.circle_radius;
    let E = params.edge_thickness;
    let P = PI * 0.5;
    let centers = array<vec2<f32>, 4>(
        vec2<f32>(S, S),
        vec2<f32>(-S, S),
        vec2<f32>(-S, -S),
        vec2<f32>(S, -S)
    );
    
    for(var i: i32 = 0; i < 4; i = i + 1) {
        let p = uv - centers[i];
        
        if(length(p) < R) {
            let mx = sign(-centers[i].x);
            let my = sign(-centers[i].y);
            let m = step(0.0, mx * p.x) * step(0.0, my * p.y);
            
            if(m < 0.5) {
                let color = vec3<f32>(luminance_modulation(t, 0.0));
                fragColor = vec4<f32>(color, fragColor.a);
            } else {
                let h = step(abs(p.y), E);
                let v = step(abs(p.x), E);
                
                if(h > 0.5 && mx * p.x > 0.0) {
                    // Horizontal edges (up/down)
                    let hp = P * (1.0 - 2.0 * f32(i & 1));
                    let color = vec3<f32>(luminance_modulation(t, hp) * params.edge_color_intensity);
                    fragColor = vec4<f32>(color, fragColor.a);
                } else if(v > 0.5 && my * p.y > 0.0) {
                    // Vertical edges (left/right)
                    let vp = P * (f32(i & 1) * 2.0 - 1.0);
                    let color = vec3<f32>(luminance_modulation(t, vp) * params.edge_color_intensity);
                    fragColor = vec4<f32>(color, fragColor.a);
                }
            }
        }
    }
    
    return fragColor;
}