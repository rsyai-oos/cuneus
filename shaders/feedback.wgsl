//test shader for testing if feedback work
@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;

struct TimeUniform {
    time: f32,
};
@group(1) @binding(0)
var<uniform> u_time: TimeUniform;

struct Params {
    feedback: f32,
    speed: f32,
    scale: f32,
};
@group(2) @binding(0)
var<uniform> params: Params;

@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    // Get previous frame
    let previous_color = textureSample(prev_frame, tex_sampler, tex_coords);
    
    // Calculate moving pixel position
    let pos = vec2<f32>(0.5) + 0.4 * vec2<f32>(
        sin(u_time.time * params.speed),
        cos(u_time.time * params.speed * 0.5)
    );
    
    // Calculate distance to the moving pixel
    let dist = length(tex_coords - pos);
    
    // Create the current frame's pixel
    let pixel_color = vec4<f32>(1.0, 1.0, 1.0, 1.0) * smoothstep(0.02, 0.0, dist);
    
    // Mix previous frame with current frame
    let mixed = mix(previous_color, pixel_color, 0.15);
    
    // Apply decay to create the trail effect
    return mixed * 0.995;
}

@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    // Simply output the result from the first pass
    return textureSample(prev_frame, tex_sampler, tex_coords);
}