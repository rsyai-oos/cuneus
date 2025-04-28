struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dimensions = textureDimensions(output);
    if (global_id.x >= dimensions.x || global_id.y >= dimensions.y) {
        return;
    }
    // Calculate normalized pixel coordinates (0.0 to 1.0)
    let uv = vec2<f32>(
        f32(global_id.x) / f32(dimensions.x),
        f32(global_id.y) / f32(dimensions.y)
    );
    let col = 0.5 + 0.5 * cos(
        u_time.time + 
        uv.xyx * 1.0 + 
        vec3<f32>(0.0, 2.0, 4.0)
    );
    textureStore(output, global_id.xy, vec4<f32>(col, 1.0));
}