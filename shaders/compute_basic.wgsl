struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
 //simple usage: On this example I show how can we use them on debug scree
 //note that, WebGPU only supports a maximum of 4 bind groups (0-3). 
//When adding this, you can also pass this to the “parameters” struct in other shaders. 
struct MouseUniform {           
    position: vec2<f32>,         // Normalized position (0.0 to 1.0)
    click_position: vec2<f32>,   // Position of last click
    wheel: vec2<f32>,            // Accumulated wheel delta
    buttons: vec2<u32>,          // Button state bitfield
};
@group(2) @binding(0) var<uniform> u_mouse: MouseUniform;

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
    
    let mouse_dist = distance(uv, u_mouse.position);
    
    let base_col = 0.5 + 0.5 * cos(
        u_time.time + 
        uv.xyx * 1.0 + 
        vec3<f32>(0.0, 2.0, 4.0)
    );
    
    let circle_radius = 0.1 + sin(u_time.time) * 0.05;
    let circle_effect = smoothstep(circle_radius, circle_radius - 0.02, mouse_dist);
    
    let circle_col = vec3<f32>(
        0.5 + 0.5 * sin(u_time.time * 1.1),
        0.5 + 0.5 * sin(u_time.time * 0.7),
        0.5 + 0.5 * sin(u_time.time * 0.5)
    );
    
    let left_button_pressed = (u_mouse.buttons.x & 1u) != 0u;
    var final_col = mix(base_col, circle_col, circle_effect);
    if (left_button_pressed && mouse_dist < circle_radius) {
        final_col = vec3<f32>(1.0) - final_col;
    }
    let wheel_effect = abs(u_mouse.wheel.y) * 0.2;
    let pulse = sin(u_time.time * 5.0 + mouse_dist * 20.0) * wheel_effect;
    if (wheel_effect > 0.01) {
        final_col = final_col * (1.0 + pulse);
    }
    textureStore(output, global_id.xy, vec4<f32>(final_col, 1.0));
}