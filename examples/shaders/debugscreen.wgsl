// @group(0): Per-Frame Resources (TimeUniform)
// @group(1): Primary Pass I/O & Parameters (output texture)
// @group(2): Global Engine Resources (mouse, fonts, audio)
// @group(3): User-Defined Data Buffers (not used in this example)
// note that, WebGPU only supports a maximum of 4 groups (0-3). But you can use more bindings :) 

// Group 0: Per-Frame Resources
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

// Group 1: Primary I/O - Output texture only for this simple shader
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

// Group 2: Global Engine Resources - Mouse, Fonts, and Audio
struct MouseUniform {           
    position: vec2<f32>,         // Normalized position (0.0 to 1.0)
    click_position: vec2<f32>,   // Position of last click
    wheel: vec2<f32>,            // Accumulated wheel delta
    buttons: vec2<u32>,          // Button state bitfield
}
@group(2) @binding(0) var<uniform> u_mouse: MouseUniform;

struct FontUniforms {
    atlas_size: vec2<f32>,
    char_size: vec2<f32>,
    screen_size: vec2<f32>,
    _padding: vec2<f32>,
}
@group(2) @binding(1) var<uniform> u_font: FontUniforms;
@group(2) @binding(2) var t_font_atlas: texture_2d<f32>;
@group(2) @binding(3) var s_font_atlas: sampler;
@group(2) @binding(4) var<storage, read_write> audio_buffer: array<f32>;

// Group 3: User-Defined Data Buffers (not used in this simple example)

// Crisp SDF character rendering with minimal anti-aliasing
fn render_char_sdf(pos: vec2<f32>, char_pos: vec2<f32>, ascii: u32, size: f32) -> f32 {
    let char_size = vec2<f32>(size, size);
    let local_pos = pos - char_pos;
    
    // Check if we're inside the character bounds
    if (local_pos.x < 0.0 || local_pos.x >= char_size.x || 
        local_pos.y < 0.0 || local_pos.y >= char_size.y) {
        return 0.0;
    }
    // Direct sampling for crisp text
    return sample_sdf_at_position(local_pos, char_size, ascii, size);
}

// Core SDF sampling function
fn sample_sdf_at_position(local_pos: vec2<f32>, char_size: vec2<f32>, ascii: u32, size: f32) -> f32 {
    // Get character position in 16x16 grid
    let char_x = ascii % 16u;
    let char_y = ascii / 16u;
    
    // Convert to normalized UV within character (0.0 to 1.0)
    let uv_local = local_pos / char_size;
    
    // Atlas dimensions (have to match Rust code! see font.rs)
    let atlas_size = 1024.0;
    let cell_size = 64.0; // Each character cell is 64x64 pixels
    let padding = 4.0;    // Padding around each character
    
    // Calculate UV coordinates in the atlas with padding
    let effective_cell_size = cell_size - padding * 2.0;
    let cell_uv = (padding + uv_local * effective_cell_size) / atlas_size;
    
    // Offset by character position in grid
    let char_offset = vec2<f32>(f32(char_x), f32(char_y)) * cell_size / atlas_size;
    let final_uv = char_offset + cell_uv;
    
    // Sample the font atlas using UV coordinates
    // Bounds check
    if (final_uv.x < 0.0 || final_uv.x >= 1.0 || 
        final_uv.y < 0.0 || final_uv.y >= 1.0) {
        return 0.0;
    }
    
    // Convert UV to pixel coordinates for textureLoad
    let atlas_coord = vec2<i32>(i32(final_uv.x * atlas_size), i32(final_uv.y * atlas_size));
    let pixel = textureLoad(t_font_atlas, atlas_coord, 0);
    
    // Since we're using direct alpha texture (not SDF), just return alpha
    return pixel.a;
}

// Character rendering using modern SDF method
fn render_char(pos: vec2<f32>, char_pos: vec2<f32>, ch: u32) -> f32 {
    return render_char_sdf(pos, char_pos, ch, 32.0);
}

// Large character rendering using SDF method
fn render_large_char(pos: vec2<f32>, char_pos: vec2<f32>, ch: u32) -> f32 {
    return render_char_sdf(pos, char_pos, ch, 64.0);
}

// Helper function: Render character at any size with SDF scaling
fn render_char_sized(pos: vec2<f32>, char_pos: vec2<f32>, ch: u32, size: f32) -> f32 {
    return render_char_sdf(pos, char_pos, ch, size);
}

// Helper function: Render a digit (0-9)
fn render_digit(pos: vec2<f32>, char_pos: vec2<f32>, digit: u32, size: f32) -> f32 {
    return render_char_sdf(pos, char_pos, digit + 48u, size); // 48 = ASCII '0'
}

// Helper function: Render letter (A-Z, a-z)
fn render_letter(pos: vec2<f32>, char_pos: vec2<f32>, letter_code: u32, size: f32) -> f32 {
    return render_char_sdf(pos, char_pos, letter_code, size);
}

// text positioning
fn get_char_advance(size: f32) -> f32 {
    return size * 0.9; // Slightly tighter spacing than before
}

// Easy-to-use text rendering functions

// Render a single word at position with specified size
fn render_word(pixel_pos: vec2<f32>, word_pos: vec2<f32>, word: array<u32, 16>, word_length: u32, size: f32) -> f32 {
    let char_spacing = get_char_advance(size);
    var alpha = 0.0;
    
    for (var i = 0u; i < word_length && i < 16u; i++) {
        let char_alpha = render_char_sized(pixel_pos, word_pos + vec2<f32>(f32(i) * char_spacing, 0.0), word[i], size);
        alpha = max(alpha, char_alpha);
    }
    
    return alpha;
}

// Render a number (up to 8 digits) at position
fn render_number(pixel_pos: vec2<f32>, num_pos: vec2<f32>, number: u32, size: f32) -> f32 {
    let char_spacing = get_char_advance(size);
    var alpha = 0.0;
    var temp_num = number;
    var digit_count = 0u;
    
    // Count digits
    if (temp_num == 0u) {
        digit_count = 1u;
    } else {
        var count_temp = temp_num;
        while (count_temp > 0u) {
            count_temp = count_temp / 10u;
            digit_count++;
        }
    }
    
    // Render digits from right to left
    temp_num = number;
    for (var i = 0u; i < digit_count; i++) {
        let digit = temp_num % 10u;
        let digit_pos = vec2<f32>(f32(digit_count - 1u - i) * char_spacing, 0.0);
        let char_alpha = render_digit(pixel_pos, num_pos + digit_pos, digit, size);
        alpha = max(alpha, char_alpha);
        temp_num = temp_num / 10u;
    }
    
    return alpha;
}

// Render floating point number with one decimal place
fn render_float_1dp(pixel_pos: vec2<f32>, num_pos: vec2<f32>, value: f32, size: f32) -> f32 {
    let char_spacing = get_char_advance(size);
    var alpha = 0.0;
    
    let whole_part = u32(value);
    let decimal_part = u32((value - f32(whole_part)) * 10.0);
    
    // Render whole part
    alpha = max(alpha, render_number(pixel_pos, num_pos, whole_part, size));
    
    // Calculate width of whole part
    var digit_count = 0u;
    if (whole_part == 0u) {
        digit_count = 1u;
    } else {
        var temp = whole_part;
        while (temp > 0u) {
            temp = temp / 10u;
            digit_count++;
        }
    }
    
    // Render decimal point
    let dot_pos = num_pos + vec2<f32>(f32(digit_count) * char_spacing, 0.0);
    alpha = max(alpha, render_char_sized(pixel_pos, dot_pos, 46u, size)); // ASCII '.'
    
    // Render decimal digit
    let decimal_pos = dot_pos + vec2<f32>(char_spacing, 0.0);
    alpha = max(alpha, render_digit(pixel_pos, decimal_pos, decimal_part, size));
    
    return alpha;
}

// "Hello Cuneus" :)
fn render_hello_cuneus(pixel_pos: vec2<f32>, screen_center: vec2<f32>) -> vec3<f32> {

    let base_size = 100.0;
    let size_pulse = sin(u_time.time * 2.0) * 20.0;
    let size = base_size + size_pulse;
    
    let char_spacing = get_char_advance(size);
    var result = vec3<f32>(0.0);
    
    // Calculate text width to center it
    let text_width = 12.0 * char_spacing;
    let start_x = screen_center.x - text_width * 0.5;
    let start_y = screen_center.y - size * 0.5;
    
    // Wave animation offset for each character
    let wave_amplitude = 30.0;
    let wave_frequency = 0.5;
    
    // Character data for "Hello Cuneus"
    let chars = array<u32, 12>(72u, 101u, 108u, 108u, 111u, 32u, 67u, 117u, 110u, 101u, 117u, 115u);
    //anim stuff
    for (var i = 0u; i < 12u; i++) {
        let wave_offset = sin(u_time.time * 3.0 + f32(i) * wave_frequency) * wave_amplitude;
        
        let hue = (u_time.time + f32(i) * 0.3) % 6.28318; // 2π
        let char_color = hsv_to_rgb(hue, 0.8, 1.0);
        
        let char_pos = vec2<f32>(
            start_x + f32(i) * char_spacing,
            start_y + wave_offset
        );
        let char_alpha = render_char_sized(pixel_pos, char_pos, chars[i], size);
        // Add to result with color
        result += char_color * char_alpha;
    }
    
    return result;
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let x = c * (1.0 - abs((h / 1.047197551) % 2.0 - 1.0)); 
    let m = v - c;
    
    var rgb = vec3<f32>(0.0);
    
    if (h < 1.047197551) {       
        rgb = vec3<f32>(c, x, 0.0);
    } else if (h < 2.094395102) { 
        rgb = vec3<f32>(x, c, 0.0);
    } else if (h < 3.141592654) {
        rgb = vec3<f32>(0.0, c, x);
    } else if (h < 4.188790205) { 
        rgb = vec3<f32>(0.0, x, c);
    } else if (h < 5.235987756) { 
        rgb = vec3<f32>(x, 0.0, c);
    } else {                     
        rgb = vec3<f32>(c, 0.0, x);
    }
    
    return rgb + vec3<f32>(m);
}

// color debugging system
fn render_color_debug(pixel_pos: vec2<f32>, debug_pos: vec2<f32>, current_color: vec3<f32>) -> f32 {
    let size = 24.0;
    let char_spacing = get_char_advance(size);
    var alpha = 0.0;
    
    // "RGB:" label
    var rgb_label = array<u32, 16>(82u, 71u, 66u, 58u, 32u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    alpha = max(alpha, render_word(pixel_pos, debug_pos, rgb_label, 5u, size));
    

    let r_val = u32(current_color.r * 255.0);
    let g_val = u32(current_color.g * 255.0);
    let b_val = u32(current_color.b * 255.0);
    
    let r_pos = debug_pos + vec2<f32>(5.0 * char_spacing, 0.0);
    alpha = max(alpha, render_number(pixel_pos, r_pos, r_val, size));
    
    // Comma separator
    let comma1_pos = r_pos + vec2<f32>(3.0 * char_spacing, 0.0);
    alpha = max(alpha, render_char_sized(pixel_pos, comma1_pos, 44u, size));
    
    // G
    let g_pos = comma1_pos + vec2<f32>(char_spacing, 0.0);
    alpha = max(alpha, render_number(pixel_pos, g_pos, g_val, size));
    
    // Comma separator
    let comma2_pos = g_pos + vec2<f32>(3.0 * char_spacing, 0.0);
    alpha = max(alpha, render_char_sized(pixel_pos, comma2_pos, 44u, size));
    
    // B
    let b_pos = comma2_pos + vec2<f32>(char_spacing, 0.0);
    alpha = max(alpha, render_number(pixel_pos, b_pos, b_val, size));
    
    return alpha;
}

// Mouse position color sampling debug
fn render_mouse_color_debug(pixel_pos: vec2<f32>, debug_pos: vec2<f32>, sampled_color: vec3<f32>) -> f32 {
    let size = 20.0;
    let char_spacing = get_char_advance(size);
    var alpha = 0.0;
    
    // "Mouse RGB:"
    var mouse_label = array<u32, 16>(77u, 111u, 117u, 115u, 101u, 58u, 32u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    alpha = max(alpha, render_word(pixel_pos, debug_pos, mouse_label, 7u, size));
    
    // Display color values
    let color_pos = debug_pos + vec2<f32>(7.0 * char_spacing, 0.0);
    alpha = max(alpha, render_color_debug(pixel_pos, color_pos, sampled_color));
    
    return alpha;
}


fn render_text(pixel_pos: vec2<f32>, text_pos: vec2<f32>) -> f32 {
    let size = 32.0;
    
    // "Debug" using the word helper - need to pad to array size 16
    var debug_word = array<u32, 16>(68u, 101u, 98u, 117u, 103u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    return render_word(pixel_pos, text_pos, debug_word, 5u, size);
}

// Render time display using new helper functions
fn render_time_display(pixel_pos: vec2<f32>, text_pos: vec2<f32>) -> f32 {
    let size = 32.0;
    let char_spacing = get_char_advance(size);
    var alpha = 0.0;
    
    // "Time: " using individual characters
    var time_label = array<u32, 16>(84u, 105u, 109u, 101u, 58u, 32u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u); // "Time: "
    alpha = max(alpha, render_word(pixel_pos, text_pos, time_label, 6u, size));
    
    // Display time with decimal point using the float helper
    let time_number_pos = text_pos + vec2<f32>(6.0 * char_spacing, 0.0);
    alpha = max(alpha, render_float_1dp(pixel_pos, time_number_pos, u_time.time, size));
    
    return alpha;
}

// fps. since its same like above I will not use comments
fn render_fps_display(pixel_pos: vec2<f32>, text_pos: vec2<f32>) -> f32 {
    let size = 32.0;
    let char_spacing = get_char_advance(size);
    var alpha = 0.0;
    // "FPS: " 
    var fps_label = array<u32, 16>(70u, 80u, 83u, 58u, 32u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    alpha = max(alpha, render_word(pixel_pos, text_pos, fps_label, 5u, size));
    let fps = u32(1.0 / max(u_time.delta, 0.001));
    let fps_number_pos = text_pos + vec2<f32>(5.0 * char_spacing, 0.0);
    alpha = max(alpha, render_number(pixel_pos, fps_number_pos, fps, size));
    return alpha;
}

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
    
    // Add text rendering
    let pixel_pos = vec2<f32>(f32(global_id.x), f32(global_id.y));
    let screen_center = vec2<f32>(f32(dimensions.x) * 0.5, f32(dimensions.y) * 0.5);
    
    // Render "Hello Cuneus" center
    let hello_color = render_hello_cuneus(pixel_pos, screen_center);
    final_col += hello_color; // Directly add the colored text
    
    // debug info at top-left
    let debug_alpha = render_text(pixel_pos, vec2<f32>(20.0, 20.0));
    if (debug_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(1.0, 1.0, 1.0), debug_alpha);
    }
    
    // time 
    let time_alpha = render_time_display(pixel_pos, vec2<f32>(20.0, 70.0));
    if (time_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(1.0, 1.0, 0.0), time_alpha);
    }
    
    //FPS 
    let fps_alpha = render_fps_display(pixel_pos, vec2<f32>(20.0, 120.0));
    if (fps_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(0.0, 1.0, 0.0), fps_alpha);
    }
    
    //show current pixel color
    let color_debug_alpha = render_color_debug(pixel_pos, vec2<f32>(20.0, 170.0), final_col);
    if (color_debug_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(0.8, 0.8, 0.8), color_debug_alpha);
    }
    
    // Mouse position color sampling
    let mouse_pixel_x = u32(u_mouse.position.x * f32(dimensions.x));
    let mouse_pixel_y = u32(u_mouse.position.y * f32(dimensions.y));
    
    // Sample color at mouse position
    let mouse_sampled_color = base_col; // For demo, using base color
    let mouse_debug_alpha = render_mouse_color_debug(pixel_pos, vec2<f32>(20.0, 220.0), mouse_sampled_color);
    if (mouse_debug_alpha > 0.01) {
        final_col = mix(final_col, vec3<f32>(0.7, 0.9, 1.0), mouse_debug_alpha);
    }
    
    // Simple audio note generation - write audio parameters to buffer for CPU synthesis
    // This generates a simple musical note when audio is enabled
    if (global_id.x == 0u && global_id.y == 0u) {
        // Generate a simple musical note (C4 = 261.63 Hz)
        let base_frequency = 261.63; // C4 note
        let note_frequency = base_frequency * (1.0 + sin(u_time.time * 0.5) * 0.1); // Slight vibrato
        let note_amplitude = 0.2; // Moderate volume
        let waveform_type = 0.0; // Sine wave
        
        // Write to audio buffer for CPU synthesis
        audio_buffer[0] = note_frequency;
        audio_buffer[1] = note_amplitude;
        audio_buffer[2] = waveform_type;
        // in screen, we are playing first note.
        let base_frequencies = array<f32, 9>(
            261.63, 293.66, 329.63, 349.23, 392.00, 
            440.00, 493.88, 523.25, 587.33
        );
        
        for (var i = 0u; i < 9u; i++) {
            audio_buffer[3 + i] = base_frequencies[i];
        }
    }
    
    textureStore(output, global_id.xy, vec4<f32>(final_col, 1.0));
}