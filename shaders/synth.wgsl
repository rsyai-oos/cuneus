// This example demonstrates a how to generate audio using cunes via compute shaders
@group(0) @binding(0) var<uniform> u_time: ComputeTimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(2) @binding(0) var<uniform> params: SynthParams;
@group(3) @binding(0) var<storage, read_write> audio_buffer: array<f32>;

struct ComputeTimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}

struct SynthParams {
    tempo: f32,
    waveform_type: u32,
    octave: f32,
    volume: f32,
    beat_enabled: u32,
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    key_states: array<vec4<f32>, 3>,
    key_decay: array<vec4<f32>, 3>,
}

const PI = 3.14159265359;

fn get_background_beat(time: f32, tempo: f32) -> f32 {
    let beat_duration = 60.0 / tempo;
    let beat_time = fract(time / beat_duration);
    
    if beat_time < 0.1 {
        return 0.25 * exp(-beat_time * 10.0);
    }
    return 0.0;
}


fn generate_waveform(phase: f32, waveform_type: u32) -> f32 {
    switch waveform_type {
        case 0u: {
            return sin(phase);
        }
        case 1u: {
            return 2.0 * fract(phase / (2.0 * PI)) - 1.0;
        }
        case 2u: {
            return select(-1.0, 1.0, sin(phase) > 0.0);
        }
        default: {
            return sin(phase);
        }
    }
}

fn get_note_frequency(note_index: u32, octave: f32) -> f32 {
    let notes = array<f32, 9>(
        233.63, 122.66, 329.63, 349.23, 392.00,
        440.00, 466.16, 523.25, 587.33
    );
    return notes[note_index] * pow(2.0, octave - 4.0);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output);
    let coord = vec2<i32>(global_id.xy);
    
    if coord.x >= i32(dims.x) || coord.y >= i32(dims.y) {
        return;
    }
    
    let uv = vec2<f32>(f32(coord.x) / f32(dims.x), f32(coord.y) / f32(dims.y));
    
    var beat_sample = 0.0;
    var key_sample = 0.0;
    var dominant_freq = 440.0;
    var max_key_amp = 0.0;
    var active_keys = 0.0;
    
    let beat_amp = select(0.0, get_background_beat(u_time.time, params.tempo), params.beat_enabled > 0u);
    beat_sample = beat_amp;
    for (var i = 0u; i < 9u; i++) {
        let vec_idx = i / 4u;
        let comp_idx = i % 4u;
        
        var key_state = 0.0;
        var key_decay_val = 0.0;
        
        if comp_idx == 0u {
            key_state = params.key_states[vec_idx].x;
            key_decay_val = params.key_decay[vec_idx].x;
        } else if comp_idx == 1u {
            key_state = params.key_states[vec_idx].y;
            key_decay_val = params.key_decay[vec_idx].y;
        } else if comp_idx == 2u {
            key_state = params.key_states[vec_idx].z;
            key_decay_val = params.key_decay[vec_idx].z;
        } else {
            key_state = params.key_states[vec_idx].w;
            key_decay_val = params.key_decay[vec_idx].w;
        }
        
        if key_state > 0.5 {
            let freq = get_note_frequency(i, params.octave);
            let envelope = key_decay_val;
            let phase = u_time.time * freq * 2.0 * PI;
            let waveform_sample = generate_waveform(phase, params.waveform_type);
            let key_amp = envelope * 0.2;
            
            key_sample += waveform_sample * key_amp;
            active_keys += 1.0;
            
            if key_amp > max_key_amp {
                max_key_amp = key_amp;
                dominant_freq = freq;
            }
        }
    }
    
    // Normalize key sample if multiple keys are playing
    if active_keys > 1.0 {
        key_sample = key_sample / sqrt(active_keys);
    }
    
    // Mix beat and keys independently - beat should ALWAYS be audible
    var mixed_sample = beat_sample + key_sample * 0.6;
    
    // Apply volume control
    mixed_sample = mixed_sample * params.volume;
    
    // Gentle limiting to prevent harsh clipping but preserve both signals
    let limit = 0.95;
    if abs(mixed_sample) > limit {
        mixed_sample = sign(mixed_sample) * limit;
    }
    
    let final_amplitude = abs(mixed_sample);
    
    if global_id.x == 0u && global_id.y == 0u {
        audio_buffer[0] = dominant_freq;
        audio_buffer[1] = final_amplitude;
        audio_buffer[2] = f32(params.waveform_type);
    }
    
    var color = vec3<f32>(0.02, 0.02, 0.1) * (1.0 - uv.y * 0.3);
    
    let bar_top = 0.9;
    let bar_max_height = 0.6;
    let bar_width = 0.08;
    let bar_spacing = 0.02;
    let total_width = 9.0 * bar_width + 8.0 * bar_spacing;
    let start_x = (1.0 - total_width) * 0.5;
    for (var i = 0u; i < 9u; i++) {
        let bar_x_left = start_x + f32(i) * (bar_width + bar_spacing);
        let bar_x_right = bar_x_left + bar_width;
        
        let vec_idx = i / 4u;
        let comp_idx = i % 4u;
        
        var key_state = 0.0;
        var key_decay_val = 0.0;
        
        if comp_idx == 0u {
            key_state = params.key_states[vec_idx].x;
            key_decay_val = params.key_decay[vec_idx].x;
        } else if comp_idx == 1u {
            key_state = params.key_states[vec_idx].y;
            key_decay_val = params.key_decay[vec_idx].y;
        } else if comp_idx == 2u {
            key_state = params.key_states[vec_idx].z;
            key_decay_val = params.key_decay[vec_idx].z;
        } else {
            key_state = params.key_states[vec_idx].w;
            key_decay_val = params.key_decay[vec_idx].w;
        }
        
        let key_active = key_state > 0.5;
        let intensity = max(0.1, key_decay_val);
        let bar_height = bar_max_height * intensity;
        let bar_bottom = bar_top - bar_height;
        if uv.x >= bar_x_left && uv.x <= bar_x_right && uv.y >= bar_bottom && uv.y <= bar_top {
            if key_active && key_decay_val > 0.1 {
                let key_hue = f32(i) / 8.0 * 6.28;
                let rainbow_color = vec3<f32>(
                    0.5 + 0.5 * sin(key_hue),
                    0.5 + 0.5 * sin(key_hue + 2.094),
                    0.5 + 0.5 * sin(key_hue + 4.188)
                );
                
                let height_factor = (bar_top - uv.y) / bar_height;
                let gradient = 1.0 - height_factor * 0.3;
                let pulse = sin(u_time.time * 10.0) * 0.1 + 0.9;
                color = rainbow_color * intensity * gradient * pulse;
            } else {
                let key_hue = f32(i) / 8.0;
                let height_factor = (bar_top - uv.y) / bar_height;
                let gradient = 1.0 - height_factor * 0.5;
                color = vec3<f32>(0.1 + key_hue * 0.3, 0.2, 0.4 - key_hue * 0.1) * gradient;
            }
        }
        
        if uv.x >= bar_x_left && uv.x <= bar_x_right && uv.y >= 0.92 && uv.y <= 0.98 {
            color = vec3<f32>(0.8, 0.8, 0.9);
        }
    }
    
    if params.beat_enabled > 0u && uv.y > 0.98 {
        let beat_intensity = beat_amp * 8.0;
        color = vec3<f32>(beat_intensity, beat_intensity * 0.5, beat_intensity * 0.2);
    }
    
    
    if uv.y < 0.05 {
        var waveform_color = vec3<f32>(0.5, 0.5, 0.5);
        switch params.waveform_type {
            case 0u: {
                waveform_color = vec3<f32>(0.3, 0.8, 0.3);
            }
            case 1u: {
                waveform_color = vec3<f32>(0.8, 0.8, 0.3);
            }
            case 2u: {
                waveform_color = vec3<f32>(0.8, 0.3, 0.3);
            }
            default: {
                waveform_color = vec3<f32>(0.3, 0.8, 0.3);
            }
        }
        color = waveform_color;
    }
    
    textureStore(output, global_id.xy, vec4<f32>(color, 1.0));
}