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
    reverb_mix: f32,
    delay_time: f32,
    delay_feedback: f32,
    filter_cutoff: f32,
    filter_resonance: f32,
    distortion_amount: f32,
    chorus_rate: f32,
    chorus_depth: f32,
    attack_time: f32,
    decay_time: f32,
    sustain_level: f32,
    release_time: f32,
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
        case 3u: {
            // Pure triangle wave
            let t = fract(phase / (2.0 * PI));
            return select(4.0 * t - 1.0, 3.0 - 4.0 * t, t > 0.5);
        }
        case 4u: {
            let seed = phase * 12.9898;
            return 2.0 * fract(sin(seed) * 43758.5453) - 1.0;
        }
        default: {
            return sin(phase);
        }
    }
}

fn get_note_frequency(note_index: u32, octave: f32) -> f32 {
    let notes = array<f32, 9>(
        261.63, 293.66, 329.63, 349.23, 392.00,
        440.00, 493.88, 523.25, 587.33
    );
    return notes[note_index] * pow(2.0, octave - 4.0);
}

fn apply_lowpass_filter(sample: f32, cutoff: f32, resonance: f32, time: f32) -> f32 {
    if cutoff > 0.95 {
        return sample;
    }
    let freq = cutoff * cutoff * 0.8;
    let filtered = sample * (0.3 + freq * 0.7);
    let resonant = sample * sin(time * 50.0) * resonance * 0.1;
    return filtered + resonant;
}

fn apply_distortion(sample: f32, amount: f32) -> f32 {
    if amount < 0.01 {
        return sample;
    }
    let drive = 1.0 + amount * 5.0;
    let driven = sample * drive;
    let distorted = driven / (1.0 + abs(driven));
    return mix(sample, distorted, amount);
}

fn apply_chorus(sample: f32, time: f32, rate: f32, depth: f32) -> f32 {
    if depth < 0.01 {
        return sample;
    }
    let lfo1 = sin(time * rate) * depth;
    let lfo2 = sin(time * rate * 1.3 + 1.57) * depth;
    let delayed1 = sample * (1.0 + lfo1 * 0.5);
    let delayed2 = sample * (1.0 + lfo2 * 0.3);
    return (sample + delayed1 * 0.4 + delayed2 * 0.3) / 1.7;
}

fn apply_reverb(sample: f32, mix: f32, time: f32) -> f32 {
    if mix < 0.01 {
        return sample;
    }
    let delay1 = sin(time * 0.7) * 0.01 + 0.03;
    let delay2 = sin(time * 0.5) * 0.015 + 0.08;
    let delay3 = sin(time * 0.3) * 0.02 + 0.15;
    
    let reverb_sample = sample * 0.7 + 
                       sample * sin(time * 100.0) * 0.15 * mix + 
                       sample * sin(time * 150.0 + delay2) * 0.1 * mix + 
                       sample * sin(time * 80.0 + delay3) * 0.08 * mix;
    
    return mix(sample, reverb_sample, mix);
}

fn apply_delay(sample: f32, time: f32, delay_time: f32, feedback: f32) -> f32 {
    if feedback < 0.01 {
        return sample;
    }
    let delayed_time = time - delay_time;
    let delayed_sample = sample * sin(delayed_time * 10.0) * feedback;
    let multi_tap = sample * sin(delayed_time * 15.0) * feedback * 0.3;
    return sample + delayed_sample * 0.6 + multi_tap * 0.4;
}

fn piano_envelope(key_state: f32, key_decay: f32, attack: f32, decay: f32, sustain: f32, release: f32) -> f32 {
    if key_state > 0.5 {
        return sustain * 1.4;
    } else {
        // Key released - 
        if key_decay > 0.7 {
            // Fast initial fade when key first released
            return sustain * key_decay * 0.75;
        } else {
            let smooth_decay = key_decay * key_decay * 1.2;
            return sustain * smooth_decay;
        }
    }
}

fn get_voice_time(key_state: f32, key_decay: f32, global_time: f32) -> f32 {
    // Use key_decay as a simple time progression value
    // When key pressed: key_decay stays at 1.0 (sustain phase)
    // When key released: key_decay slowly decreases (release phase)
    return key_decay * 2.0;
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
        
        if key_state > 0.5 || key_decay_val > 0.01 {
            let freq = get_note_frequency(i, params.octave);
            
            // Simple envelope calculation
            let envelope = piano_envelope(
                key_state,
                key_decay_val,
                params.attack_time, 
                params.decay_time, 
                params.sustain_level, 
                params.release_time
            );
            
            // Anti-crackling: detuning and phase offsets for all waveforms to prevent interference
            var adjusted_freq = freq;
            var phase_offset = 0.0;
            
            // Apply subtle detuning to each voice to prevent phase alignment crackling
            let detune_amount = (f32(i) - 4.0) * 0.002; // Small detuning per voice
            adjusted_freq = freq * (1.0 + detune_amount);
            
            // Golden ratio phase offset for natural harmonic spacing
            phase_offset = f32(i) * 0.61803398875;
            
            let phase = (u_time.time * adjusted_freq + phase_offset) * 2.0 * PI;
            var waveform_sample = generate_waveform(phase, params.waveform_type);
            
            waveform_sample = apply_lowpass_filter(waveform_sample, params.filter_cutoff, params.filter_resonance, u_time.time);
            waveform_sample = apply_distortion(waveform_sample, params.distortion_amount);
            waveform_sample = apply_chorus(waveform_sample, u_time.time + f32(i) * 0.1, params.chorus_rate, params.chorus_depth);
            waveform_sample = apply_delay(waveform_sample, u_time.time + f32(i) * 0.05, params.delay_time, params.delay_feedback);
            waveform_sample = apply_reverb(waveform_sample, params.reverb_mix, u_time.time);
            
            let key_amp = envelope * 0.6;
            
            key_sample += waveform_sample * key_amp;
            active_keys += 1.0;
            
            if envelope > max_key_amp {
                max_key_amp = envelope;
                dominant_freq = freq;
            }
        }
    }
    
    if active_keys > 1.0 {
        key_sample = key_sample / sqrt(active_keys);
    }
    
    var mixed_sample = beat_sample * 0.2 + key_sample;
    
    mixed_sample = apply_reverb(mixed_sample, params.reverb_mix * 0.4, u_time.time);
    mixed_sample = apply_delay(mixed_sample, u_time.time, params.delay_time * 0.8, params.delay_feedback * 0.5);
    
    mixed_sample = mixed_sample * params.volume * 3.5;
    
    // Simple limiting
    let limit = 0.9;
    if abs(mixed_sample) > limit {
        mixed_sample = sign(mixed_sample) * limit;
    }
    
    let final_amplitude = abs(mixed_sample);
    
    if global_id.x == 0u && global_id.y == 0u {
        audio_buffer[0] = dominant_freq;
        audio_buffer[1] = final_amplitude;
        audio_buffer[2] = f32(params.waveform_type);
        
        // Output per-voice frequencies and smooth envelope amplitudes
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
            
            let frequency = get_note_frequency(i, params.octave);
            audio_buffer[3 + i] = frequency;
            
            // Calculate and output smooth envelope amplitude for each voice
            if key_state > 0.5 || key_decay_val > 0.01 {
                let envelope = piano_envelope(
                    key_state,
                    key_decay_val,
                    params.attack_time, 
                    params.decay_time, 
                    params.sustain_level, 
                    params.release_time
                );
                
                audio_buffer[12 + i] = envelope * params.volume * 0.4;
            } else {
                audio_buffer[12 + i] = 0.0;
            }
        }
        
        audio_buffer[21] = beat_sample;
        audio_buffer[22] = params.tempo * 2.0;
        
        audio_buffer[23] = params.reverb_mix;
        audio_buffer[24] = params.delay_time;
        audio_buffer[25] = params.delay_feedback;
        audio_buffer[26] = params.filter_cutoff;
        audio_buffer[27] = params.distortion_amount;
        audio_buffer[28] = params.chorus_rate;
        audio_buffer[29] = params.chorus_depth;
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
            case 3u: {
                waveform_color = vec3<f32>(0.3, 0.3, 0.8);
            }
            case 4u: {
                waveform_color = vec3<f32>(0.8, 0.3, 0.8);
            }
            default: {
                waveform_color = vec3<f32>(0.3, 0.8, 0.3);
            }
        }
        
        let effect_intensity = max(params.reverb_mix, max(params.delay_feedback, params.distortion_amount));
        waveform_color = waveform_color * (1.0 + effect_intensity * 0.5);
        color = waveform_color;
    }
    
    if uv.y > 0.05 && uv.y < 0.08 {
        let filter_viz = params.filter_cutoff;
        let resonance_viz = params.filter_resonance;
        let filter_color = vec3<f32>(filter_viz, resonance_viz, 0.5);
        color = mix(color, filter_color, 0.3);
    }
    
    textureStore(output, global_id.xy, vec4<f32>(color, 1.0));
}