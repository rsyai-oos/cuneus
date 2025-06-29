// Enes Altun, 2025; MIT License
// Veridis Quo - Mathematical/Shader Approach
// Base frequencies for the main notes used in the song. This is also my first shader song, probably WIP for a while.
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;


struct FontUniforms {
    atlas_size: vec2<f32>,
    char_size: vec2<f32>,
    screen_size: vec2<f32>,
    _padding: vec2<f32>,
};
@group(3) @binding(0) var<uniform> u_font: FontUniforms;
@group(3) @binding(1) var t_font_atlas: texture_2d<f32>;
@group(3) @binding(2) var s_font_atlas: sampler;
@group(3) @binding(3) var<storage, read_write> audio_buffer: array<f32>;

struct SongParams {
    volume: f32,
    octave_shift: f32,
    tempo_multiplier: f32,
    waveform_type: u32,
    crossfade: f32,
};
@group(2) @binding(0) var<uniform> u_song: SongParams;

const PI = 3.14159265359;


const F5 = 698.46;  // Fret 13
const E5 = 659.25;  // Fret 12  
const D5 = 587.33;  // Fret 10
const C5 = 523.25;  // Fret 8
const B4 = 493.88;  // Fret 7
const A4 = 440.00;  // Fret 5

fn get_veridis_quo_frequency_and_envelope(time_in_song: f32) -> vec3<f32> {
    // 107 BPM = 107 quarter notes per minute = 1.785 quarter notes per second
    // Each measure in 4/4 = 4 quarter notes = 4/1.785 = 2.24 seconds per measure
    let measure_duration = 60.0 / 107.0 * 4.0;
    let total_pattern = measure_duration * 7.0; 
    let loop_time = time_in_song % total_pattern;
    let measure = u32(loop_time / measure_duration);
    let progress = fract(loop_time / measure_duration);
    
    var frequency = 440.0;
    var envelope = 0.8;
    var note_type = 0.0;
    
    switch measure {
        case 0u: {
            // Measure 1: 13-12-13-10 (F5-E5-F5-D5)
            // first 4 notes in veridis quo is fast right? 
            let phase = progress * 7.0;
            let current_note = u32(min(phase, 3.99));
            let note_progress = fract(phase);
            
            switch current_note {
                case 0u: { 
                    // Very subtle transitions
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(F5, E5, transition);
                    } else {
                        frequency = F5;
                    }
                    note_type = 5.0;
                }
                case 1u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(E5, F5, transition);
                    } else {
                        frequency = E5;
                    }
                    note_type = 4.0; 
                }
                case 2u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(F5, D5, transition);
                    } else {
                        frequency = F5;
                    }
                    note_type = 5.0; 
                }
                default: { 
                    if phase > 4.0 {
                        let padding_progress = (phase - 4.0) / 3.0; // Adjusted for phase * 7.0
                        frequency = mix(D5, F5, smoothstep(0.0, 1.0, padding_progress * 0.3));
                        envelope = 0.8 * (1.0 + padding_progress * 0.05);
                    } else {
                        frequency = D5;
                    }
                    note_type = 3.0; 
                }
            }
            envelope = 0.8;
        }
        case 1u: {
            // Measure 2: 13-12-13-7 (F5-E5-F5-B4)
            let phase = progress * 7.0;
            let current_note = u32(min(phase, 3.99));
            let note_progress = fract(phase);
            
            switch current_note {
                case 0u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(F5, E5, transition);
                    } else {
                        frequency = F5;
                    }
                    note_type = 5.0; 
                }
                case 1u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(E5, F5, transition);
                    } else {
                        frequency = E5;
                    }
                    note_type = 4.0; 
                }
                case 2u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(F5, B4, transition);
                    } else {
                        frequency = F5;
                    }
                    note_type = 5.0; 
                }
                default: { 
                    if phase > 4.0 {
                        let padding_progress = (phase - 4.0) / 3.0; 
                        frequency = mix(B4, B4, smoothstep(0.0, 1.0, padding_progress));
                        envelope = 0.8 * (1.0 + padding_progress * 0.1);
                    } else {
                        frequency = B4;
                    }
                    note_type = 1.0; 
                }
            }
            // Don't override envelope if it was set in padding area
            if phase <= 4.0 {
                envelope = 0.8;
            }
        }
        case 2u: {
            // Measure 3: (7) - B4 TIE
            if progress < 0.5 {
                // B4 tie for first
                frequency = B4;
                note_type = 1.0;
                let fade_progress = progress / 0.3;
                envelope = 0.57 * (1.0 - fade_progress * 0.8); // Fade out B4
            } else {
                // Start preparing for next section (E5-D5-E5-C5)
                let prep_progress = (progress - 0.5) / 0.5;
                frequency = mix(B4, E5, smoothstep(0.8, 1.0, prep_progress));
                note_type = 4.0;
                 // Crescendo into next section
                envelope = 0.1 + prep_progress * 0.7;
            }
        }
        case 3u: {
            // Measure 4: 12-10-12-8 (E5-D5-E5-C5) - fast rhythm
            let phase = progress * 7.0;
            let current_note = u32(min(phase, 3.99));
            let note_progress = fract(phase);
            
            switch current_note {
                case 0u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(E5, D5, transition);
                    } else {
                        frequency = E5;
                    }
                    note_type = 4.0; 
                }
                case 1u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(D5, E5, transition);
                    } else {
                        frequency = D5;
                    }
                    note_type = 3.0; 
                }
                case 2u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(E5, C5, transition);
                    } else {
                        frequency = E5;
                    }
                    note_type = 4.0; 
                }
                default: { 
                    if phase > 4.0 {
                        let padding_progress = (phase - 4.0) / 3.0;
                        frequency = mix(C5, E5, smoothstep(0.0, 1.0, padding_progress * 0.3));
                        envelope = 0.8 * (1.0 + padding_progress * 0.05);
                    } else {
                        frequency = C5;
                    }
                    note_type = 2.0; 
                }
            }
            envelope = 0.8;
        }
        
        case 4u: {
            // Measure 5: 12-10-12-5 (E5-D5-E5-A4) - ending in A4 tie
            let phase = progress * 7.0;
            let current_note = u32(min(phase, 3.99));
            let note_progress = fract(phase);
            
            switch current_note {
                case 0u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(E5, D5, transition);
                    } else {
                        frequency = E5;
                    }
                    note_type = 4.0; 
                }
                case 1u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(D5, E5, transition);
                    } else {
                        frequency = D5;
                    }
                    note_type = 3.0; 
                }
                case 2u: { 
                    if note_progress > u_song.crossfade {
                        let transition = smoothstep(u_song.crossfade, 1.0, note_progress);
                        frequency = mix(E5, A4, transition);
                    } else {
                        frequency = E5;
                    }
                    note_type = 4.0; 
                }
                default: { 
                    if phase > 4.0 {
                        let padding_progress = (phase - 4.0) / 3.0;
                        frequency = mix(A4, A4, smoothstep(0.0, 1.0, padding_progress));
                        envelope = 0.8 * (1.0 + padding_progress * 0.1);
                    } else {
                        frequency = A4;
                    }
                    note_type = 0.0; 
                }
            }
            envelope = 0.8;
        }
        case 5u: {
            // Measure 6: (5) - A4 TIE 
            if progress < 0.5 {

                frequency = A4;
                note_type = 0.0;
                envelope = 0.8 * (1.0 - progress * 0.6);
            } else {
                let fade_progress = (progress - 0.5) / 0.5;
                frequency = mix(A4, F5, smoothstep(0.0, 1.0, fade_progress * 0.3)); // Gentle transition
                note_type = 5.0;
                envelope = 0.5 * (1.0 - fade_progress * 0.8); // Fade out for loop
            }
        }
        case 6u: {
            // Measure 7: Rest/Pause before loop
            frequency = F5;
            note_type = 5.0;
            envelope = 0.1 + progress * 0.2;
        }
        default: {
            frequency = A4;
            note_type = 0.0;
            envelope = 0.5;
        }
    }
    envelope = max(envelope, 0.05);
    
    return vec3<f32>(frequency, envelope, note_type);
}


fn get_current_melody_info(song_time: f32, tempo_multiplier: f32, volume: f32, octave_shift: f32) -> vec4<f32> {
    // Apply tempo multiplier
    let adjusted_time = song_time * tempo_multiplier;
    
    let melody_result = get_veridis_quo_frequency_and_envelope(adjusted_time);
    let frequency = melody_result.x;
    let envelope = melody_result.y;
    let note_type = melody_result.z;
    
    let final_envelope = envelope * volume;
    
    let adjusted_frequency = frequency * pow(2.0, octave_shift);
    
    return vec4<f32>(adjusted_frequency, final_envelope, 0.0, note_type);
}

fn generate_waveform(phase: f32, waveform_type: u32) -> f32 {
    switch waveform_type {
        case 0u: { //sine
            return sin(phase);
        }
        case 1u: { // saw
            return 2.0 * fract(phase / (2.0 * PI)) - 1.0;
        }
        case 2u: { // square
            return select(-1.0, 1.0, sin(phase) > 0.0);
        }
        default: {
            return sin(phase);
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output);
    let coord = vec2<i32>(global_id.xy);
    
    if coord.x >= i32(dims.x) || coord.y >= i32(dims.y) {
        return;
    }
    
    let uv = vec2<f32>(f32(coord.x) / f32(dims.x), f32(coord.y) / f32(dims.y));
    
    // CPU
    let volume = u_song.volume; 
    let octave_shift = u_song.octave_shift; 
    let tempo_multiplier = u_song.tempo_multiplier; 
    let waveform_type = u_song.waveform_type;
    
    let melody_info = get_current_melody_info(u_time.time, tempo_multiplier, volume, octave_shift);
    let frequency = melody_info.x;
    let envelope = melody_info.y;
    let note_index = u32(melody_info.z);
    let note_name = u32(melody_info.w);
    
    // Generate audio buffer (only from thread 0,0): same approach: debugscreen.wgsl/synth.wgsl
    if global_id.x == 0u && global_id.y == 0u {
        // Write audio metadata to buffer
        audio_buffer[0] = frequency;
        audio_buffer[1] = envelope * 0.4;
        // Pass waveform type to CPU
        audio_buffer[2] = f32(waveform_type);
        
        let sample_start = 3u;
        var previous_phase = 0.0;
        
        for (var i = 0u; i < 1024u; i++) {
            let sample_time = u_time.time + f32(i) * (1.0 / 44100.0);
            let sample_melody_info = get_current_melody_info(sample_time, tempo_multiplier, volume, octave_shift);
            let sample_freq = sample_melody_info.x;
            let sample_env = sample_melody_info.y;
            
            var final_sample = 0.0;
            if sample_freq > 0.0 && sample_env > 0.001 {
                let phase_increment = sample_freq * 2.0 * PI * (1.0 / 44100.0);
                previous_phase += phase_increment;
                
                let waveform_sample = generate_waveform(previous_phase, waveform_type);
                
                final_sample = waveform_sample * sample_env * 0.3;
                
                final_sample = final_sample * 0.8;
            } else {

                final_sample = 0.0;
                previous_phase = 0.0;
            }
            
            audio_buffer[sample_start + i] = final_sample;
        }
    }
    var color = vec3<f32>(0.02, 0.01, 0.08);
    let wave_distortion = sin(uv.x * 6.0 + u_time.time * 1.5) * 0.3;
    let freq_factor = frequency / 700.0;
    color += vec3<f32>(0.05, 0.02, 0.1) * wave_distortion * freq_factor;
    let progress_bar_height = 0.03;
    let progress_y = 0.95;
    if uv.y > progress_y && uv.y < progress_y + progress_bar_height {
        let song_duration = 14.0;
        let song_progress = (u_time.time % song_duration) / song_duration;
        if uv.x < song_progress {
            color = vec3<f32>(0.0, 0.7, 1.0);
        } else {
            color = vec3<f32>(0.15, 0.15, 0.3);
        }
    }
    
    let visualizer_center_y = 0.5;
    let pattern_width = 0.8;
    let pattern_start_x = 0.1;
    let pattern_height = 0.4;
    
    for (var measure = 0u; measure < 7u; measure++) {
        let measure_width = pattern_width / 7.0;
        let measure_x = pattern_start_x + f32(measure) * measure_width;
        
        if uv.x >= measure_x && uv.x <= measure_x + measure_width * 0.9 {
            let measure_progress = (uv.x - measure_x) / measure_width;
            
            let measure_time = f32(measure) * 2.0;
            let test_melody = get_veridis_quo_frequency_and_envelope(measure_time + measure_progress * 2.0);
            let measure_freq = test_melody.x;
            let measure_note_type = test_melody.z;
            
            let freq_norm = (measure_freq - 440.0) / (698.46 - 440.0);
            let bar_height = mix(0.1, pattern_height, freq_norm);
            let bar_bottom = visualizer_center_y - bar_height * 0.5;
            let bar_top = visualizer_center_y + bar_height * 0.5;
            
            if uv.y >= bar_bottom && uv.y <= bar_top {
                let current_measure = u32((u_time.time % (7.0 * 2.0)) / 2.0);
                
                if measure == current_measure && frequency > 0.0 {
                    let pulse = sin(u_time.time * 8.0) * 0.4 + 0.8;
                    color = vec3<f32>(1.0, 0.9, 0.2) * pulse;
                } else {
                    let note_color = get_note_color(u32(measure_note_type));
                    color = note_color * 0.6;
                }
            }
        }
    }
    
    if uv.y < 0.12 {
        let spectrum_freq = mix(400.0, 800.0, uv.x);
        let freq_distance = abs(spectrum_freq - frequency);
        let freq_response = exp(-freq_distance / 30.0);
        let spectrum_intensity = freq_response * envelope;
        
        let spectrum_bar_height = uv.y / 0.12;
        if spectrum_bar_height < spectrum_intensity && frequency > 0.0 {
            color += vec3<f32>(spectrum_intensity * 0.8, spectrum_intensity * 0.4, 0.1);
        }
    }
    
    if uv.y > 0.88 && uv.y < 0.95 {
        let title_glow = sin(uv.x * 12.0 + u_time.time * 2.0) * 0.3 + 0.5;
        color = mix(color, vec3<f32>(0.8, 0.8, 1.0), title_glow * 0.4);
    }
    
    let ambient_glow = envelope * 0.15;
    color += vec3<f32>(ambient_glow * 0.3, ambient_glow * 0.5, ambient_glow * 0.8);
    
    textureStore(output, global_id.xy, vec4<f32>(color, 1.0));
}

fn get_note_color(note_name: u32) -> vec3<f32> {
    switch note_name {
        case 0u: { return vec3<f32>(1.0, 0.3, 0.3); }
        case 1u: { return vec3<f32>(1.0, 0.6, 0.0); }
        case 2u: { return vec3<f32>(1.0, 1.0, 0.2); }
        case 3u: { return vec3<f32>(0.3, 1.0, 0.3); }
        case 4u: { return vec3<f32>(0.2, 0.7, 1.0); }
        case 5u: { return vec3<f32>(0.8, 0.3, 1.0); }
        default: { return vec3<f32>(0.5, 0.5, 0.5); }
    }
}