// Enes Altun, 2025; MIT License
// Veridis Quo - Mathematical/Shader Approach
// Base frequencies for the main notes used in the song. This is also my first shader song, but I think it could be a nice example for cuneus. 
// This song also probably always WIP, I will keep improving it over time by the time I implement more advanced audio synthesis techniques on cuneus.
// Note numbers (basically tabs) based on my guitar feelings :-P so don't be confuse about those numbers and sorry for ignorance about music theory :D

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

// --- Note Frequencies (Bassline - 2 octaves lower) ---
const F3 = F5 / 4.0;
const B2 = B4 / 4.0;
const E3 = E5 / 4.0; 
const A2 = A4 / 4.0;

fn get_note_color(note_name: u32) -> vec3<f32> {
    switch note_name {
        case 0u: { return vec3<f32>(1.0, 0.3, 0.3); } // A
        case 1u: { return vec3<f32>(1.0, 0.6, 0.0); } // B
        case 2u: { return vec3<f32>(1.0, 1.0, 0.2); } // C
        case 3u: { return vec3<f32>(0.3, 1.0, 0.3); } // D
        case 4u: { return vec3<f32>(0.2, 0.7, 1.0); } // E
        case 5u: { return vec3<f32>(0.8, 0.3, 1.0); } // F
        default: { return vec3<f32>(0.5, 0.5, 0.5); }
    }
}

// Generates a legato (smooth) transition between two frequencies.
fn legato(freq_from: f32, freq_to: f32, progress: f32) -> f32 {
    let start_point = 1.0 - clamp(u_song.crossfade, 0.01, 1.0);
    if (progress < start_point) {
        return freq_from;
    }
    let transition = smoothstep(start_point, 1.0, progress);
    return mix(freq_from, freq_to, transition);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output);
    if (global_id.x >= dims.x || global_id.y >= dims.y) {
        return;
    }

    var melody_freq_visualizer = 0.0;
    var envelope_visualizer = 0.0;
    var note_type_visualizer = 0.0;

    // Audio Generation (executed only on the first thread) ---
    if (global_id.x == 0u && global_id.y == 0u) {
        let adjusted_time = u_time.time * u_song.tempo_multiplier;
        
        let measure_duration = (60.0 / 107.0) * 4.0;
        let total_pattern_duration = measure_duration * 8.0;
        let loop_time = adjusted_time % total_pattern_duration;
        let measure = u32(loop_time / measure_duration);
        let progress_in_measure = fract(loop_time / measure_duration);

        var melody_freq = 0.0;
        var melody_amp = 1.0;
        var bass_freq = 0.0;
        var bass_amp = 0.7;

        switch (measure) {
            // -- SECTION 1 (Normal Speed) --
            case 0u: { // Measure 1: 13-12-13-10, then 13-12-13-7
                bass_freq = F3;
                if (progress_in_measure < 0.5) { // First half
                    let phrase_progress = progress_in_measure * 2.0;
                    let note_index = u32(floor(phrase_progress * 4.0));
                    let note_progress = fract(phrase_progress * 4.0);
                    switch (note_index) {
                        case 0u: { melody_freq = legato(F5, E5, note_progress); note_type_visualizer = 5.0; }
                        case 1u: { melody_freq = legato(E5, F5, note_progress); note_type_visualizer = 4.0; }
                        case 2u: { melody_freq = legato(F5, D5, note_progress); note_type_visualizer = 5.0; }
                        default: { melody_freq = D5; note_type_visualizer = 3.0; }
                    }
                } else { // Second half
                    let phrase_progress = (progress_in_measure - 0.5) * 2.0;
                    let note_index = u32(floor(phrase_progress * 4.0));
                    let note_progress = fract(phrase_progress * 4.0);
                    switch (note_index) {
                        case 0u: { melody_freq = legato(F5, E5, note_progress); note_type_visualizer = 5.0; }
                        case 1u: { melody_freq = legato(E5, F5, note_progress); note_type_visualizer = 4.0; }
                        case 2u: { melody_freq = legato(F5, B4, note_progress); note_type_visualizer = 5.0; }
                        default: { melody_freq = B4; note_type_visualizer = 1.0; }
                    }
                }
            }
            case 1u: { // Measure 2: Hold B4
                melody_freq = B4; bass_freq = B2; note_type_visualizer = 1.0;
                melody_amp = (1.0 - progress_in_measure);
            }

            // -- SECTION 2 (Normal Speed) --
            case 2u: { // Measure 3: 12-10-12-8, then 12-10-12-5
                bass_freq = E3;
                if (progress_in_measure < 0.5) { // First half
                    let phrase_progress = progress_in_measure * 2.0;
                    let note_index = u32(floor(phrase_progress * 4.0));
                    let note_progress = fract(phrase_progress * 4.0);
                    switch (note_index) {
                        case 0u: { melody_freq = legato(E5, D5, note_progress); note_type_visualizer = 4.0; }
                        case 1u: { melody_freq = legato(D5, E5, note_progress); note_type_visualizer = 3.0; }
                        case 2u: { melody_freq = legato(E5, C5, note_progress); note_type_visualizer = 4.0; }
                        default: { melody_freq = C5; note_type_visualizer = 2.0; }
                    }
                } else { // Second half
                    let phrase_progress = (progress_in_measure - 0.5) * 2.0;
                    let note_index = u32(floor(phrase_progress * 4.0));
                    let note_progress = fract(phrase_progress * 4.0);
                    switch (note_index) {
                        case 0u: { melody_freq = legato(E5, D5, note_progress); note_type_visualizer = 4.0; }
                        case 1u: { melody_freq = legato(D5, E5, note_progress); note_type_visualizer = 3.0; }
                        case 2u: { melody_freq = legato(E5, A4, note_progress); note_type_visualizer = 4.0; }
                        default: { melody_freq = A4; note_type_visualizer = 0.0; }
                    }
                }
            }
            case 3u: { // Measure 4: Hold A4
                melody_freq = A4; bass_freq = A2; note_type_visualizer = 0.0;
                melody_amp = (1.0 - progress_in_measure);
            }

            // -- SECTION 3 (Normal Speed - Repeat of Section 1) --
            case 4u: { // Measure 5: Same as Measure 1
                bass_freq = F3;
                if (progress_in_measure < 0.5) {
                    let phrase_progress = progress_in_measure * 2.0; let note_index = u32(floor(phrase_progress * 4.0)); let note_progress = fract(phrase_progress * 4.0);
                    switch (note_index) { case 0u:{melody_freq=legato(F5,E5,note_progress);note_type_visualizer=5.0;} case 1u:{melody_freq=legato(E5,F5,note_progress);note_type_visualizer=4.0;} case 2u:{melody_freq=legato(F5,D5,note_progress);note_type_visualizer=5.0;} default:{melody_freq=D5;note_type_visualizer=3.0;} }
                } else {
                    let phrase_progress = (progress_in_measure - 0.5) * 2.0; let note_index = u32(floor(phrase_progress * 4.0)); let note_progress = fract(phrase_progress * 4.0);
                    switch (note_index) { case 0u:{melody_freq=legato(F5,E5,note_progress);note_type_visualizer=5.0;} case 1u:{melody_freq=legato(E5,F5,note_progress);note_type_visualizer=4.0;} case 2u:{melody_freq=legato(F5,B4,note_progress);note_type_visualizer=5.0;} default:{melody_freq=B4;note_type_visualizer=1.0;} }
                }
            }
            case 5u: { // Measure 6: Same as Measure 2
                melody_freq = B4; bass_freq = B2; note_type_visualizer = 1.0;
                melody_amp = (1.0 - progress_in_measure);
            }

            // -- SECTION 4 (THE FAST PART) FOR A COOL ENDING --
            case 6u: { // Measure 7: The FAST 8-note run, then hold
                bass_freq = E3;
                if (progress_in_measure < 0.5) { // The entire fast run happens in the first half
                    let phrase_progress = progress_in_measure * 2.0;
                    let note_index = u32(floor(phrase_progress * 8.0));
                    let note_progress = fract(phrase_progress * 8.0);
                    switch (note_index) {
                        case 0u: { melody_freq=legato(E5,D5,note_progress);note_type_visualizer=4.0;} // 12
                        case 1u: { melody_freq=legato(D5,E5,note_progress);note_type_visualizer=3.0;} // 10
                        case 2u: { melody_freq=legato(E5,C5,note_progress);note_type_visualizer=4.0;} // 12
                        case 3u: { melody_freq=legato(C5,E5,note_progress);note_type_visualizer=2.0;} // 8
                        case 4u: { melody_freq=legato(E5,D5,note_progress);note_type_visualizer=4.0;} // 12
                        case 5u: { melody_freq=legato(D5,E5,note_progress);note_type_visualizer=3.0;} // 10
                        case 6u: { melody_freq=legato(E5,A4,note_progress);note_type_visualizer=4.0;} // 12
                        default: { melody_freq=A4;note_type_visualizer=0.0;}                             // 5
                    }
                } else { // Hold the final note for the rest of the measure
                    melody_freq = A4; note_type_visualizer = 0.0;
                }
            }
            case 7u: { // Measure 8: Hold A4
                melody_freq = A4; bass_freq = A2; note_type_visualizer = 0.0;
                melody_amp = (1.0 - progress_in_measure);
            }
            default: {}
        }
        
        // Fade out bass amp during holds to make it less prominent
        if (measure == 1u || measure == 3u || measure == 5u || measure == 7u) {
            bass_amp = 0.7 * (1.0 - progress_in_measure);
        }

        melody_freq_visualizer = melody_freq;
        envelope_visualizer = (melody_amp + bass_amp) * u_song.volume;
        let final_melody_freq = melody_freq * pow(2.0, u_song.octave_shift);
        let final_bass_freq = bass_freq * pow(2.0, u_song.octave_shift);
        let final_melody_amp = melody_amp * u_song.volume;
        let final_bass_amp = bass_amp * u_song.volume * 0.7;

        audio_buffer[0] = melody_freq_visualizer;
        audio_buffer[1] = envelope_visualizer;
        audio_buffer[2] = f32(u_song.waveform_type);
        audio_buffer[3] = final_melody_freq;
        audio_buffer[4] = final_melody_amp;
        audio_buffer[5] = final_bass_freq;
        audio_buffer[6] = final_bass_amp;
        for (var i = 2u; i < 16u; i++) {
             audio_buffer[3u + i * 2u] = 0.0;
             audio_buffer[3u + i * 2u + 1u] = 0.0;
        }
    }

    let frequency = audio_buffer[0];
    let envelope = audio_buffer[1];
    let uv = vec2<f32>(global_id.xy) / vec2<f32>(dims);
    var color = vec3<f32>(0.02, 0.01, 0.08); 
    let wave_distortion = sin(uv.x * 6.0 + u_time.time * 1.5) * 0.3;
    let freq_factor = (frequency - 400.0) / 300.0;
    color += vec3<f32>(0.05, 0.02, 0.1) * wave_distortion * freq_factor * envelope;
    let progress_bar_height = 0.02;
    if (uv.y > 0.95 && uv.y < 0.95 + progress_bar_height) {
        let measure_duration = (60.0 / 107.0) * 4.0;
        let total_pattern_duration = measure_duration * 8.0;
        let song_progress = (u_time.time * u_song.tempo_multiplier % total_pattern_duration) / total_pattern_duration;
        if (uv.x < song_progress) { color = mix(color, vec3<f32>(0.0, 0.7, 1.0), 0.8); } 
        else { color = mix(color, vec3<f32>(0.15, 0.15, 0.3), 0.8); }
    }
    let visualizer_center_y = 0.5;
    let pattern_width = 0.8;
    let pattern_start_x = (1.0 - pattern_width) / 2.0;
    let measure_duration = (60.0 / 107.0) * 4.0;
    let total_pattern_duration = measure_duration * 8.0;
    let current_song_time = u_time.time * u_song.tempo_multiplier;
    let current_progress_in_pattern = (current_song_time % total_pattern_duration) / total_pattern_duration;
    if (abs(uv.y - visualizer_center_y) < 0.25) {
        let viz_progress = (uv.x - pattern_start_x) / pattern_width;
        if (viz_progress >= 0.0 && viz_progress <= 1.0) {
            let viz_measure = u32(floor(viz_progress * 8.0));
            let viz_measure_progress = fract(viz_progress * 8.0);
            let freq_norm = (frequency - A4) / (F5 - A4);
            let bar_y = visualizer_center_y + (freq_norm - 0.5) * 0.4;
            if (abs(uv.y - bar_y) < 0.01) {
                if (abs(viz_progress - current_progress_in_pattern) < 0.005) {
                     let pulse = sin(u_time.time * 15.0) * 0.5 + 0.5;
                     color = mix(color, vec3<f32>(1.0, 1.0, 0.5) * pulse, 0.9);
                }
            }
        }
    }
    let ambient_glow = envelope * 0.15;
    color += vec3<f32>(ambient_glow * 0.3, ambient_glow * 0.5, ambient_glow * 0.8);
    textureStore(output, global_id.xy, vec4<f32>(color, 1.0));
}