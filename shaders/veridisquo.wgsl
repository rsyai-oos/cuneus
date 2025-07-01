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
        
        // Music timing based on 107 BPM and 4/4 time signature
        let measure_duration = (60.0 / 107.0) * 4.0;
        let total_pattern_duration = measure_duration * 4.0; // 4-measure pattern
        let loop_time = adjusted_time % total_pattern_duration;
        let measure = u32(loop_time / measure_duration);
        let progress_in_measure = fract(loop_time / measure_duration);

        var melody_freq = 0.0;
        var melody_amp = 0.0;
        var bass_freq = 0.0;
        var bass_amp = 0.0;

        switch (measure) {
            // Measure 1: 13-12-13-10-13-12-13-7 
            case 0u: {
                let note_index = u32(floor(progress_in_measure * 8.0));
                let note_progress = fract(progress_in_measure * 8.0);
                bass_freq = F3;
                bass_amp = 0.7;
                melody_amp = 1.0;

                switch (note_index) {
                    case 0u: { melody_freq = legato(F5, E5, note_progress); note_type_visualizer = 5.0; } // 13
                    case 1u: { melody_freq = legato(E5, F5, note_progress); note_type_visualizer = 4.0; } // 12
                    case 2u: { melody_freq = legato(F5, D5, note_progress); note_type_visualizer = 5.0; } // 13
                    case 3u: { melody_freq = legato(D5, F5, note_progress); note_type_visualizer = 3.0; } // 10
                    case 4u: { melody_freq = legato(F5, E5, note_progress); note_type_visualizer = 5.0; } // 13
                    case 5u: { melody_freq = legato(E5, F5, note_progress); note_type_visualizer = 4.0; } // 12
                    case 6u: { melody_freq = legato(F5, B4, note_progress); note_type_visualizer = 5.0; } // 13
                    default: { melody_freq = B4; note_type_visualizer = 1.0; } // 7
                }
            }
            // Measure 2: Hold note 7 (B4)
            case 1u: {
                melody_freq = B4;
                bass_freq = B2;
                note_type_visualizer = 1.0;
                // Fade out the held notes over the measure
                let decay = 1.0 - progress_in_measure;
                melody_amp = 1.0 * decay;
                bass_amp = 0.7 * decay;
            }
            // Measure 3: 12-10-12-8-12-10-12-5 (as eight 16th notes)
            case 2u: {
                let note_index = u32(floor(progress_in_measure * 8.0));
                let note_progress = fract(progress_in_measure * 8.0);
                bass_freq = E3;
                bass_amp = 0.7;
                melody_amp = 1.0;

                switch (note_index) {
                    case 0u: { melody_freq = legato(E5, D5, note_progress); note_type_visualizer = 4.0; } // 12
                    case 1u: { melody_freq = legato(D5, E5, note_progress); note_type_visualizer = 3.0; } // 10
                    case 2u: { melody_freq = legato(E5, C5, note_progress); note_type_visualizer = 4.0; } // 12
                    case 3u: { melody_freq = legato(C5, E5, note_progress); note_type_visualizer = 2.0; } // 8
                    case 4u: { melody_freq = legato(E5, D5, note_progress); note_type_visualizer = 4.0; } // 12
                    case 5u: { melody_freq = legato(D5, E5, note_progress); note_type_visualizer = 3.0; } // 10
                    case 6u: { melody_freq = legato(E5, A4, note_progress); note_type_visualizer = 4.0; } // 12
                    default: { melody_freq = A4; note_type_visualizer = 0.0; } // 5
                }
            }
            // Measure 4: Hold note 5 (A4)
            case 3u: {
                melody_freq = A4;
                bass_freq = A2;
                note_type_visualizer = 0.0;
                let decay = 1.0 - progress_in_measure;
                melody_amp = 1.0 * decay;
                bass_amp = 0.7 * decay;
            }
            default: {}
        }
        
        melody_freq_visualizer = melody_freq;
        envelope_visualizer = (melody_amp + bass_amp) * u_song.volume;

        // Apply global controls (volume, octave)
        let final_melody_freq = melody_freq * pow(2.0, u_song.octave_shift);
        let final_bass_freq = bass_freq * pow(2.0, u_song.octave_shift);
        let final_melody_amp = melody_amp * u_song.volume;
        let final_bass_amp = bass_amp * u_song.volume * 0.7;

        // --- Write data to audio_buffer for Rust host ---
        // Metadata
        audio_buffer[0] = melody_freq_visualizer; // Main frequency for visualizer
        audio_buffer[1] = envelope_visualizer;    // Overall envelope for visualizer
        audio_buffer[2] = f32(u_song.waveform_type);
        
        // Voice 0: Melody
        audio_buffer[3] = final_melody_freq;
        audio_buffer[4] = final_melody_amp;
        
        // Voice 1: Bass
        audio_buffer[5] = final_bass_freq;
        audio_buffer[6] = final_bass_amp;

        // Clear any other voices to ensure they are silent
        for (var i = 2u; i < 16u; i++) {
             audio_buffer[3u + i * 2u] = 0.0;
             audio_buffer[3u + i * 2u + 1u] = 0.0;
        }
    }

    // Read the main melody frequency and envelope from the buffer
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
        let total_pattern_duration = measure_duration * 4.0;
        let song_progress = (u_time.time * u_song.tempo_multiplier % total_pattern_duration) / total_pattern_duration;
        if (uv.x < song_progress) {
            color = mix(color, vec3<f32>(0.0, 0.7, 1.0), 0.8);
        } else {
            color = mix(color, vec3<f32>(0.15, 0.15, 0.3), 0.8);
        }
    }
    
    let visualizer_center_y = 0.5;
    let pattern_width = 0.8;
    let pattern_start_x = (1.0 - pattern_width) / 2.0;
    
    let measure_duration = (60.0 / 107.0) * 4.0;
    let total_pattern_duration = measure_duration * 4.0;
    let current_song_time = u_time.time * u_song.tempo_multiplier;
    let current_progress_in_pattern = (current_song_time % total_pattern_duration) / total_pattern_duration;

    if (abs(uv.y - visualizer_center_y) < 0.25) {
        let viz_progress = (uv.x - pattern_start_x) / pattern_width;
        if (viz_progress >= 0.0 && viz_progress <= 1.0) {
            let viz_time = viz_progress * total_pattern_duration;
            let viz_measure = u32(floor(viz_progress * 4.0));
            let viz_measure_progress = fract(viz_progress * 4.0);
            
            var viz_freq = 0.0;
            var viz_note_type = 6.0;

            switch (viz_measure) {
                 case 0u: {
                    let note_index = u32(floor(viz_measure_progress * 8.0));
                    switch(note_index) { case 0u,2u,4u,6u: {viz_freq=F5;viz_note_type=5.0;} case 1u,5u: {viz_freq=E5;viz_note_type=4.0;} case 3u: {viz_freq=D5;viz_note_type=3.0;} default: {viz_freq=B4;viz_note_type=1.0;} }
                 }
                 case 1u: { viz_freq = B4; viz_note_type = 1.0; }
                 case 2u: {
                    let note_index = u32(floor(viz_measure_progress * 8.0));
                    switch(note_index) { case 0u,2u,4u,6u: {viz_freq=E5;viz_note_type=4.0;} case 1u,5u: {viz_freq=D5;viz_note_type=3.0;} case 3u: {viz_freq=C5;viz_note_type=2.0;} default: {viz_freq=A4;viz_note_type=0.0;} }
                 }
                 case 3u: { viz_freq = A4; viz_note_type = 0.0; }
                 default: {}
            }

            let freq_norm = (viz_freq - A4) / (F5 - A4);
            let bar_y = visualizer_center_y + (freq_norm - 0.5) * 0.4;
            
            if (abs(uv.y - bar_y) < 0.01) {
                let note_color = get_note_color(u32(viz_note_type));
                 if (abs(viz_progress - current_progress_in_pattern) < 0.01) {
                     let pulse = sin(u_time.time * 15.0) * 0.5 + 0.5;
                     color = mix(color, vec3<f32>(1.0, 1.0, 0.5) * pulse, 0.9);
                 } else {
                     color = mix(color, note_color, 0.7);
                 }
            }
        }
    }
    
    let ambient_glow = envelope * 0.15;
    color += vec3<f32>(ambient_glow * 0.3, ambient_glow * 0.5, ambient_glow * 0.8);
    textureStore(output, global_id.xy, vec4<f32>(color, 1.0));
}