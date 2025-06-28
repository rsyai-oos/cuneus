// This example demonstrates a how to generate audio using cunes via compute shaders
@group(0) @binding(0) var<uniform> u_time: ComputeTimeUniform;
@group(1) @binding(0) var output_texture: texture_storage_2d<rgba16float, write>;
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
    scale_type: u32,
    visualizer_mode: u32,
    bass_boost: f32,
    melody_octave: f32,
    harmony_mix: f32,
    reverb_amount: f32,
    volume: f32,
}

const PI = 3.14159265359;

// Convert time and tempo into musical notes
fn get_melody_note(time: f32, tempo: f32, octave: f32) -> vec2<f32> {
    let beat_duration = 60.0 / tempo;
    let note_time = time / beat_duration;
    let note_index = u32(note_time) % 8u;
    
    // Some arbitrary scales for demonstration
    let scale_notes = array<f32, 8>(1.0, 9.0/8.0, 5.0/4.0, 4.0/3.0, 3.0/2.0, 5.0/3.0, 15.0/8.0, 2.0);
    let base_freq = 261.63 * pow(2.0, octave - 4.0);
    
    let frequency = base_freq * scale_notes[note_index];
    
    // Note envelope: fade out near end of each note
    let note_progress = fract(note_time);
    let amplitude = select(0.3, 0.3 * (1.0 - note_progress), note_progress > 0.7);
    
    return vec2<f32>(frequency, amplitude);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    let coord = vec2<i32>(global_id.xy);
    
    if coord.x >= i32(dims.x) || coord.y >= i32(dims.y) {
        return;
    }
    
    let uv = vec2<f32>(f32(coord.x) / f32(dims.x), f32(coord.y) / f32(dims.y));
    
    let melody = get_melody_note(u_time.time, params.tempo, params.melody_octave);
    let frequency = melody.x;
    let amplitude = melody.y * params.volume;
    
    // GPU → CPU communication: write audio data for real-time playback
    if global_id.x == 0u && global_id.y == 0u {
        audio_buffer[0] = frequency;
        audio_buffer[1] = amplitude;
        audio_buffer[2] = 0.0;
    }
    
    var color = vec3<f32>(0.0);
    
    let x = uv.x;
    let y = uv.y;
    
    // simple visualization
    let bar_width = 0.1;
    let freq_normalized = (frequency - 200.0) / 800.0;
    let bar_x = freq_normalized;
    
    if abs(x - bar_x) < bar_width * 0.5 && y < amplitude * 2.0 {
        let hue = freq_normalized;
        if hue < 0.5 {
            color = vec3<f32>(1.0 - hue * 2.0, hue * 2.0, 0.0);
        } else {
            color = vec3<f32>(0.0, 2.0 - hue * 2.0, (hue - 0.5) * 2.0);
        }
    }
    
    let bg_gradient = vec3<f32>(0.05, 0.02, 0.1) * (1.0 - uv.y);
    color = max(color, bg_gradient);
    
    textureStore(output_texture, coord, vec4<f32>(color, 1.0));
}