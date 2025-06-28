@group(0) @binding(0) var<uniform> u_time: ComputeTimeUniform;
@group(1) @binding(0) var output_texture: texture_storage_2d<rgba16float, write>;
@group(2) @binding(0) var<uniform> u_audio_synth: AudioSynthUniform;
@group(3) @binding(0) var<storage, read_write> audio_buffer: array<f32>;

struct ComputeTimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}

struct AudioSynthUniform {
    note_frequencies: array<vec4<f32>, 4>,
    note_amplitudes: array<vec4<f32>, 4>,
    master_volume: f32,
    waveform_type: u32,
    active_note_count: u32,
}

const SAMPLE_RATE: f32 = 44100.0;
const AUDIO_BUFFER_SIZE: u32 = 1024u;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    let coord = vec2<i32>(global_id.xy);
    
    if coord.x >= i32(dims.x) || coord.y >= i32(dims.y) {
        return;
    }
    
    if (global_id.x == 0u && global_id.y == 0u) {
        let frequency = 440.0 + sin(u_time.time * 0.5) * 200.0;
        let amplitude = 0.1 * (1.0 + sin(u_time.time * 2.0) * 0.5);
        let waveform_type = f32((u32(u_time.time * 0.3) % 4u));
        
        audio_buffer[0] = frequency;
        audio_buffer[1] = amplitude;
        audio_buffer[2] = waveform_type;
    }
    
    textureStore(output_texture, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
}