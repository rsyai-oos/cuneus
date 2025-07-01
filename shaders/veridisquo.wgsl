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
    reverb_mix: f32,
    chorus_rate: f32,
    _padding: f32,
};
@group(2) @binding(0) var<uniform> u_song: SongParams;

const PI = 3.14159265359;

// --- Note Frequencies ---
const F5=698.46; const E5=659.25; const D5=587.33; const C5=523.25; const B4=493.88; const A4=440.00;
// --- Bass Frequencies ---
const F3=F5/4.0; const B2=B4/4.0; const E3=E5/4.0; const A2=A4/4.0;

fn legato(freq_from:f32, freq_to:f32, progress:f32) -> f32 {
    let start_point = 1.0 - clamp(u_song.crossfade, 0.0, 1.0);
    if (progress <= start_point) { return freq_from; }
    let transition = smoothstep(start_point, 1.0, progress);
    return mix(freq_from, freq_to, transition);
}

fn get_note_color(note_type: f32) -> vec3<f32> {
    let note_id = u32(note_type);
    switch note_id {
        case 0u: { return vec3<f32>(1.0, 0.3, 0.3); }  // A4 - Red
        case 1u: { return vec3<f32>(1.0, 0.6, 0.0); }  // B4 - Orange  
        case 2u: { return vec3<f32>(1.0, 1.0, 0.2); }  // C5 - Yellow
        case 3u: { return vec3<f32>(0.3, 1.0, 0.3); }  // D5 - Green
        case 4u: { return vec3<f32>(0.2, 0.7, 1.0); }  // E5 - Blue
        case 5u: { return vec3<f32>(0.8, 0.3, 1.0); }  // F5 - Purple
        default: { return vec3<f32>(0.5, 0.5, 0.5); }  // Default - Gray
    }
}
// Returns frequency and note_type for each measure
fn get_measure_preview(measure: u32) -> vec2<f32> {
    switch measure {
        case 0u, 4u: { return vec2<f32>(F5, 5.0); }    // F5 - Purple
        case 1u, 5u: { return vec2<f32>(B4, 1.0); }    // B4 - Orange
        case 2u: { return vec2<f32>(E5, 4.0); }        // E5 - Blue  
        case 3u, 7u: { return vec2<f32>(A4, 0.0); }    // A4 - Red
        case 6u: { return vec2<f32>(E5, 4.0); }        // E5 - Blue (fast run)
        default: { return vec2<f32>(440.0, 0.0); }     // Default A4
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dims=textureDimensions(output); if(global_id.x>=dims.x||global_id.y>=dims.y){return;}
    var melody_freq_visualizer=0.0; var envelope_visualizer=0.0; var note_type_visualizer=0.0;

    if (global_id.x == 0u && global_id.y == 0u) {
        let adjusted_time = u_time.time * u_song.tempo_multiplier;
        let measure_duration = (60.0 / 107.0) * 4.0;
        let total_pattern_duration = measure_duration * 8.0;
        let loop_time = adjusted_time % total_pattern_duration;
        let measure = u32(loop_time / measure_duration);
        let progress_in_measure = fract(loop_time / measure_duration);

        var melody_freq=0.0; var melody_amp=1.0; var bass_freq=0.0; var bass_amp=0.7;

        let phrase_dur = 0.25;
        let short_hold_dur = 0.125;
        let phrase2_start = phrase_dur + short_hold_dur;
        let phrase2_end = phrase2_start + phrase_dur;
        let hold_end_point = 0.875; // Hold note for 7/8 of the measure, leaving a 1/8 rest.

        switch (measure) {
            case 0u, 4u: { // Measures 1 & 5
                bass_freq = F3;
                if(progress_in_measure<phrase_dur){let p=progress_in_measure/phrase_dur;let n=fract(p*4.0);switch(u32(floor(p*4.0))){case 0u:{melody_freq=legato(F5,E5,n);note_type_visualizer=5.0;}case 1u:{melody_freq=legato(E5,F5,n);note_type_visualizer=4.0;}case 2u:{melody_freq=legato(F5,D5,n);note_type_visualizer=5.0;}default:{melody_freq=D5;note_type_visualizer=3.0;}}}
                else if(progress_in_measure<phrase2_start){melody_freq=D5;note_type_visualizer=3.0;}
                else if(progress_in_measure<phrase2_end){let p=(progress_in_measure-phrase2_start)/phrase_dur;let n=fract(p*4.0);switch(u32(floor(p*4.0))){case 0u:{melody_freq=legato(F5,E5,n);note_type_visualizer=5.0;}case 1u:{melody_freq=legato(E5,F5,n);note_type_visualizer=4.0;}case 2u:{melody_freq=legato(F5,B4,n);note_type_visualizer=5.0;}default:{melody_freq=B4;note_type_visualizer=1.0;}}}
                else{melody_freq=B4;note_type_visualizer=1.0;}
            }
            case 1u, 5u: { // Measures 2 & 6: Hold B4, then short rest
                melody_freq = B4; bass_freq = B2; note_type_visualizer = 1.0;
                if (progress_in_measure < hold_end_point) {
                    let hold_progress = progress_in_measure / hold_end_point;
                    let sustain_level = mix(1.0, 0.7, hold_progress);
                    let tremolo = sin(adjusted_time * 8.0) * 0.05;
                    melody_amp = sustain_level + tremolo;
                } else { melody_amp = 0.0; bass_amp = 0.0; }
            }
            case 2u: { // Measure 3
                bass_freq = E3;
                if(progress_in_measure<phrase_dur){let p=progress_in_measure/phrase_dur;let n=fract(p*4.0);switch(u32(floor(p*4.0))){case 0u:{melody_freq=legato(E5,D5,n);note_type_visualizer=4.0;}case 1u:{melody_freq=legato(D5,E5,n);note_type_visualizer=3.0;}case 2u:{melody_freq=legato(E5,C5,n);note_type_visualizer=4.0;}default:{melody_freq=C5;note_type_visualizer=2.0;}}}
                else if(progress_in_measure<phrase2_start){melody_freq=C5;note_type_visualizer=2.0;}
                else if(progress_in_measure<phrase2_end){let p=(progress_in_measure-phrase2_start)/phrase_dur;let n=fract(p*4.0);switch(u32(floor(p*4.0))){case 0u:{melody_freq=legato(E5,D5,n);note_type_visualizer=4.0;}case 1u:{melody_freq=legato(D5,E5,n);note_type_visualizer=3.0;}case 2u:{melody_freq=legato(E5,A4,n);note_type_visualizer=4.0;}default:{melody_freq=A4;note_type_visualizer=0.0;}}}
                else{melody_freq=A4;note_type_visualizer=0.0;}
            }
            case 3u, 7u: { // Measures 4 & 8: Hold A4, then short rest
                melody_freq = A4; bass_freq = A2; note_type_visualizer = 0.0;
                if (progress_in_measure < hold_end_point) {
                    let hold_progress = progress_in_measure / hold_end_point;
                    let sustain_level = mix(1.0, 0.75, hold_progress);
                    let tremolo = sin(adjusted_time * 8.0) * 0.05;
                    melody_amp = sustain_level + tremolo;
                } else { melody_amp = 0.0; bass_amp = 0.0; } // Silence for anticipation
            }
            case 6u: { // Measure 7: The FAST 8-note run
                bass_freq = E3; let run_dur=0.5;
                if(progress_in_measure<run_dur){let p=progress_in_measure/run_dur;let n=fract(p*8.0);switch(u32(floor(p*8.0))){case 0u:{melody_freq=legato(E5,D5,n);note_type_visualizer=4.0;}case 1u:{melody_freq=legato(D5,E5,n);note_type_visualizer=3.0;}case 2u:{melody_freq=legato(E5,C5,n);note_type_visualizer=4.0;}case 3u:{melody_freq=legato(C5,E5,n);note_type_visualizer=2.0;}case 4u:{melody_freq=legato(E5,D5,n);note_type_visualizer=4.0;}case 5u:{melody_freq=legato(D5,E5,n);note_type_visualizer=3.0;}case 6u:{melody_freq=legato(E5,A4,n);note_type_visualizer=4.0;}default:{melody_freq=A4;note_type_visualizer=0.0;}}}
                else{melody_freq=A4;note_type_visualizer=0.0;}
            }
            default: {}
        }
        if(measure==1u||measure==3u||measure==5u||measure==7u){if(progress_in_measure>=hold_end_point){bass_amp=0.0;}else{bass_amp=0.7*(1.0-progress_in_measure/hold_end_point*0.5);}}
        melody_freq_visualizer=melody_freq;envelope_visualizer=(melody_amp+bass_amp)*u_song.volume;
        let final_melody_freq=melody_freq*pow(2.0,u_song.octave_shift);let final_bass_freq=bass_freq*pow(2.0,u_song.octave_shift);
        
        var final_melody_amp=melody_amp*u_song.volume;
        var final_bass_amp=bass_amp*u_song.volume*0.7;
        
        if(u_song.reverb_mix > 0.0) {
            let reverb_delay = sin(adjusted_time - 0.2) * u_song.reverb_mix * 0.3;
            final_melody_amp += reverb_delay * final_melody_amp;
            final_bass_amp += reverb_delay * final_bass_amp;
        }
        
        var chorus_freq_mod = 1.0;
        if(u_song.chorus_rate > 0.0) {
            chorus_freq_mod = 1.0 + sin(adjusted_time * u_song.chorus_rate) * 0.02;
        }
        
        audio_buffer[0]=melody_freq_visualizer;audio_buffer[1]=envelope_visualizer;audio_buffer[2]=f32(u_song.waveform_type);
        audio_buffer[3]=final_melody_freq*chorus_freq_mod;audio_buffer[4]=final_melody_amp;
        audio_buffer[5]=final_bass_freq*chorus_freq_mod;audio_buffer[6]=final_bass_amp;
        for(var i=2u;i<16u;i++){audio_buffer[3u+i*2u]=0.0;audio_buffer[3u+i*2u+1u]=0.0;}
    }
    let frequency=audio_buffer[0];let envelope=audio_buffer[1];let uv=vec2<f32>(global_id.xy)/vec2<f32>(dims);
    
    var color=vec3<f32>(0.02,0.01,0.08);
    
    let visualizer_center_y = 0.5;
    let pattern_width = 0.8;
    let pattern_start_x = 0.1;
    let pattern_height = 0.4;
    
    let measure_duration = (60.0 / 107.0) * 4.0;
    let total_pattern_duration = measure_duration * 8.0;
    let song_time = u_time.time * u_song.tempo_multiplier;
    let current_measure = u32((song_time % total_pattern_duration) / measure_duration);
    
    for (var measure = 0u; measure < 8u; measure++) {
        let measure_width = pattern_width / 8.0;
        let measure_x = pattern_start_x + f32(measure) * measure_width;
        
        if uv.x >= measure_x && uv.x <= measure_x + measure_width * 0.9 {
            let measure_info = get_measure_preview(measure);
            let measure_freq = measure_info.x;
            let measure_note_type = measure_info.y;
            
            let freq_norm = (measure_freq - 440.0) / (698.46 - 440.0);
            let bar_height = mix(0.1, pattern_height, freq_norm);
            let bar_bottom = visualizer_center_y - bar_height * 0.5;
            let bar_top = visualizer_center_y + bar_height * 0.5;
            
            if uv.y >= bar_bottom && uv.y <= bar_top {
                if measure == current_measure && envelope > 0.01 {
                    let pulse = sin(u_time.time * 8.0) * 0.4 + 0.8;
                    color = vec3<f32>(1.0, 0.9, 0.2) * pulse;
                } else {
                    let note_color = get_note_color(measure_note_type);
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
        if spectrum_bar_height < spectrum_intensity && envelope > 0.01 {
            color += vec3<f32>(spectrum_intensity * 0.8, spectrum_intensity * 0.4, 0.1);
        }
    }
    
    let progress_bar_height=0.02;
    if(uv.y>0.95&&uv.y<0.95+progress_bar_height){
        let measure_duration=(60.0/107.0)*4.0;
        let total_pattern_duration=measure_duration*8.0;
        let song_progress=(u_time.time*u_song.tempo_multiplier%total_pattern_duration)/total_pattern_duration;
        if(uv.x<song_progress){
            color=mix(color,vec3<f32>(0.0,0.7,1.0),0.8);
        }else{
            color=mix(color,vec3<f32>(0.15,0.15,0.3),0.8);
        }
    }
    let ambient_glow = envelope * 0.1;
    color += vec3<f32>(ambient_glow * 0.1, ambient_glow * 0.3, ambient_glow * 0.1);
    
    textureStore(output,global_id.xy,vec4<f32>(color,1.0));
}