
//This is an example shader that uses audio data to create a visualizer effect.
//Currently still TODO.
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(1) @binding(0) var<uniform> u_time: TimeUniform;
@group(2) @binding(0) var<uniform> params: Params;
@group(3) @binding(0) var<uniform> u_resolution: ResolutionUniform;

struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
    audio_data: array<vec4<f32>, 32>,  // 128 processed bands
    audio_raw: array<vec4<f32>, 32>,   // 128 raw bands
    bpm: f32,
};

struct TimeUniform {
    time: f32,
};

struct Params { 
    red_power: f32,
    green_power: f32,
    blue_power: f32,
    green_boost: f32,
    contrast: f32, 
    gamma: f32,
    glow: f32,
}
// Get processed audio value at any frequency (0-1 range)
fn getAudioValue(freq: f32) -> f32 {
    let idx = freq * 128.0;
    let i = u32(idx);
    let fract_part = idx - f32(i);
    
    // Safety bounds check
    if (i >= 128u) {
        return 0.0;
    }
    
    // Calculate which vec4 and component
    let vec_idx = i / 4u;
    let vec_component = i % 4u;
    
    var val1 = 0.0;
    if (vec_component == 0u) {
        val1 = u_resolution.audio_data[vec_idx].x;
    } else if (vec_component == 1u) {
        val1 = u_resolution.audio_data[vec_idx].y;
    } else if (vec_component == 2u) {
        val1 = u_resolution.audio_data[vec_idx].z;
    } else {
        val1 = u_resolution.audio_data[vec_idx].w;
    }
    
    // If at the last band, no interpolation needed
    if (i >= 127u) {
        return val1;
    }
    
    // Get next value for interpolation
    let next_i = i + 1u;
    let next_vec_idx = next_i / 4u;
    let next_vec_component = next_i % 4u;
    
    var val2 = 0.0;
    if (next_vec_component == 0u) {
        val2 = u_resolution.audio_data[next_vec_idx].x;
    } else if (next_vec_component == 1u) {
        val2 = u_resolution.audio_data[next_vec_idx].y;
    } else if (next_vec_component == 2u) {
        val2 = u_resolution.audio_data[next_vec_idx].z;
    } else {
        val2 = u_resolution.audio_data[next_vec_idx].w;
    }
    
    // Linear interpolation between bands
    return mix(val1, val2, fract_part);
}

// Get raw audio value at any frequency (0-1 range)
fn getRawAudioValue(freq: f32) -> f32 {
    let idx = freq * 128.0;
    let i = u32(idx);
    let fract_part = idx - f32(i);
    
    // Safety bounds check
    if (i >= 128u) {
        return 0.0;
    }
    
    // Calculate which vec4 and component
    let vec_idx = i / 4u;
    let vec_component = i % 4u;
    
    var val1 = 0.0;
    if (vec_component == 0u) {
        val1 = u_resolution.audio_raw[vec_idx].x;
    } else if (vec_component == 1u) {
        val1 = u_resolution.audio_raw[vec_idx].y;
    } else if (vec_component == 2u) {
        val1 = u_resolution.audio_raw[vec_idx].z;
    } else {
        val1 = u_resolution.audio_raw[vec_idx].w;
    }
    
    // If at the last band, no interpolation needed
    if (i >= 127u) {
        return val1;
    }
    
    // Get next value for interpolation
    let next_i = i + 1u;
    let next_vec_idx = next_i / 4u;
    let next_vec_component = next_i % 4u;
    
    var val2 = 0.0;
    if (next_vec_component == 0u) {
        val2 = u_resolution.audio_raw[next_vec_idx].x;
    } else if (next_vec_component == 1u) {
        val2 = u_resolution.audio_raw[next_vec_idx].y;
    } else if (next_vec_component == 2u) {
        val2 = u_resolution.audio_raw[next_vec_idx].z;
    } else {
        val2 = u_resolution.audio_raw[next_vec_idx].w;
    }
    
    // Linear interpolation between bands
    return mix(val1, val2, fract_part);
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let resolution = u_resolution.dimensions;
    let uv = tex_coords;
    var finalColor = vec3<f32>(0.05, 0.05, 0.1);
    if (fract(uv.x * 20.0) < 0.05 || fract(uv.y * 20.0) < 0.05) {
        finalColor = mix(finalColor, vec3<f32>(0.2, 0.2, 0.25), 0.2);
    }
    
    // Calculate total energy for effects
    var totalEnergy = 0.0;
    for (var i = 0; i < 32; i++) {
        totalEnergy += (u_resolution.audio_data[i].x + 
                       u_resolution.audio_data[i].y + 
                       u_resolution.audio_data[i].z + 
                       u_resolution.audio_data[i].w) / 128.0;
    }
    finalColor *= 1.0 + totalEnergy * 0.5;
    // ==================
    // Draw multi-band equalizer
    // ==================
    let eqBottom = 0.15;
    let eqHeight = 0.7;
    let bandWidth = 1.0 / 130.0;  // 128 bands + small margins
    let bandSpacing = bandWidth * 0.1;
    for (var i = 0; i < 128; i++) {
        // Calculate position for this frequency band
        let bandX = (f32(i) + 1.0) * bandWidth;
        // Get band energy (processed audio)
        let bandEnergy = getAudioValue(f32(i) / 128.0);
        
        let barHeight = bandEnergy * eqHeight;
        let barBottom = 1.0 - eqBottom - barHeight; // Invert Y
        let barTop = 1.0 - eqBottom; // Invert Y
        // Draw bar if we're within its boundaries
        if (uv.x >= bandX && uv.x < bandX + bandWidth - bandSpacing) {
            if (uv.y <= barTop && uv.y >= barBottom) {
                // Color gradient based on frequency and height
                let freqT = f32(i) / 128.0;  // 0-1 range for frequency
                let heightT = (barTop - uv.y) / max(barHeight, 0.001);  // 0-1 range for height within bar
                // Create a color spectrum from red (low) to blue (high) frequencies
                let baseColor = mix(
                    mix(
                        vec3<f32>(0.9, 0.1, 0.1),  // Low/bass (red)
                        vec3<f32>(0.9, 0.6, 0.1),  // Low-mid (orange)
                        min(freqT * 3.0, 1.0)
                    ),
                    mix(
                        vec3<f32>(0.1, 0.8, 0.2),  // Mid (green)
                        vec3<f32>(0.1, 0.2, 0.9),  // High (blue)
                        (freqT - 0.33) * 1.5
                    ),
                    min(max((freqT - 0.1) * 1.5, 0.0), 1.0)
                );
                
                // Add brightness gradient based on height
                let color = mix(
                    baseColor * 0.7,             // Darker at bottom
                    baseColor * 1.3,             // Brighter at top
                    1.0 - heightT
                );
                
                // Add animated stripes based on frequency
                let stripeSpeed = 1.0 + freqT * 3.0;
                let stripeWidth = 0.1 + bandEnergy * 0.2;
                if (fract(uv.y * 15.0 + u_time.time * stripeSpeed) < stripeWidth) {
                    finalColor = color * 1.3;
                } else {
                    finalColor = color;
                }
                
                // Add highlight at top of bar
                if (abs(uv.y - barBottom) < 0.01) {
                    finalColor = mix(finalColor, vec3<f32>(1.0), 0.5);
                }
            }
        }
    }
    
    // ==================
    // Draw baseline
    // ==================
    if (abs(uv.y - (1.0 - eqBottom)) < 0.002) {
        finalColor = vec3<f32>(0.4, 0.4, 0.5);
    }
    
    // ==================
    // RAW audio data
    // ==================
    let specY = 0.95;
    let specHeight = 0.03;
    
    if (uv.y > specY - specHeight && uv.y < specY) {
        for (var i = 0; i < 128; i++) {
            let specBandWidth = 1.0 / 130.0;
            let specBandX = (f32(i) + 1.0) * specBandWidth;
            let freqT = f32(i) / 128.0;
            
            // Use RAW audio data for this display
            let rawValue = getRawAudioValue(freqT);
            let peakHeight = rawValue * specHeight;
            
            // Draw peak line
            if (uv.x >= specBandX && uv.x < specBandX + specBandWidth - (specBandWidth * 0.1)) {
                // Draw frequency band (grows DOWN from specY)
                if (uv.y <= specY && uv.y >= specY - peakHeight) {
                    // Color based on frequency
                    let color = mix(
                        vec3<f32>(1.0, 0.2, 0.1),  // Low/bass (red)
                        mix(
                            vec3<f32>(1.0, 0.8, 0.1),  // Mid (yellow)
                            vec3<f32>(0.2, 0.4, 1.0),  // High (blue)
                            min(max((freqT - 0.3) * 1.5, 0.0), 1.0)
                        ),
                        min(freqT * 2.0, 1.0)
                    );
                    
                    finalColor = color;
                }
            }
        }
        
        // Draw baseline for spectrum
        if (abs(uv.y - specY) < 0.001) {
            finalColor = vec3<f32>(0.3, 0.3, 0.3);
        }
    }
    // ==================
    // BPM: TODO
    // ==================
    if (u_resolution.bpm > 0.0) {
        // Calculate beat phase (0-1 for each beat)
        let beat_duration = 60.0 / max(u_resolution.bpm, 1.0);
        let beat_phase = fract(u_time.time / beat_duration);
        let beat_pulse = smoothstep(0.0, 0.1, beat_phase) * smoothstep(1.0, 0.8, beat_phase);
        finalColor = mix(finalColor, vec3<f32>(1.0, 1.0, 1.0), beat_pulse * 0.3);
        if (beat_phase < 0.1) {
            // First 10% of the beat has increased saturation
            finalColor *= 1.3;
        }
    }
    // Get alpha from original texture
    let alpha = textureSample(tex, tex_sampler, uv).a;
    
    return vec4<f32>(finalColor, alpha);
}