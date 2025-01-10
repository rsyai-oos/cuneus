struct TimeUniform {
    time: f32,
};

@group(0) @binding(0) var<uniform> u_time: TimeUniform;

struct Params {
    base_color: vec3<f32>,
    _pad1: f32,
    rim_color: vec3<f32>,
    _pad2: f32,
    accent_color: vec3<f32>,
    _pad3: f32,
    
    light_intensity: f32,
    rim_power: f32,
    ao_strength: f32,
    env_light_strength: f32,
    
    iridescence_power: f32,
    falloff_distance: f32,
    vignette_strength: f32,
    num_cells: i32,
    
    rotation_speed: f32,
    wave_speed: f32,
    fold_intensity: f32,
    _pad4: f32,
};

@group(1) @binding(0) var<uniform> params: Params;

const PI: f32 = 3.14159265359;



fn waveSDF(p: vec2<f32>, time: f32, frequency: f32, amplitude: f32) -> f32 {
    var minDist: f32 = 1000.0;
    let waves = 3;
    for(var i: i32 = 0; i < waves; i = i + 1) {
        let phase = f32(i) * PI * 2.0 / f32(waves);
        let wave_time = time + phase;
        let wave_x = amplitude * sin(p.y * frequency + wave_time);
        let dist = abs(p.x - wave_x);
        let thickness = 0.1 + 0.05 * sin(p.y * 2.0 + time);
        minDist = min(minDist, dist - thickness);
    }
    
    return minDist;
}


fn getWaveLightPositions(time: f32, layer: f32) -> array<vec2<f32>, 4> {
    var positions: array<vec2<f32>, 4>;
    
    for(var i: i32 = 0; i < 4; i = i + 1) {
        let angle = f32(i) * PI * 0.5 + time * 0.5;
        let radius = 1.0 + 0.3 * sin(time * 0.7 + f32(i));
        positions[i] = vec2<f32>(
            cos(angle) * radius,
            sin(angle) * radius + 0.2 * sin(time + layer * 2.0)
        );
    }
    
    return positions;
}

fn getLightIntensity(uv: vec2<f32>, wave_dist: f32, layer: f32, time: f32) -> f32 {
    let lights = getWaveLightPositions(time, layer);
    var intensity: f32 = 0.0;
    for(var i: i32 = 0; i < 4; i = i + 1) {
        let light_pos = lights[i];
        let dist = length(uv - light_pos);
        let falloff = 1.0 / (1.0 + dist * 2.5);
        let interference = 0.2 + 0.1 * sin(layer * 5.0 + time * (0.5 + f32(i) * 0.1));
        intensity = intensity + falloff * interference;
    }
    //  ambient
    let ao = 0.5 - (layer * 0.1) * (1.0 + 0.2 * sin(layer * 8.0 + time));
    // rim lighting
    let normal = normalize(vec2<f32>(wave_dist, 1.0));
    let rim = params.fold_intensity - abs(dot(normalize(uv), normal));
    let rim_light = pow(rim, 4.0);
    
    //  shimmer effect
    let shimmer = sin(layer * 6.0 + time) * cos(layer * 4.0 - time * 0.8);
    intensity = intensity * (1.0 + 0.1 * shimmer);
    
    return intensity * ao * 5.0 + rim_light * 0.7;
}

fn getEnvironmentLight(uv: vec2<f32>, layer: f32, time: f32) -> f32 {
    let light_dir = normalize(vec2<f32>(cos(time * 0.5), sin(time * 0.5)));
    let normal = normalize(uv);
    var env_light = dot(normal, light_dir);
    env_light = env_light * 0.5 + 0.5;
    
    let depth = 1.0 - (layer / 1.5);
    let layer_effect = sin(layer * 3.0 + time) * 0.3 + 0.7;
    
    return mix(env_light, layer_effect, 0.5) * depth * 0.4;
}

fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    let bg = 0.4;
    var frag_color = vec4<f32>(bg, bg, bg, 1.0);
    let screen_size = vec2<f32>(1920.0, 1080.0);
    let t = u_time.time * 0.5;
    
    var i: f32 = params.light_intensity;
    while(i > 1.1) {
        let layer = i * 3.0;
        
        var uv = (FragCoord.xy - screen_size.xy * 0.5) / min(screen_size.x, screen_size.y);
        
        let wave = waveSDF(uv * params.falloff_distance, t, 3.0 + sin(layer + t) * 0.5, 0.5);
        let alpha = smoothstep(params.rim_power, 0.1, (wave + 0.01) * screen_size.y * 0.2);
        
        let light_intensity = getLightIntensity(uv, wave, i, t);
        let env_light = getEnvironmentLight(uv, layer, t);
        
        let color_intensity = 0.7 + 0.3 * sin(layer * 15.0 + t) * cos(layer * 10.0 - t * 0.5);
        let scaled_color = params.rim_color * vec3<f32>(1.0, 2.0, 3.0) + vec3<f32>(1.0, 1.0, 1.0);
        let hue = sin(i * 0.7 + vec4<f32>(scaled_color, 1.0) + wave * 0.5) * 0.25 + color_intensity;
        
        let lit_color = hue * (light_intensity + env_light);
        
        let depth_factor = 2.0 - (i - 0.003) / (1.5 - 0.003);
        let iridescence = sin(dot(uv, uv) * 4.0 + t) * 0.15 * depth_factor + 0.9;
        let lit_color_irid = lit_color * vec4<f32>(iridescence, iridescence * 0.98, iridescence * 1.02, 1.0);
        
        let mix_factor = 0.3 / (abs(wave) + 0.01) * (params.iridescence_power- depth_factor * 0.15);
        frag_color = mix(lit_color_irid, frag_color, alpha) *
                    mix(vec4<f32>(params.base_color,1.0), hue + alpha * params.wave_speed * (uv.x / (abs(wave) + 0.001) + light_intensity), mix_factor);
        
        i = i -1.0;
    }
    
    frag_color = vec4<f32>(frag_color.rgb * 1.1, 1.0);
    
    let vignette_uv = (FragCoord.xy - 0.5 * screen_size) / screen_size.y;
    let vignette = 1.0 - dot(vignette_uv, vignette_uv) * 0.25;
    
    let final_color = gamma(frag_color.rgb * vignette, 0.45);
    return vec4<f32>(final_color, frag_color.a);
}