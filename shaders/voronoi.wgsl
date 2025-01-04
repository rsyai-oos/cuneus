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

const PI: f32 = 3.14159265358979323846;

fn rotate(angle: f32) -> mat2x2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return mat2x2<f32>(
        vec2<f32>(c, -s),
        vec2<f32>(s, c)
    );
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn getVoronoiCellCenter(index: i32, time: f32) -> vec2<f32> {
    let angle = f32(index) * PI * 0.5;
    return vec2<f32>(
        cos(angle) * (1.0 + 0.5 * sin(time * params.wave_speed + f32(index))),
        sin(angle) * (1.0 + 0.5 * cos(time * params.wave_speed * 0.8 + f32(index)))
    ) * 1.8;
}

fn voronoi(p: vec2<f32>, time: f32) -> f32 {
    var minDist: f32 = 1e5;
    let rotP = rotate(time * params.rotation_speed) * p;
    
    for(var i: i32 = 0; i < params.num_cells; i = i + 1) {
        let cellCenter = getVoronoiCellCenter(i, time);
        let dist = abs(rotP.x - cellCenter.x) + abs(rotP.y - cellCenter.y);
        minDist = min(minDist, dist);
    }
    
    return minDist - 0.3;
}

fn getLightIntensity(uv: vec2<f32>, voronoiDist: f32, layer: f32, time: f32, angle: f32) -> f32 {
    var intensity: f32 = 0.0;
    
    for(var i: i32 = 0; i < params.num_cells; i = i + 1) {
        let lightPos = getVoronoiCellCenter(i, time);
        let dist = length(uv - lightPos);
        let falloff = 1.0 / (1.0 + dist * params.falloff_distance);
        let weight = 0.1 + 0.05 * sin(layer * (5.0 + f32(i)) + time * (0.2 + f32(i) * 0.03));
        intensity = intensity + falloff * weight;
    }

    let ao = 0.6 - (layer * params.ao_strength) * (1.0 + 0.2 * sin(layer * 12.0 + time * 0.7));
    let normal = normalize(uv);
    let rim = 1.3 - abs(dot(normalize(uv), normal));
    let rimLight = pow(rim, params.rim_power);
    
    let shimmer = sin(layer * 7.0 + time * 1.1) * cos(layer * 4.0 - time * 0.8);
    intensity = intensity * (1.0 + 0.08 * shimmer);

    return intensity * ao * params.light_intensity + rimLight * 0.2;
}

fn getEnvironmentLight(uv: vec2<f32>, angle: f32, layer: f32, time: f32) -> f32 {
    let lightDir = normalize(vec2<f32>(cos(time * 0.7), sin(time * 0.7)));
    let normal = normalize(uv);
    var envLight = dot(normal, lightDir);
    envLight = envLight * 0.5 + 0.5;

    let depth = 1.0 - (layer / 1.5);
    let layerEffect = sin(layer * 2.5 + time * 0.6) * 0.3 + 0.7;

    return mix(envLight, layerEffect, 0.5) * depth * params.env_light_strength;
}

fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}

@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>) -> @location(0) vec4<f32> {
    var hue: vec4<f32>;
    let bg = params.base_color.r;
    var fragColor = vec4<f32>(bg, bg, bg, 1.0);
    let screen_size = vec2<f32>(1920.0, 1080.0);
    let t = u_time.time * params.wave_speed;
    var angle: f32 = 0.25;
    let foldPattern = cos(t * 0.5) * PI * 0.25 * 0.5;
    let globalLight = 1.0;
    let asd = 0.1;

    var i: f32 = 1.5;
    while(i > 0.003) {
        let layer = i * 1.0;
        let fold = sin(t + layer * 0.2) * cos(t * 0.5 + layer * 0.1);
        let alternatingFold = sign(sin(layer * 0.5)) * sin(t + i * 2.0);

        // Fixed UV calculation for proper centering and aspect ratio
        var uv = FragCoord.xy/screen_size.xy;  // Normalize to [0,1]
        uv = uv * 2.0 - 1.0;           // Convert to [-1,1]
        uv.x *= screen_size.x/screen_size.y;  // Correct aspect ratio
        uv = rotate(i + (angle + alternatingFold) + foldPattern) * uv;

        let voro = voronoi(uv * 8.0, t);
        let alpha = smoothstep(0.0, 0.1, (voro + 0.01) * screen_size.y * 0.2);
        
        let lightIntensity = getLightIntensity(uv, voro, i, t, angle);
        let envLight = getEnvironmentLight(uv, angle, i, t);

        let des = 0.75;
        let des2 = 0.55;
        
        let colorIntensity = 0.7 +
            0.3 * sin(layer * 18.37 + t * 1.2) *
            cos(layer * 13.54 - t * 0.4) *
            sin(angle * 2.0 + t * 0.7);

        let colorShift = 0.25;
        let scaled_color = params.rim_color * vec3<f32>(1.0, 2.0, 3.0) + vec3<f32>(1.0, 1.0, 1.0);
        hue = sin(i / des2 + angle / des + vec4<f32>(scaled_color, 1.0) + fold * params.fold_intensity) * colorShift + colorIntensity;

        let litColor = hue * (lightIntensity * globalLight + envLight);

        let depthFactor = 2.0 - (i - 0.003) / (1.5 - 0.003);
        let iridescence = sin(dot(uv, uv) * 6.0 + t * 1.4) * params.iridescence_power * depthFactor + 0.9;
        let litColorIrid = litColor * vec4<f32>(iridescence, iridescence * 0.98, iridescence * 1.02, 1.0);

        let des4 = 0.3;
        let des12 = 0.2;

        let mixFactor = des4 / (abs(voro) + 0.01) * (0.4 - depthFactor * 0.15);
        fragColor = mix(litColorIrid, fragColor, alpha) *
                   mix(vec4<f32>(1.0), hue + 1.5 * alpha * des12 * (uv.x / (abs(voro) + 0.001) + lightIntensity), mixFactor);

        i = i - asd;
    }

    let des2_final = 1.1;
    fragColor = vec4<f32>(fragColor.rgb * (des2_final + globalLight * 0.1), 1.0);

    let vignetteUV = (FragCoord.xy - 0.5 * screen_size) / screen_size.y;
    let vignette = 1.0 - dot(vignetteUV, vignetteUV) * params.vignette_strength;
    
    let finalColor = gamma(fragColor.rgb * vignette, 0.41);
    return vec4<f32>(finalColor, fragColor.a);
}