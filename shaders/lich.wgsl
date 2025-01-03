//// inspired by https://www.shadertoy.com/view/3sl3WH
///
@group(0) @binding(0) var prev_frame: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(3) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<i32>>;

struct TimeUniform {
    time: f32,
};
@group(1) @binding(0)
var<uniform> time_data: TimeUniform;

struct Params {
    cloud_density: f32,
    lightning_intensity: f32,
    branch_count: f32,
    feedback_decay: f32,
    base_color: vec3<f32>,
    _pad1: f32,
    color_shift: f32,
    spectrum_mix: f32,
    _pad2: vec2<f32>,
};
@group(2) @binding(0)
var<uniform> params: Params;

const ATOMIC_SCALE: f32 = 2048.0;
const SPECTRUM_SIZE: i32 = 45;

fn wavelength_to_rgb(wave: f32) -> vec3<f32> {
    let x = (wave - 380.0) / 10.0;
    let factor = select(0.0, 
                       select(1.0 - (x - 750.0) / 50.0,
                             select(1.0, 1.0 - (380.0 - x) / 50.0, x >= 380.0),
                             x <= 750.0),
                       x <= 800.0);
    
    var r = select(0.0,
                  select(1.0,
                        select((x - 440.0) / (510.0 - 440.0),
                              select(1.0,
                                    (750.0 - x) / (750.0 - 610.0),
                                    x >= 610.0),
                              x >= 510.0),
                        x >= 440.0),
                  x >= 380.0);
    
    var g = select(0.0,
                  select((x - 440.0) / (490.0 - 440.0),
                        select(1.0,
                              (580.0 - x) / (580.0 - 510.0),
                              x >= 510.0),
                        x >= 490.0),
                  x >= 440.0);
    
    var b = select(0.0,
                  select(1.0,
                        (490.0 - x) / (490.0 - 440.0),
                        x >= 440.0),
                  x >= 380.0);
    
    return vec3<f32>(r, g, b) * factor;
}

fn IHash(a: i32) -> i32 {
    var x = a;
    x = (x ^ 61) ^ (x >> 16);
    x = x + (x << 3);
    x = x ^ (x >> 4);
    x = x * 0x27d4eb; 
    x = x ^ (x >> 15);
    return x;
}

fn Hash(a: i32) -> f32 {
    return f32(IHash(a)) / f32(0x7FFFFFFF);
}

fn rand4(seed: i32) -> vec4<f32> {
    return vec4<f32>(
        Hash(seed ^ 348593),
        Hash(seed ^ 859375),
        Hash(seed ^ 625384),
        Hash(seed ^ 253625)
    );
}

fn rand2(seed: i32) -> vec2<f32> {
    return vec2<f32>(
        Hash(seed ^ 348593),
        Hash(seed ^ 859375)
    );
}

fn randn(randuniform: vec2<f32>) -> vec2<f32> {
    var r = randuniform;
    r.x = sqrt(-2.0 * log(1e-9 + abs(r.x)));
    r.y = r.y * 6.28318;
    return r.x * vec2<f32>(cos(r.y), sin(r.y));
}

fn lineDist(a: vec2<f32>, b: vec2<f32>, uv: vec2<f32>) -> f32 {
    return length(uv-(a+normalize(b-a)*min(length(b-a),max(0.0,dot(normalize(b-a),(uv-a))))));
}

fn process_color(base_color: vec3<f32>, wave: f32) -> vec3<f32> {
    let spectral = wavelength_to_rgb(wave * 380.0 + 400.0);
    let mixed = mix(base_color, spectral, params.spectrum_mix);
    
    let temp_adjust = vec3<f32>(
        1.0 + 0.0 * 0.2, 
        1.0,                            
        1.0 - 0.0 * 0.1  
    );
    
    return mixed * temp_adjust;
}

@fragment
fn fs_pass1(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(prev_frame));
    let pixel_pos = vec2<i32>(FragCoord.xy);
    let pixel_index = pixel_pos.y * i32(dimensions.x) + pixel_pos.x;
    
    atomicStore(&atomic_buffer[pixel_index * 3], 0);
    atomicStore(&atomic_buffer[pixel_index * 3 + 1], 0);
    atomicStore(&atomic_buffer[pixel_index * 3 + 2], 0);
    
    let uv = (FragCoord.xy * 2.0 - dimensions.xy) / dimensions.y;
    var ds = 1e4;
    
    for(var q = 0; q < 1; q = q + 1) {
        let anim_frame = i32(time_data.time * 20.0);
        let f = anim_frame + 123457 * (q + 1); 
        var seed = i32(params.cloud_density);
        
        var a = vec2<f32>(0.0, 1.0);
        var b = vec2<f32>(0.2, 0.7) + 0.4 * randn(rand2(seed ^ 859375)) / 8.0;
        
        // Apply branch_count parameter to control branching intensity
        let branch_factor = 30.0 * params.branch_count;
        for(var k = 0; k < i32(branch_factor); k = k + 1) {
            let l = length(b - a);
            
            let c = (a + b) / 2.0 + l * randn(rand2(seed ^ 859375)) / 8.0;
            let d = b * 1.9 - a * 0.9 + l * randn(rand2(seed ^ 935375)) / 4.0;
            let e = b * 1.9 - a * 0.9 + l * randn(rand2(seed ^ 687643)) / 4.0;
            
            let j = 1.0 + 0.5 * rand4(seed ^ IHash(anim_frame * 574595 ^ q));
            
            let d0 = lineDist(a, c, uv) * j.x;
            let d1 = lineDist(c, b, uv) * j.y;
            let d2 = lineDist(b, d, uv) * j.z;
            let d3 = lineDist(b, e, uv) * j.w;
            
            if(d0 < min(d1, min(d2, d3))) {
                b = c;
                seed = IHash(seed ^ 796489);
            } else if(d1 < min(d2, d3)) {
                a = c;
                seed = IHash(seed ^ 879235);
            } else if(d2 < d3) {
                a = b;
                b = d;
                seed = IHash(seed ^ 574595);
            } else {
                a = b;
                b = e;
                seed = IHash(seed ^ 630658);
            }
        }
        
        ds = min(ds, lineDist(a, b, uv));
    }
    
    let intensity = max(0.0, 1.0 - ds * dimensions.y / params.color_shift) * params.lightning_intensity;
    if(intensity > 0.001) {
        let wave = Hash(i32(time_data.time * 1000.0));
        let color = process_color(params.base_color, wave) * intensity;
        
        let scaled_color = vec3<i32>(floor(ATOMIC_SCALE * color));
        atomicAdd(&atomic_buffer[pixel_index * 3], scaled_color.x);
        atomicAdd(&atomic_buffer[pixel_index * 3 + 1], scaled_color.y);
        atomicAdd(&atomic_buffer[pixel_index * 3 + 2], scaled_color.z);
    }
    
    let current = vec3<f32>(
        f32(atomicLoad(&atomic_buffer[pixel_index * 3])),
        f32(atomicLoad(&atomic_buffer[pixel_index * 3 + 1])),
        f32(atomicLoad(&atomic_buffer[pixel_index * 3 + 2]))
    ) / ATOMIC_SCALE;
    
    let prev = textureSample(prev_frame, tex_sampler, tex_coords);
    return vec4<f32>(current + prev.rgb * params.feedback_decay, 1.0);
}

@fragment
fn fs_pass2(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(prev_frame, tex_sampler, tex_coords);
}

@fragment
fn fs_pass3(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(prev_frame, tex_sampler, tex_coords);
    let exposed = pow(color.rgb * (1.0 + 0.0), vec3<f32>(1.0/2.2));
    return vec4<f32>(exposed, color.a);
}