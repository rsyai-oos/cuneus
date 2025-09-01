// Simple Kuwahara Filter
// MIT License Enes Altun, 2025
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

// Output and parameters
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> params: KuwaharaParams;

// Channel textures for media input
@group(2) @binding(0) var channel0: texture_2d<f32>;
@group(2) @binding(1) var channel0_sampler: sampler;

// Multi-pass inputs (Group 3 has input_texture0, input_sampler0, input_texture1, input_sampler1, input_texture2, input_sampler2)
@group(3) @binding(0) var input_texture0: texture_2d<f32>;
@group(3) @binding(1) var input_sampler0: sampler;
@group(3) @binding(2) var input_texture1: texture_2d<f32>;
@group(3) @binding(3) var input_sampler1: sampler;
@group(3) @binding(4) var input_texture2: texture_2d<f32>;
@group(3) @binding(5) var input_sampler2: sampler;

struct KuwaharaParams {
    radius: f32,
    q: f32,
    alpha: f32,
    filter_strength: f32,
    sigma_d: f32,
    sigma_r: f32,
    edge_threshold: f32,
    color_enhance: f32,
    filter_mode: i32,
    show_tensors: i32,
    _pad1: u32,
    _pad2: u32, 
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

// Constants
const PI: f32 = 3.14159265359;

// Fabrice's fast blur for tensor smoothing
// https://www.shadertoy.com/view/ltScRG
const BLUR_SAMPLES: i32 = 35;
const BLUR_LOD: i32 = 2; 
const BLUR_SLOD: i32 = 4;
const BLUR_SIGMA: f32 = 8.75;

fn gaussian_weight(i: vec2f, sigma: f32) -> f32 {
    let sigma_i = i / sigma;
    return exp(-0.5 * dot(sigma_i, sigma_i)) / (6.28 * sigma * sigma);
}

fn blur_tensor(uv: vec2f, texel_size: vec2f) -> vec3f {
    var result = vec3f(0.0);
    var total_weight = 0.0;
    let s = BLUR_SAMPLES / BLUR_SLOD;
    let effective_sigma = params.sigma_r * 2.5;
    let blur_lod = max(0.0, params.sigma_d - 0.5);
    
    for (var i = 0; i < s * s; i++) {
        let d = vec2f(f32(i % s), f32(i / s)) * f32(BLUR_SLOD) - f32(BLUR_SAMPLES) / 2.0;
        let weight = gaussian_weight(d, effective_sigma);
        let sample_uv = clamp(uv + texel_size * d, vec2f(0.0), vec2f(1.0));
        let tensor_data = textureSampleLevel(input_texture0, input_sampler0, sample_uv, blur_lod);
        
        result += tensor_data.xyz * weight;
        total_weight += weight;
    }
    
    return result / total_weight;
}

// region calculation for classical Kuwahara
fn calc_region_stats(uv: vec2f, lower: vec2i, upper: vec2i, texel_size: vec2f) -> vec2f {
    var color_sum = vec3f(0.0);
    var color_variance_sum = vec3f(0.0);
    var count = 0;
    
    for (var j = lower.y; j <= upper.y; j++) {
        for (var i = lower.x; i <= upper.x; i++) {
            let offset = vec2f(f32(i), f32(j)) * texel_size;
            let sample_uv = clamp(uv + offset, vec2f(0.0), vec2f(1.0));
            let sample_color = get_input_color(sample_uv);
            
            color_sum += sample_color;
            color_variance_sum += sample_color * sample_color;
            count++;
        }
    }
    
    if (count > 0) {
        let mean_color = color_sum / f32(count);
        
        // RGB-based variance
        let rgb_variance = color_variance_sum / f32(count) - (mean_color * mean_color);
        let total_variance = rgb_variance.r + rgb_variance.g + rgb_variance.b;
        
        let mean_luminance = dot(mean_color, vec3f(0.299, 0.587, 0.114));
        let combined_variance = total_variance * 0.7 + (rgb_variance.r * 0.299 + rgb_variance.g * 0.587 + rgb_variance.b * 0.114) * 0.3;
        
        return vec2f(mean_luminance, combined_variance);
    }
    return vec2f(0.0, 999999.0);
}

fn ACESFilm(color: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), vec3f(0.0), vec3f(1.0));
}

fn saturate(rgb: vec3f, adjustment: f32) -> vec3f {
    let W = vec3f(0.2125, 0.7154, 0.0721);
    let intensity = vec3f(dot(rgb, W));
    return mix(intensity, rgb, adjustment);
}

fn get_input_color(uv: vec2f) -> vec3f {
    // Check uploaded textures (channel0) 
    let channel_dims = textureDimensions(channel0);
    if (channel_dims.x > 1 && channel_dims.y > 1) {
        return textureSampleLevel(channel0, channel0_sampler, uv, 0.0).rgb;
    }
    let center = vec2f(0.5);
    let dist = distance(uv, center);
    let circle = smoothstep(0.2, 0.21, dist);
    return mix(vec3f(0.8, 0.4, 0.2), vec3f(0.1, 0.1, 0.2), circle);
}

// Structure Tensor Computation (Pass 1)
@compute @workgroup_size(16, 16, 1)
fn structure_tensor(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let texel_size = 1.0 / vec2f(dims);

    // Sobel gradient computation: sigma_d to control derivative kernel size
    let d = texel_size * params.sigma_d;
    
    // Sobel X kernel: [-1,-2,-1; 0,0,0; 1,2,1] / 4
    let sobel_x = (
        -1.0 * get_input_color(clamp(uv + vec2f(-d.x, -d.y), vec2f(0.0), vec2f(1.0))) +
        -2.0 * get_input_color(clamp(uv + vec2f(-d.x,  0.0), vec2f(0.0), vec2f(1.0))) + 
        -1.0 * get_input_color(clamp(uv + vec2f(-d.x,  d.y), vec2f(0.0), vec2f(1.0))) +
        1.0 * get_input_color(clamp(uv + vec2f( d.x, -d.y), vec2f(0.0), vec2f(1.0))) +
        2.0 * get_input_color(clamp(uv + vec2f( d.x,  0.0), vec2f(0.0), vec2f(1.0))) + 
        1.0 * get_input_color(clamp(uv + vec2f( d.x,  d.y), vec2f(0.0), vec2f(1.0)))
    ) / (4.0;

    // Sobel Y kernel: [-1,0,1; -2,0,2; -1,0,1] / 4
    let sobel_y = (
        -1.0 * get_input_color(clamp(uv + vec2f(-d.x, -d.y), vec2f(0.0), vec2f(1.0))) + 
        -2.0 * get_input_color(clamp(uv + vec2f( 0.0, -d.y), vec2f(0.0), vec2f(1.0))) + 
        -1.0 * get_input_color(clamp(uv + vec2f( d.x, -d.y), vec2f(0.0), vec2f(1.0))) +
        1.0 * get_input_color(clamp(uv + vec2f(-d.x,  d.y), vec2f(0.0), vec2f(1.0))) +
        2.0 * get_input_color(clamp(uv + vec2f( 0.0,  d.y), vec2f(0.0), vec2f(1.0))) + 
        1.0 * get_input_color(clamp(uv + vec2f( d.x,  d.y), vec2f(0.0), vec2f(1.0)))
    ) / 4.0;
    
    // RGB gradient magnitudes (per-channel edge detection)
    let grad_r = length(vec2f(sobel_x.r, sobel_y.r));
    let grad_g = length(vec2f(sobel_x.g, sobel_y.g));
    let grad_b = length(vec2f(sobel_x.b, sobel_y.b));
    
    // gradient using color-sensitive weighting
    let color_weights = vec3f(0.299, 0.587, 0.114);
    let weighted_grad = grad_r * color_weights.r + grad_g * color_weights.g + grad_b * color_weights.b;
    
    let gx = dot(sobel_x, color_weights) + (grad_r + grad_g + grad_b) * 0.1;
    let gy = dot(sobel_y, color_weights) + (grad_r + grad_g + grad_b) * 0.1;
    
    // Structure tensor components
    let Jxx = gx * gx + weighted_grad * 0.05;
    let Jyy = gy * gy + weighted_grad * 0.05;
    let Jxy = gx * gy;
    
    textureStore(output, id.xy, vec4f(Jxx, Jyy, Jxy, weighted_grad));
}

// Tensor Field Computation (Pass 2)
@compute @workgroup_size(16, 16, 1)
fn tensor_field(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let texel_size = 1.0 / vec2f(dims);

    // Fabrice's fast blur for tensor smoothing
    let smoothed_tensor = blur_tensor(uv, texel_size);
    
    let Jxx = smoothed_tensor.x;
    let Jyy = smoothed_tensor.y;
    let Jxy = smoothed_tensor.z;

    // Eigenvalue computation 
    let lambda1 = 0.5 * (Jyy + Jxx + sqrt(Jyy * Jyy - 2.0 * Jxx * Jyy + Jxx * Jxx + 4.0 * Jxy * Jxy));
    let lambda2 = 0.5 * (Jyy + Jxx - sqrt(Jyy * Jyy - 2.0 * Jxx * Jyy + Jxx * Jxx + 4.0 * Jxy * Jxy));

    // Compute eigenvector for dominant direction
    var v = vec2f(lambda1 - Jxx, -Jxy);
    var orientation: vec2f;
    if (length(v) > 0.0) { 
        orientation = normalize(v);
    } else {
        orientation = vec2f(0.0, 1.0);
    }

    let phi = atan2(orientation.y, orientation.x);
    
    // Anisotropy measure (coherence)
    var anisotropy = 0.0;
    if (lambda1 + lambda2 > 0.0) {
        anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2);
    }
    
    // Store: orientation.xy, phi, anisotropy
    textureStore(output, id.xy, vec4f(orientation.x, orientation.y, phi, anisotropy));
}

// Kuwahara Filter (Pass 3)
@compute @workgroup_size(16, 16, 1) 
fn kuwahara_filter(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    let texel_size = 1.0 / vec2f(dims);
    
    // Get original color
    let original_color = get_input_color(uv);
    var result_color = vec4f(original_color, 1.0);
    
    if (params.filter_mode == 0) {
        // Classic Kuwahara Filter (fast, symmetric)
        let radius = i32(params.radius);
        
        var quadrant_mean: array<vec4f, 4>;
        var quadrant_variance: array<f32, 4>;
        
        // Calculate quadrant means and variances
        for (var dy = -radius; dy <= radius; dy++) {
            for (var dx = -radius; dx <= radius; dx++) {
                let offset = vec2f(f32(dx), f32(dy)) * texel_size;
                let sample_uv = clamp(uv + offset, vec2f(0.0), vec2f(1.0));
                let sample_color = get_input_color(sample_uv);
                
                // Determine quadrant (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
                var quadrant = 0;
                if (dx >= 0 && dy < 0) { quadrant = 1; }   // top-right
                else if (dx < 0 && dy >= 0) { quadrant = 2; }   // bottom-left  
                else if (dx >= 0 && dy >= 0) { quadrant = 3; }   // bottom-right
                
                quadrant_mean[quadrant] += vec4f(sample_color, 1.0);
                let rgb_intensity = length(sample_color);
                quadrant_variance[quadrant] += rgb_intensity * rgb_intensity;
            }
        }
        
        // Find quadrant with minimum variance
        var min_variance = 999999.0;
        var selected_quadrant = 0;
        
        for (var q = 0; q < 4; q++) {
            if (quadrant_mean[q].w > 0.0) {
                let mean_color = quadrant_mean[q].rgb / quadrant_mean[q].w;
                let mean_intensity = length(mean_color);
                let variance = (quadrant_variance[q] / quadrant_mean[q].w) - (mean_intensity * mean_intensity);
                
                let adjusted_variance = variance * params.q;
                
                if (adjusted_variance < min_variance) {
                    min_variance = adjusted_variance;
                    selected_quadrant = q;
                }
            }
        }
        
        // Use the mean color of the selected quadrant
        if (quadrant_mean[selected_quadrant].w > 0.0) {
            let selected_color = quadrant_mean[selected_quadrant].rgb / quadrant_mean[selected_quadrant].w;
            result_color = vec4f(mix(original_color, selected_color, params.filter_strength), 1.0);
        }
    } else {
        // Classical Anisotropic Kuwahara
        let tensor_data = textureSampleLevel(input_texture1, input_sampler1, uv, 0.0);
        let orientation = tensor_data.xy;
        let anisotropy = tensor_data.w;
        
        let alpha = params.alpha;
        let radius = params.radius;
        
        // edge threshold to control anisotropic effect
        let effective_anisotropy = select(0.0, anisotropy, anisotropy > params.edge_threshold);
        
        // elliptical sampling
        let a = radius * (1.0 + effective_anisotropy * alpha * 0.8);
        let b = radius * max(0.3, 1.0 - effective_anisotropy * alpha * 0.6);
        
        // 4 overlapping quadrants (classical Kuwahara approach)
        var quadrant_means: array<vec3f, 4>;
        var quadrant_variances: array<f32, 4>;
        var quadrant_counts: array<f32, 4>;
        
        // init
        for (var k = 0; k < 4; k++) {
            quadrant_means[k] = vec3f(0.0);
            quadrant_variances[k] = 0.0;
            quadrant_counts[k] = 0.0;
        }
        
        // Sample in simple oriented ellipse
        let max_r = i32(min(radius + 2.0, 10.0));
        for (var j = -max_r; j <= max_r; j++) {
            for (var i = -max_r; i <= max_r; i++) {
                let offset = vec2f(f32(i), f32(j));
                
                // Simple elliptical constraint
                let ellipse_x = offset.x * orientation.x + offset.y * orientation.y;
                let ellipse_y = -offset.x * orientation.y + offset.y * orientation.x;
                let ellipse_dist = (ellipse_x * ellipse_x) / (a * a) + (ellipse_y * ellipse_y) / (b * b);
                
                if (ellipse_dist <= 1.0) {
                    let sample_uv = clamp(uv + offset * texel_size, vec2f(0.0), vec2f(1.0));
                    let sample_color = get_input_color(sample_uv);
                    let rgb_intensity = length(sample_color);
                    
                    // Simple quadrant assignment (overlapping like classical)
                    if (i <= 0 && j <= 0) { // Top-left
                        quadrant_means[0] += sample_color;
                        quadrant_variances[0] += rgb_intensity * rgb_intensity;
                        quadrant_counts[0] += 1.0;
                    }
                    if (i >= 0 && j <= 0) { // Top-right  
                        quadrant_means[1] += sample_color;
                        quadrant_variances[1] += rgb_intensity * rgb_intensity;
                        quadrant_counts[1] += 1.0;
                    }
                    if (i <= 0 && j >= 0) { // Bottom-left
                        quadrant_means[2] += sample_color;
                        quadrant_variances[2] += rgb_intensity * rgb_intensity;
                        quadrant_counts[2] += 1.0;
                    }
                    if (i >= 0 && j >= 0) { // Bottom-right
                        quadrant_means[3] += sample_color;
                        quadrant_variances[3] += rgb_intensity * rgb_intensity;  
                        quadrant_counts[3] += 1.0;
                    }
                }
            }
        }
        
        // Find quadrant with minimum variance (classical)
        var min_variance = 999999.0;
        var best_color = original_color;
        
        for (var q = 0; q < 4; q++) {
            if (quadrant_counts[q] > 0.0) {
                let mean_color = quadrant_means[q] / quadrant_counts[q];
                let mean_intensity = length(mean_color);
                let variance = (quadrant_variances[q] / quadrant_counts[q]) - (mean_intensity * mean_intensity);
                
                let adjusted_variance = variance * params.q;
                
                if (adjusted_variance < min_variance) {
                    min_variance = adjusted_variance;
                    best_color = mean_color;
                }
            }
        }
        
        result_color = vec4f(mix(original_color, best_color, params.filter_strength), 1.0);
    }

    // Tensor visualization - uses pre-computed tensor data
    if (params.show_tensors == 1) {
        if (params.filter_mode == 1) {
            // For anisotropic mode, use the pre-computed tensor data
            let viz_tensor_data = textureSampleLevel(input_texture1, input_sampler1, uv, 0.0);
            let viz_orientation = viz_tensor_data.xy;
            let viz_phi = viz_tensor_data.z;
            let viz_anisotropy = viz_tensor_data.w;
            
            // Create directional lines for visualization
            let pixel_pos = uv * vec2f(dims);
            let line_coord = dot((pixel_pos % 16.0) - 8.0, viz_orientation);
            let line_pattern = 1.0 - smoothstep(0.5, 2.0, abs(line_coord));
            
            // Color mapping for tensor visualization
            let tensor_viz_color = vec3f(
                clamp(length(viz_orientation) * 5.0, 0.0, 1.0),   // Red: structure strength
                clamp(viz_anisotropy, 0.0, 1.0),                  // Green: anisotropy
                0.3                                               // Blue: constant
            ) * (0.7 + 0.3 * line_pattern);
            
            // Blend tensor visualization with filtered result
            result_color = vec4f(mix(result_color.rgb * 0.3, tensor_viz_color, 0.8), 1.0);
        } else {
            // For classic mode, show simple edge visualization
            let grad_x = get_input_color(clamp(uv + vec2f(texel_size.x, 0.0), vec2f(0.0), vec2f(1.0))) - 
                        get_input_color(clamp(uv - vec2f(texel_size.x, 0.0), vec2f(0.0), vec2f(1.0)));
            let grad_y = get_input_color(clamp(uv + vec2f(0.0, texel_size.y), vec2f(0.0), vec2f(1.0))) - 
                        get_input_color(clamp(uv - vec2f(0.0, texel_size.y), vec2f(0.0), vec2f(1.0)));
            
            let gradient_magnitude = length(vec2f(dot(grad_x, vec3f(0.333)), dot(grad_y, vec3f(0.333))));
            let edge_viz_color = vec3f(gradient_magnitude * 10.0, 0.2, 0.8);
            
            result_color = vec4f(mix(result_color.rgb * 0.5, edge_viz_color, 0.6), 1.0);
        }
    }
    
    // Stable and predictable color enhancement
    var final_color = result_color.rgb;
    
    // Simple linear saturation adjustment
    if (abs(params.color_enhance - 1.0) > 0.01) {
        let enhancement = params.color_enhance;
        
        // Linear saturation boost
        let luminance = dot(final_color, vec3f(0.299, 0.587, 0.114));
        let saturation_factor = mix(1.0, enhancement * 1.2, 0.5);
        final_color = mix(vec3f(luminance), final_color, saturation_factor);
        

        let contrast_factor = 0.9 + (enhancement - 1.0) * 0.1;
        final_color = (final_color - 0.5) * contrast_factor + 0.5;
        
        final_color = clamp(final_color, vec3f(0.0), vec3f(1.0));
    }
    
    result_color = vec4f(final_color, result_color.a);
    
    textureStore(output, id.xy, result_color);
}

// Main Image Display (Pass 4) - Final output to screen
@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3u) {
    let dims = textureDimensions(output);
    if (id.x >= dims.x || id.y >= dims.y) { return; }
    
    let uv = (vec2f(id.xy) + 0.5) / vec2f(dims);
    
    // result from the kuwahara_filter pass
    let filtered_result = textureSampleLevel(input_texture2, input_sampler2, uv, 0.0);
    
    textureStore(output, id.xy, filtered_result);
}