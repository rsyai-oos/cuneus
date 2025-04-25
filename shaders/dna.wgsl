struct TimeUniform {
    time: f32,
};
struct ResolutionUniform {
    dimensions: vec2<f32>,
    _padding: vec2<f32>,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var<uniform> u_resolution: ResolutionUniform;
@group(2) @binding(0) var<uniform> params: Params;

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

const PI = 3.1416;

fn rot(a: f32) -> mat2x2<f32> {
    let s = sin(a); let c = cos(a);
    return mat2x2<f32>(c, -s, s, c);
}

// DNA Helix SDF 
fn helixSDF(p: vec3<f32>) -> f32 {
    // Create two helical strands with direct vector ops
    let sd = 0.9;  // strand distance
    let s1 = length(p.xz + vec2(sd, 0.)) - 0.25;
    let s2 = length(p.xz - vec2(sd, 0.)) - 0.25;
    
    // Base pairs using modulo trick
    let py = p.y - 0.6 * floor(p.y/0.6) - 0.3;
    let bar = max(length(vec2(py, p.z)) - 0.1, abs(p.x) - sd*0.98);
    
    return min(min(s1, s2), bar);
}

fn map(p: vec3<f32>, t: f32) -> f32 {
    var q = p * (params.falloff_distance + 3.);
    let rot_matrix = rot(t * params.rotation_speed) * rot(q.y * 0.6);
    let rotated = rot_matrix * q.xz;
    q.x = rotated.x;
    q.z = rotated.y;
    
    return helixSDF(q) * params.falloff_distance;
}

fn calcNormal(p: vec3<f32>, t: f32) -> vec3<f32> {
    const e = vec2(0.001, 0.);
    return normalize(vec3(
        map(p + e.xyy, t) - map(p - e.xyy, t),
        map(p + e.yxy, t) - map(p - e.yxy, t),
        map(p + e.yyx, t) - map(p - e.yyx, t)
    ));
}

fn calcAO(p: vec3<f32>, n: vec3<f32>, t: f32) -> f32 {
    var o = 0.; var s = 1.;
    for(var i = 0; i < 5; i++) {
        let h = 0.01 + 0.03*f32(i);
        o += -(map(p + n*h, t) - h) * s;
        s *= 0.95;
    }
    return max(0., min(1., 1. - o*params.ao_strength));
}

fn cda(p: vec3<f32>, n: vec3<f32>, t: f32, d: f32) -> f32 {
    let aoRange = d/20.0;
    let occlusion = max(0.0, 1.0 - map(p + n*aoRange, t)/aoRange);
    return exp2(-2.0*occlusion*occlusion);
}

// Specular reflection occlusion
fn cso(p: vec3<f32>, reflection: vec3<f32>, t: f32, d: f32) -> f32 {
    let aoRange = d/40.0;
    let specOcclusion = max(0.0, 1.0 - map(p + reflection*aoRange, t)/aoRange);
    return exp2(-2.0*specOcclusion*specOcclusion);
}

// Subsurface scattering approximation
fn cs(p: vec3<f32>, lightDir: vec3<f32>, t: f32, d: f32) -> f32 {
    let range = d/10.0;
    let transmission = map(p + lightDir*range, t)/range;
    return smoothstep(0.0, 1.0, transmission);
}

fn raymarch(ro: vec3<f32>, rd: vec3<f32>, t: f32) -> vec2<f32> {
    var d = 0.;
    for(var i = 0; i < 100; i++) {
        let h = map(ro + rd*d, t);
        if(h < 0.001) { return vec2(d, 1.); }
        d += h * 0.5;
        if(d > 20.) { break; }
    }
    return vec2(d, 0.);
}

// Light positions with optimal vector construction
fn glo(t: f32, l: f32) -> array<vec3<f32>, 4> {
    var p: array<vec3<f32>, 4>;
    
    for(var i = 0; i < 4; i++) {
        let a = f32(i)*PI/2. + t*0.3;
        let r = 4. + 0.8*sin(t*0.4 + f32(i)); 
        p[i] = vec3(cos(a)*r, 2.*sin(t*0.2 + f32(i)*0.8)*l, sin(a)*r);
    }
    
    return p;
}
// Light intensity
fn gli(p: vec3<f32>, n: vec3<f32>, rd: vec3<f32>, d: f32, l: f32, t: f32) -> f32 {
    let lights = glo(t, l);
    var diffuse = 0.;
    var specular = 0.;
    var subsurface = 0.;
    for(var j = 0; j < 4; j++) {
        let lv = lights[j] - p;
        let ldist = length(lv);
        let ldir = normalize(lv);
        let ndotl = max(0., dot(n, ldir));
        let lightIntensity = (0.2 + 0.1*sin(l*5. + t*(0.5 + f32(j)*0.1))) 
           / (1. + ldist*0.2);
        diffuse += ndotl * lightIntensity;
        let h = normalize(ldir - rd);
        let specPower = exp2(3.0 + 5.0*params.rim_power);
        let spec = pow(max(0., dot(n, h)), specPower) * specPower/32.0;
        specular += spec * lightIntensity;
        let sss = cs(p, ldir, t, d);
        subsurface += sss * lightIntensity;
    }
    let ao = 0.5 - l*0.1*(1. + 0.2*sin(l*8. + t));
    let rim = pow(1. - abs(dot(n, -rd)), 1.) * params.fold_intensity;
    let diffuseColor = diffuse * (1. + 0.1*sin(l*6. + t)*cos(l*4. - t*0.8)) * ao * 5.;
    return diffuseColor + specular + rim*0.7 + subsurface * 0.3;
}

// Environment light with vector dot product
fn enlig(p: vec3<f32>, n: vec3<f32>, l: f32, t: f32) -> f32 {
    let ld = normalize(vec3(cos(t*0.5), sin(t*0.5), 0.5));
    let depth = 1. - l/1.5;
    let layer_fx = sin(l*3. + t)*0.3 + 0.7;
    return mix(dot(n, ld)*0.5 + 0.5, layer_fx, 0.5) * depth * params.env_light_strength;
}

fn calcFresnel(n: vec3<f32>, rd: vec3<f32>, specularity: f32) -> f32 {
    let fresnel = pow(1.0 + dot(n, rd), 5.0);
    return mix(mix(0.0, 0.01, specularity), mix(0.4, 1.0, specularity), fresnel);
}
fn gamma(c: vec3<f32>, g: f32) -> vec3<f32> {
    return pow(c, vec3(1./g));
}

@fragment
fn fs_main(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    let t = u_time.time * params.wave_speed;
    let ss = u_resolution.dimensions;
    let uv = (fc.xy - ss*0.5) / min(ss.x, ss.y);
    
    var col = vec4(0.3, 0.3, 0.33, 1.);
    
    let ca = t*0.05;
    let ro = vec3(sin(ca)*5., 0., cos(ca)*5.);
    
    let ww = normalize(-ro);
    let uu = normalize(cross(ww, vec3(0.,1.,0.)));
    let vv = normalize(cross(uu, ww));
    
    let rd = normalize(uv.x*uu + uv.y*vv + 1.5*ww);
    
    if(raymarch(ro, rd, t).y > 0.5) {
        // Layer processing with optimized loop
        for(var i = 2.0; i > 1.1; i -= 0.3) {
            let l = i*params.rim_power;
            
            // Layer-specific ray with parallax offset
            let lr = normalize(rd + 0.01*vec3(sin(l*0.2), cos(l*0.3), 0.));
            let r = raymarch(ro, lr, t);
            
            if(r.y > 0.5 && r.x < 20.) {
                // Surface properties
                let p = ro + r.x*lr;
                let n = calcNormal(p, t);
                let d = map(p, t);
                
                let a = smoothstep(0.0, 0.1, (d + 0.01)*0.2);
                
                let ao = cda(p, n, t, r.x);
                
                // Lighting and color
                let li = gli(p, n, lr, d, i, t);
                let el = enlig(p, n, l, t);
                
                let ci = 0.7 + 0.3*sin(l*15. + t)*cos(l*10. - t*0.5);
                let sc = params.rim_color*vec3(1.,2.,3.) + 1.;
                let h = sin(i*0.7 + vec4(sc, 1.) + d*0.5)*0.25 + ci;
                
                let lc = h * ((li + el) * ao);
                
                let reflection = reflect(lr, n);
                let specOcclusion = cso(p, reflection, t, r.x);
                
                let fresnel = calcFresnel(n, lr, params.rim_power);
                
                let df = 2. - (i - 0.003)/1.497;
                let ir = sin(dot(p.xy, p.xy)*4. + t)*0.15*df + 0.9;
                let lic = lc * vec4(ir, ir*0.98, ir*1.02, 1.);
                let mf = 0.3/(abs(d) + 0.01)*(params.iridescence_power - df*0.15);
                let finalColor = mix(lic, lic * specOcclusion, fresnel);
                
                col = mix(finalColor, col, a) * mix(
                    vec4(params.base_color, 1.),
                    h + a*params.wave_speed*(uv.x/(abs(d) + 0.001) + li),
                    mf
                );
            }
        }
    }
    
    let v = 1. - dot((fc.xy - ss*0.5)/ss.y, (fc.xy - ss*0.5)/ss.y)*params.vignette_strength;
    
    return vec4(gamma(col.rgb*v*1.1, params.light_intensity), 1.);
}