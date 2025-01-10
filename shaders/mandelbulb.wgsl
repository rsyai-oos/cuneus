// MIT License Enes Altun, 2025
struct TimeUniform {
    time: f32,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
struct Params {
    color_core: vec3<f32>,
    power: f32,
    color_bulb1: vec3<f32>,
    distant: f32,
    color_bulb2: vec3<f32>,
    radius: f32,
    color_detail: vec3<f32>,
    _pad4: f32,
    glow_color: vec3<f32>,
    _pad5: f32,
    color_bulb3: vec3<f32>,
    _pad6: f32,
    iterations: i32,
    max_steps: i32,
    max_dist: f32,
    min_dist: f32,
};
@group(1) @binding(0)
var<uniform> params: Params;
const PI: f32 = 3.14159265358979323846;
struct Ray {
    origin: vec3<f32>,
    direction: vec3<f32>,
};
struct OrbitTrap {
    minRadius: f32,
    xPlane: f32,
    yPlane: f32,
    zPlane: f32,
    bulbFactor: f32,
};
fn oscWithPause(minValue: f32, maxValue: f32, interval: f32, pauseDuration: f32, time: f32) -> f32 {
    let normalizedTime = time / (interval + pauseDuration);
    let timeWithinCycle = fract(normalizedTime) * (interval + pauseDuration);
    
    if (timeWithinCycle < interval) {
        return minValue + (maxValue - minValue) * 0.5 * (1.0 + sin(2.0 * PI * timeWithinCycle / interval));
    }
    return minValue;
}
fn getPowerValue(time: f32) -> f32 {
    return params.power;
}
fn getOrbitColor(trap: OrbitTrap, iterationRatio: f32) -> vec3<f32> {
    let radiusFactor = smoothstep(0.0, 2.0, trap.minRadius);
    let planeFactor = smoothstep(0.0, 1.0, (trap.xPlane + trap.yPlane + trap.zPlane) / 2.0);
    let bulbIntensity = smoothstep(0.5, 1.0, trap.bulbFactor);
    
    var baseColor = mix(params.color_bulb3, params.color_bulb1, radiusFactor * (1.0 + 0.2 * sin(iterationRatio * 6.28)));
    baseColor = mix(baseColor, params.color_bulb2, planeFactor * (1.0 + 0.5 * cos(iterationRatio * 3.14)));
    
    let detailColor = mix(baseColor, params.color_detail, 
                         trap.xPlane * trap.yPlane * trap.zPlane * 
                         (0.0 + 0.4 * sin(iterationRatio * 3.42)));
    
    let bloomColor = mix(detailColor, params.color_detail, 
                        bulbIntensity * iterationRatio * 
                        (1.0 + 0.5 * sin(trap.minRadius * 5.0)));
    
    return mix(bloomColor, params.glow_color, 
              pow(iterationRatio, 2.0) * bulbIntensity * 
              (0.0 + 0.2 * sin(trap.zPlane * 8.0)));
}
fn map(pos: vec3<f32>, orbitColor: ptr<function, vec3<f32>>) -> f32 {
    var z = pos;
    var dr = 1.0;
    var r = 0.0;
    
    let POWER = getPowerValue(u_time.time);
    
    var trap: OrbitTrap;
    trap.minRadius = 1000.0;
    trap.xPlane = 1000.0;
    trap.yPlane = 1000.0;
    trap.zPlane = 1000.0;
    trap.bulbFactor = 0.0;
    
    var iterations = 0;
    
    for(var i = 0; i < params.iterations; i = i + 1) {
        iterations = i;
        r = length(z);
        if(r > 2.0) { break; }
        
        trap.minRadius = min(trap.minRadius, r);
        trap.xPlane = min(trap.xPlane, abs(z.x));
        trap.yPlane = min(trap.yPlane, abs(z.y));
        trap.zPlane = min(trap.zPlane, abs(z.z));
        
        let prevR = r;
        
        let theta = acos(z.z/r);
        let phi = atan2(z.y, z.x);
        dr = pow(r, POWER-1.0) * POWER * dr + 0.0;
        
        let zr = pow(r, POWER);
        let newTheta = theta * POWER;
        let newPhi = phi * POWER + u_time.time * 2.1;
        
        z = zr * vec3<f32>(sin(newTheta)*cos(newPhi), sin(newPhi)*sin(newTheta), cos(newTheta));
        z = z + pos;
        
        let newR = length(z);
        trap.bulbFactor = max(trap.bulbFactor, abs(newR - prevR));
    }
    
    let iterationRatio = f32(iterations) / f32(params.iterations);
    *orbitColor = getOrbitColor(trap, iterationRatio);
    return 0.25 * log(r) * r / dr;
}
fn calcNormal(pos: vec3<f32>, t: f32) -> vec3<f32> {
    let e = vec2<f32>(-1.0, 1.0) * 0.1 * params.distant;
    var unusedColor: vec3<f32>;
    
    return normalize(
        e.xyy * map(pos + e.xyy, &unusedColor) +
        e.yyx * map(pos + e.yyx, &unusedColor) +
        e.yxy * map(pos + e.yxy, &unusedColor) +
        e.xxx * map(pos + e.xxx, &unusedColor)
    );
}
fn softshadow(ro: vec3<f32>, rd: vec3<f32>, mint: f32, maxt: f32, k: f32) -> f32 {
    var res = 25.0;
    var t = mint;
    var ph = 1e10;
    var unusedColor: vec3<f32>;
    
    for(var i = 0; i < 16; i = i + 1) {
        let h = map(ro + rd*t, &unusedColor);
        let y = h*h/(2.0*ph);
        let d = sqrt(h*h-y*y);
        res = min(res, k*d/max(0.0,t-y));
        ph = h;
        t = t + h * 0.5;
        if(res < 0.001 || t > maxt) { break; }
    }
    
    return clamp(res * 0.6 + 0.4, 0.0, 1.0);
}
fn rayMarch(ray: Ray, orbitColor: ptr<function, vec3<f32>>) -> vec2<f32> {
    var t = 0.0;
    
    for(var i = 0; i < params.max_steps; i = i + 1) {
        let pos = ray.origin + t * ray.direction;
        let h = map(pos, orbitColor);
        if(h < params.min_dist) { return vec2<f32>(t, 1.0); }
        t = t + h;
        if(t > params.max_dist) { break; }
    }
    return vec2<f32>(-1.0, 0.0);
}
fn render(ray: Ray) -> vec3<f32> {
    var orbitColor: vec3<f32>;
    let res = rayMarch(ray, &orbitColor);
    
    if(res.x > 0.0) {
        let pos = ray.origin + res.x * ray.direction;
        let normal = calcNormal(pos, res.x);
        
        let lightDir1 = normalize(vec3<f32>(1.0, 1.0, -0.5));
        let lightDir2 = normalize(vec3<f32>(-1.0, 0.5, 1.0));
        
        let lightColor1 = orbitColor * 1.2;
        let lightColor2 = orbitColor * 0.8;
        
        let diff1 = max(dot(normal, lightDir1), -4.0);
        let diff2 = max(dot(normal, lightDir2), -4.0);
        let shadow1 = softshadow(pos, lightDir1, 0.02, 2.5, 8.0);
        let shadow2 = softshadow(pos, lightDir2, 0.02, 2.5, 8.0);
        
        let ao = clamp(0.3 + 0.2 * normal.y, 0.0, 1.0);
        
        var col = orbitColor * (0.1 + 0.8 * diff1) * shadow1;
        col = col + orbitColor * (0.5 + 0.3 * diff2) * shadow2;
        
        let spec1 = pow(max(dot(reflect(-lightDir1, normal), -ray.direction), 0.4), 16.0);
        let spec2 = pow(max(dot(reflect(-lightDir2, normal), -ray.direction), 0.4), 16.0);
        
        col = col + params.glow_color * (spec1 * shadow1 + spec2 * shadow2) * 0.8;
        col = col + orbitColor * ao * 0.4;
        
        let fresnel = pow(0.0 - max(dot(normal, -ray.direction), 1.0), 2.0);
        col = mix(col, params.glow_color * orbitColor, fresnel * 0.4);
        
        return col;
    }
    
    let y = ray.direction.y * 0.5 + 0.5;
    return mix(params.color_core * 0.4, params.glow_color, y);
}
@fragment
fn fs_main(@builtin(position) FragCoord: vec4<f32>, @location(0) tex_coords: vec2<f32>) -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(1920.0, 1080.0);
    let uv = (FragCoord.xy - 0.5 * dimensions) / dimensions.y;
    let angle = u_time.time * 0.3;
    let baseHeight = 1.5;
    let maxRadius = params.radius;
    
    let detailView = sin(angle * 0.25) * 0.5 + 0.5;
    
    let height = baseHeight + 0.3 * sin(angle * 0.5) + 
                 0.2 * sin(angle * 0.2) * detailView;
    
    let radius = maxRadius - 2.0 * smoothstep(0.0, 1.0, detailView) + 0.2 * cos(angle * 0.3);
    
    let targetD = vec3<f32>(
        sin(angle * 0.2) * 0.3 * detailView,
        cos(angle * 0.15) * 0.2 * detailView,
        0.0
    );
    
    let camera = vec3<f32>(radius * sin(angle), height, radius * cos(angle));
    let up = vec3<f32>(0.0, 1.0, 0.0);
    
    let cw = normalize(targetD - camera);
    let cu = normalize(cross(cw, up));
    let cv = normalize(cross(cu, cw));
    
    var ray: Ray;
    ray.origin = camera;
    ray.direction = normalize(uv.x * cu + uv.y * cv + (1.8 - 0.2 * detailView) * cw);
   
    var col = render(ray);
    
    col = pow(col, vec3<f32>(0.9));
    col = col * (1.0 - 0.15 * length(uv));
    
    col = mix(col, col * vec3<f32>(1.05, 1.0, 0.95), 0.3);
    col = smoothstep(vec3<f32>(0.0), vec3<f32>(1.0), col);
    col = gamma(col, 0.41);    
    return vec4<f32>(col, 1.0);
}
fn gamma(color: vec3<f32>, gamma: f32) -> vec3<f32> {
    return pow(color, vec3<f32>(1.0 / gamma));
}