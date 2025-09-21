// Enes Altun, 21 Sep 2025 
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct VortexParams {
    a: f32,
    b: f32,
    c: f32,
    dof_amount: f32,
    dof_focal_dist: f32,
    brightness: f32,
    color1_r: f32,
    color1_g: f32,
    color1_b: f32,
    color2_r: f32,
    color2_g: f32,
    color2_b: f32,
    zoom: f32,
    camera_rotation_x: f32,
    camera_rotation_y: f32,
    camera_auto_rotate: f32,
    _padding: f32,
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> custom: VortexParams;

@group(2) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m2 = mat2x2<f32>;
alias m3 = mat3x3<f32>;
alias m4 = mat4x4<f32>;
const pi = 3.14159265359;
const tau = 6.28318530718;

var<private> R: v2;
var<private> seed: u32;

fn rot(a: f32)-> m2{ return m2(cos(a), -sin(a),sin(a), cos(a));}
fn rotX(a: f32) -> m3{
    let r = rot(a); return m3(1.,0.,0.,0.,r[0][0],r[0][1],0.,r[1][0],r[1][1]);
}
fn rotY(a: f32) -> m3{
    let r = rot(a); return m3(r[0][0],0.,r[0][1],0.,1.,0.,r[1][0],0.,r[1][1]);
}
fn rotZ(a: f32) -> m3{
    let r = rot(a); return m3(r[0][0],r[0][1],0.,r[1][0],r[1][1],0.,0.,0.,1.);
}

fn hash_u(_a: u32) -> u32{ var a = _a; a ^= a >> 16u;a *= 0x7feb352du;a ^= a >> 15u;a *= 0x846ca68bu;a ^= a >> 16u;return a; }
fn hash_f() -> f32{ var s = hash_u(seed); seed = s;return ( f32( s ) / f32( 0xffffffffu ) ); }
fn hash_v2() -> v2{ return v2(hash_f(), hash_f()); }
fn hash_v3() -> v3{ return v3(hash_f(), hash_f(), hash_f()); }

fn sample_disk() -> v2{
    let r = hash_v2();
    return v2(sin(r.x*tau),cos(r.x*tau))*sqrt(r.y);
}

fn noise2d(p: v2) -> f32 {
    return sin(p.x * 3.0 + sin(p.y * 2.7)) * cos(p.y * 1.1 + cos(p.x * 2.3));
}

fn fbm_noise(p: v3) -> f32 {
    var v = 0.0;
    var a = 1.0;
    var pp = p;

    for(var i = 0; i < 7; i++) {
        v += noise2d(pp.xy + pp.z * 0.5) * a;
        pp *= 2.0;
        a *= 0.5;
    }
    return v;
}

fn projParticle(_p: v3) -> v3{
    var p = _p;
    p.z += 0.5;
    p /= p.z*0.3*custom.zoom;
    p.z = _p.z;
    p.x /= R.x/R.y;
    return p;
}

@compute @workgroup_size(256, 1, 1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let Ru = vec2<u32>(textureDimensions(output));
    R = v2(Ru);

    seed = hash_u(id.x + hash_u(Ru.x * id.y * 200u) * 20u + hash_u(id.z) * 250u);
    seed = hash_u(seed);

    let iters = 20;

    var p = v3(0.0, 0.0, 0.0);
    //angle
    let ang = hash_f() * tau;
    //radius
    let rad = 0.3 + hash_f() * 0.7;
    p.x = cos(ang) * rad;
    p.y = sin(ang) * rad;
    p.z = hash_f() * 2.0 - 1.0;

    //tunnel_speed
    var tspeed = 1.0 + custom.a * 2.0;
    //rotation_speed
    var rspeed = 1.1 + custom.b * 2.0;
    //noise_strength
    var nstr = 0.3 + custom.c * 0.7;

    let focusDist = (custom.dof_focal_dist*2. - 1.)*2.;
    let dofFac = 1./v2(R.x/R.y,1.)*custom.dof_amount;

    for(var i = 0; i < iters; i++){
        let fi = f32(i) / f32(iters);
        let r = hash_f();

        p.z += tspeed * 0.06;
        p.z = (p.z + 2.0) % 4.0 - 2.0;

        let rot_angle = p.z * rspeed + time_data.time;
        let rotated_xy = rot(rot_angle) * p.xy;
        p = v3(rotated_xy.x, rotated_xy.y, p.z);

        let noise_p = p + time_data.time * 0.1;
        let N = fbm_noise(noise_p);

        if(r < 0.8) {
            //target_radius
            let trad = 2.0 - N * nstr;
            //current_radius
            let crad = length(p.xy);

            if(crad > 0.01) {
                let scale = trad / crad;
                p = v3(p.xy * scale, p.z);
            }

            if(hash_f() < 0.1) {
                //turb_angle
                let turb_ang = N * pi * 0.5;
                let turb_xy = rot(turb_ang) * p.xy;
                p = v3(turb_xy.x, turb_xy.y, p.z);
            }
        }
        else if(r < 0.01) {
            //flow_radius
            let frad = length(p.xy);

            //spiral_angle
            let spir_ang = time_data.time * 2.0 + fi * tau * 2.0;
            //spiral_offset
            let spir_off = v2(cos(spir_ang), sin(spir_ang)) * 0.1;
            p = v3(p.xy + spir_off * (1.0 - frad), p.z);

            //radial_pulse
            let rad_pulse = 0.5 + 0.5 * sin(N * 2.0 + time_data.time * 3.0);
            p = v3(p.xy * (0.8 + rad_pulse * 0.4), p.z);
        }
        else {
            p = v3(p.xy * 0.1, p.z);

            //vortex_angle
            let vort_ang = atan2(p.y, p.x) + N * 1.5;
            //vortex_radius
            let vort_rad = length(p.xy) * (0.5 + 0.5 * sin(time_data.time + N));
            p = v3(
                cos(vort_ang) * vort_rad,
                sin(vort_ang) * vort_rad,
                p.z
            );

            if(hash_f() < 0.1) {
                p += (hash_v3() - 0.5) * 0.3 * abs(N);
            }
        }

        if(custom.camera_auto_rotate > 0.5) {
            p = rotY(custom.camera_rotation_y + time_data.time  * 0.5) * p;
        } else {
            p = rotY(custom.camera_rotation_y) * p;
            p = rotX(custom.camera_rotation_x) * p;
        }

        var q = projParticle(p);
        var k = q.xy;

        k += sample_disk()*abs(q.z - focusDist)*0.02*dofFac;

        let uv = k.xy/4. + 0.5;
        let cc = vec2<u32>(uv.xy*R.xy);
        let idx = cc.x + Ru.x * cc.y;
        if (
            uv.x > 0. && uv.x < 1.
            && uv.y > 0. && uv.y < 1.
            && idx < (Ru.x*Ru.y)
            ){
            //glow_weight
            let glow_w = 1.0 / (length(p.xy) * 0.8 + 0.2);
            atomicAdd(&atomic_buffer[idx], u32(glow_w * 100.0));

            //color_data
            let col_data = u32((N + 1.0) * 50.0 + length(p.xy) * 30.0);
            atomicAdd(&atomic_buffer[idx + Ru.x*Ru.y], col_data);
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }

    R = v2(res);
    let hist_id = id.x + u32(R.x) * id.y;

    //col1_intensity
    let col1_int = f32(atomicLoad(&atomic_buffer[hist_id]));
    //col2_intensity
    let col2_int = f32(atomicLoad(&atomic_buffer[hist_id + res.x*res.y]));

    var col1 = col1_int * v3(custom.color1_r, custom.color1_g, custom.color1_b);
    var col2 = col2_int * v3(custom.color2_r, custom.color2_g, custom.color2_b);
    var col = (col1 + col2);

    let sc = 25000.0;
    col = log(col * custom.brightness * .5 + 1.0) / log(sc);
    col = smoothstep(v3(0.), v3(1.), col);

    col = tanh(col * 2.0);
    col = pow(col, v3(1. / 0.4));

    textureStore(output, vec2<i32>(id.xy), v4(col, 1.));

    atomicStore(&atomic_buffer[hist_id], 0u);
    atomicStore(&atomic_buffer[hist_id + res.x*res.y], 0u);
}
