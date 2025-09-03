// Enes Altun, 3 Sep 2025 
// This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> time_data: TimeUniform;

struct SystemParams {
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
    _padding: u32,
}
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> custom: SystemParams;

@group(2) @binding(0) var<storage, read_write> atomic_buffer: array<atomic<u32>>;

alias v4 = vec4<f32>;
alias v3 = vec3<f32>;
alias v2 = vec2<f32>;
alias m2 = mat2x2<f32>;
alias m3 = mat3x3<f32>;
alias m4 = mat4x4<f32>;

var<workgroup> texCoords: array<array<vec2<f32>, 16>, 16>;
const pi = 3.14159265359;
const tau = 6.28318530718;
var<private> R: v2;
var<private> U: v2;
var<private> seed: u32;
//hash and some interesting rots from: https://compute.toys/view/90; wrighter
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

fn hash_u(_a: u32) -> u32{ var a = _a; a ^= a >> 16;a *= 0x7feb352du;a ^= a >> 15;a *= 0x846ca68bu;a ^= a >> 16;return a; }
fn hash_f() -> f32{ var s = hash_u(seed); seed = s;return ( f32( s ) / f32( 0xffffffffu ) ); }
fn hash_v2() -> v2{ return v2(hash_f(), hash_f()); }
fn hash_v3() -> v3{ return v3(hash_f(), hash_f(), hash_f()); }

fn sample_disk() -> v2{
    let r = hash_v2();
    return v2(sin(r.x*tau),cos(r.x*tau))*sqrt(r.y);
}

const COL_CNT = 4;
var<private> kCols = array<v3, COL_CNT>( 
     vec3(1.0, 1.0, 1.0), vec3(0.2, 0.9, 1.0),
     vec3(1.0, 1.0, 1.0) * 1.5, vec3(1.0, 0.3, 0.7)
);

fn mix_cols(_idx: f32)->v3{
    let idx = _idx%1.;
    var cols_idx = i32(idx*f32(COL_CNT));
    var fract_idx = fract(idx*f32(COL_CNT));
    fract_idx = smoothstep(0.,1.,fract_idx);
    return mix( kCols[cols_idx], kCols[(cols_idx + 1)%COL_CNT], fract_idx );
}

fn projParticle(_p: v3) -> v3{
    var p = _p;
    
    p += sin(v3(1.8, 2.9, 1.4) + time_data.time*0.5)*0.08;
    
    p.z += 0.99;
    p /= p.z*0.315;
    p.z = _p.z;
    p.x /= R.x/R.y;
    return p;
}

@compute @workgroup_size(256, 1,1)
fn Splat(@builtin(global_invocation_id) id: vec3<u32>) {
    let Ru = vec2<u32>(textureDimensions(output));
    R = v2(Ru); U = v2(id.xy);
    
    seed = hash_u(id.x + hash_u(Ru.x*id.y*200u)*20u + hash_u(id.z)*250u);
    seed = hash_u(seed);

    let iters = 90;
    
    var t = time_data.time*0.6 - hash_f()*0.4;
    var env = t + sin(t*0.4);

    var p = (hash_v3() - 0.5)*1.2;
    
    var charge: f32;
    if hash_f() < 0.5 {
        charge = 3.0;
    } else {
        charge = -1.0;
    }
    let focusDist = (custom.dof_focal_dist*2. - 1.)*2.;
    let dofFac = 1./v2(R.x/R.y,1.)*custom.dof_amount;
    
    for(var i = 0; i < iters; i++){
        let r = hash_f();
        if(r < 0.1){
            let field_strength = 1.4 + custom.a*1.2;
            p = p/(dot(p,p)*field_strength*charge + 0.2);
            if(hash_f() < 0.3){
                charge = -charge;
                p += v3(sin(env), cos(env*1.2), sin(env*0.8))*1.1;
            }
        } 
        else if(r < 0.6){
            let gradient_factor = 0.9 + custom.b*1.1;
            p = p/(dot(p,p)*gradient_factor + 0.25); 
            
            let field_dir = normalize(p + v3(0.001, 0.001, 0.001));
            p += field_dir * charge * 1.15;
        }
        else {
            p += v3(custom.c*charge, 1.2, 0.1*charge);
            
            if(length(p) > 1.2){
                p *= 0.1;
                charge = -charge;
            } else if(length(p) < 1.3){
                p *= 1.8;
            }
            if(hash_f() < 0.1){
                p += (hash_v3() - 0.5)*0.3*abs(charge);
            }
        }
        
        var q = projParticle(p);
        var k = q.xy;

        k += sample_disk()*abs(q.z - focusDist)*0.03*dofFac;
        
        let uv = k.xy/2. + 0.5;
        let cc = vec2<u32>(uv.xy*R.xy);
        let idx = cc.x + Ru.x * cc.y;
        if ( 
            uv.x > 0. && uv.x < 1. 
            && uv.y > 0. && uv.y < 1. 
            && idx < u32(Ru.x*Ru.y)
            ){     
            let field_intensity = abs(charge) + length(p) * 0.5;
            if (charge > 0.0) {
                atomicAdd(&atomic_buffer[idx], u32(field_intensity * 100.0));
            } else {
                atomicAdd(&atomic_buffer[idx + Ru.x*Ru.y], u32(field_intensity * 100.0));
            }
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main_image(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = vec2<u32>(textureDimensions(output));
    if (id.x >= res.x || id.y >= res.y) { return; }

    R = v2(res);
    let hist_id = id.x + u32(R.x) * id.y;

    var col1 = f32(atomicLoad(&atomic_buffer[hist_id])) * v3(custom.color1_r, custom.color1_g, custom.color1_b);
    var col2 = f32(atomicLoad(&atomic_buffer[hist_id + res.x*res.y])) * v3(custom.color2_r, custom.color2_g, custom.color2_b);
    var col = (col1 + col2);
    
    let sc = 25000.0;
    col = log(col*custom.brightness*10000.0 + 1.0)/ log(sc);
    col = smoothstep(v3(0.),v3(1.),col);

    col = pow(col, v3(1./0.1));
    
    textureStore(output, vec2<i32>(id.xy), v4(col, 1.));
    
    // Clear
    atomicStore(&atomic_buffer[hist_id], 0u);
    atomicStore(&atomic_buffer[hist_id + res.x*res.y], 0u);
}