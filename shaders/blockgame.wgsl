// Block Game, Enes Altun, 2025, MIT License

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

struct MouseUniform {
    position: vec2<f32>,         
    click_position: vec2<f32>,   
    wheel: vec2<f32>,            
    buttons: vec2<u32>,          
};
@group(2) @binding(0) var<uniform> u_mouse: MouseUniform;

// Group 3: Font system + storage buffer for game state
struct FontUniforms {
    atlas_size: vec2<f32>,
    char_size: vec2<f32>,
    screen_size: vec2<f32>,
    _padding: vec2<f32>,
};
@group(3) @binding(0) var<uniform> u_font: FontUniforms;
@group(3) @binding(1) var t_font_atlas: texture_2d<f32>;
@group(3) @binding(2) var s_font_atlas: sampler;
@group(3) @binding(3) var<storage, read_write> game_data: array<f32>;

// game indices
const O = array<u32,7>(0,1,2,3,4,5,6); // state,score,block,click,cam_y,cam_h,cam_a
const BD = 100u; // block data start
const BS = 10u;  // block size

// stuff
struct Block { p: vec3<f32>, s: vec3<f32>, c: vec3<f32>, perf: f32, };
struct Mat { alb: vec3<f32>, r: f32, m: f32, f: f32, }; // material
struct Light { p: vec3<f32>, c: vec3<f32>, i: f32, }; // light

// sdf sampling
fn sdf(p: vec2<f32>, sz: vec2<f32>, a: u32, size: f32) -> f32 {
    let uv = p / sz;
    let c = vec2<u32>(a % 16u, a / 16u);
    let fuv = (vec2<f32>(c) * 64. + 4. + uv * 56.) / 1024.;
    
    if (any(fuv < vec2(0.)) || any(fuv >= vec2(1.))) { return 0.; }
    return textureLoad(t_font_atlas, vec2<i32>(fuv * 1024.), 0).a;
}

fn ch(pos: vec2<f32>, cpos: vec2<f32>, a: u32, sz: f32) -> f32 { // char
    let lp = pos - cpos;
    let s = vec2(sz);
    return select(0., sdf(lp, s, a, sz), all(lp >= vec2(0.)) && all(lp < s));
}

fn adv(sz: f32) -> f32 { return sz * .9; } // char advance

fn word(p: vec2<f32>, wp: vec2<f32>, w: array<u32, 16>, len: u32, sz: f32) -> f32 {
    let a = adv(sz);
    var alpha = 0.;
    for (var i = 0u; i < len && i < 16u; i++) {
        alpha = max(alpha, ch(p, wp + vec2(f32(i) * a, 0.), w[i], sz));
    }
    return alpha;
}

fn num(p: vec2<f32>, np: vec2<f32>, n: u32, sz: f32) -> f32 { // number
    let a = adv(sz);
    var alpha = 0.;
    var tn = n;
    var dc = select(1u, 0u, n == 0u);
    
    // count digits
    var ct = tn;
    while (ct > 0u) { ct /= 10u; dc++; }
    
    tn = n;
    for (var i = 0u; i < dc; i++) {
        let digit = tn % 10u;
        alpha = max(alpha, ch(p, np + vec2(f32(dc-1u-i) * a, 0.), digit + 48u, sz));
        tn /= 10u;
    }
    return alpha;
}

// get block material
fn mat(id: u32) -> Mat {
    let h = fract(f32(id) * .618034);
    var alb: vec3<f32>;
    
    if (h < .33) { alb = vec3(.8, .2 + h * 1.8, .1); }
    else if (h < .66) { alb = vec3(.1 + (.66 - h) * 2.1, .8, .2); }
    else { alb = vec3(.2, .1 + (h - .66) * 2.1, .9); }
    
    return Mat(alb, .1 + h * .7, select(.1, .8, id % 3u == 0u), .04);
}

// ggx stuff
fn dggx(nh: f32, r: f32) -> f32 {
    let a2 = r * r * r * r;
    let d = nh * nh * (a2 - 1.) + 1.;
    return a2 / (3.14159 * d * d);
}
 // geometry smith
fn gsmith(nv: f32, nl: f32, r: f32) -> f32 {
    let k = (r + 1.) * (r + 1.) *.125;
    return (nl / (nl * (1. - k) + k)) * (nv / (nv * (1. - k) + k));
}

// fresnel
fn fschlick(ct: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3(1.) - f0) * pow(clamp(1. - ct, 0., 1.), 5.);
}

// cook torrance lighting
fn ct(ld: vec3<f32>, vd: vec3<f32>, n: vec3<f32>, m: Mat, lc: vec3<f32>, li: f32) -> vec3<f32> {
    let h = normalize(ld + vd);
    let nl = max(dot(n, ld), 0.);
    let nv = max(dot(n, vd), 0.);
    let nh = max(dot(n, h), 0.);
    let hv = max(dot(h, vd), 0.);
    
    let d = dggx(nh, m.r);
    let g = gsmith(nv, nl, m.r);
    let f0 = mix(vec3(m.f), m.alb, m.m);
    let f = fschlick(hv, f0);
    
    let spec = d * g * f / (4. * nv * nl + .0001);
    let kd = (vec3(1.) - f) * (1. - m.m);
    let diff = kd * m.alb / 3.14159;
    
    return (diff + spec) * lc * li * nl;
}

// ambient calc
fn amb(m: Mat, ao: f32) -> vec3<f32> {
    return m.alb * vec3(.1, .15, .25) * ao;
}

// game state getters/setters
fn gs() -> u32 { return u32(game_data[O[0]]); } // get state
fn ss(s: u32) { game_data[O[0]] = f32(s); } // set state
fn gsc() -> u32 { return u32(game_data[O[1]]); } // get score  
fn ssc(s: u32) { game_data[O[1]] = f32(s); } // set score
fn gcb() -> u32 { return u32(game_data[O[2]]); } // get current block
fn scb(b: u32) { game_data[O[2]] = f32(b); } // set current block
fn gct() -> bool { return game_data[O[3]] > .5; } // get click triggered
fn sct(t: bool) { game_data[O[3]] = select(0., 1., t); } // set click triggered
fn gcy() -> f32 { return game_data[O[4]]; } // get camera y
fn scy(y: f32) { game_data[O[4]] = y; } // set camera y
fn gch() -> f32 { return game_data[O[5]]; } // get camera height
fn sch(h: f32) { game_data[O[5]] = h; } // set camera height
fn gca() -> f32 { return game_data[O[6]]; } // get camera angle
fn sca(a: f32) { game_data[O[6]] = a; } // set camera angle

// update camera for tower
fn updcam() {
    let cb = gcb();
    if (cb > 0u) {
        let th = f32(cb) * .6;
        let tcy = th * 40.;
        let ccy = gcy();
        scy(mix(ccy, tcy, .1));
    }
}

// get stored block
fn gb(id: u32) -> Block {
    if (id >= 50u) { return Block(vec3(0.), vec3(0.), vec3(0.), 0.); }
    
    let i = BD + id * BS;
    return Block(
        vec3(game_data[i], game_data[i+1u], game_data[i+2u]),
        vec3(game_data[i+3u], game_data[i+4u], game_data[i+5u]), 
        vec3(game_data[i+6u], game_data[i+7u], game_data[i+8u]),
        game_data[i+9u]
    );
}

// set stored block  
fn sb(id: u32, b: Block) {
    if (id < 50u) {
        let i = BD + id * BS;
        game_data[i] = b.p.x;   game_data[i+1u] = b.p.y; game_data[i+2u] = b.p.z;
        game_data[i+3u] = b.s.x; game_data[i+4u] = b.s.y; game_data[i+5u] = b.s.z;
        game_data[i+6u] = b.c.x; game_data[i+7u] = b.c.y; game_data[i+8u] = b.c.z;
        game_data[i+9u] = b.perf;
    }
}

// world to isometric
fn w2i(wp: vec3<f32>) -> vec2<f32> {
    let ap = wp - vec3(0., gch(), 0.);
    let a = gca();
    let rp = vec3(ap.x * cos(a) - ap.z * sin(a), ap.y, ap.x * sin(a) + ap.z * cos(a));
    return vec2((rp.x - rp.z) * .866, (rp.x + rp.z) * .5 - rp.y);
}

fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 { return a.x * b.y - a.y * b.x; }

// point in quad
fn piq(p: vec2<f32>, v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>, v3: vec2<f32>) -> bool {
    let d = vec4(cross2(v1-v0, p-v0), cross2(v2-v1, p-v1), cross2(v3-v2, p-v2), cross2(v0-v3, p-v3));
    return all(d >= vec4(0.)) || all(d <= vec4(0.));
}

fn lights() -> array<Light, 3> {
    return array<Light, 3>(
        Light(normalize(vec3(2., 3., 1.5)), vec3(1., .95, .8), 2.5),
        Light(normalize(vec3(-1.5, 2., -1.)), vec3(.4, .6, .9), 1.2),
        Light(normalize(vec3(0., 1., -2.)), vec3(.9, .7, .3), .8)
    );
}

// simple ao
fn ao(wp: vec3<f32>, n: vec3<f32>) -> f32 {
    return .3 + .7 * (clamp(wp.y / 10., 0., 1.) + clamp(dot(n, vec3(0., 1., 0.)), 0., 1.) * .5);
}

// render block with lighting  
fn rbl(pp: vec2<f32>, b: Block, ss: vec2<f32>, id: u32) -> vec3<f32> {
    if (any(b.s <= vec3(0.))) { return vec3(0.); }
    
    let m = mat(id);
    var fm = m;
    if (b.perf > .5) { fm = Mat(m.alb + vec3(.1, .05, 0.), m.r, m.m, m.f); } // golden tint
    
    let scale = 80.;
    let cy = gcy();
    let co = vec2(ss.x * .5, ss.y * .7 + cy);
    
    let hw = b.s.x * .5;
    let hd = b.s.z * .5;
    
    // bottom/top corners
    let bc = array<vec3<f32>, 4>(
        b.p + vec3(-hw, 0., -hd), b.p + vec3(hw, 0., -hd),
        b.p + vec3(hw, 0., hd), b.p + vec3(-hw, 0., hd)
    );
    let tc = array<vec3<f32>, 4>(
        b.p + vec3(-hw, b.s.y, -hd), b.p + vec3(hw, b.s.y, -hd),
        b.p + vec3(hw, b.s.y, hd), b.p + vec3(-hw, b.s.y, hd)
    );
    
    // to screen space
    var bs = array<vec2<f32>, 4>();
    var ts = array<vec2<f32>, 4>();
    for (var i = 0u; i < 4u; i++) {
        bs[i] = w2i(bc[i]) * scale + co;
        ts[i] = w2i(tc[i]) * scale + co;
    }
    
    let ls = lights();
    var fc = vec3(0.);
    var hit = false;
    let vd = normalize(vec3(0., 0., -1.));
    
    // check faces - top
    if (piq(pp, ts[0], ts[1], ts[2], ts[3])) {
        let n = vec3(0., 1., 0.);
        let wsp = b.p + vec3(0., b.s.y, 0.);
        var lc = amb(fm, ao(wsp, n));
        for (var i = 0; i < 3; i++) { lc += ct(ls[i].p, vd, n, fm, ls[i].c, ls[i].i); }
        fc = lc; hit = true;
    }
    
    // left face
    if (!hit && piq(pp, bs[0], ts[0], ts[3], bs[3])) {
        let n = vec3(-1., 0., 0.);
        let wsp = b.p + vec3(-hw, b.s.y * .5, 0.);
        var lc = amb(m, ao(wsp, n));
        for (var i = 0; i < 3; i++) { lc += ct(ls[i].p, vd, n, m, ls[i].c, ls[i].i) * .8; }
        fc = lc; hit = true;
    }
    
    // right face  
    if (!hit && piq(pp, bs[1], bs[2], ts[2], ts[1])) {
        let n = vec3(1., 0., 0.);
        let wsp = b.p + vec3(hw, b.s.y * .5, 0.);
        var lc = amb(m, ao(wsp, n));
        for (var i = 0; i < 3; i++) { lc += ct(ls[i].p, vd, n, m, ls[i].c, ls[i].i) * .7; }
        fc = lc; hit = true;
    }
    
    // front face
    if (!hit && piq(pp, bs[0], bs[1], ts[1], ts[0])) {
        let n = vec3(0., 0., 1.);
        let wsp = b.p + vec3(0., b.s.y * .5, hd);
        var lc = amb(m, ao(wsp, n));
        for (var i = 0; i < 3; i++) { lc += ct(ls[i].p, vd, n, m, ls[i].c, ls[i].i) * .9; }
        fc = lc; hit = true;
    }
    
    // back face
    if (!hit && piq(pp, bs[2], bs[3], ts[3], ts[2])) {
        let n = vec3(0., 0., -1.);
        let wsp = b.p + vec3(0., b.s.y * .5, -hd);
        var lc = amb(m, ao(wsp, n));
        for (var i = 0; i < 3; i++) { lc += ct(ls[i].p, vd, n, m, ls[i].c, ls[i].i) * .6; }
        fc = lc; hit = true;
    }
    
    return fc;
}

// get moving block pos
fn gmbp() -> vec3<f32> {
    let cb = gcb();
    if (gs() != 1u || cb == 0u) { return vec3(0., -100., 0.); }
    
    let th = f32(cb - 1u) * .6;
    let osc = sin(u_time.time * 2.) * 2.5;
    return vec3(osc, th + .6, 0.);
}

// init game
fn init() {
    if (u_time.frame == 1u) {
        // foundation
        sb(0u, Block(vec3(0., 0., 0.), vec3(4., .6, 4.), vec3(.8, .6, .4), 0.));
        
        ss(0u); ssc(0u); scb(1u); sct(false); scy(0.); sch(8.); sca(0.);
    }
}

// update game logic
fn upd() {
    let mc = (u_mouse.buttons.x & 1u) != 0u;
    let state = gs();
    
    updcam();
    
    // click detection
    let wc = gct();
    if (mc && !wc) {
        sct(true);
        
        if (state == 0u) {
            // start
            ss(1u); ssc(0u); scb(1u); scy(0.);
        }
        else if (state == 1u) {
            // drop block
            let cb = gcb();
            if (cb < 30u) {
                let mp = gmbp();
                let pb = gb(cb - 1u);
                
                var nb = Block(vec3(mp.x, f32(cb) * .6, mp.z), vec3(0., .6, pb.s.z), vec3(0.), 0.);
                
                // trimming
                let ox = abs(mp.x - pb.p.x);
                nb.s.x = max(pb.s.x - ox, .2);
                
                // perfect match check
                nb.perf = select(0., 1., ox < .05);
                
                // adjust pos
                nb.p.x = select(pb.p.x - (pb.s.x - nb.s.x) * .5, 
                               pb.p.x + (pb.s.x - nb.s.x) * .5, mp.x > pb.p.x);
                nb.p.z = pb.p.z;
                
                // material color
                let m = mat(cb);
                nb.c = m.alb;
                
                if (nb.s.x < .5) { ss(2u); } // game over
                else { sb(cb, nb); scb(cb + 1u); ssc(gsc() + 10u); }
            }
        }
        else if (state == 2u) {
            // reset
            ss(0u); ssc(0u); scb(1u); scy(0.);
            for (var i = 1u; i < 30u; i++) {
                sb(i, Block(vec3(0.), vec3(0.), vec3(0.), 0.));
            }
        }
    } else if (!mc) { sct(false); }
}

fn txt(pp: vec2<f32>, ss: vec2<f32>) -> vec3<f32> {
    let state = gs();
    var tc = vec3(0.);
    
    if (state == 0u) {
        // menu
        let title = array<u32, 16>(66u, 76u, 79u, 67u, 75u, 32u, 84u, 79u, 87u, 69u, 82u, 0u, 0u, 0u, 0u, 0u);
        if (word(pp, vec2(ss.x * .5 - 280., 100.), title, 11u, 64.) > .01) { tc = vec3(1., 1., 0.); }
        
        let sub = array<u32, 16>(67u, 76u, 73u, 67u, 75u, 32u, 84u, 79u, 32u, 83u, 84u, 65u, 82u, 84u, 0u, 0u);
        if (word(pp, vec2(ss.x * .5 - 200., 200.), sub, 14u, 32.) > .01) { tc = vec3(.8, .8, 1.); }
        
    } else if (state == 1u) {
        // playing
        let scl = array<u32, 16>(83u, 67u, 79u, 82u, 69u, 58u, 32u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        if (word(pp, vec2(50., 50.), scl, 7u, 48.) > .01) { tc = vec3(1.); }
        if (num(pp, vec2(50. + 7. * adv(48.), 50.), gsc(), 48.) > .01) { tc = vec3(.01, .01, .01); }
        
    } else if (state == 2u) {
        // game over
        let go = array<u32, 16>(71u, 65u, 77u, 69u, 32u, 79u, 86u, 69u, 82u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
        if (word(pp, vec2(ss.x * .5 - 240., ss.y * .5), go, 9u, 60.) > .01) { tc = vec3(1., .2, .2); }
        
        let rst = array<u32, 16>(67u, 76u, 73u, 67u, 75u, 32u, 84u, 79u, 32u, 82u, 69u, 83u, 84u, 65u, 82u, 84u);
        if (word(pp, vec2(ss.x * .5 - 220., ss.y * .5 + 100.), rst, 16u, 32.) > .01) { tc = vec3(.1); }
    }
    
    return tc;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ss = vec2<f32>(textureDimensions(output));
    let pp = vec2<f32>(gid.xy);
    
    if any(pp >= ss) { return; }
    
    // single thread updates
    if (all(gid.xy == vec2(0u))) { init(); upd(); }
    
    // background
    let ny = pp.y / ss.y;
    let nx = pp.x / ss.x;
    
    let sky = vec3(.4, .7, 1.);
    let hor = vec3(.9, .6, .3);
    let gnd = vec3(.2, .3, .4);
    
    let noise = sin(pp.x * .01 + u_time.time * .5) * sin(pp.y * .01) * .1;
    
    var col = select(mix(gnd, hor, ny * 1.67), mix(hor, sky, (ny - .6) * 2.5), ny > .6);
    col += noise * vec3(.02, .02, .04);
    
    // vignette
    let vig = smoothstep(0., .3, min(nx, min(1. - nx, min(ny, 1. - ny))));
    col = mix(col * .7, col, vig);
    
    let state = gs();
    
    if (state == 1u) {
        // playing - render blocks
        let cb = gcb();
        
        // placed blocks
        for (var i = 0u; i < cb && i < 30u; i++) {
            let b = gb(i);
            if (b.s.x > 0.) {
                let bc = rbl(pp, b, ss, i);
                if (length(bc) > 0.) { col = bc; }
            }
        }
        
        // moving block
        let mp = gmbp();
        if (mp.y > -50.) {
            let pb = gb(cb - 1u);
            let m = mat(cb);
            let pulse = sin(u_time.time * 8.) * .3 + .7;
            let mb = Block(mp, vec3(pb.s.x, .6, pb.s.z), m.alb * pulse, 0.);
            
            let mbc = rbl(pp, mb, ss, cb);
            if (length(mbc) > 0.) { col = mbc; }
        }
    } else {
        // menu/gameover - foundation
        let f = gb(0u);
        let fc = rbl(pp, f, ss, 0u);
        if (length(fc) > 0.) { col = fc; }
        
        if (state == 0u) { col *= sin(u_time.time * 3.) * .2 + .8; }
        else if (state == 2u) { col = mix(col, vec3(1., .3, .3), .3); }
    }
    
    // text overlay
    let tc = txt(pp, ss);
    if (length(tc) > 0.) { col = tc; }
    
    // post processing: tone map vig, gamma etc etc
    col *= 1.2; // exposure
    col = (col * (2.51 * col + .03)) / (col * (2.43 * col + .59) + .14);
    col = pow(col, vec3(1. / 2.2)); // gamma
    col = col * .8 + .2 * col * col * (3. - 2. * col); 
    col *= vec3(1.05, 1., 1.02); 
    
    let uv = pp / ss;
    let vf = 1. - .15 * smoothstep(.3, 1., distance(uv, vec2(.5)));
    col = clamp(col * vf, vec3(0.), vec3(1.));
    
    textureStore(output, vec2<i32>(gid.xy), vec4(col, 1.));
}