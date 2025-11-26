// Enes Altun 2025, CC BY-NC-SA 3.0
struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
}
@group(0) @binding(0) var<uniform> u_time: TimeUniform;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var<uniform> p: LegoParams;

@group(2) @binding(0) var ch0: texture_2d<f32>;
@group(2) @binding(1) var ch0_s: sampler;

struct LegoParams {
    scl: f32,       // brick_scale
    lx: f32, ly: f32, // lightdir
    grn: f32,       // grain
    gam: f32,       // gamma
    sh_s: f32, sh_d: f32, // shadow str, dist
    ao_s: f32,      // ao str
    sp_p: f32, sp_s: f32, // spec pow, str
    enh: f32,       // edge enh
    sh: f32, bh: f32, // stud h, base h
    rim: f32,       // rim str
    rsm: f32,       // res scale mult
    shm: f32,       // stud h mult
    lr: f32, lg: f32, lb: f32, // light rgb
    dsc: f32,       // depth scale
    ebl: f32,       // edge blend
    _p1: f32, _p2: f32, _p3: f32,
}

alias v2 = vec2<f32>;
alias v3 = vec3<f32>;
alias v4 = vec4<f32>;

fn aces(x:v3)->v3{
    let a=2.51; let b=.03; let c=2.43; let d=.59; let e=.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e),v3(0.),v3(1.));
}

fn rnd(s:v2)->f32{
    return fract(sin(dot(s,v2(12.9898,78.233)))*43758.5453);
}

fn hmap(uv:v2, rs:f32, ls:f32)->f32{
    let sr=.3;
    let sb=.005+ls*.2;
    let br=.7;
    let bb=.005+ls*.1;
    let sm=(.5/rs)+ls;
    
    let db=abs(uv)-(br-bb);
    let bs=length(max(db,v2(0.)))+min(max(db.x,db.y),0.)-bb;
    let h=(1.-smoothstep(0.,sm,bs))*p.bh;
    
    let ds=length(uv)-(sr-sb);
    let hs=(1.-smoothstep(0.,sm*.5+ls,ds))*p.sh;
    
    return h+hs*(p.shm-smoothstep(0.,sm,bs));
}

fn nmap(uv:v2, scl:f32, rs:f32, ls:f32)->v3{
    let em=1.+ls*5.;
    let e=v2(scl*2.*em,0.);
    let dx=hmap(uv+e.xy,rs,ls)-hmap(uv-e.xy,rs,ls);
    let dy=hmap(uv+e.yx,rs,ls)-hmap(uv-e.yx,rs,ls);
    return normalize(v3(dx,dy,.1+ls*.5));
}

@compute @workgroup_size(16,16,1)
fn main_image(@builtin(global_invocation_id) g:vec3<u32>){
    let d=textureDimensions(output);
    if(g.x>=d.x||g.y>=d.y){return;}
    
    let R=v2(f32(d.x),f32(d.y));
    let fc=v2(f32(g.x),f32(g.y));
    let uv=fc/R;
    
    let su=fc*p.scl;
    let bid=floor(su+.5);
    let lu=su-bid;
    let mu=bid/p.scl/R;
    
    // LOD & AA
    let ppb=1./p.scl;
    let ls=smoothstep(12.,3.,ppb);
    let ml=max(0.,log2(p.scl*1000.)*-1.);
    
    let bc=textureSampleLevel(ch0,ch0_s,mu,ml).rgb;
    let rs=R.y*p.scl*p.rsm;
    let h=hmap(lu,rs,ls);
    
    if(h<=.001){
        let gc=mix(v3(.05),bc*.5,ls);
        textureStore(output,vec2<i32>(g.xy),v4(gc,1.));
        return;
    }
    
    let n=nmap(lu,p.scl,rs,ls);
    
    // Edges
    let bd=abs(lu)-.7;
    let bs=length(max(bd,v2(0.)))+min(max(bd.x,bd.y),0.);
    let sd=length(lu)-.3;
    let ed=min(abs(bs),abs(sd));
    let ee=(1.1/max(ed,.01))*(1.-ls);
    let ef=smoothstep(0.,2.,ee);
    
    // AO & Shadow
    var ao=1.;
    let l2=v2(p.lx,p.ly);
    
    if(ls<.8){
        let dc=length(lu);
        if(dc<.3){ao*=1.-((1.-(dc/.3))*.4);}
        
        var sh=0.;
        let sam=12;
        for(var i=0;i<sam;i++){
            let t=f32(i)/f32(sam-1);
            let sp=lu-l2*(.05+t*p.sh_d);
            let ho=hmap(sp,rs,ls);
            sh+=max(0.,ho-h)*(1.-t)*p.sh_s;
        }
        ao*=1.-smoothstep(0.,1.,sh);
    }
    
    let eu=lu+.5;
    let ex=smoothstep(.75,.95,eu.x);
    let ey=smoothstep(.25,.05,eu.y);
    ao*=pow(1.-max(ex,ey)*.75,p.ao_s);
    
    // Lighting
    let l=normalize(v3(l2,1.));
    let lc=v3(p.lr,p.lg,p.lb);
    let v=normalize(v3(0.,0.,1.));
    
    var acc=v3(0.);
    var tw=0.;
    let dsam=4;
    
    for(var i=0;i<dsam;i++){
        let dl=f32(i)/f32(dsam-1);
        let sc=p.dsc+dl*(1.-p.dsc);
        let su=lu*sc;
        let sh=hmap(su,rs,ls);
        if(sh>0.){
            let sn=nmap(su,p.scl,rs,ls);
            let ld=max(0.,dot(sn,l));
            let la=sn.z*.5+.5;
            let w=(1.-dl*.3)*(.5+.5*ld);
            acc+=(la+ld)*w;
            tw+=w;
        }
    }
    acc/=max(tw,.1);
    
    let sky=v3(.2,.25,.3);
    let gnd=bc*.1;
    let amb=mix(gnd,sky,n.z*.5+.5)*ao*acc.x;
    let diff=max(0.,dot(n,l))*acc.y*lc;
    
    let h_vec=normalize(l+v);
    let spec=pow(max(0.,dot(n,h_vec)),p.sp_p)*lc*p.sp_s;
    
    let rn=normalize(v3(lu,h*2.));
    let rim=pow(1.-abs(dot(rn,v)),2.)*diff*p.rim;
    
    var col=bc*(amb+diff)+spec;
    
    // Soft Shade 
    let ssd=lu.y/max(length(lu),.01);
    let ss=.5+.5*ssd*diff;
    
    col=mix(col,col+bc*ss*ee*p.enh,ef*p.ebl);
    col+=rim*lc;
    
    let grn=(rnd(uv)-.5)*p.grn*(1.-ls);
    col+=grn;
    
    col=pow(aces(col),v3(1./p.gam));
    textureStore(output,vec2<i32>(g.xy),v4(col,1.));
}