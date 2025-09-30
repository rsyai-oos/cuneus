// Enes Altun, 2025; MIT License
// Veridis Quo - Mathematical/Shader Approach
// Base frequencies for the main notes used in the song. This is also my first shader song, but I think it could be a nice example for cuneus. 
// This song also probably always WIP, I will keep improving it over time by the time I implement more advanced audio synthesis techniques on cuneus.
// Note numbers (basically tabs) based on my guitar feelings :-P so don't be confuse about those numbers and sorry for ignorance about music theory :D

struct TimeUniform {
    time: f32,
    delta: f32,
    frame: u32,
    _padding: u32,
};
@group(0) @binding(0) var<uniform> u_time: TimeUniform;
@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

struct SongParams {
    volume: f32,
    octave_shift: f32,
    tempo_multiplier: f32,
    waveform_type: u32,
    crossfade: f32,
    reverb_mix: f32,
    chorus_rate: f32,
    _padding: f32,
};
@group(1) @binding(1) var<uniform> u_song: SongParams;

struct FontUniforms {
    atlas_size: vec2<f32>,
    char_size: vec2<f32>,
    screen_size: vec2<f32>,
    _padding: vec2<f32>,
};
@group(2) @binding(0) var<uniform> u_font: FontUniforms;
@group(2) @binding(1) var t_font_atlas: texture_2d<f32>;
@group(2) @binding(2) var<storage, read_write> audio_buffer: array<f32>;

const PI=3.14159265;
const F5=698.46;
const E5=659.25;
const D5=587.33;
const C5=523.25;
const B4=493.88;
const A4=440.;
const F3=F5/4.;
const B2=B4/4.;
const E3=E5/4.;
const A2=A4/4.;

fn legato(a:f32,b:f32,t:f32)->f32{
    let s=1.-clamp(u_song.crossfade,0.,1.);
    if(t<=s){return a;}
    return mix(a,b,smoothstep(s,1.,t));
}

fn note_col(n:f32)->vec3<f32>{
    switch u32(n){
        case 0u:{return vec3(1.,.3,.3);}
        case 1u:{return vec3(1.,.6,0.);}
        case 2u:{return vec3(1.,1.,.2);}
        case 3u:{return vec3(.3,1.,.3);}
        case 4u:{return vec3(.2,.7,1.);}
        case 5u:{return vec3(.8,.3,1.);}
        default:{return vec3(.5);}
    }
}

fn measure_data(m:u32)->vec2<f32>{
    switch m{
        case 0u,4u:{return vec2(F5,5.);}
        case 1u,5u:{return vec2(B4,1.);}
        case 2u,6u:{return vec2(E5,4.);}
        case 3u,7u:{return vec2(A4,0.);}
        default:{return vec2(A4,0.);}
    }
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) g:vec3<u32>){
    let d=textureDimensions(output);
    if(g.x>=d.x||g.y>=d.y){return;}
    var mf=0.;
    var ev=0.;
    var nt=0.;

    if(g.x<1u&&g.y<1u){
        let T=u_time.time*u_song.tempo_multiplier;
        let md=(60./107.)*4.;
        let td=md*8.;
        let lt=T%td;
        let m=u32(lt/md);
        let pm=fract(lt/md);
        var mel=0.;
        var ma=1.;
        var bas=0.;
        var ba=.7;
        let pd=.25;
        let sh=.125;
        let p2s=pd+sh;
        let p2e=p2s+pd;
        
        switch m{
            case 0u,4u:{
                bas=F3;
                if(pm<pd){
                    let p=pm/pd;
                    let n=fract(p*4.);
                    switch u32(floor(p*4.)){
                        case 0u:{mel=legato(F5,E5,n);nt=5.;}
                        case 1u:{mel=legato(E5,F5,n);nt=4.;}
                        case 2u:{mel=legato(F5,D5,n);nt=5.;}
                        default:{mel=D5;nt=3.;}
                    }
                }
                else if(pm<p2s){mel=D5;nt=3.;}
                else if(pm<p2e){
                    let p=(pm-p2s)/pd;
                    let n=fract(p*4.);
                    switch u32(floor(p*4.)){
                        case 0u:{mel=legato(F5,E5,n);nt=5.;}
                        case 1u:{mel=legato(E5,F5,n);nt=4.;}
                        case 2u:{mel=legato(F5,B4,n);nt=5.;}
                        default:{mel=B4;nt=1.;}
                    }
                }
                else{mel=B4;nt=1.;}
            }
            case 1u,5u:{
                mel=B4;bas=B2;nt=1.;
                let dc=mix(1.,.1,pm);
                let tr=sin(T*8.)*.05;
                let rb=sin(pm*PI)*u_song.reverb_mix*.5;
                ma=dc+tr+rb;
            }
            case 2u:{
                bas=E3;
                if(pm<pd){
                    let p=pm/pd;
                    let n=fract(p*4.);
                    switch u32(floor(p*4.)){
                        case 0u:{mel=legato(E5,D5,n);nt=4.;}
                        case 1u:{mel=legato(D5,E5,n);nt=3.;}
                        case 2u:{mel=legato(E5,C5,n);nt=4.;}
                        default:{mel=C5;nt=2.;}
                    }
                }
                else if(pm<p2s){mel=C5;nt=2.;}
                else if(pm<p2e){
                    let p=(pm-p2s)/pd;
                    let n=fract(p*4.);
                    switch u32(floor(p*4.)){
                        case 0u:{mel=legato(E5,D5,n);nt=4.;}
                        case 1u:{mel=legato(D5,E5,n);nt=3.;}
                        case 2u:{mel=legato(E5,A4,n);nt=4.;}
                        default:{mel=A4;nt=0.;}
                    }
                }
                else{mel=A4;nt=0.;}
            }
            case 3u,7u:{
                mel=A4;bas=A2;nt=0.;
                let dc=mix(1.,.15,pm);
                let tr=sin(T*8.)*.05;
                let rb=sin(pm*PI)*u_song.reverb_mix*.5;
                ma=dc+tr+rb;
            }
            case 6u:{
                bas=E3;
                let rd=.5;
                if(pm<rd){
                    let p=pm/rd;
                    let n=fract(p*8.);
                    switch u32(floor(p*8.)){
                        case 0u:{mel=legato(E5,D5,n);nt=4.;}
                        case 1u:{mel=legato(D5,E5,n);nt=3.;}
                        case 2u:{mel=legato(E5,C5,n);nt=4.;}
                        case 3u:{mel=legato(C5,E5,n);nt=2.;}
                        case 4u:{mel=legato(E5,D5,n);nt=4.;}
                        case 5u:{mel=legato(D5,E5,n);nt=3.;}
                        case 6u:{mel=legato(E5,A4,n);nt=4.;}
                        default:{mel=A4;nt=0.;}
                    }
                }
                else{mel=A4;nt=0.;}
            }
            default:{}
        }
        if(m==1u||m==3u||m==5u||m==7u){ba*=1.-pm*.5;}
        mf=mel;
        ev=clamp(ma,0.,1.);
        let fm=mel*pow(2.,u_song.octave_shift);
        let fb=bas*pow(2.,u_song.octave_shift);
        let fma=ma*u_song.volume;
        let fba=ba*u_song.volume*.7;
        var cm=1.;
        if(u_song.chorus_rate>0.){
            cm=1.+sin(T*u_song.chorus_rate)*.005;
        }
        
        audio_buffer[0]=mf;
        audio_buffer[1]=ev;
        audio_buffer[2]=f32(u_song.waveform_type);
        audio_buffer[3]=fm*cm;
        audio_buffer[4]=fma;
        audio_buffer[5]=fb*cm;
        audio_buffer[6]=fba;
        for(var i=2u;i<16u;i++){
            audio_buffer[3u+i*2u]=0.;
            audio_buffer[3u+i*2u+1u]=0.;
        }
    }
    
    let freq=audio_buffer[0];
    let env=audio_buffer[1];
    let uv=vec2<f32>(g.xy)/vec2<f32>(d);
    var col=vec3(.02,.01,.08);
    let cy=.5;
    let pw=.8;
    let px=.1;
    let ph=.4;
    let md=(60./107.)*4.;
    let td=md*8.;
    let st=u_time.time*u_song.tempo_multiplier;
    let cm=u32((st%td)/md);
    
    for(var m=0u;m<8u;m++){
        let mw=pw/8.;
        let mx=px+f32(m)*mw;
        if(uv.x>=mx&&uv.x<=mx+mw*.9){
            let mi=measure_data(m);
            let mf=mi.x;
            let mn=mi.y;
            let fr=(mf-440.)/(698.46-440.);
            let bh=mix(.1,ph,fr);
            let bb=cy-bh*.5;
            let bt=cy+bh*.5;
            if(uv.y>=bb&&uv.y<=bt){
                if(m==cm&&env>.01){
                    col=vec3(1.,.9,.2)*(sin(u_time.time*8.)*.4+.8);
                }else{
                    col=note_col(mn)*.6;
                }
            }
        }
    }
    
    if(uv.y<.12){
        let sf=mix(400.,800.,uv.x);
        let fd=abs(sf-freq);
        let fr=exp(-fd/30.);
        let si=fr*env;
        let sh=uv.y/.12;
        if(sh<si&&env>.01){col+=vec3(si*.8,si*.4,.1);}
    }
    
    let pbh=.02;
    if(uv.y>.95&&uv.y<.95+pbh){
        let sp=(st%td)/td;
        if(uv.x<sp){
            col=mix(col,vec3(0.,.7,1.),.8);
        }else{
            col=mix(col,vec3(.15,.15,.3),.8);
        }
    }
    
    let ag=env*.1;
    col+=vec3(ag*.1,ag*.3,ag*.1);
    textureStore(output,g.xy,vec4(col,1.));
}