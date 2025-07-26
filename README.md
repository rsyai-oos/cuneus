[![Shader Binary Release](https://github.com/altunenes/cuneus/actions/workflows/release.yaml/badge.svg)](https://github.com/altunenes/cuneus/actions/workflows/release.yaml) [![crates.io](https://img.shields.io/crates/v/Cuneus.svg)](https://crates.io/crates/Cuneus)

<img src="https://github.com/user-attachments/assets/590dbd91-5eaa-4c04-b3f9-d579924fa4c3" alt="cuneus sdf" width="320" height="120" />


A tool for experimenting with WGSL shaders, it uses `wgpu` for rendering, `egui` for the UI and `winit` for windowing :-)

### Current Features

- Hot shader reloading
- Compute & Fragment shader support 
- Multi-pass, atomics etc
- Interactive parameter adjustment, ez media imports through egui
- Easily use HDR textures via UI
- Easily use your own videos/webcam as textures
- Audio/Visual synchronization: Spectrum and BPM detection
- Real-time audio synthesis: Generate music directly from wgsl shaders
- Export HQ frames via egui


## Current look

  <a href="https://github.com/user-attachments/assets/25d47df4-45f5-4455-b2cf-ba673a8c081c">
    <img src="https://github.com/user-attachments/assets/25d47df4-45f5-4455-b2cf-ba673a8c081c" width="300" alt="Cuneus IDE Interface"/>
  </a>

## Keys

- `F` full screen/minimal screen, `H` hide egui

#### Usage

- If you want to try your own shaders, check out the [usage.md](usage.md).
- **Optional Media Support**: GStreamer dependencies are optional - use `--no-default-features` for lightweight builds with pure GPU compute shaders.
- **When using cuneus as a dependency** (via `cargo add`):
  - Add `bytemuck = { version = "1", features = ["derive"] }` to dependencies (derive macros can't be re-exported)
  - Copy [build.rs](build.rs) to your project root to configure `GStreamer` paths (only needed for media features)
  - then simply use `use cuneus::prelude::*;`


#### Run examples

- `cargo run --release --bin *file*`
- Or download on the [releases](https://github.com/altunenes/cuneus/releases)
- Or, as the best method, use tui browser via ratatui (thanks to `davehorner`): 
    
     `cargo run --example tui_browser`



# Gallery

| **Sinh** | **JFA** | **Volumetric Passage** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/a80d2415-fbb2-4335-bbc3-b74b7a8170ad"><img src="https://github.com/user-attachments/assets/823a3def-b822-42ed-906b-e419fa490634" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/sinh.rs) | <a href="https://github.com/user-attachments/assets/f07023a3-0d93-4740-a95c-49f16d815e29"><img src="https://github.com/user-attachments/assets/8c71ce99-58ff-4354-9c0a-0a0fd4e5032d" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/jfa.rs) | <a href="https://github.com/user-attachments/assets/c19365ac-267f-4301-a9c8-42097d4b167a"><img src="https://github.com/user-attachments/assets/5ef301cd-cb11-4850-b013-13537939fd22" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/volumepassage.rs)|

| **PathTracing Mandelbulb** | **Lich** | **Galaxy** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/24083cae-7e96-4726-8509-fb3d5973308a"><img src="https://github.com/user-attachments/assets/e454b395-a1a0-4b91-a776-9afd1a789d23" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/mandelbulb.rs) | <a href="https://github.com/user-attachments/assets/9589d2ec-43b8-4373-8dce-9cd2c74d862f"><img src="https://github.com/user-attachments/assets/9589d2ec-43b8-4373-8dce-9cd2c74d862f" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/lich.rs) | <a href="https://github.com/user-attachments/assets/a2647904-55bd-4912-9713-4558203ee6aa"><img src="https://github.com/user-attachments/assets/a2647904-55bd-4912-9713-4558203ee6aa" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/galaxy.rs) |

| **Buddhabrot** | **FFT(Butterworth filter)** | **Clifford** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/93a17f27-695a-4249-9ff8-be2742926358"><img src="https://github.com/user-attachments/assets/93a17f27-695a-4249-9ff8-be2742926358" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/buddhabrot.rs) | <a href="https://github.com/user-attachments/assets/5806af3b-a640-433c-b7ec-1ca051412300"><img src="https://github.com/user-attachments/assets/e1e7f7e9-5979-43fe-8bb0-ccda8e428fe5" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/fft.rs) | <a href="https://github.com/user-attachments/assets/8b078f40-a989-4d07-bb2f-d19d8232cc9f"><img src="https://github.com/user-attachments/assets/8b078f40-a989-4d07-bb2f-d19d8232cc9f" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/cliffordcompute.rs) |


| **Block Tower: 3D Game** | **Code Sound:Veridis Quo** | **Vision LM via fastvlm** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/9ce52cc1-31c0-4e50-88c7-2fb06d1a57b3"><img src="https://github.com/user-attachments/assets/9ce52cc1-31c0-4e50-88c7-2fb06d1a57b3" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/blockgame.rs) | <a href="https://github.com/user-attachments/assets/e629cb9c-2f22-40e3-8cb1-9b9fb867c1d2"><img src="https://github.com/user-attachments/assets/e629cb9c-2f22-40e3-8cb1-9b9fb867c1d2" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/veridisquo.rs) | <a href="https://github.com/user-attachments/assets/b0596266-882c-4231-97bd-5deb59e5f79e"><img src="https://github.com/user-attachments/assets/b0596266-882c-4231-97bd-5deb59e5f79e" width="250"/></a><br/>[Code](https://github.com/altunenes/calcarine) |


| **water** | **path tracer** | **audio visualizer** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/465dae75-2bbc-4b4e-8384-054cfdf9f129"><img src="https://github.com/user-attachments/assets/dbcc8c37-4cf0-4c46-99f0-2f33ceed395b" width="250" height ="200"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/water.rs) | <a href="https://github.com/user-attachments/assets/45b8f532-f3fb-453c-b356-1d3c153d614a"><img src="https://github.com/user-attachments/assets/896228c3-7583-40de-9643-8b58aaec6050" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/pathtracing.rs) | <a href="https://github.com/user-attachments/assets/3eda9c33-7961-4dd4-aad1-170ae32640e7"><img src="https://github.com/user-attachments/assets/3eda9c33-7961-4dd4-aad1-170ae32640e7" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/audiovis.rs) |

| **Poe2:loading** | **tree** | **voronoi** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/fa588334-dd8d-492d-9caa-1aaeaecf024b"><img src="https://github.com/user-attachments/assets/fa588334-dd8d-492d-9caa-1aaeaecf024b" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/poe2.rs) | <a href="https://github.com/user-attachments/assets/2f0bdc7c-d226-4091-bae7-b96561c1fb4f"><img src="https://github.com/user-attachments/assets/2f0bdc7c-d226-4091-bae7-b96561c1fb4f" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/tree.rs) | <a href="https://github.com/user-attachments/assets/6c68d718-872c-4e14-bccb-f2339cf121d2"><img src="https://github.com/user-attachments/assets/6c68d718-872c-4e14-bccb-f2339cf121d2" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/voronoi.rs) |
