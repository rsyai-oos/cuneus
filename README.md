[![Shader Binary Release](https://github.com/altunenes/cuneus/actions/workflows/release.yaml/badge.svg)](https://github.com/altunenes/cuneus/actions/workflows/release.yaml) [![crates.io](https://img.shields.io/crates/v/Cuneus.svg)](https://crates.io/crates/Cuneus)

<img src="https://github.com/user-attachments/assets/590dbd91-5eaa-4c04-b3f9-d579924fa4c3" alt="cuneus sdf" width="320" height="120" />


A tool for experimenting with WGSL shaders, it uses `wgpu` for rendering, `egui` for the UI, `winit` for windowing, and `notify` for hot-reload. :-)

### Current Features

- Hot shader reloading
- Multi-pass, atomics etc
- Interactive parameter adjustment, ez Texture loading through egui
- Easily use your own videos as textures (thanks to the `gstreamer`)
- Audio/Visual synchronization: Spectrum and BPM detection via `gstreamer`
- Export HQ frames via egui


## Current look

  <a href="https://github.com/user-attachments/assets/7eea9b94-875a-4e01-9204-3da978d3cd65">
    <img src="https://github.com/user-attachments/assets/7eea9b94-875a-4e01-9204-3da978d3cd65" width="300" alt="Cuneus IDE Interface"/>
  </a>

## Keys

- `F` full screen/minimal screen, `H` hide egui

#### Usage

- If you want to try your own shaders, check out the [usage.md](usage.md).

#### Run examples

- `cargo run --release --bin *file*`
- Or download on the [releases](https://github.com/altunenes/cuneus/releases)
- Or, as the best method, use tui browser via ratatui (thanks to `davehorner`): 
    
     `cargo run --example tui_browser`


# Gallery

| **Sinh** | **Signed Distance** | **Satan** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/a80d2415-fbb2-4335-bbc3-b74b7a8170ad"><img src="https://github.com/user-attachments/assets/823a3def-b822-42ed-906b-e419fa490634" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/sinh.rs) | <a href="https://github.com/user-attachments/assets/1847c374-5719-4fee-b74d-3418e5fa4d7b"><img src="https://github.com/user-attachments/assets/1847c374-5719-4fee-b74d-3418e5fa4d7b" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/sdvert.rs) | <a href="https://github.com/user-attachments/assets/8f86a3b4-8d31-499f-b9fa-8b23266291ae"><img src="https://github.com/user-attachments/assets/8f86a3b4-8d31-499f-b9fa-8b23266291ae" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/satan.rs) |

| **Mandelbulb** | **Lich** | **Galaxy** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/2405334c-f13e-4d8d-863f-bab7dcc676ab"><img src="https://github.com/user-attachments/assets/2405334c-f13e-4d8d-863f-bab7dcc676ab" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/mandelbulb.rs) | <a href="https://github.com/user-attachments/assets/9589d2ec-43b8-4373-8dce-9cd2c74d862f"><img src="https://github.com/user-attachments/assets/9589d2ec-43b8-4373-8dce-9cd2c74d862f" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/lich.rs) | <a href="https://github.com/user-attachments/assets/a2647904-55bd-4912-9713-4558203ee6aa"><img src="https://github.com/user-attachments/assets/a2647904-55bd-4912-9713-4558203ee6aa" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/galaxy.rs) |

| **Xmas** | **Droste** | **Clifford** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/4f1f0cc0-12a5-4158-90e1-ac205fa2d28a"><img src="https://github.com/user-attachments/assets/4f1f0cc0-12a5-4158-90e1-ac205fa2d28a" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/xmas.rs) | <a href="https://github.com/user-attachments/assets/ffe1e193-9a9a-4784-8193-177d6b8648af"><img src="https://github.com/user-attachments/assets/ffe1e193-9a9a-4784-8193-177d6b8648af" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/droste.rs) | <a href="https://github.com/user-attachments/assets/42868686-bad9-4ce3-b5bd-346d880c8540"><img src="https://github.com/user-attachments/assets/42868686-bad9-4ce3-b5bd-346d880c8540" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/clifford.rs) |


| **orbits** | **hilbert room** | **genuary6** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/54dcd781-30af-46fb-aeda-2d2d607b0742"><img src="https://github.com/user-attachments/assets/951b30d6-6f8d-4fc7-884f-eec496fb3885" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/orbits.rs) | <a href="https://github.com/user-attachments/assets/bc596e6b-9304-48ba-b509-140544450f5d"><img src="https://github.com/user-attachments/assets/bc596e6b-9304-48ba-b509-140544450f5d" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/hilbert.rs) | <a href="https://github.com/user-attachments/assets/be2e132a-a473-462d-8b5b-2277336c7e78"><img src="https://github.com/user-attachments/assets/be2e132a-a473-462d-8b5b-2277336c7e78" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/genuary2025_6.rs) |


| **rorschach** | **nebula** | **audio visualizer** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/320c977d-1e64-4e44-9a8c-03779b70f025"><img src="https://github.com/user-attachments/assets/320c977d-1e64-4e44-9a8c-03779b70f025" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/rorschach.rs) | <a href="https://github.com/user-attachments/assets/5f230955-4115-4695-955c-8df2d4bba5af"><img src="https://github.com/user-attachments/assets/26d4b3a4-f9b5-45df-b43a-160e00520bfe" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/nebula.rs) | <a href="https://github.com/user-attachments/assets/3eda9c33-7961-4dd4-aad1-170ae32640e7"><img src="https://github.com/user-attachments/assets/3eda9c33-7961-4dd4-aad1-170ae32640e7" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/audiovis.rs) |

| **Poe2:loading** | **tree** | **voronoi** |
|:---:|:---:|:---:|
| <a href="https://github.com/user-attachments/assets/fa588334-dd8d-492d-9caa-1aaeaecf024b"><img src="https://github.com/user-attachments/assets/fa588334-dd8d-492d-9caa-1aaeaecf024b" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/poe2.rs) | <a href="https://github.com/user-attachments/assets/2f0bdc7c-d226-4091-bae7-b96561c1fb4f"><img src="https://github.com/user-attachments/assets/2f0bdc7c-d226-4091-bae7-b96561c1fb4f" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/tree.rs) | <a href="https://github.com/user-attachments/assets/6c68d718-872c-4e14-bccb-f2339cf121d2"><img src="https://github.com/user-attachments/assets/6c68d718-872c-4e14-bccb-f2339cf121d2" width="250"/></a><br/>[Code](https://github.com/altunenes/cuneus/blob/main/src/bin/voronoi.rs) |

