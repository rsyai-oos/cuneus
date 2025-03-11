use std::time::Instant;
use egui_wgpu::ScreenDescriptor;
use egui::ViewportId;
use crate::gst::video::VideoTextureManager;
use std::path::Path;
use log::{warn, info, error};
use crate::{Core, Renderer, TextureManager, UniformProvider, UniformBinding,KeyInputHandler,ExportManager,ShaderControls,ControlsRequest,ResolutionUniform};
#[cfg(target_os = "macos")]
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8UnormSrgb;
#[cfg(not(target_os = "macos"))]
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimeUniform {
    pub time: f32,
    pub frame: u32,
}
impl UniformProvider for TimeUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}
pub struct BaseShader {
    pub renderer: Renderer,
    pub video_texture_manager: Option<VideoTextureManager>,
    pub using_video_texture: bool,
    pub texture_manager: Option<TextureManager>,
    pub egui_renderer: egui_wgpu::Renderer,
    pub egui_state: egui_winit::State,
    pub context: egui::Context,
    pub texture_bind_group_layout: wgpu::BindGroupLayout,
    pub start_time: Instant,
    pub time_uniform: UniformBinding<TimeUniform>,
    pub resolution_uniform: UniformBinding<ResolutionUniform>,
    pub key_handler: KeyInputHandler,
    pub export_manager: ExportManager,
    pub controls: ShaderControls,
    pub prev_audio_data: [[f32; 4]; 32],
}
impl BaseShader {
    pub fn new(
        core: &Core,
        vs_source: &str,
        fs_source: &str,
        bind_group_layouts: &[&wgpu::BindGroupLayout],
        fragment_entry: Option<&str>,
    ) -> Self {
        let time_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("time_bind_group_layout"),
        });
        let resolution_bind_group_layout = core.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("resolution_bind_group_layout"),
        });
        let time_uniform = UniformBinding::new(
            &core.device,
            "Time Uniform",
            TimeUniform { 
                time: 0.0,
                frame: 0,
            },
            &time_bind_group_layout,
            0,
        );
        let resolution_uniform = UniformBinding::new(
            &core.device,
            "Resolution Uniform",
            ResolutionUniform {
                dimensions: [core.size.width as f32, core.size.height as f32],
                _padding: [0.0, 0.0],
                audio_data: [[0.0; 4]; 32],
                audio_raw: [[0.0; 4]; 32],
                bpm: 0.0,
                _bpm_padding: [0.0, 0.0, 0.0],
            },
            &resolution_bind_group_layout,
            0,
        );
        let vs_shader = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Vertex Shader"),
            source: wgpu::ShaderSource::Wgsl(vs_source.into()),
        });
        let fs_shader = core.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fragment Shader"),
            source: wgpu::ShaderSource::Wgsl(fs_source.into()),
        });
        let texture_bind_group_layout = core.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            },
        );
        let pipeline_layout = core.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts,
            push_constant_ranges: &[],
        });
        let renderer = Renderer::new(
            &core.device,
            &vs_shader,
            &fs_shader,
            core.config.format,
            &pipeline_layout,
            fragment_entry, 
        );
        let context = egui::Context::default();
        let egui_state = egui_winit::State::new(
            context.clone(),
            ViewportId::default(),
            core.window(),
            None,
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &core.device,
            core.config.format,
            None,
            1,
            false,
        );

        //  default texture manager
        let texture_manager = Self::create_default_texture_manager(core, &texture_bind_group_layout);

        Self {
            renderer,
            video_texture_manager: None,
            using_video_texture: false,
            texture_manager: Some(texture_manager),
            egui_renderer,
            egui_state,
            context,
            texture_bind_group_layout,
            start_time: Instant::now(),
            time_uniform,
            resolution_uniform,
            key_handler: KeyInputHandler::new(),
            export_manager: ExportManager::new(),
            controls: ShaderControls::new(),
            prev_audio_data: [[0.0; 4]; 32],
        }
    }

    pub fn update_time(&mut self, queue: &wgpu::Queue) {
        self.time_uniform.data.time = self.start_time.elapsed().as_secs_f32();
        self.time_uniform.update(queue);
    }
    pub fn update_resolution(&mut self, queue: &wgpu::Queue, new_size: winit::dpi::PhysicalSize<u32>) {
        self.resolution_uniform.data.dimensions = [new_size.width as f32, new_size.height as f32];
        self.resolution_uniform.update(queue);
    }
    fn create_default_texture_manager(
        core: &Core,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> TextureManager {
        let default_texture = core.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Default Texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let default_view = default_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = core.device.create_sampler(&wgpu::SamplerDescriptor::default());

        let bind_group = core.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&default_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("Default Texture Bind Group"),
        });

        TextureManager {
            texture: default_texture,
            view: default_view,
            sampler,
            bind_group,
        }
    }

    pub fn render_ui<F>(&mut self, core: &Core, mut ui_builder: F) -> egui::FullOutput 
    where
        F: FnMut(&egui::Context),
    {
        let raw_input = self.egui_state.take_egui_input(core.window());
        self.context.run(raw_input, |ctx| ui_builder(ctx))
    }

    pub fn handle_render_output(
        &mut self,
        core: &Core,
        view: &wgpu::TextureView,
        full_output: egui::FullOutput,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [core.config.width, core.config.height],
            pixels_per_point: core.window().scale_factor() as f32,
        };

        let clipped_primitives = self.context.tessellate(
            full_output.shapes,
            screen_descriptor.pixels_per_point,
        );
        // Update egui textures
        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer.update_texture(
                &core.device,
                &core.queue,
                *id,
                image_delta,
            );
        }

        self.egui_renderer.update_buffers(
            &core.device,
            &core.queue,
            encoder,
            &clipped_primitives,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let mut render_pass = render_pass.forget_lifetime();
            self.egui_renderer.render(
                &mut render_pass,
                &clipped_primitives,
                &screen_descriptor,
            );
        }
        // Cleanup egui textures
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
    pub fn load_media<P: AsRef<Path>>(&mut self, core: &Core, path: P) -> anyhow::Result<()> {
        let path_ref = path.as_ref();
        let extension = path_ref.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase());
        
        match extension {
            // Image formats
            Some(ext) if ["png", "jpg", "jpeg", "bmp", "gif", "tiff", "webp"].contains(&ext.as_str()) => {
                info!("Loading image: {:?}", path_ref);
                if let Ok(img) = image::open(path_ref) {
                    let rgba_image = img.into_rgba8();
                    let new_texture_manager = TextureManager::new(
                        &core.device,
                        &core.queue,
                        &rgba_image,
                        &self.texture_bind_group_layout,
                    );
                    self.texture_manager = Some(new_texture_manager);
                    self.using_video_texture = false;
                    self.video_texture_manager = None;
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("Failed to open image"))
                }
            },
            Some(ext) if ["mp4", "avi", "mkv", "mov", "webm"].contains(&ext.as_str()) => {
                info!("Loading video: {:?}", path_ref);
                match VideoTextureManager::new(
                    &core.device,
                    &core.queue,
                    &self.texture_bind_group_layout,
                    path_ref,
                ) {
                    Ok(video_manager) => {
                        self.video_texture_manager = Some(video_manager);
                        self.using_video_texture = true;
                        if let Err(e) = self.play_video() {
                            warn!("Failed to play video: {}", e);
                        }
                        self.set_video_loop(true);
                        
                        Ok(())
                    },
                    Err(e) => {
                        error!("Failed to load video: {}", e);
                        Err(e)
                    }
                }
            },
            _ => {
                Err(anyhow::anyhow!("Unsupported media format: {:?}", path_ref))
            }
        }
    }
    pub fn update_video_texture(&mut self, core: &Core, queue: &wgpu::Queue) -> bool {
        if self.using_video_texture {
            if let Some(video_manager) = &mut self.video_texture_manager {
                if let Ok(updated) = video_manager.update_texture(
                    &core.device,
                    queue,
                    &self.texture_bind_group_layout
                ) {
                    return updated;
                }
            }
        }
        false
    }
    pub fn play_video(&mut self) -> anyhow::Result<()> {
        if let Some(video_manager) = &mut self.video_texture_manager {
            video_manager.play()?;
        }
        Ok(())
    }
    pub fn pause_video(&mut self) -> anyhow::Result<()> {
        if let Some(video_manager) = &mut self.video_texture_manager {
            video_manager.pause()?;
        }
        Ok(())
    }
    pub fn seek_video(&mut self, position_seconds: f64) -> anyhow::Result<()> {
        if let Some(video_manager) = &mut self.video_texture_manager {
            let position = gstreamer::ClockTime::from_seconds(position_seconds as u64);
            video_manager.seek(position)?;
        }
        Ok(())
    }
    pub fn set_video_loop(&mut self, should_loop: bool) {
        if let Some(video_manager) = &mut self.video_texture_manager {
            video_manager.set_loop(should_loop);
        }
    }
    pub fn load_image(&mut self, core: &Core, path: std::path::PathBuf) {
        if let Ok(img) = image::open(path) {
            let rgba_image = img.into_rgba8();
            let new_texture_manager = TextureManager::new(
                &core.device,
                &core.queue,
                &rgba_image,
                &self.texture_bind_group_layout,
            );
            self.texture_manager = Some(new_texture_manager);
        }
    }
    pub fn create_capture_texture(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::Buffer) {
        let capture_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Capture Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CAPTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let align = 256;
        let unpadded_bytes_per_row = width * 4;
        let padding = (align - unpadded_bytes_per_row % align) % align;
        let padded_bytes_per_row = unpadded_bytes_per_row + padding;
        let buffer_size = padded_bytes_per_row * height;
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Capture Buffer"),
            size: buffer_size as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        (capture_texture, output_buffer)
    }
    pub fn apply_control_request(&mut self, request: ControlsRequest) {
        if request.should_reset {
            self.start_time = Instant::now();
        }
        self.controls.apply_ui_request(request);
    }
    pub fn update_audio_spectrum(&mut self, queue: &wgpu::Queue) {
        // Initialize audio data arrays to zero
        for i in 0..32 {
            for j in 0..4 {
                self.resolution_uniform.data.audio_data[i][j] = 0.0;
                self.resolution_uniform.data.audio_raw[i][j] = 0.0;
            }
        }
        
        if self.using_video_texture {
            if let Some(video_manager) = &self.video_texture_manager {
                if video_manager.has_audio() {
                    let spectrum_data = video_manager.spectrum_data();
                    self.resolution_uniform.data.bpm = video_manager.get_bpm();
                    print!("BPM: {}", self.resolution_uniform.data.bpm);
                    if !spectrum_data.magnitudes.is_empty() {
                        let bands = spectrum_data.bands;
                        // Highly sensitive threshold for detecting subtle high frequencies
                        let threshold: f32 = -60.0;
                        // Store raw audio data with frequency-dependent processing
                        for i in 0..128.min(bands) {
                            // Linear frequency index for accurate frequency representation
                            let band_percent = i as f32 / 128.0;
                            // Map to source index with slight emphasis on higher frequencies
                            let src_idx = (band_percent * (0.8 + band_percent * 0.2) * bands as f32) as usize;
                            if src_idx < bands {
                                // Get value
                                let val = spectrum_data.magnitudes[src_idx];
                                // Normalize from dB to 0-1 with frequency-dependent boost
                                // Higher frequencies get progressively more boost
                                let freq_boost = if band_percent < 0.6 {
                                    // No boost for low/mid frequencies
                                    1.0
                                } else {
                                    // Progressive boost for high frequencies
                                    // Linear increase from 1.0 at 60% to 4.0 at 100%
                                    1.0 + (band_percent - 0.6) * 7.5
                                };
                                
                                // Basic normalization
                                let normalized = ((val - threshold) / -threshold).max(0.0).min(1.0);
                                // Apply frequency boost with limiting
                                let boosted = (normalized * freq_boost).min(1.0);
                                // Store in raw audio array
                                let vec_idx = i / 4;
                                let vec_component = i % 4;
                                if vec_idx < 32 {
                                    self.resolution_uniform.data.audio_raw[vec_idx][vec_component] = boosted;
                                }
                            }
                        }
                        // Process enhanced audio data with accurate representation
                        for i in 0..128.min(bands) {
                            let band_percent = i as f32 / 128.0;
                            // Similar conservative mapping as above
                            let source_idx = (band_percent * (0.8 + band_percent * 0.2) * bands as f32) as usize;
                            // Use narrow width for all frequencies for accuracy
                            let width = 1;
                            let end_idx = (source_idx + width).min(bands);
                            if source_idx < bands {
                                // Get peak value in this range
                                let mut peak: f32 = -120.0;
                                for j in source_idx..end_idx {
                                    if j < bands {
                                        let val = spectrum_data.magnitudes[j];
                                        peak = peak.max(val);
                                    }
                                }
                                // Map from dB scale to 0-1
                                let normalized = ((peak - threshold) / -threshold).max(0.0).min(1.0);
                                // Apply frequency-specific processing that's balanced
                                // Lower boost for bass, higher boost for treble
                                let enhanced = if band_percent < 0.2 {
                                    // Bass - slightly reduced
                                    (normalized.powf(0.75) * 0.85).min(1.0)
                                } else if band_percent < 0.4 {
                                    // Low-mids - neutral
                                    normalized.powf(0.7).min(1.0)
                                } else if band_percent < 0.6 {
                                    // Mids - slight boost
                                    (normalized.powf(0.65) * 1.1).min(1.0)
                                } else if band_percent < 0.8 {
                                    // Upper-mids - moderate boost
                                    (normalized.powf(0.55) * 1.6).min(1.0)
                                } else {
                                    // Highs - significant boost with lower power
                                    // The critical adjustment for high frequency sensitivity
                                    (normalized.powf(0.4) * 3.0).min(1.0)
                                };
                                // No minimum thresholds - let silent frequencies be silent
                                // Temporal smoothing with frequency-specific parameters
                                let vec_idx = i / 4;
                                let vec_component = i % 4;
                                if vec_idx < 32 {
                                    let prev_value = self.prev_audio_data[vec_idx][vec_component];
                                    // Fast attack for all frequencies - slightly faster for highs
                                    let attack = if band_percent < 0.6 {
                                        0.6 
                                    } else {
                                        0.7 
                                    };
                                    let decay = if band_percent < 0.6 {
                                        0.3 
                                    } else {
                                        0.25 
                                    };
                                    
                                    // Apply smoothing
                                    let smoothing_factor = if enhanced > prev_value {
                                        attack  // Rising
                                    } else {
                                        decay   // Falling
                                    };
                                    // Calculate smoothed value
                                    let smoothed = prev_value * (1.0 - smoothing_factor) + 
                                                  enhanced * smoothing_factor;
                                    // Store the result
                                    self.resolution_uniform.data.audio_data[vec_idx][vec_component] = smoothed;
                                    // Store for next frame
                                    self.prev_audio_data[vec_idx][vec_component] = smoothed;
                                }
                            }
                        }
                        
                        // Beat detection with balanced boost across frequency spectrum
                        let mut bass_energy: f32 = 0.0;
                        let bass_bands = 128 / 16;
                        for i in 0..(bass_bands / 4) {
                            for j in 0..4 {
                                bass_energy += self.resolution_uniform.data.audio_data[i][j];
                            }
                        }
                        bass_energy /= bass_bands as f32;
                        
                        // If we detect a beat, provide progressive boost to mid/high frequencies
                        if bass_energy > 0.5 {
                            // First quarter - bass 
                            let q1 = 32 / 4;
                            // Second quarter - low-mids
                            let q2 = 32 / 2;
                            // Third quarter - upper-mids
                            let q3 = 3 * 32 / 4;
                            
                            for i in 0..32 {
                                for j in 0..4 {
                                    if i < q1 {
                                        // No boost for bass (prevent dominance)
                                        // Actually reduce bass slightly on beats
                                        self.resolution_uniform.data.audio_data[i][j] *= 0.9;
                                    } else if i < q2 {
                                        // Small boost for low-mids
                                        self.resolution_uniform.data.audio_data[i][j] *= 1.1;
                                    } else if i < q3 {
                                        // Moderate boost for upper-mids
                                        self.resolution_uniform.data.audio_data[i][j] *= 1.3;
                                    } else {
                                        // Strong boost for highs during beats
                                        self.resolution_uniform.data.audio_data[i][j] *= 1.7;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            self.resolution_uniform.update(queue);
        }
    }
    pub fn handle_video_requests(&mut self, core: &Core, request: &ControlsRequest) {
        if let Some(path) = &request.load_media_path {
            if let Err(e) = self.load_media(core, path) {
                error!("Failed to load media: {}", e);
            }
        }
        
        if request.play_video {
            let _ = self.play_video();
        }
        
        if request.pause_video {
            let _ = self.pause_video();
        }
        
        if request.restart_video {
            let _ = self.seek_video(0.0);
            let _ = self.play_video();
        }
        
        if let Some(position) = request.seek_position {
            let _ = self.seek_video(position);
        }
        
        if let Some(should_loop) = request.set_loop {
            self.set_video_loop(should_loop);
        }
        
        // Handle audio control requests
        if let Some(volume) = request.set_volume {
            if let Some(vm) = &mut self.video_texture_manager {
                let _ = vm.set_volume(volume);
            }
        }
        
        if let Some(muted) = request.mute_audio {
            if let Some(vm) = &mut self.video_texture_manager {
                let _ = vm.set_mute(muted);
            }
        }
        if request.toggle_mute {
            if let Some(vm) = &mut self.video_texture_manager {
                let _ = vm.toggle_mute();
            }
        }
    }
    
    /// Get video information if a video texture is loaded
    pub fn get_video_info(&self) -> Option<(Option<f32>, f32, (u32, u32), Option<f32>, bool, bool, f64, bool)> {
        if self.using_video_texture {
            if let Some(vm) = &self.video_texture_manager {
                Some((
                    vm.duration().map(|d| d.seconds() as f32),
                    vm.position().seconds() as f32,
                    vm.dimensions(),
                    vm.framerate().map(|(num, den)| num as f32 / den as f32),
                    vm.is_looping(),
                    vm.has_audio(),
                    vm.volume(),
                    vm.is_muted()
                ))
            } else {
                None
            }
        } else {
            None
        }
    }
}