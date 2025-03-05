use anyhow::{Result, anyhow};
use gstreamer as gst;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use log::{debug, error, info, warn};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use gst::prelude::*;
use crate::texture::TextureManager;
use wgpu;
/// Here I created a struct to organize the video text mang.
/// Manages a video texture that can be updated frame by frame
pub struct VideoTextureManager {
    /// The underlying TextureManager that handles the WGPU resources
    texture_manager: TextureManager,
    /// The GStreamer pipeline for video decoding
    pipeline: gst::Pipeline,
    /// The AppSink element that receives decoded frames
    appsink: gst_app::AppSink,
    /// Whether the video has an audio track
    has_audio: bool,
    /// Audio volume (0.0 to 1.0)
    volume: Arc<Mutex<f64>>,
    /// Whether audio is muted
    is_muted: Arc<Mutex<bool>>,
    /// Current video dimensions
    dimensions: (u32, u32),
    /// Video duration in nanoseconds (if available)
    duration: Option<gst::ClockTime>,
    /// Current position in the video
    position: Arc<Mutex<gst::ClockTime>>,
    /// Frame rate of the video
    framerate: Option<gst::Fraction>,
    /// Whether the video is currently playing
    is_playing: Arc<Mutex<bool>>,
    /// Whether to loop the video when it ends
    loop_playback: Arc<Mutex<bool>>,
    /// Last frame update time
    last_update: Instant,
    /// Frame buffer for the most recently decoded frame
    current_frame: Arc<Mutex<Option<image::RgbaImage>>>,
    /// Path to the video file
    video_path: String,
    texture_initialized: bool,
    /// Frame counter for debugging
    frame_count: usize,
}
impl VideoTextureManager {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bind_group_layout: &wgpu::BindGroupLayout,
        video_path: impl AsRef<Path>,
    ) -> Result<Self> {
        // Create a default 1x1 texture initially, note that, this going to be replaced with first video frame
        let default_image = image::RgbaImage::new(1, 1);
        let texture_manager = TextureManager::new(device, queue, &default_image, bind_group_layout);
        
        let path_str = video_path.as_ref()
            .to_str()
            .ok_or_else(|| anyhow!("Invalid video path"))?
            .to_string();
            
        info!("Creating video texture from: {}", path_str);
        
        let pipeline = gst::Pipeline::new();
        
        // Source element - read from file
        let filesrc = gst::ElementFactory::make("filesrc")
            .name("source")
            .property("location", &path_str)
            .build()
            .map_err(|_| anyhow!("Failed to create filesrc element"))?;
        // Decoding element
        let decodebin = gst::ElementFactory::make("decodebin")
            .name("decoder")
            .build()
            .map_err(|_| anyhow!("Failed to create decodebin element"))?;
            
        // videorate element to enforce correct frame timing
        let videorate = gst::ElementFactory::make("videorate")
            .name("rate")
            .build()
            .map_err(|_| anyhow!("Failed to create videorate element"))?;
        // Convert to proper format
        let videoconvert = gst::ElementFactory::make("videoconvert")
            .name("convert")
            .build()
            .map_err(|_| anyhow!("Failed to create videoconvert element"))?;
        //caps filter to force framerate if needed
        let capsfilter = gst::ElementFactory::make("capsfilter")
            .name("capsfilter")
            .build()
            .map_err(|_| anyhow!("Failed to create capsfilter element"))?;
            
        // Output sink for video
        let appsink = gst::ElementFactory::make("appsink")
            .name("sink")
            .build()
            .map_err(|_| anyhow!("Failed to create appsink element"))?;
        
        let appsink = appsink.dynamic_cast::<gst_app::AppSink>()
            .map_err(|_| anyhow!("Failed to cast to AppSink"))?;
            
        // Configure appsink
        appsink.set_caps(Some(&gst::Caps::builder("video/x-raw")
            .field("format", gst_video::VideoFormat::Rgba.to_str())
            .build()));
        
        appsink.set_max_buffers(2);
        appsink.set_drop(true);  // Drop old buffers when full
        appsink.set_sync(true);
            
        // video elements goes to the pipeline
        pipeline.add_many(&[
            &filesrc, 
            &decodebin, 
            &videorate, 
            &videoconvert, 
            &capsfilter, 
            &appsink.upcast_ref()
        ])
        .map_err(|_| anyhow!("Failed to add video elements to pipeline"))?;
            
        // Link elements that can be linked statically
        gst::Element::link_many(&[&videorate, &videoconvert, &capsfilter, &appsink.upcast_ref()])
            .map_err(|_| anyhow!("Failed to link video elements"))?;
            
        gst::Element::link_many(&[&filesrc, &decodebin])
            .map_err(|_| anyhow!("Failed to link filesrc to decodebin"))?;
        
        // Set up pad-added signal for dynamic linking from decodebin -> videorate
        let videorate_weak = videorate.downgrade();
        let has_audio = Arc::new(Mutex::new(false));
        let has_audio_clone = has_audio.clone();
        
        // now audio elements reference holders
        let audioconvert_weak = Arc::new(Mutex::new(None));
        
        decodebin.connect_pad_added(move |_, pad| {
            let caps = match pad.current_caps() {
                Some(caps) => caps,
                _ => return,
            };
            
            let structure = match caps.structure(0) {
                Some(s) => s,
                _ => return,
            };
            // Check if this is a video or audio stream. maybe there could be other way to handle this but I love simpicity
            if structure.name().starts_with("video/") {
                // Handle video path
                if let Some(videorate) = videorate_weak.upgrade() {
                    let sink_pad = match videorate.static_pad("sink") {
                        Some(pad) => pad,
                        _ => return,
                    };
                    
                    if !sink_pad.is_linked() {
                        let _ = pad.link(&sink_pad);
                        info!("Linked decoder to videorate successfully");
                    }
                }
            } else if structure.name().starts_with("audio/") {
                // has_audio flag to true - we've detected an audio stream
                if let Ok(mut has_audio_lock) = has_audio_clone.lock() {
                    *has_audio_lock = true;
                    info!("Audio track detected in video");
                }
                
                // Now lets dynamically create the audio processing chain
                if let Ok(mut audioconvert_lock) = audioconvert_weak.lock() {
                    // Only create audio elements once when first audio pad is detected
                    if audioconvert_lock.is_none() {
                        // Create audio elements
                        let audioconvert = match gst::ElementFactory::make("audioconvert")
                            .name("audioconvert")
                            .build() {
                                Ok(e) => e,
                                Err(_) => {
                                    warn!("Failed to create audioconvert");
                                    return;
                                }
                            };
                            
                        let audioresample = match gst::ElementFactory::make("audioresample")
                            .name("audioresample")
                            .build() {
                                Ok(e) => e,
                                Err(_) => {
                                    warn!("Failed to create audioresample");
                                    return;
                                }
                            };
                            
                        let volume = match gst::ElementFactory::make("volume")
                            .name("volume")
                            .property("volume", 1.0)
                            .build() {
                                Ok(e) => e,
                                Err(_) => {
                                    warn!("Failed to create volume");
                                    return;
                                }
                            };
                            
                        // autoaudiosink should works on all platforms: https://gstreamer.freedesktop.org/documentation/autodetect/autoaudiosink.html?gi-language=c
                        let audio_sink = match gst::ElementFactory::make("autoaudiosink")
                            .name("audiosink")
                            .build() {
                                Ok(e) => e,
                                Err(_) => {
                                    warn!("Failed to create autoaudiosink");
                                    return;
                                }
                            };
                            
                        // Add elements to pipeline
                        if let Err(e) = pad.parent_element().unwrap().parent().unwrap()
                                         .downcast_ref::<gst::Pipeline>().unwrap()
                                         .add_many(&[&audioconvert, &audioresample, &volume, &audio_sink]) {
                            warn!("Failed to add audio elements: {:?}", e);
                            return;
                        }
                        
                        // Link audio elements
                        if let Err(e) = gst::Element::link_many(&[&audioconvert, &audioresample, &volume, &audio_sink]) {
                            warn!("Failed to link audio elements: {:?}", e);
                            return;
                        }
                        
                        // Set elements to PAUSED state
                        let _ = audioconvert.sync_state_with_parent();
                        let _ = audioresample.sync_state_with_parent();
                        let _ = volume.sync_state_with_parent();
                        let _ = audio_sink.sync_state_with_parent();
                        
                        *audioconvert_lock = Some(audioconvert.clone());
                    }
                    
                    // Link decoder pad to audioconvert
                    if let Some(audioconvert) = &*audioconvert_lock {
                        let sink_pad = match audioconvert.static_pad("sink") {
                            Some(pad) => pad,
                            _ => return,
                        };
                        
                        if !sink_pad.is_linked() {
                            match pad.link(&sink_pad) {
                                Ok(_) => {
                                    info!("Linked decoder to audioconvert successfully");
                                },
                                Err(err) => {
                                    warn!("Failed to link audio pad: {:?}", err);
                                }
                            }
                        }
                    }
                }
            }
        });
        
        // Create shared state
        let current_frame = Arc::new(Mutex::new(None));
        let current_frame_clone = current_frame.clone();
        let position = Arc::new(Mutex::new(gst::ClockTime::ZERO));
        let is_playing = Arc::new(Mutex::new(false));
        let volume_val = Arc::new(Mutex::new(1.0));
        let is_muted = Arc::new(Mutex::new(false));
        
        // Setup callbacks to receive frames
        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |sink| {
                    let sample = match sink.pull_sample() {
                        Ok(sample) => sample,
                        Err(_) => return Err(gst::FlowError::Eos),
                    };
                    
                    let buffer = match sample.buffer() {
                        Some(buffer) => buffer,
                        _ => return Err(gst::FlowError::Error),
                    };
                    
                    let caps = match sample.caps() {
                        Some(caps) => caps,
                        _ => return Err(gst::FlowError::Error),
                    };
                    
                    let video_info = match gst_video::VideoInfo::from_caps(caps) {
                        Ok(info) => info,
                        Err(_) => return Err(gst::FlowError::Error),
                    };
                    
                    let map = match buffer.map_readable() {
                        Ok(map) => map,
                        Err(_) => return Err(gst::FlowError::Error),
                    };
                    
                    // Access the raw frame data
                    let frame_data = map.as_slice();
                    let width = video_info.width() as usize;
                    let height = video_info.height() as usize;
                    
                    // Create an RgbaImage from the frame data
                    // (We need to copy the data because buffer will be unmapped after this function)
                    let mut rgba_image = image::RgbaImage::new(width as u32, height as u32);
                    
                    // Stride might be larger than width * 4
                    let stride = video_info.stride()[0] as usize;
                    
                    for y in 0..height {
                        let src_start = y * stride;
                        let src_end = src_start + width * 4;
                        let dst_start = y * width * 4;
                        let dst_end = dst_start + width * 4;
                        
                        // Copy row by row to handle stride correctly
                        let dst_buffer = rgba_image.as_mut();
                        if src_end <= frame_data.len() && dst_end <= dst_buffer.len() {
                            dst_buffer[dst_start..dst_end]
                                .copy_from_slice(&frame_data[src_start..src_end]);
                        }
                    }
                    
                    // Store the frame
                    if let Ok(mut frame_lock) = current_frame_clone.lock() {
                        *frame_lock = Some(rgba_image);
                    }
                    
                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );
        
        // init the object
        let mut video_texture = Self {
            texture_manager,
            pipeline,
            appsink,
            has_audio: false,
            volume: volume_val,
            is_muted: is_muted,
            dimensions: (1, 1),
            duration: None,
            position,
            framerate: None,
            is_playing,
            loop_playback: Arc::new(Mutex::new(true)),
            last_update: Instant::now(),
            current_frame,
            video_path: path_str,
            texture_initialized: false,
            frame_count: 0,
        };
        // Start pipeline in paused state to get video info
        if video_texture.pipeline.set_state(gst::State::Paused).is_err() {
            return Err(anyhow!("Failed to set pipeline to PAUSED state"));
        }
        
        // Wait a bit for pipeline to settle
        std::thread::sleep(Duration::from_millis(200));
        
        let state_result = video_texture.pipeline.state(gst::ClockTime::from_seconds(1));
        if let (_, gst::State::Paused, _) = state_result {
            video_texture.query_video_info()?;
        } else {
            warn!("Pipeline not in PAUSED state, may not be able to query info");
        }
        // Set specific framerate in the caps filter if we detected one
        if let Some(framerate) = video_texture.framerate {
            if let Some(capsfilter_elem) = video_texture.pipeline.by_name("capsfilter") {
                let caps = gst::Caps::builder("video/x-raw")
                    .field("framerate", framerate)
                    .build();
                capsfilter_elem.set_property("caps", &caps);
                info!("Set capsfilter to force framerate {}/{}", framerate.numer(), framerate.denom());
            }
        }
        video_texture.has_audio = *has_audio.lock().unwrap();
        info!("Video has audio: {}", video_texture.has_audio);
        
        info!("Video texture manager created successfully");
        Ok(video_texture)
    }
    
    /// Query video information (dimensions, duration, framerate)
    fn query_video_info(&mut self) -> Result<()> {
        // Query duration
        if let Some(duration) = self.pipeline.query_duration::<gst::ClockTime>() {
            self.duration = Some(duration);
            info!("Video duration: {:?} ({:.2} seconds)", 
                 duration, duration.seconds() as f64);
        }
        
        // Now try to get video info
        if let Some(pad) = self.appsink.static_pad("sink") {
            if let Some(caps) = pad.current_caps() {
                if let Some(s) = caps.structure(0) {
                    //  dims
                    if let (Ok(width), Ok(height)) = (s.get::<i32>("width"), s.get::<i32>("height")) {
                        self.dimensions = (width as u32, height as u32);
                        info!("Video dimensions: {}x{}", width, height);
                    }
                    
                    // framerate 
                    if let Ok(framerate) = s.get::<gst::Fraction>("framerate") {
                        self.framerate = Some(framerate);
                        info!("Video framerate: {}/{} (approx. {:.2} fps)", 
                             framerate.numer(), framerate.denom(),
                             framerate.numer() as f64 / framerate.denom() as f64);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Get the texture manager (for binding to shaders)
    pub fn texture_manager(&self) -> &TextureManager {
        &self.texture_manager
    }
    
    /// Update the texture with the current video frame
    pub fn update_texture(
        &mut self, 
        device: &wgpu::Device, 
        queue: &wgpu::Queue, 
        bind_group_layout: &wgpu::BindGroupLayout
    ) -> Result<bool> {
        // No update needed if video is not playing
        if !*self.is_playing.lock().unwrap() {
            return Ok(false);
        }
        
        // Check if we have a NEW frame to process
        let frame_to_process = {
            let mut frame_lock = self.current_frame.lock().unwrap();
            frame_lock.take()
        };
        
        // If we have a frame, update the texture
        if let Some(frame) = frame_to_process {
            self.frame_count += 1;
            
            // Get frame dimensions
            let width = frame.width();
            let height = frame.height();
            
            // Log less frequently to reduce spam
            if self.frame_count % 30 == 0 {
                debug!("Processing video frame #{} (dimensions: {}x{})", 
                     self.frame_count, width, height);
            }
            
            // ALWAYS recreate the texture for the first frame or if dimensions don't match
            let should_recreate = !self.texture_initialized || 
                                 self.dimensions != (width, height) ||
                                 self.dimensions.0 <= 1 || 
                                 self.dimensions.1 <= 1 ||
                                 self.frame_count <= 3;
            
            if should_recreate {
                info!("Creating new texture with dimensions: {}x{}", width, height);
                
                // Create a completely new texture with the frame's dimensions
                let new_texture_manager = TextureManager::new(
                    device, 
                    queue, 
                    &frame, 
                    bind_group_layout
                );
                
                self.texture_manager = new_texture_manager;
                self.dimensions = (width, height);
                self.texture_initialized = true;
            } else {
                self.texture_manager.update(queue, &frame);
            }
            
            // Get current position
            if let Some(position) = self.pipeline.query_position::<gst::ClockTime>() {
                *self.position.lock().unwrap() = position;
                
                // Check if we reached the end of the video
                if let Some(duration) = self.duration {
                    let near_end_threshold = duration.saturating_sub(
                        gst::ClockTime::from_mseconds(100)
                    );
                    
                    if position >= near_end_threshold {
                        debug!("Near end of video (position: {:?}, duration: {:?})", position, duration);
                        
                        if *self.loop_playback.lock().unwrap() {
                            debug!("Looping video");
                            self.seek(gst::ClockTime::ZERO)?;
                        } else {
                            debug!("Pausing at end of video");
                            self.pause()?;
                        }
                    }
                }
            }
            
            // Update the last update time
            self.last_update = Instant::now();
            
            Ok(true) // Texture was updated
        } else {
            Ok(false) // No update
        }
    }
    
    /// Start playing the video
    pub fn play(&mut self) -> Result<()> {
        info!("Playing video");
        match self.pipeline.set_state(gst::State::Playing) {
            Ok(_) => {
                *self.is_playing.lock().unwrap() = true;
                Ok(())
            },
            Err(e) => Err(anyhow!("Failed to start playback: {:?}", e))
        }
    }
    
    /// Pause the video
    pub fn pause(&mut self) -> Result<()> {
        info!("Pausing video");
        match self.pipeline.set_state(gst::State::Paused) {
            Ok(_) => {
                *self.is_playing.lock().unwrap() = false;
                Ok(())
            },
            Err(e) => Err(anyhow!("Failed to pause playback: {:?}", e))
        }
    }
    
    /// Seek to a specific position in the video
    pub fn seek(&mut self, position: gst::ClockTime) -> Result<()> {
        debug!("Seeking to position: {:?}", position);
        
        // are we sure the pipeline is in a state that can handle seeking??
        let current_state = self.pipeline.current_state();
        if current_state == gst::State::Null || current_state == gst::State::Ready {
            warn!("Cannot seek in current state: {:?}", current_state);
            return Ok(());
        }
        
        // exec the seek operation
        let seek_flags = gst::SeekFlags::FLUSH | gst::SeekFlags::KEY_UNIT;
        if self.pipeline.seek_simple(seek_flags, position).is_ok() {
            debug!("Seek successful");
            *self.position.lock().unwrap() = position;
            Ok(())
        } else {
            let err = anyhow!("Failed to seek to {:?}", position);
            error!("{}", err);
            Err(err)
        }
    }
    
    pub fn set_loop(&mut self, should_loop: bool) {
        *self.loop_playback.lock().unwrap() = should_loop;
        info!("Video loop set to: {}", should_loop);
    }
    
    /// audio volume (between 0.0 and 1.0)
    pub fn set_volume(&mut self, volume: f64) -> Result<()> {
        if !self.has_audio {
            debug!("Ignoring volume change request - video has no audio");
            return Ok(());
        }
        
        let clamped_volume = volume.max(0.0).min(1.0);
        *self.volume.lock().unwrap() = clamped_volume;
        
        if let Some(volume_elem) = self.pipeline.by_name("volume") {
            volume_elem.set_property("volume", clamped_volume);
            debug!("Set volume to {:.2}", clamped_volume);
            Ok(())
        } else {
            warn!("Volume element not found in pipeline");
            Ok(()) // Don't fail if element not found - video might still work
        }
    }
    
    /// Mutes or unmutes the audio
    pub fn set_mute(&mut self, muted: bool) -> Result<()> {
        if !self.has_audio {
            debug!("Ignoring mute request - video has no audio");
            return Ok(());
        }
        
        *self.is_muted.lock().unwrap() = muted;
        
        if let Some(volume_elem) = self.pipeline.by_name("volume") {
            volume_elem.set_property("mute", muted);
            debug!("Set mute to {}", muted);
            Ok(())
        } else {
            warn!("Volume element not found in pipeline");
            Ok(()) // Don't fail if element not found - video might still work
        }
    }
    
    pub fn toggle_mute(&mut self) -> Result<()> {
        let current_mute = *self.is_muted.lock().unwrap();
        self.set_mute(!current_mute)
    }
    
    /// Returns true if the video has an audio track
    pub fn has_audio(&self) -> bool {
        self.has_audio
    }
    
    /// Gets the current volume (0.0 to 1.0)
    pub fn volume(&self) -> f64 {
        *self.volume.lock().unwrap()
    }
    
    /// Returns true if audio is currently muted
    pub fn is_muted(&self) -> bool {
        *self.is_muted.lock().unwrap()
    }
    
    pub fn position(&self) -> gst::ClockTime {
        *self.position.lock().unwrap()
    }
    
    pub fn duration(&self) -> Option<gst::ClockTime> {
        self.duration
    }
    
    pub fn dimensions(&self) -> (u32, u32) {
        self.dimensions
    }
    
    pub fn framerate(&self) -> Option<(i32, i32)> {
        self.framerate.map(|f| (f.numer(), f.denom()))
    }
    
    pub fn is_playing(&self) -> bool {
        *self.is_playing.lock().unwrap()
    }
    
    pub fn is_looping(&self) -> bool {
        *self.loop_playback.lock().unwrap()
    }
    
    pub fn path(&self) -> &str {
        &self.video_path
    }
}

impl Drop for VideoTextureManager {
    fn drop(&mut self) {
        info!("Shutting down video pipeline");
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}
