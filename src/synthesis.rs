use anyhow::Result;
use std::time::Instant;
use cpal::{Stream, SampleFormat, OutputCallbackInfo};
use cpal::traits::{HostTrait, DeviceTrait, StreamTrait};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct AudioState {
    sample_rate: f32,
    time: f32,
    frequency: f32,
    amplitude: f32,
    waveform_type: u32,
    target_frequency: f32,
    target_amplitude: f32,
    target_waveform_type: u32,
}

impl AudioState {
    fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            time: 0.0,
            frequency: 440.0,
            amplitude: 0.1,
            waveform_type: 0,
            target_frequency: 440.0,
            target_amplitude: 0.1,
            target_waveform_type: 0,
        }
    }

    fn generate_sample(&mut self) -> f32 {
        let dt = 1.0 / self.sample_rate;
        self.time += dt;

        let smoothing = 0.999;
        self.frequency = self.frequency * smoothing + self.target_frequency * (1.0 - smoothing);
        self.amplitude = self.amplitude * smoothing + self.target_amplitude * (1.0 - smoothing);
        if self.target_waveform_type != self.waveform_type {
            self.waveform_type = self.target_waveform_type;
        }

        let phase = 2.0 * std::f32::consts::PI * self.frequency * self.time;
        let sample = match self.waveform_type {
            0 => phase.sin(),
            1 => {
                if phase.sin() > 0.0 { 1.0 } else { -1.0 }
            },
            2 => {
                let t = (phase / (2.0 * std::f32::consts::PI)) % 1.0;
                2.0 * t - 1.0
            },
            3 => {
                let t = (phase / (2.0 * std::f32::consts::PI)) % 1.0;
                if t < 0.5 { 4.0 * t - 1.0 } else { 3.0 - 4.0 * t }
            },
            _ => phase.sin(),
        };
        
        sample * self.amplitude
    }

    fn update_params(&mut self, frequency: f32, amplitude: f32, waveform_type: u32) {
        self.target_frequency = frequency;
        self.target_amplitude = amplitude;
        self.target_waveform_type = waveform_type;
    }
}

pub struct SynthesisStreamer {
    stream: Option<Stream>,
    audio_state: Arc<Mutex<AudioState>>,
    is_playing: bool,
}

impl SynthesisStreamer {
    pub fn new() -> Result<Self> {
        Ok(Self {
            stream: None,
            audio_state: Arc::new(Mutex::new(AudioState::new(44100.0))),
            is_playing: false,
        })
    }
    
    pub fn start(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host.default_output_device()
            .ok_or_else(|| anyhow::anyhow!("No output device available"))?;
        
        let config = device.default_output_config()?;
        let sample_rate = config.sample_rate().0 as f32;
        
        
        {
            let mut state = self.audio_state.lock().unwrap();
            state.sample_rate = sample_rate;
        }
        
        let audio_state_clone = Arc::clone(&self.audio_state);
        
        let stream = match config.sample_format() {
            SampleFormat::F32 => self.build_stream::<f32>(&device, &config.into(), audio_state_clone)?,
            SampleFormat::I16 => self.build_stream::<i16>(&device, &config.into(), audio_state_clone)?,
            SampleFormat::U16 => self.build_stream::<u16>(&device, &config.into(), audio_state_clone)?,
            sample_format => return Err(anyhow::anyhow!("Unsupported sample format: {}", sample_format)),
        };
        
        stream.play()?;
        self.stream = Some(stream);
        self.is_playing = true;
        
        Ok(())
    }
    
    fn build_stream<T>(
        &self,
        device: &cpal::Device,
        config: &cpal::StreamConfig,
        audio_state: Arc<Mutex<AudioState>>,
    ) -> Result<Stream>
    where
        T: cpal::SizedSample + cpal::FromSample<f32>,
    {
        let channels = config.channels as usize;
        let err_fn = |err| eprintln!("Audio stream error: {}", err);
        
        let stream = device.build_output_stream(
            config,
            move |data: &mut [T], _: &OutputCallbackInfo| {
                if let Ok(mut state) = audio_state.lock() {
                    for frame in data.chunks_mut(channels) {
                        let sample = state.generate_sample();
                        let value: T = T::from_sample(sample);
                        
                        for channel_sample in frame.iter_mut() {
                            *channel_sample = value;
                        }
                    }
                }
            },
            err_fn,
            None,
        )?;
        
        Ok(stream)
    }
    
    pub fn update_frequency(&mut self, frequency: f32) {
        if let Ok(mut state) = self.audio_state.lock() {
            state.frequency = frequency;
        }
    }
    
    pub fn update_params(&mut self, frequency: f32, amplitude: f32, waveform_type: u32) {
        if let Ok(mut state) = self.audio_state.lock() {
            state.update_params(frequency, amplitude, waveform_type);
        }
    }
    
    pub fn stop(&mut self) -> Result<()> {
        if let Some(stream) = self.stream.take() {
            drop(stream);
            self.is_playing = false;
        }
        Ok(())
    }
    
    pub fn buffer_info(&self) -> (usize, bool) {
        (0, self.is_playing)
    }
}

pub struct SynthesisManager {
    gpu_streamer: Option<SynthesisStreamer>,
    sample_rate: u32,
    last_update: Instant,
    gpu_synthesis_enabled: bool,
}

impl SynthesisManager {
    pub fn new() -> Result<Self> {
        
        let gpu_streamer = match SynthesisStreamer::new() {
            Ok(streamer) => {
                Some(streamer)
            },
            Err(e) => {
                eprintln!("Failed to initialize Real-time Synthesis Streamer: {}", e);
                None
            }
        };
        
        Ok(Self {
            gpu_streamer,
            sample_rate: 44100,
            last_update: Instant::now(),
            gpu_synthesis_enabled: false,
        })
    }
    
    pub fn start_gpu_synthesis(&mut self) -> Result<()> {
        if let Some(ref mut streamer) = self.gpu_streamer {
            streamer.start()?;
            self.gpu_synthesis_enabled = true;
        }
        Ok(())
    }
    
    pub fn stop_gpu_synthesis(&mut self) -> Result<()> {
        if let Some(ref mut streamer) = self.gpu_streamer {
            streamer.stop()?;
            self.gpu_synthesis_enabled = false;
        }
        Ok(())
    }
    
    pub fn update_frequency(&mut self, frequency: f32) {
        if let Some(ref mut streamer) = self.gpu_streamer {
            streamer.update_frequency(frequency);
        }
    }
    
    pub fn update_synth_params(&mut self, frequency: f32, amplitude: f32, waveform_type: u32) {
        if let Some(ref mut streamer) = self.gpu_streamer {
            streamer.update_params(frequency, amplitude, waveform_type);
        }
    }
    
    pub fn stream_gpu_samples(&mut self, _samples: &[f32]) {
    }
    
    pub fn update(&mut self) {
        self.last_update = Instant::now();
    }
    
    pub fn get_buffer_info(&self) -> Option<(usize, bool)> {
        self.gpu_streamer.as_ref().map(|s| s.buffer_info())
    }
    
    pub fn is_gpu_synthesis_enabled(&self) -> bool {
        self.gpu_synthesis_enabled
    }
    
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SynthesisUniform {
    pub note_frequencies: [[f32; 4]; 4],
    pub note_amplitudes: [[f32; 4]; 4],
    pub master_volume: f32,
    pub waveform_type: u32,
    pub active_note_count: u32,
}

unsafe impl bytemuck::Pod for SynthesisUniform {}
unsafe impl bytemuck::Zeroable for SynthesisUniform {}

impl crate::UniformProvider for SynthesisUniform {
    fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

impl SynthesisUniform {
    pub fn new() -> Self {
        Self {
            note_frequencies: [[0.0; 4]; 4],
            note_amplitudes: [[0.0; 4]; 4],
            master_volume: 0.3,
            waveform_type: 0,
            active_note_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthesisWaveform {
    Sine,
    Square,
    Saw,
    Triangle,
}

impl SynthesisWaveform {
    pub fn to_u32(self) -> u32 {
        match self {
            SynthesisWaveform::Sine => 0,
            SynthesisWaveform::Square => 1,
            SynthesisWaveform::Saw => 2,
            SynthesisWaveform::Triangle => 3,
        }
    }
}