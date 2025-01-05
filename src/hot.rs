use notify::{Watcher, RecursiveMode};
use std::sync::Arc;
use std::path::PathBuf;
use std::fs;

pub struct ShaderHotReload {
    pub vs_module: wgpu::ShaderModule,
    pub fs_module: wgpu::ShaderModule,
    device: Arc<wgpu::Device>,
    shader_paths: Vec<PathBuf>,
    last_vs_content: String,
    last_fs_content: String,
}

impl ShaderHotReload {
    pub fn new(
        device: Arc<wgpu::Device>,
        shader_paths: Vec<PathBuf>,
        vs_module: wgpu::ShaderModule,
        fs_module: wgpu::ShaderModule,
    ) -> notify::Result<Self> {
        let (tx, _rx) = std::sync::mpsc::channel();
        
        if let Some(first_path) = shader_paths.first() {
            if let Some(parent) = first_path.parent() {
                fs::create_dir_all(parent).unwrap_or_else(|_| {
                    println!("Could not create shader directory, hot reload might be limited");
                });
            }
        }

        let mut watcher = notify::recommended_watcher(move |res| {
            if let Ok(event) = res {
                tx.send(event).unwrap_or_default();
            }
        })?;
        for path in &shader_paths {
            if let Some(parent) = path.parent() {
                if parent.exists() {
                    watcher.watch(parent, RecursiveMode::NonRecursive).unwrap_or_else(|_| {
                        println!("Could not watch shader directory: {}", parent.display());
                    });
                } else {
                    println!("Shader directory does not exist: {}", parent.display());
                }
            }
        }
        let last_vs_content = fs::read_to_string(&shader_paths[0]).unwrap_or_default();
        let last_fs_content = fs::read_to_string(&shader_paths[1]).unwrap_or_default();

        Ok(Self {
            vs_module,
            fs_module,
            device,
            shader_paths,
            last_vs_content,
            last_fs_content,
        })
    }

    pub fn check_and_reload(&mut self) -> Option<(&wgpu::ShaderModule, &wgpu::ShaderModule)> {
        let vs_content = match fs::read_to_string(&self.shader_paths[0]) {
            Ok(content) => content,
            Err(_) => {
                return None;
            }
        };

        let fs_content = match fs::read_to_string(&self.shader_paths[1]) {
            Ok(content) => content,
            Err(_) => {
                return None;
            }
        };

        if vs_content == self.last_vs_content && fs_content == self.last_fs_content {
            return None;
        }

        let new_vs = match self.create_shader_module(&vs_content, "Vertex Shader") {
            Ok(module) => module,
            Err(e) => {
                eprintln!("Failed to compile vertex shader: {}", e);
                return None;
            }
        };

        let new_fs = match self.create_shader_module(&fs_content, "Fragment Shader") {
            Ok(module) => module,
            Err(e) => {
                eprintln!("Failed to compile fragment shader: {}", e);
                return None;
            }
        };

        self.last_vs_content = vs_content;
        self.last_fs_content = fs_content;
        self.vs_module = new_vs;
        self.fs_module = new_fs;

        Some((&self.vs_module, &self.fs_module))
    }

    fn create_shader_module(&self, source: &str, label: &str) -> Result<wgpu::ShaderModule, String> {
        let desc = wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        };
        Ok(self.device.create_shader_module(desc))
    }
}