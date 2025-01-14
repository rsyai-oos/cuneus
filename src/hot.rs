use notify::{Watcher, RecursiveMode, Event, EventKind };
use notify::event::ModifyKind;
use std::sync::Arc;
use std::path::PathBuf;
use std::fs;
use std::sync::mpsc::channel;

pub struct ShaderHotReload {
    pub vs_module: wgpu::ShaderModule,
    pub fs_module: wgpu::ShaderModule,
    device: Arc<wgpu::Device>,
    shader_paths: Vec<PathBuf>,
    last_vs_content: String,
    last_fs_content: String,
    #[allow(dead_code)]
    watcher: notify::RecommendedWatcher,
    rx: std::sync::mpsc::Receiver<notify::Event>,
    _watcher_tx: std::sync::mpsc::Sender<notify::Event>,
}

impl ShaderHotReload {
    pub fn new(
        device: Arc<wgpu::Device>,
        shader_paths: Vec<PathBuf>,
        vs_module: wgpu::ShaderModule,
        fs_module: wgpu::ShaderModule,
    ) -> notify::Result<Self> {
        let (tx, rx) = channel();
        let watcher_tx = tx.clone();
        let mut watcher = notify::recommended_watcher(move |res: Result<Event, _>| {
            if let Ok(event) = res {
                if let EventKind::Modify(ModifyKind::Data(_)) = event.kind {
                    tx.send(event).unwrap_or_default();
                }
            }
        })?;

        if let Some(first_path) = shader_paths.first() {
            if let Some(parent) = first_path.parent() {
                fs::create_dir_all(parent).unwrap_or_else(|_| {
                    println!("Could not create shader directory, hot reload might be limited");
                });
            }
        }

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
            watcher,
            rx,
            _watcher_tx: watcher_tx,
        })
    }
    // on here, we are create shader module cathing the panics and it will return Option<wgpu::ShaderModule> - Never panics
    fn create_shader_module(&self, source: &str, label: &str) -> Option<wgpu::ShaderModule> {
        let desc = wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        };

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.device.create_shader_module(desc)
        }));

        match result {
            Ok(module) => Some(module),
            Err(e) => {
                if let Some(error_msg) = e.downcast_ref::<String>() {
                    eprintln!("Shader compilation error in {}: {}", label, error_msg);
                } else {
                    eprintln!("Shader compilation error in {}", label);
                }
                None
            }
        }
    }

    pub fn check_and_reload(&mut self) -> Option<(&wgpu::ShaderModule, &wgpu::ShaderModule)> {
    if let Ok(_event) = self.rx.try_recv() {
        let vs_content = match fs::read_to_string(&self.shader_paths[0]) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Failed to read vertex shader: {}", e);
                return None;
            }
        };

        let fs_content = match fs::read_to_string(&self.shader_paths[1]) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Failed to read fragment shader: {}", e);
                return None;
            }
        };

        if vs_content == self.last_vs_content && fs_content == self.last_fs_content {
            return None;
        }

        let new_vs = match self.create_shader_module(&vs_content, "Vertex Shader") {
            Some(module) => module,
            None => return None,
        };

        let new_fs = match self.create_shader_module(&fs_content, "Fragment Shader") {
            Some(module) => module,
            None => return None,
        };

        self.last_vs_content = vs_content;
        self.last_fs_content = fs_content;
        self.vs_module = new_vs;
        self.fs_module = new_fs;

        Some((&self.vs_module, &self.fs_module))
    } else {
        None
    }
    }

    pub fn has_shader_changed(&self, shader_type: &str) -> bool {
        let (path, last_content) = match shader_type {
            "vertex" => (&self.shader_paths[0], &self.last_vs_content),
            "fragment" => (&self.shader_paths[1], &self.last_fs_content),
            _ => return false,
        };
        match fs::read_to_string(path) {
            Ok(current_content) => current_content != *last_content,
            Err(_) => false,
        }
    }
}