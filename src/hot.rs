use notify::{Watcher, RecursiveMode, Event, EventKind};
use std::sync::Arc;
use std::path::{PathBuf, Path};
use std::fs;
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant};
use std::collections::HashMap;

pub struct ShaderHotReload {
    pub vs_module: wgpu::ShaderModule,
    pub fs_module: wgpu::ShaderModule,
    device: Arc<wgpu::Device>,
    shader_paths: Vec<PathBuf>,
    last_vs_content: String,
    last_fs_content: String,
    #[allow(dead_code)]
    watcher: notify::RecommendedWatcher,
    rx: Receiver<notify::Event>,
    _watcher_tx: std::sync::mpsc::Sender<notify::Event>,
    last_update_times: HashMap<PathBuf, Instant>, //Keeps track of when each shader file was last updated.
    debounce_duration: Duration, //Defines how long to wait before allowing another reload of the same file. The default is 100ms.
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
                match event.kind {
                    EventKind::Modify(_) |
                    EventKind::Create(_) |
                    EventKind::Remove(_)
                    => {
                        tx.send(event).unwrap_or_default();
                    },
                    _ => {}
                }
            }
        })?;

        //normalize for Windows
        let normalized_paths: Vec<PathBuf> = shader_paths.iter()
            .map(|path| Self::normalize_path(path))
            .collect();

        for path in &normalized_paths {
            if let Some(parent) = path.parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent).unwrap_or_else(|e| {
                        println!("Failed to create shader directory: {}", e);
                    });
                }
                
                if let Err(e) = watcher.watch(parent, RecursiveMode::Recursive) {
                    println!("Warning: Could not watch shader directory {}: {}", parent.display(), e);
                    if cfg!(windows) {
                        if let Err(e) = watcher.watch(parent, RecursiveMode::NonRecursive) {
                            println!("Fallback watch failed: {}", e);
                        }
                    }
                }
            }
        }

        let last_vs_content = fs::read_to_string(&normalized_paths[0]).unwrap_or_default();
        let last_fs_content = fs::read_to_string(&normalized_paths[1]).unwrap_or_default();

        Ok(Self {
            vs_module,
            fs_module,
            device,
            shader_paths: normalized_paths,
            last_vs_content,
            last_fs_content,
            watcher,
            rx,
            _watcher_tx: watcher_tx,
            last_update_times: HashMap::new(),
            debounce_duration: Duration::from_millis(100),
        })
    }

    fn normalize_path(path: &Path) -> PathBuf {
        if cfg!(windows) {
            
            path.components()
                .collect::<PathBuf>()
                .canonicalize()
                .unwrap_or_else(|_| path.to_path_buf())
        } else {
            path.to_path_buf()
        }
    }

    pub fn check_and_reload(&mut self) -> Option<(&wgpu::ShaderModule, &wgpu::ShaderModule)> {
        let mut should_reload = false;
        
        // Process all pending events
        while let Ok(event) = self.rx.try_recv() {
            for path in event.paths {
                let now = Instant::now();
                
                if let Some(last_update) = self.last_update_times.get(&path) {
                    if now.duration_since(*last_update) < self.debounce_duration {
                        continue;
                    }
                }
                
                self.last_update_times.insert(path.clone(), now);
                should_reload = true;
            }
        }

        if !should_reload {
            return None;
        }

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
    }

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
}