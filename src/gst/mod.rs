pub mod video;

use log::info;

pub fn init() -> anyhow::Result<()> {
    // Set very verbose logging for spectrum and element messages
    std::env::set_var("RUST_LOG", "info,warn,debug,gstreamer=debug,cuneus=debug,spectrum=trace");
    std::env::set_var("GST_DEBUG", "spectrum:5,bus:5,message:5");
    info!("Setting up GStreamer with enhanced logging");
    gstreamer::init()?;
    info!("GStreamer initialized successfully");
    Ok(())
}