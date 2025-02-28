pub mod video;

use log::info;

pub fn init() -> anyhow::Result<()> {
    info!("Initializing GStreamer subsystem");
    gstreamer::init()?;
    info!("GStreamer initialized successfully");
    Ok(())
}