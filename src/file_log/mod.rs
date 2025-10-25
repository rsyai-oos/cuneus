use tracing::{level_filters::LevelFilter, Level};
use tracing_subscriber::{fmt::writer::MakeWriterExt, EnvFilter};

pub fn init_tracing() {
    {
        // set dep crate log level
        // Silence wgpu log spam (https://github.com/gfx-rs/wgpu/issues/3206)
        let mut rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".to_owned());
        for loud_crate in ["naga", "wgpu_core", "wgpu_hal"] {
            if !rust_log.contains(&format!("{loud_crate}=")) {
                rust_log += &format!(",{loud_crate}=warn");
            }
        }
        std::env::set_var("RUST_LOG", rust_log);
    }
    let log_info =
        tracing_appender::rolling::never("logs", "info_log.txt").with_max_level(Level::INFO);
    let _log_debug =
        tracing_appender::rolling::never("logs", "debug_log.txt").with_max_level(Level::DEBUG);
    let _log_warn =
        tracing_appender::rolling::never("logs", "warn_log.txt").with_max_level(Level::WARN);

    let all_files = log_info; //.and(log_debug).and(log_warn);
                              // let all_files = all_files.and(debug_file).and(trace_file);
    tracing_subscriber::fmt()
        .with_writer(all_files)
        .with_ansi(false)
        .with_max_level(LevelFilter::INFO)
        // .with_thread_ids(true)
        // .with_thread_names(true)
        .with_file(true)
        .with_line_number(true)
        .without_time()
        // .with_env_filter(env_filter)
        .with_env_filter(EnvFilter::from_env("RUST_LOG"))
        .init();

    tracing::debug!("main program start");
}
