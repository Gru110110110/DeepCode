mod cli;
mod commands;
mod session;
mod ui;
mod workspace_trust;

use clap::Parser;
use cli::{Cli, Commands};
use std::fs::OpenOptions;
use std::path::PathBuf;
use tracing_subscriber::filter::LevelFilter;

fn home_dir() -> PathBuf {
    deepcode_core::paths::home_dir()
}

fn default_log_path() -> PathBuf {
    if let Some(path) = std::env::var_os("DEEPCODE_LOG_FILE") {
        return PathBuf::from(path);
    }

    let state_dir = std::env::var_os("DEEPCODE_STATE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join(".local/state/deepcode"));
    state_dir.join("logs").join("deepcode.log")
}

fn parse_log_level(value: &str) -> LevelFilter {
    match value.to_ascii_lowercase().as_str() {
        "trace" => LevelFilter::TRACE,
        "debug" => LevelFilter::DEBUG,
        "warn" | "warning" => LevelFilter::WARN,
        "error" => LevelFilter::ERROR,
        "off" => LevelFilter::OFF,
        _ => LevelFilter::INFO,
    }
}

fn init_logging(log_level: &str) -> anyhow::Result<PathBuf> {
    let log_path = default_log_path();
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let max_level = parse_log_level(log_level);

    tracing_subscriber::fmt()
        .with_writer(move || {
            log_file
                .try_clone()
                .expect("failed to clone DeepCode log file handle")
        })
        .with_ansi(false)
        .with_target(true)
        .with_max_level(max_level)
        .init();

    tracing::info!(
        log_path = %log_path.display(),
        log_level = %log_level,
        "DeepCode logging initialized"
    );

    Ok(log_path)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = crossterm::terminal::disable_raw_mode();
    let cli = Cli::parse();
    let _log_path = init_logging(&cli.log_level)?;
    let config_path = cli
        .config
        .clone()
        .unwrap_or_else(commands::default_config_path);

    match &cli.command {
        Some(Commands::Config) => commands::config_command(config_path).await?,
        Some(Commands::Models { refresh }) => {
            commands::models_command(config_path, cli.provider.clone(), *refresh).await?;
        }
        Some(Commands::Run { prompt }) => {
            commands::run_command(
                prompt.clone(),
                cli.provider.clone(),
                cli.model.clone(),
                config_path,
            )
            .await?;
        }
        Some(Commands::Chat) => {
            commands::chat_command(cli.provider.clone(), cli.model.clone(), config_path, None)
                .await?;
        }
        Some(Commands::Sessions { all, limit }) => {
            commands::sessions_command(*all, *limit, &config_path)?;
        }
        Some(Commands::Resume { session_id, last }) => {
            commands::resume_command(session_id.clone(), *last, config_path).await?;
        }
        None => {
            // Default to chat if no subcommand
            commands::chat_command(cli.provider.clone(), cli.model.clone(), config_path, None)
                .await?;
        }
    }

    Ok(())
}
