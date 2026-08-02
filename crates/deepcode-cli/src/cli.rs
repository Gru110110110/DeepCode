use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "deepcode",
    about = "Multi-provider AI coding assistant",
    version = "0.1.0"
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Config file path (default: ~/.config/deepcode/config.toml)
    #[arg(long, global = true)]
    pub config: Option<std::path::PathBuf>,

    /// Provider to use
    #[arg(short, long, global = true)]
    pub provider: Option<String>,

    /// Model to use
    #[arg(short, long, global = true)]
    pub model: Option<String>,

    /// Log level
    #[arg(long, global = true, default_value = "info")]
    pub log_level: String,
}

#[derive(Subcommand)]
pub(crate) enum Commands {
    /// Start an interactive coding session
    Chat,
    /// Run a one-shot prompt
    Run {
        /// The prompt to send
        prompt: Vec<String>,
    },
    /// Inspect configuration
    Config,
    /// List models discovered for the active provider
    Models {
        /// Ignore cache TTL and failure backoff
        #[arg(long)]
        refresh: bool,
    },
    /// List saved sessions
    Sessions {
        /// Include sessions from every workspace
        #[arg(long)]
        all: bool,
        /// Maximum number of sessions to show
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Resume a saved session
    Resume {
        /// Session UUID
        session_id: Option<String>,
        /// Resume the most recent session in the current workspace
        #[arg(long)]
        last: bool,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_session_management_commands() {
        let sessions =
            Cli::try_parse_from(["deepcode", "sessions", "--all", "--limit", "20"]).unwrap();
        assert!(matches!(
            sessions.command,
            Some(Commands::Sessions {
                all: true,
                limit: Some(20)
            })
        ));

        let resume = Cli::try_parse_from(["deepcode", "resume", "session-id"]).unwrap();
        assert!(matches!(
            resume.command,
            Some(Commands::Resume {
                session_id: Some(id),
                last: false
            }) if id == "session-id"
        ));
    }

    #[test]
    fn chat_resume_flag_is_rejected() {
        assert!(Cli::try_parse_from(["deepcode", "chat", "--resume"]).is_err());
    }

    #[test]
    fn parses_model_refresh() {
        let models = Cli::try_parse_from(["deepcode", "models", "--refresh"]).unwrap();
        assert!(matches!(
            models.command,
            Some(Commands::Models { refresh: true })
        ));
    }
}
