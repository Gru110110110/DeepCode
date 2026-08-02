use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeepCodeError {
    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Provider '{provider}' does not support requested feature '{feature}'")]
    UnsupportedFeature { provider: String, feature: String },

    #[error("Config error: {0}")]
    Config(String),

    #[error("Tool execution error: {tool} -- {message}")]
    ToolExecution { tool: String, message: String },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(String),

    #[error("Serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Context limit exceeded")]
    ContextLimitExceeded,

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, DeepCodeError>;
