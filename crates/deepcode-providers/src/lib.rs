mod anthropic;
pub mod catalog;
mod compress_common;
mod deepseek;
mod kimi;
mod ollama;
mod openai;
mod openai_compat;
mod sse;
mod transport;

use deepcode_core::config::ProviderConfig;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::LlmProvider;
use std::sync::Arc;

/// Create the appropriate provider from configuration.
pub fn create_provider(config: &ProviderConfig) -> Result<Arc<dyn LlmProvider>> {
    match config.kind.as_str() {
        "openai" => Ok(Arc::new(openai::OpenAiProvider::new(config)?)),
        "anthropic" => Ok(Arc::new(anthropic::AnthropicProvider::new(config)?)),
        "deepseek" => Ok(Arc::new(deepseek::DeepSeekProvider::new(config)?)),
        "ollama" => Ok(Arc::new(ollama::OllamaProvider::new(config)?)),
        "kimi" => Ok(Arc::new(kimi::KimiProvider::new(config)?)),
        other => Err(deepcode_core::error::DeepCodeError::Config(format!(
            "Unknown provider type: {}. Supported: openai, anthropic, deepseek, ollama, kimi",
            other
        ))),
    }
}
