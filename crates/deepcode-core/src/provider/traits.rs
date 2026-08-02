use async_trait::async_trait;
use futures::stream::Stream;
use std::collections::BTreeMap;
use std::pin::Pin;

use crate::config::ReasoningEffort;
use crate::error::Result;
use crate::types::{ContentBlock, Message, ToolDefinition};

// ── Strategy trait 1: Request Building ──

/// Build provider-specific HTTP request bodies from canonical types.
///
/// Each provider has a different API format (OpenAI chat/completions,
/// Anthropic Messages, Ollama /api/chat, etc.). This trait encapsulates
/// that conversion so the agent loop never touches raw JSON.
#[async_trait]
pub trait RequestBuilder: Send + Sync {
    /// Capabilities for this concrete provider/model/wire-format combination.
    fn capabilities(&self, _model: &str) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    /// Convert canonical Message/tool/param types into a provider-specific
    /// JSON request body (the full POST body).
    fn build_request(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_prompt: Option<&str>,
        params: &GenerateParams,
    ) -> Result<serde_json::Value>;
}

// ── Strategy trait 2: Response Parsing ──

/// Parse provider-specific response bodies into canonical types.
///
/// Providers return different JSON structures and SSE event formats.
/// This trait normalizes them into `GenerateResponse` / `StreamDelta`.
#[async_trait]
pub trait ResponseParser: Send + Sync {
    /// Parse a complete (non-streaming) response body.
    fn parse_response(&self, raw_body: &serde_json::Value) -> Result<GenerateResponse>;

    /// Parse a single line from an SSE stream into a delta.
    /// Returns `Ok(None)` for lines that don't contain a meaningful delta
    /// (comments, empty data, [DONE] markers).
    fn parse_stream_chunk(&self, raw_line: &str) -> Result<Option<StreamDelta>>;
}

// ── Strategy trait 3: Context Compression ──

/// Compress conversation context for a specific provider.
///
/// Different providers have different tokenizers and respond better to
/// different compression strategies. This trait lets each provider
/// customize how it estimates tokens and compresses history.
#[async_trait]
pub trait ContextCompressor: Send + Sync {
    /// Check whether compression is needed given current usage vs window.
    fn needs_compression(&self, token_count: usize, context_window: usize) -> bool;

    /// Estimate the token count for a list of messages.
    ///
    /// This should be fast — it runs before every LLM call.
    /// Providers may use rough heuristics (chars/4) or integrate with
    /// real tokenizers (tiktoken, anthropic-tokenizer, etc.).
    fn estimate_tokens(&self, messages: &[Message]) -> usize;

    /// Compress message history to fit within `target_tokens`.
    ///
    /// Called only when `needs_compression` returns true.
    /// Returns the compressed messages and updated token count.
    async fn compress(
        &self,
        messages: &[Message],
        current_tokens: usize,
        target_tokens: usize,
    ) -> Result<(Vec<Message>, usize)>;
}

// ── Unified Provider Trait ──

/// The top-level LLM provider abstraction.
///
/// Composes the three strategy traits and adds HTTP transport.
/// The agent loop only depends on this trait — never on concrete providers.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Human-readable provider identifier (e.g., "openai", "anthropic").
    fn name(&self) -> &str;

    /// Strategy: how this provider builds HTTP request bodies.
    fn request_builder(&self) -> &dyn RequestBuilder;

    fn capabilities(&self, model: &str) -> ProviderCapabilities {
        self.request_builder().capabilities(model)
    }

    /// Strategy: how this provider parses HTTP response bodies.
    fn response_parser(&self) -> &dyn ResponseParser;

    /// Strategy: how this provider compresses context.
    fn context_compressor(&self) -> &dyn ContextCompressor;

    // ── Default methods (shared logic using the strategies above) ──

    /// Send a non-streaming request and return a complete response.
    async fn generate(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_prompt: Option<&str>,
        params: &GenerateParams,
    ) -> Result<GenerateResponse> {
        let body =
            self.request_builder()
                .build_request(model, messages, tools, system_prompt, params)?;
        let raw = self.send_request(&body).await?;
        self.response_parser().parse_response(&raw)
    }

    /// Generate a streaming response.
    /// Each provider must implement its own streaming logic because
    /// SSE formats and response parsing differ significantly.
    async fn generate_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_prompt: Option<&str>,
        params: &GenerateParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamDelta>> + Send>>>;

    // ── Transport layer (provider-specific HTTP) ──

    /// Send a non-streaming HTTP request to this provider's API.
    async fn send_request(&self, body: &serde_json::Value) -> Result<serde_json::Value>;
}

// ── Shared Parameter / Response Types ──

#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
    /// Provider reasoning depth (for example `low`, `high`, or `max`).
    pub reasoning_effort: Option<ReasoningEffort>,
    pub reasoning_summary: Option<ReasoningSummary>,
    pub reasoning_mode: Option<ReasoningMode>,
    pub reasoning_context: Option<ReasoningContext>,
    pub reasoning_display: Option<ReasoningDisplay>,
    pub response_format: Option<ResponseFormat>,
    pub verbosity: Option<TextVerbosity>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub strict_tools: Option<bool>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<String>,
    pub prediction: Option<String>,
    pub seed: Option<i64>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u8>,
    pub safety_identifier: Option<String>,
    pub metadata: BTreeMap<String, String>,
    pub store: Option<bool>,
    pub previous_response_id: Option<String>,
    /// Provider-specific escape hatch. Keys are provider names, values are
    /// merged into the top-level request after typed fields are lowered.
    pub provider_options: BTreeMap<String, serde_json::Value>,
    /// Explicit opt-in for dropping fields unsupported by the selected model.
    pub unsupported_feature_policy: UnsupportedFeaturePolicy,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: Some(4096),
            top_p: None,
            stop_sequences: vec![],
            reasoning_effort: None,
            reasoning_summary: None,
            reasoning_mode: None,
            reasoning_context: None,
            reasoning_display: None,
            response_format: None,
            verbosity: None,
            tool_choice: None,
            parallel_tool_calls: None,
            strict_tools: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            prediction: None,
            seed: None,
            logprobs: None,
            top_logprobs: None,
            safety_identifier: None,
            metadata: BTreeMap::new(),
            store: None,
            previous_response_id: None,
            provider_options: BTreeMap::new(),
            unsupported_feature_policy: UnsupportedFeaturePolicy::Error,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnsupportedFeaturePolicy {
    #[default]
    Error,
    AllowFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSummary {
    Auto,
    Concise,
    Detailed,
}

impl ReasoningSummary {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Concise => "concise",
            Self::Detailed => "detailed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningMode {
    Standard,
    Pro,
}

impl ReasoningMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::Pro => "pro",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningContext {
    Relevant,
    AllTurns,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningDisplay {
    Summarized,
    Omitted,
}

impl ReasoningDisplay {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Summarized => "summarized",
            Self::Omitted => "omitted",
        }
    }
}

impl ReasoningContext {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Relevant => "relevant",
            Self::AllTurns => "all_turns",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextVerbosity {
    Low,
    Medium,
    High,
}

impl TextVerbosity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        description: Option<String>,
        schema: serde_json::Value,
        strict: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Function(String),
}

#[derive(Debug, Clone, Copy, Default)]
pub struct ProviderCapabilities {
    pub provider: &'static str,
    pub temperature: bool,
    pub top_p: bool,
    pub stop_sequences: bool,
    pub reasoning_effort: bool,
    pub reasoning_efforts: &'static [ReasoningEffort],
    pub reasoning_can_disable: bool,
    pub reasoning_summary: bool,
    pub reasoning_mode: bool,
    pub reasoning_context: bool,
    pub reasoning_display: bool,
    pub response_format: bool,
    pub verbosity: bool,
    pub tool_choice: bool,
    pub parallel_tool_calls: bool,
    pub strict_tools: bool,
    pub prompt_cache_key: bool,
    pub prompt_cache_retention: bool,
    pub prediction: bool,
    pub seed: bool,
    pub logprobs: bool,
    pub safety_identifier: bool,
    pub metadata: bool,
    pub store: bool,
    pub previous_response_id: bool,
    pub image_input: bool,
    pub audio_input: bool,
    pub file_input: bool,
    pub provider_item_replay: bool,
    pub provider_options: bool,
}

impl ProviderCapabilities {
    pub fn validate(&self, params: &GenerateParams) -> Result<()> {
        if params.unsupported_feature_policy == UnsupportedFeaturePolicy::AllowFallback {
            return Ok(());
        }
        let requested = [
            (
                params.temperature.is_some(),
                self.temperature,
                "temperature",
            ),
            (params.top_p.is_some(), self.top_p, "top_p"),
            (
                !params.stop_sequences.is_empty(),
                self.stop_sequences,
                "stop_sequences",
            ),
            (
                params
                    .reasoning_effort
                    .is_some_and(|effort| effort != ReasoningEffort::Off),
                self.reasoning_effort,
                "reasoning_effort",
            ),
            (
                params.reasoning_summary.is_some(),
                self.reasoning_summary,
                "reasoning_summary",
            ),
            (
                params.reasoning_mode.is_some(),
                self.reasoning_mode,
                "reasoning_mode",
            ),
            (
                params.reasoning_context.is_some(),
                self.reasoning_context,
                "reasoning_context",
            ),
            (
                params.reasoning_display.is_some(),
                self.reasoning_display,
                "reasoning_display",
            ),
            (
                params.response_format.is_some(),
                self.response_format,
                "response_format",
            ),
            (params.verbosity.is_some(), self.verbosity, "verbosity"),
            (
                params.tool_choice.is_some(),
                self.tool_choice,
                "tool_choice",
            ),
            (
                params.parallel_tool_calls.is_some(),
                self.parallel_tool_calls,
                "parallel_tool_calls",
            ),
            (
                params.strict_tools.is_some(),
                self.strict_tools,
                "strict_tools",
            ),
            (
                params.prompt_cache_key.is_some(),
                self.prompt_cache_key,
                "prompt_cache_key",
            ),
            (
                params.prompt_cache_retention.is_some(),
                self.prompt_cache_retention,
                "prompt_cache_retention",
            ),
            (params.prediction.is_some(), self.prediction, "prediction"),
            (params.seed.is_some(), self.seed, "seed"),
            (
                params.logprobs.is_some() || params.top_logprobs.is_some(),
                self.logprobs,
                "logprobs",
            ),
            (
                params.safety_identifier.is_some(),
                self.safety_identifier,
                "safety_identifier",
            ),
            (!params.metadata.is_empty(), self.metadata, "metadata"),
            (params.store.is_some(), self.store, "store"),
            (
                params.previous_response_id.is_some(),
                self.previous_response_id,
                "previous_response_id",
            ),
            (
                !params.provider_options.is_empty(),
                self.provider_options,
                "provider_options",
            ),
        ];
        for (is_requested, is_supported, feature) in requested {
            if is_requested && !is_supported {
                return Err(crate::error::DeepCodeError::UnsupportedFeature {
                    provider: self.provider.to_string(),
                    feature: feature.to_string(),
                });
            }
        }
        if let Some(effort) = params
            .reasoning_effort
            .filter(|effort| *effort != ReasoningEffort::Off)
        {
            if !self.reasoning_efforts.is_empty() && !self.reasoning_efforts.contains(&effort) {
                return Err(crate::error::DeepCodeError::UnsupportedFeature {
                    provider: self.provider.to_string(),
                    feature: format!("reasoning_effort:{}", effort.as_str()),
                });
            }
        }
        if params.reasoning_effort == Some(ReasoningEffort::Off)
            && self.reasoning_effort
            && !self.reasoning_can_disable
        {
            return Err(crate::error::DeepCodeError::UnsupportedFeature {
                provider: self.provider.to_string(),
                feature: "reasoning_effort:off".to_string(),
            });
        }
        Ok(())
    }

    pub fn validate_request(&self, messages: &[Message], params: &GenerateParams) -> Result<()> {
        self.validate(params)?;
        if params.unsupported_feature_policy == UnsupportedFeaturePolicy::AllowFallback {
            return Ok(());
        }
        for block in messages.iter().flat_map(|message| &message.content) {
            let feature = match block {
                ContentBlock::Image { .. } if !self.image_input => Some("image_input"),
                ContentBlock::Audio { .. } if !self.audio_input => Some("audio_input"),
                ContentBlock::File { .. } if !self.file_input => Some("file_input"),
                ContentBlock::ProviderItem { provider, .. }
                    if provider == self.provider && !self.provider_item_replay =>
                {
                    Some("provider_item_replay")
                }
                ContentBlock::ProviderItem { provider, .. } if provider != self.provider => {
                    Some("foreign_provider_item")
                }
                _ => None,
            };
            if let Some(feature) = feature {
                return Err(crate::error::DeepCodeError::UnsupportedFeature {
                    provider: self.provider.to_string(),
                    feature: feature.to_string(),
                });
            }
        }
        if params
            .provider_options
            .keys()
            .any(|provider| provider != self.provider)
        {
            return Err(crate::error::DeepCodeError::UnsupportedFeature {
                provider: self.provider.to_string(),
                feature: "foreign_provider_options".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub content: Vec<ResponseContentBlock>,
    pub usage: Usage,
    pub finish_reason: FinishReason,
}

#[derive(Debug, Clone)]
pub enum ResponseContentBlock {
    Text(String),
    Reasoning {
        text: String,
        metadata: Option<serde_json::Value>,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ProviderItem {
        provider: String,
        value: serde_json::Value,
    },
}

/// A single streaming delta from the LLM.
#[derive(Debug, Clone)]
pub enum StreamDelta {
    /// A piece of text content (may be partial).
    TextDelta(String),
    /// A piece of hidden model reasoning content (may be partial).
    ReasoningDelta(String),
    /// Provider metadata needed to continue reasoning across turns.
    ReasoningMetadata(serde_json::Value),
    /// A completed provider-owned output item which must be replayed verbatim.
    ProviderItem {
        provider: String,
        value: serde_json::Value,
    },
    /// Start of a tool call block.
    ToolUseStart {
        id: String,
        name: String,
        index: Option<usize>,
        input_delta: Option<String>,
    },
    /// A chunk of tool input JSON (appended incrementally).
    ToolUseInput {
        id: String,
        index: Option<usize>,
        input_delta: String,
    },
    /// End of a tool call block.
    ToolUseEnd { id: String, index: Option<usize> },
    /// Multiple deltas encoded in one provider stream event.
    Batch(Vec<StreamDelta>),
    /// Token usage information (may appear at end of stream).
    Usage {
        input_tokens: usize,
        output_tokens: usize,
        cached_input_tokens: usize,
        cache_miss_input_tokens: usize,
        reasoning_output_tokens: usize,
    },
    /// Stream finished with this reason.
    Finished(FinishReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cached_input_tokens: usize,
    pub cache_miss_input_tokens: usize,
    pub reasoning_output_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_features_fail_unless_fallback_is_explicit() {
        let capabilities = ProviderCapabilities {
            provider: "test",
            ..ProviderCapabilities::default()
        };
        let mut params = GenerateParams {
            temperature: Some(0.5),
            ..GenerateParams::default()
        };
        assert!(matches!(
            capabilities.validate(&params),
            Err(crate::error::DeepCodeError::UnsupportedFeature { feature, .. })
                if feature == "temperature"
        ));
        params.unsupported_feature_policy = UnsupportedFeaturePolicy::AllowFallback;
        assert!(capabilities.validate(&params).is_ok());
    }
}
