// Anthropic provider - uses Messages API (different from OpenAI format).
pub(crate) mod compress;
pub(crate) mod request;
pub(crate) mod response;

use deepcode_core::config::ProviderConfig;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_core::provider::traits::{
    ContextCompressor, GenerateParams, LlmProvider, RequestBuilder, ResponseParser, StreamDelta,
};
use deepcode_core::types::{Message, ToolDefinition};
use futures::stream::Stream;
use std::pin::Pin;

use crate::transport;

pub(crate) struct AnthropicProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) anthropic_version: String,
    pub(crate) request_builder: request::AnthropicRequestBuilder,
    pub(crate) response_parser: response::AnthropicResponseParser,
    pub(crate) context_compressor: compress::AnthropicContextCompressor,
    pub(crate) limiter: transport::RequestLimiter,
}

impl AnthropicProvider {
    pub(crate) fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config.resolve_api_key().ok_or_else(|| {
            DeepCodeError::Config("Anthropic API key not found (set api_key)".to_string())
        })?;
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.anthropic.com/v1".to_string())
            .trim_end_matches('/')
            .to_string();

        Ok(Self {
            client: transport::build_client(config)?,
            api_key,
            base_url,
            anthropic_version: "2023-06-01".to_string(),
            request_builder: request::AnthropicRequestBuilder,
            response_parser: response::AnthropicResponseParser,
            context_compressor: compress::AnthropicContextCompressor,
            limiter: transport::RequestLimiter::from_config(config),
        })
    }

    fn messages_url(&self) -> String {
        format!("{}/messages", self.base_url)
    }

    fn headers(&self, body: &serde_json::Value) -> Vec<transport::Header> {
        let mut headers = vec![
            ("x-api-key", self.api_key.clone()),
            ("anthropic-version", self.anthropic_version.clone()),
        ];
        if contains_file_source(body) {
            headers.push(("anthropic-beta", "files-api-2025-04-14".to_string()));
        }
        headers
    }
}

fn contains_file_source(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Object(object) => {
            (object.get("type").and_then(serde_json::Value::as_str) == Some("file")
                && object.contains_key("file_id"))
                || object.values().any(contains_file_source)
        }
        serde_json::Value::Array(values) => values.iter().any(contains_file_source),
        _ => false,
    }
}

#[async_trait::async_trait]
impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn request_builder(&self) -> &dyn RequestBuilder {
        &self.request_builder
    }

    fn response_parser(&self) -> &dyn ResponseParser {
        &self.response_parser
    }

    fn context_compressor(&self) -> &dyn ContextCompressor {
        &self.context_compressor
    }

    async fn send_request(&self, body: &serde_json::Value) -> Result<serde_json::Value> {
        let _permit = self.limiter.acquire().await?;
        transport::send_json_request_with_retry(
            &self.client,
            self.messages_url(),
            self.headers(body),
            body,
            transport::RetryPolicy::REMOTE,
        )
        .await
    }

    async fn generate_stream(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_prompt: Option<&str>,
        params: &GenerateParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamDelta>> + Send>>> {
        let permit = self.limiter.acquire().await?;
        let mut body =
            self.request_builder
                .build_request(model, messages, tools, system_prompt, params)?;
        if let Some(obj) = body.as_object_mut() {
            obj.insert("stream".into(), true.into());
        }

        let raw_stream = transport::send_sse_request_with_retry(
            &self.client,
            self.messages_url(),
            self.headers(&body),
            &body,
            transport::RetryPolicy::REMOTE,
        )
        .await?;
        Ok(transport::hold_permit(
            transport::parse_sse_lines(raw_stream, self.response_parser),
            permit,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_files_api_sources_without_matching_unrelated_file_blocks() {
        assert!(contains_file_source(&serde_json::json!({
            "messages": [{
                "content": [{
                    "type": "image",
                    "source": {"type": "file", "file_id": "file_123"}
                }]
            }]
        })));
        assert!(!contains_file_source(&serde_json::json!({
            "messages": [{"content": [{"type": "file", "name": "notes.txt"}]}]
        })));
    }
}
