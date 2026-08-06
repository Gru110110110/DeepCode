mod request;

use deepcode_core::config::ProviderConfig;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_core::provider::traits::{
    ContextCompressor, GenerateParams, LlmProvider, RequestBuilder, ResponseParser, StreamDelta,
};
use deepcode_core::types::{Message, ToolDefinition};
use futures::stream::Stream;
use std::pin::Pin;

use crate::openai::compress::OpenAiContextCompressor;
use crate::openai::response::OpenAiResponseParser;
use crate::transport;

pub(crate) const DEFAULT_BASE_URL: &str = "https://api.kimi.com/coding/v1";
pub(crate) const USER_AGENT: &str = concat!("deepcode/", env!("CARGO_PKG_VERSION"));

pub(crate) struct KimiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    request_builder: request::KimiRequestBuilder,
    response_parser: OpenAiResponseParser,
    context_compressor: OpenAiContextCompressor,
    limiter: transport::RequestLimiter,
}

impl KimiProvider {
    pub(crate) fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config.resolve_api_key().ok_or_else(|| {
            DeepCodeError::Config("Kimi API key not found (set api_key)".to_string())
        })?;
        Ok(Self {
            client: transport::build_client(config)?,
            api_key,
            base_url: config
                .base_url
                .clone()
                .unwrap_or_else(|| DEFAULT_BASE_URL.to_string())
                .trim_end_matches('/')
                .to_string(),
            request_builder: request::KimiRequestBuilder,
            response_parser: OpenAiResponseParser,
            context_compressor: OpenAiContextCompressor,
            limiter: transport::RequestLimiter::from_config(config),
        })
    }

    fn url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn headers(&self) -> Vec<transport::Header> {
        vec![
            ("Authorization", format!("Bearer {}", self.api_key)),
            ("User-Agent", USER_AGENT.to_string()),
        ]
    }
}

#[async_trait::async_trait]
impl LlmProvider for KimiProvider {
    fn name(&self) -> &str {
        "kimi"
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
        transport::send_json_request(&self.client, self.url(), self.headers(), body).await
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
        if let Some(object) = body.as_object_mut() {
            object.insert("stream".into(), true.into());
            object.insert(
                "stream_options".into(),
                serde_json::json!({"include_usage": true}),
            );
        }
        let raw =
            transport::send_sse_request(&self.client, self.url(), self.headers(), &body).await?;
        Ok(transport::hold_permit(
            transport::parse_sse_lines(raw, self.response_parser),
            permit,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(kind: &str) -> ProviderConfig {
        ProviderConfig {
            kind: kind.to_string(),
            api_key: Some("secret".to_string()),
            base_url: None,
            max_concurrent_requests: None,
            request_timeout_secs: None,
            model: None,
            reasoning_effort: None,
            wire_api: None,
            models: Default::default(),
        }
    }

    #[test]
    fn kimi_uses_code_endpoint_by_default() {
        let provider = KimiProvider::new(&config("kimi")).unwrap();

        assert_eq!(provider.name(), "kimi");
        assert_eq!(provider.base_url, "https://api.kimi.com/coding/v1");
        assert_eq!(
            provider.url(),
            "https://api.kimi.com/coding/v1/chat/completions"
        );
        assert_eq!(
            provider.headers(),
            vec![
                ("Authorization", "Bearer secret".to_string()),
                (
                    "User-Agent",
                    format!("deepcode/{}", env!("CARGO_PKG_VERSION"))
                ),
            ]
        );
    }
}
