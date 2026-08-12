// Ollama provider - uses native /api/chat endpoint.
pub(crate) mod compress;
pub(crate) mod request;
pub(crate) mod response;

use deepcode_core::config::ProviderConfig;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::{
    ContextCompressor, GenerateParams, LlmProvider, RequestBuilder, ResponseParser, StreamDelta,
};
use deepcode_core::types::{Message, ToolDefinition};
use futures::stream::Stream;
use std::pin::Pin;

use crate::transport;

pub(crate) struct OllamaProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: Option<String>,
    pub(crate) base_url: String,
    pub(crate) request_builder: request::OllamaRequestBuilder,
    pub(crate) response_parser: response::OllamaResponseParser,
    pub(crate) context_compressor: compress::OllamaContextCompressor,
    pub(crate) limiter: transport::RequestLimiter,
}

impl OllamaProvider {
    pub(crate) fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config.resolve_api_key();
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "http://localhost:11434".to_string())
            .trim_end_matches('/')
            .to_string();

        Ok(Self {
            client: transport::build_client(config)?,
            api_key,
            base_url,
            request_builder: request::OllamaRequestBuilder,
            response_parser: response::OllamaResponseParser,
            context_compressor: compress::OllamaContextCompressor,
            limiter: transport::RequestLimiter::from_config(config),
        })
    }

    fn chat_url(&self) -> String {
        let root = self.base_url.strip_suffix("/v1").unwrap_or(&self.base_url);
        format!("{}/api/chat", root)
    }

    fn headers(&self) -> Vec<transport::Header> {
        self.api_key
            .as_ref()
            .map(|key| vec![("Authorization", format!("Bearer {}", key))])
            .unwrap_or_default()
    }
}

#[async_trait::async_trait]
impl LlmProvider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
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
            self.chat_url(),
            self.headers(),
            body,
            transport::RetryPolicy::LOCAL,
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
            self.chat_url(),
            self.headers(),
            &body,
            transport::RetryPolicy::LOCAL,
        )
        .await?;
        Ok(transport::hold_permit(
            transport::parse_sse_lines(raw_stream, self.response_parser),
            permit,
        ))
    }
}
