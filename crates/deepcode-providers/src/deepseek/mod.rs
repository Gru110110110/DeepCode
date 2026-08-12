// DeepSeek provider - OpenAI-compatible API with reasoning history support.
pub(crate) mod compress;
mod history;
pub(crate) mod request;

use deepcode_core::config::ProviderConfig;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_core::provider::traits::{
    ContextCompressor, GenerateParams, GenerateResponse, LlmProvider, ProviderCapabilities,
    RequestBuilder, ResponseParser, StreamDelta,
};
use deepcode_core::types::{Message, ToolDefinition};
use futures::stream::Stream;
use std::pin::Pin;

use crate::deepseek::request::{DeepSeekRequestBuilder, DeepSeekResponsesRequestBuilder};
use crate::openai::response::OpenAiResponseParser;
use crate::openai::responses_response::ResponsesResponseParser;
use crate::transport;

pub(crate) struct DeepSeekProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) request_builder: DeepSeekRequestBuilder,
    pub(crate) response_parser: OpenAiResponseParser,
    pub(crate) responses_request_builder: DeepSeekResponsesRequestBuilder,
    pub(crate) responses_response_parser: ResponsesResponseParser,
    pub(crate) context_compressor: compress::DeepSeekContextCompressor,
    pub(crate) limiter: transport::RequestLimiter,
    wire_api: DeepSeekWireApi,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WireApi {
    ChatCompletions,
    Responses,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeepSeekWireApi {
    Auto,
    ChatCompletions,
    Responses,
}

impl DeepSeekProvider {
    pub(crate) fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config.resolve_api_key().ok_or_else(|| {
            DeepCodeError::Config("DeepSeek API key not found (set api_key)".to_string())
        })?;
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.deepseek.com".to_string())
            .trim_end_matches('/')
            .to_string();
        let wire_api = match config.wire_api.as_deref() {
            Some("responses") => DeepSeekWireApi::Responses,
            Some("chat_completions") => DeepSeekWireApi::ChatCompletions,
            _ => DeepSeekWireApi::Auto,
        };

        Ok(Self {
            client: transport::build_client(config)?,
            api_key,
            base_url,
            request_builder: DeepSeekRequestBuilder,
            response_parser: OpenAiResponseParser,
            responses_request_builder: DeepSeekResponsesRequestBuilder,
            responses_response_parser: ResponsesResponseParser::deepseek(),
            context_compressor: compress::DeepSeekContextCompressor,
            limiter: transport::RequestLimiter::from_config(config),
            wire_api,
        })
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn responses_url(&self) -> String {
        format!("{}/responses", self.base_url)
    }

    fn request_url(&self, wire_api: WireApi) -> String {
        match wire_api {
            WireApi::ChatCompletions => self.chat_completions_url(),
            WireApi::Responses => self.responses_url(),
        }
    }

    fn wire_api_for_model(&self, model: &str) -> Result<WireApi> {
        match self.wire_api {
            DeepSeekWireApi::ChatCompletions => Ok(WireApi::ChatCompletions),
            DeepSeekWireApi::Responses if supports_responses_api(model) => Ok(WireApi::Responses),
            DeepSeekWireApi::Responses => Err(DeepCodeError::Config(format!(
                "DeepSeek Responses API currently supports deepseek-v4-flash only; model '{}' should use wire_api = \"chat_completions\"",
                model
            ))),
            DeepSeekWireApi::Auto => Ok(WireApi::ChatCompletions),
        }
    }

    fn headers(&self) -> Vec<transport::Header> {
        vec![("Authorization", format!("Bearer {}", self.api_key))]
    }

    async fn send_request_to(
        &self,
        body: &serde_json::Value,
        wire_api: WireApi,
    ) -> Result<serde_json::Value> {
        let _permit = self.limiter.acquire().await?;
        transport::send_json_request_with_retry(
            &self.client,
            self.request_url(wire_api),
            self.headers(),
            body,
            transport::RetryPolicy::REMOTE,
        )
        .await
    }
}

fn supports_responses_api(model: &str) -> bool {
    model == "deepseek-v4-flash"
}

#[async_trait::async_trait]
impl LlmProvider for DeepSeekProvider {
    fn name(&self) -> &str {
        "deepseek"
    }

    fn request_builder(&self) -> &dyn RequestBuilder {
        &self.request_builder
    }

    fn capabilities(&self, model: &str) -> ProviderCapabilities {
        match self.wire_api_for_model(model) {
            Ok(WireApi::Responses) => self.responses_request_builder.capabilities(model),
            _ => self.request_builder.capabilities(model),
        }
    }

    fn response_parser(&self) -> &dyn ResponseParser {
        &self.response_parser
    }

    fn context_compressor(&self) -> &dyn ContextCompressor {
        &self.context_compressor
    }

    async fn generate(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_prompt: Option<&str>,
        params: &GenerateParams,
    ) -> Result<GenerateResponse> {
        let wire_api = self.wire_api_for_model(model)?;
        let (body, parser): (serde_json::Value, &dyn ResponseParser) = match wire_api {
            WireApi::ChatCompletions => (
                self.request_builder.build_request(
                    model,
                    messages,
                    tools,
                    system_prompt,
                    params,
                )?,
                &self.response_parser,
            ),
            WireApi::Responses => (
                self.responses_request_builder.build_request(
                    model,
                    messages,
                    tools,
                    system_prompt,
                    params,
                )?,
                &self.responses_response_parser,
            ),
        };
        let raw = self.send_request_to(&body, wire_api).await?;
        parser.parse_response(&raw)
    }

    async fn send_request(&self, body: &serde_json::Value) -> Result<serde_json::Value> {
        self.send_request_to(body, WireApi::ChatCompletions).await
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
        let wire_api = self.wire_api_for_model(model)?;
        let mut body = match wire_api {
            WireApi::ChatCompletions => {
                self.request_builder
                    .build_request(model, messages, tools, system_prompt, params)?
            }
            WireApi::Responses => self.responses_request_builder.build_request(
                model,
                messages,
                tools,
                system_prompt,
                params,
            )?,
        };
        if let Some(obj) = body.as_object_mut() {
            obj.insert("stream".into(), true.into());
            if wire_api == WireApi::ChatCompletions {
                obj.insert(
                    "stream_options".into(),
                    serde_json::json!({ "include_usage": true }),
                );
            }
        }

        let raw_stream = transport::send_sse_request_with_retry(
            &self.client,
            self.request_url(wire_api),
            self.headers(),
            &body,
            transport::RetryPolicy::REMOTE,
        )
        .await?;
        let parsed = match wire_api {
            WireApi::ChatCompletions => {
                transport::parse_sse_lines(raw_stream, self.response_parser)
            }
            WireApi::Responses => {
                transport::parse_sse_lines(raw_stream, self.responses_response_parser)
            }
        };
        Ok(transport::hold_permit(parsed, permit))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn deepseek_provider(wire_api: Option<&str>) -> DeepSeekProvider {
        let config = ProviderConfig {
            kind: "deepseek".to_string(),
            api_key: Some("test-key".to_string()),
            base_url: None,
            max_concurrent_requests: None,
            connect_timeout_secs: None,
            read_timeout_secs: None,
            model: None,
            reasoning_effort: None,
            wire_api: wire_api.map(str::to_string),
            models: HashMap::new(),
        };
        DeepSeekProvider::new(&config).unwrap()
    }

    #[test]
    fn flash_defaults_to_chat_completions_api() {
        let provider = deepseek_provider(None);
        assert!(matches!(
            provider.wire_api_for_model("deepseek-v4-flash").unwrap(),
            WireApi::ChatCompletions
        ));
        assert!(matches!(
            provider.wire_api_for_model("deepseek-v4-pro").unwrap(),
            WireApi::ChatCompletions
        ));
    }

    #[test]
    fn configured_wire_api_overrides_model_default() {
        let provider = deepseek_provider(Some("chat_completions"));
        assert!(matches!(
            provider.wire_api_for_model("deepseek-v4-flash").unwrap(),
            WireApi::ChatCompletions
        ));

        let provider = deepseek_provider(Some("responses"));
        assert!(matches!(
            provider.wire_api_for_model("deepseek-v4-flash").unwrap(),
            WireApi::Responses
        ));
    }

    #[test]
    fn configured_responses_rejects_unsupported_deepseek_model() {
        let provider = deepseek_provider(Some("responses"));
        let error = provider.wire_api_for_model("deepseek-v4-pro").unwrap_err();
        assert!(error
            .to_string()
            .contains("supports deepseek-v4-flash only"));
    }
}
