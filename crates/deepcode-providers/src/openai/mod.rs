// OpenAI provider - strategy pattern implementation.
pub(crate) mod compress;
pub(crate) mod request;
pub(crate) mod response;
pub(crate) mod responses_request;
pub(crate) mod responses_response;

use deepcode_core::config::ProviderConfig;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::{
    ContextCompressor, GenerateParams, LlmProvider, ProviderCapabilities, RequestBuilder,
    ResponseParser, StreamDelta,
};
use deepcode_core::types::{Message, ToolDefinition};
use futures::stream::Stream;
use std::pin::Pin;

use crate::transport;

pub(crate) struct OpenAiProvider {
    pub(crate) client: reqwest::Client,
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    pub(crate) request_builder: request::OpenAiRequestBuilder,
    pub(crate) response_parser: response::OpenAiResponseParser,
    pub(crate) context_compressor: compress::OpenAiContextCompressor,
    pub(crate) limiter: transport::RequestLimiter,
    wire_api: WireApi,
    responses_request_builder: responses_request::ResponsesRequestBuilder,
    responses_response_parser: responses_response::ResponsesResponseParser,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WireApi {
    ChatCompletions,
    Responses,
}

impl OpenAiProvider {
    pub(crate) fn new(config: &ProviderConfig) -> Result<Self> {
        let api_key = config.resolve_api_key().ok_or_else(|| {
            deepcode_core::error::DeepCodeError::Config(
                "OpenAI API key not found (set api_key)".to_string(),
            )
        })?;
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| "https://api.openai.com/v1".to_string())
            .trim_end_matches('/')
            .to_string();
        let wire_api = match config.wire_api.as_deref() {
            Some("responses") => WireApi::Responses,
            Some("chat_completions") => WireApi::ChatCompletions,
            _ if base_url.contains("api.openai.com") => WireApi::Responses,
            _ => WireApi::ChatCompletions,
        };

        Ok(Self {
            client: transport::build_client(config)?,
            api_key,
            base_url,
            request_builder: request::OpenAiRequestBuilder,
            response_parser: response::OpenAiResponseParser,
            context_compressor: compress::OpenAiContextCompressor,
            limiter: transport::RequestLimiter::from_config(config),
            wire_api,
            responses_request_builder: responses_request::ResponsesRequestBuilder,
            responses_response_parser: responses_response::ResponsesResponseParser::openai(),
        })
    }

    fn chat_completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    fn request_url(&self) -> String {
        match self.wire_api {
            WireApi::ChatCompletions => self.chat_completions_url(),
            WireApi::Responses => format!("{}/responses", self.base_url),
        }
    }

    fn headers(&self) -> Vec<transport::Header> {
        vec![("Authorization", format!("Bearer {}", self.api_key))]
    }
}

#[async_trait::async_trait]
impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
    }

    fn request_builder(&self) -> &dyn RequestBuilder {
        &self.request_builder
    }

    fn capabilities(&self, model: &str) -> ProviderCapabilities {
        match self.wire_api {
            WireApi::ChatCompletions => self.request_builder.capabilities(model),
            WireApi::Responses => self.responses_request_builder.capabilities(model),
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
    ) -> Result<deepcode_core::provider::traits::GenerateResponse> {
        let (body, parser): (serde_json::Value, &dyn ResponseParser) = match self.wire_api {
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
        let raw = self.send_request(&body).await?;
        parser.parse_response(&raw)
    }

    async fn send_request(&self, body: &serde_json::Value) -> Result<serde_json::Value> {
        let _permit = self.limiter.acquire().await?;
        transport::send_json_request(&self.client, self.request_url(), self.headers(), body).await
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
        let mut body = match self.wire_api {
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
            if self.wire_api == WireApi::ChatCompletions {
                obj.insert(
                    "stream_options".into(),
                    serde_json::json!({"include_usage": true}),
                );
            }
        }

        let raw_stream =
            transport::send_sse_request(&self.client, self.request_url(), self.headers(), &body)
                .await?;
        let parsed = match self.wire_api {
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
