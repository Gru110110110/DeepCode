use async_trait::async_trait;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::{GenerateParams, ProviderCapabilities, RequestBuilder};
use deepcode_core::types::{Message, ToolDefinition};

use crate::openai_compat::{
    build_chat_completions_request, ChatCompletionsOptions, ChatMaxTokensField,
};

pub(crate) struct OpenAiRequestBuilder;

#[async_trait]
impl RequestBuilder for OpenAiRequestBuilder {
    fn capabilities(&self, model: &str) -> ProviderCapabilities {
        let fixed_sampling = model.starts_with("gpt-5");
        ProviderCapabilities {
            provider: "openai",
            temperature: !fixed_sampling,
            top_p: !fixed_sampling,
            stop_sequences: !fixed_sampling,
            reasoning_effort: fixed_sampling,
            reasoning_efforts: if fixed_sampling {
                &[
                    deepcode_core::config::ReasoningEffort::Low,
                    deepcode_core::config::ReasoningEffort::Medium,
                    deepcode_core::config::ReasoningEffort::High,
                    deepcode_core::config::ReasoningEffort::Xhigh,
                    deepcode_core::config::ReasoningEffort::Max,
                ]
            } else {
                &[]
            },
            reasoning_can_disable: fixed_sampling,
            response_format: true,
            verbosity: fixed_sampling,
            tool_choice: true,
            parallel_tool_calls: true,
            strict_tools: true,
            prompt_cache_key: true,
            prediction: true,
            seed: !fixed_sampling,
            logprobs: true,
            safety_identifier: true,
            store: true,
            image_input: true,
            audio_input: model.contains("audio"),
            provider_options: true,
            ..ProviderCapabilities::default()
        }
    }

    fn build_request(
        &self,
        model: &str,
        messages: &[Message],
        tools: &[ToolDefinition],
        system_prompt: Option<&str>,
        params: &GenerateParams,
    ) -> Result<serde_json::Value> {
        build_chat_completions_request(
            model,
            messages,
            tools,
            system_prompt,
            params,
            ChatCompletionsOptions {
                max_tokens_field: ChatMaxTokensField::MaxCompletionTokens,
                capabilities: self.capabilities(model),
                ..ChatCompletionsOptions::default()
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepcode_core::types::{ContentBlock, Message, ToolDefinition};

    #[test]
    fn assistant_tool_use_serializes_as_tool_calls() {
        let builder = OpenAiRequestBuilder;
        let messages = vec![Message::assistant(vec![ContentBlock::tool_use(
            "call_1",
            "read_file",
            serde_json::json!({"path": "README.md"}),
        )])];
        let body = builder
            .build_request("gpt-test", &messages, &[], None, &GenerateParams::default())
            .unwrap();

        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert!(msg["content"].is_null());
        assert_eq!(msg["tool_calls"][0]["id"], "call_1");
        assert_eq!(msg["tool_calls"][0]["type"], "function");
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "read_file");
        assert_eq!(
            msg["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"README.md\"}"
        );
    }

    #[test]
    fn tools_serialize_as_function_definitions() {
        let builder = OpenAiRequestBuilder;
        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = builder
            .build_request(
                "gpt-test",
                &[Message::user("hi")],
                &tools,
                None,
                &GenerateParams::default(),
            )
            .unwrap();

        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn chat_completions_uses_current_completion_token_field() {
        let params = GenerateParams {
            max_tokens: Some(1234),
            ..GenerateParams::default()
        };
        let body = OpenAiRequestBuilder
            .build_request("gpt-test", &[Message::user("hi")], &[], None, &params)
            .unwrap();

        assert_eq!(body["max_completion_tokens"], 1234);
        assert!(body.get("max_tokens").is_none());
    }
}
