use async_trait::async_trait;
use deepcode_core::config::ReasoningEffort;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::{
    GenerateParams, ProviderCapabilities, RequestBuilder, UnsupportedFeaturePolicy,
};
use deepcode_core::types::{Message, ToolDefinition};

use crate::openai::responses_request::{build_responses_request, ResponsesOptions};
use crate::openai_compat::{build_chat_completions_request, ChatCompletionsOptions};

use super::history::chat_context_messages;

pub(crate) struct DeepSeekRequestBuilder;
pub(crate) struct DeepSeekResponsesRequestBuilder;

#[async_trait]
impl RequestBuilder for DeepSeekRequestBuilder {
    fn capabilities(&self, _model: &str) -> ProviderCapabilities {
        ProviderCapabilities {
            provider: "deepseek",
            temperature: true,
            top_p: true,
            stop_sequences: true,
            reasoning_effort: true,
            reasoning_efforts: &[
                ReasoningEffort::Minimal,
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
                ReasoningEffort::Xhigh,
                ReasoningEffort::Max,
            ],
            reasoning_can_disable: true,
            response_format: true,
            tool_choice: true,
            logprobs: true,
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
        let messages = chat_context_messages(messages);
        let mut body = build_chat_completions_request(
            model,
            messages.as_ref(),
            tools,
            system_prompt,
            params,
            ChatCompletionsOptions {
                include_reasoning_content: true,
                capabilities: self.capabilities(model),
                ..ChatCompletionsOptions::default()
            },
        )?;
        reject_ignored_thinking_sampling(params, "deepseek")?;
        let Some(object) = body.as_object_mut() else {
            return Ok(body);
        };

        match params.reasoning_effort {
            Some(ReasoningEffort::Off) => {
                object.remove("reasoning_effort");
                object.insert("thinking".into(), serde_json::json!({ "type": "disabled" }));
            }
            Some(effort) => {
                object.insert("thinking".into(), serde_json::json!({ "type": "enabled" }));
                object.insert(
                    "reasoning_effort".into(),
                    deepseek_chat_effort(model, effort).into(),
                );
                strip_thinking_ignored_sampling(object);
            }
            None => {
                object.insert("thinking".into(), serde_json::json!({ "type": "enabled" }));
                strip_thinking_ignored_sampling(object);
            }
        }
        Ok(body)
    }
}

fn reject_ignored_thinking_sampling(params: &GenerateParams, provider: &str) -> Result<()> {
    let thinking_enabled = params
        .reasoning_effort
        .is_none_or(|effort| effort != ReasoningEffort::Off);
    if thinking_enabled
        && (params.temperature.is_some() || params.top_p.is_some())
        && params.unsupported_feature_policy == UnsupportedFeaturePolicy::Error
    {
        return Err(deepcode_core::error::DeepCodeError::UnsupportedFeature {
            provider: provider.to_string(),
            feature: "sampling_with_thinking".to_string(),
        });
    }
    Ok(())
}

#[async_trait]
impl RequestBuilder for DeepSeekResponsesRequestBuilder {
    fn capabilities(&self, _model: &str) -> ProviderCapabilities {
        ProviderCapabilities {
            provider: "deepseek",
            reasoning_effort: true,
            reasoning_efforts: &[
                ReasoningEffort::Minimal,
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
                ReasoningEffort::Xhigh,
                ReasoningEffort::Max,
            ],
            reasoning_can_disable: true,
            tool_choice: true,
            provider_item_replay: true,
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
        let mut body = build_responses_request(
            model,
            messages,
            tools,
            system_prompt,
            params,
            ResponsesOptions {
                capabilities: self.capabilities(model),
                include_encrypted_reasoning: false,
                default_store: None,
            },
        )?;
        let Some(object) = body.as_object_mut() else {
            return Ok(body);
        };

        object.remove("include");
        object.remove("store");
        match params.reasoning_effort {
            Some(ReasoningEffort::Off) => {
                object.insert("reasoning".into(), serde_json::json!({ "effort": "none" }));
            }
            Some(effort) => {
                object.insert(
                    "reasoning".into(),
                    serde_json::json!({ "effort": deepseek_responses_effort(model, effort) }),
                );
            }
            None => {
                object.insert("reasoning".into(), serde_json::json!({ "effort": "high" }));
            }
        }
        Ok(body)
    }
}

fn deepseek_chat_effort(model: &str, effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Minimal | ReasoningEffort::Low if is_v4_flash(model) => "low",
        ReasoningEffort::Minimal
        | ReasoningEffort::Low
        | ReasoningEffort::Medium
        | ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh if is_v4_flash(model) => "high",
        ReasoningEffort::Xhigh | ReasoningEffort::Max => "max",
        ReasoningEffort::Off => "off",
    }
}

fn deepseek_responses_effort(model: &str, effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Off => "none",
        other => deepseek_chat_effort(model, other),
    }
}

fn strip_thinking_ignored_sampling(object: &mut serde_json::Map<String, serde_json::Value>) {
    object.remove("temperature");
    object.remove("top_p");
}

fn is_v4_flash(model: &str) -> bool {
    model == "deepseek-v4-flash"
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepcode_core::types::{ContentBlock, Message};

    #[test]
    fn assistant_reasoning_serializes_as_reasoning_content() {
        let builder = DeepSeekRequestBuilder;
        let messages = vec![Message::assistant(vec![
            ContentBlock::reasoning("hidden chain"),
            ContentBlock::tool_use(
                "call_1",
                "read_file",
                serde_json::json!({"path": "README.md"}),
            ),
        ])];
        let body = builder
            .build_request(
                "deepseek-test",
                &messages,
                &[],
                None,
                &GenerateParams::default(),
            )
            .unwrap();

        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["reasoning_content"], "hidden chain");
        assert!(msg["content"].is_null());
        assert_eq!(msg["tool_calls"][0]["id"], "call_1");
    }

    #[test]
    fn assistant_reasoning_is_omitted_without_tool_calls() {
        let messages = vec![
            Message::user("question"),
            Message::assistant(vec![
                ContentBlock::reasoning("hidden chain"),
                ContentBlock::text("answer"),
            ]),
        ];

        let body = DeepSeekRequestBuilder
            .build_request(
                "deepseek-test",
                &messages,
                &[],
                None,
                &GenerateParams::default(),
            )
            .unwrap();

        let msg = &body["messages"][1];
        assert_eq!(msg["content"], "answer");
        assert!(msg.get("reasoning_content").is_none());
    }

    #[test]
    fn off_disables_deepseek_thinking() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Off),
            ..GenerateParams::default()
        };
        let body = DeepSeekRequestBuilder
            .build_request(
                "deepseek-v4-pro",
                &[Message::user("hello")],
                &[],
                None,
                &params,
            )
            .unwrap();
        assert_eq!(body["thinking"]["type"], "disabled");
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn high_enables_thinking_and_omits_ignored_sampling() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::High),
            top_p: Some(0.9),
            unsupported_feature_policy: UnsupportedFeaturePolicy::AllowFallback,
            ..GenerateParams::default()
        };
        let body = DeepSeekRequestBuilder
            .build_request(
                "deepseek-v4-pro",
                &[Message::user("hello")],
                &[],
                None,
                &params,
            )
            .unwrap();
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
    }

    #[test]
    fn xhigh_maps_to_deepseek_max_effort() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Xhigh),
            ..GenerateParams::default()
        };
        let body = DeepSeekRequestBuilder
            .build_request(
                "deepseek-v4-pro",
                &[Message::user("hello")],
                &[],
                None,
                &params,
            )
            .unwrap();
        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["reasoning_effort"], "max");
    }

    #[test]
    fn flash_low_effort_stays_low() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Low),
            ..GenerateParams::default()
        };
        let body = DeepSeekRequestBuilder
            .build_request(
                "deepseek-v4-flash",
                &[Message::user("hello")],
                &[],
                None,
                &params,
            )
            .unwrap();
        assert_eq!(body["reasoning_effort"], "low");
    }

    #[test]
    fn responses_off_disables_thinking_with_none_effort() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Off),
            ..GenerateParams::default()
        };
        let body = DeepSeekResponsesRequestBuilder
            .build_request(
                "deepseek-v4-flash",
                &[Message::user("hello")],
                &[],
                None,
                &params,
            )
            .unwrap();
        assert_eq!(body["reasoning"]["effort"], "none");
        assert!(body.get("include").is_none());
        assert!(body.get("store").is_none());
    }

    #[test]
    fn responses_uses_deepseek_effort_mapping() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Max),
            ..GenerateParams::default()
        };
        let body = DeepSeekResponsesRequestBuilder
            .build_request(
                "deepseek-v4-flash",
                &[Message::user("hello")],
                &[],
                None,
                &params,
            )
            .unwrap();
        assert_eq!(body["reasoning"]["effort"], "max");
        assert!(body["reasoning"].get("summary").is_none());
    }
}
