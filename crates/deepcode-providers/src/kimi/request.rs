use async_trait::async_trait;
use deepcode_core::config::ReasoningEffort;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::{
    GenerateParams, ProviderCapabilities, RequestBuilder, ToolChoice, UnsupportedFeaturePolicy,
};
use deepcode_core::types::{Message, ToolDefinition};

use crate::openai_compat::{
    build_chat_completions_request, ChatCompletionsOptions, ChatMaxTokensField,
};

pub(crate) struct KimiRequestBuilder;

#[async_trait]
impl RequestBuilder for KimiRequestBuilder {
    fn capabilities(&self, model: &str) -> ProviderCapabilities {
        let fixed_sampling = is_fixed_sampling(model);
        ProviderCapabilities {
            provider: "kimi",
            temperature: !fixed_sampling,
            top_p: !fixed_sampling,
            stop_sequences: true,
            reasoning_effort: is_kimi_reasoning_model(model),
            reasoning_efforts: kimi_efforts(model),
            reasoning_can_disable: false,
            response_format: true,
            tool_choice: true,
            // Kimi returns independent tool calls in parallel by default and
            // does not document an OpenAI-style request toggle.
            parallel_tool_calls: false,
            strict_tools: true,
            prompt_cache_key: true,
            prediction: true,
            seed: false,
            logprobs: true,
            safety_identifier: true,
            image_input: true,
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
        let fixed_sampling = is_fixed_sampling(model);
        if fixed_sampling
            && (params.temperature.is_some() || params.top_p.is_some())
            && params.unsupported_feature_policy == UnsupportedFeaturePolicy::Error
        {
            return Err(deepcode_core::error::DeepCodeError::UnsupportedFeature {
                provider: self.capabilities(model).provider.to_string(),
                feature: "fixed_sampling".to_string(),
            });
        }
        // K3 accepts `required` and a named function; the always-thinking
        // K2.7 Code models only accept `auto` and `none`.
        let unsupported_tool_choice = match params.tool_choice.as_ref() {
            Some(ToolChoice::Required) if !is_k3(model) => Some("tool_choice.required"),
            Some(ToolChoice::Function(_)) if !is_k3(model) => Some("tool_choice.function"),
            _ => None,
        };
        if params.unsupported_feature_policy == UnsupportedFeaturePolicy::Error {
            if let Some(feature) = unsupported_tool_choice {
                return Err(deepcode_core::error::DeepCodeError::UnsupportedFeature {
                    provider: self.capabilities(model).provider.to_string(),
                    feature: feature.to_string(),
                });
            }
        }
        let mut body = build_chat_completions_request(
            model,
            messages,
            tools,
            system_prompt,
            params,
            ChatCompletionsOptions {
                include_reasoning_content: true,
                max_tokens_field: ChatMaxTokensField::MaxCompletionTokens,
                capabilities: self.capabilities(model),
            },
        )?;
        let Some(object) = body.as_object_mut() else {
            return Ok(body);
        };
        // Kimi requires the assistant's reasoning_content field to survive
        // every multi-step tool call, including when its value is empty.
        if is_kimi_reasoning_model(model) {
            preserve_reasoning_content_for_tool_calls(object);
        }
        if unsupported_tool_choice.is_some() && object.contains_key("tool_choice") {
            object.insert("tool_choice".into(), "auto".into());
        }
        match model {
            "k3" | "k3-256k" => {
                object.remove("thinking");
                strip_fixed_sampling(object);
                match params.reasoning_effort {
                    Some(ReasoningEffort::Off) => {
                        // Keep the selected K3 model instead of silently
                        // routing an off request to K2.6.
                        object.remove("reasoning_effort");
                    }
                    Some(effort) => {
                        object.insert("reasoning_effort".into(), kimi_k3_effort(effort).into());
                    }
                    None => {}
                }
            }
            "kimi-for-coding" | "kimi-for-coding-highspeed" => {
                object.remove("reasoning_effort");
                object.remove("thinking");
                strip_fixed_sampling(object);
            }
            _ => {}
        }
        Ok(body)
    }
}

fn kimi_efforts(model: &str) -> &'static [ReasoningEffort] {
    match model {
        "k3" | "k3-256k" => &[
            ReasoningEffort::Low,
            ReasoningEffort::High,
            ReasoningEffort::Max,
        ],
        "kimi-for-coding" | "kimi-for-coding-highspeed" => &[ReasoningEffort::High],
        _ => &[],
    }
}

fn is_k3(model: &str) -> bool {
    matches!(model, "k3" | "k3-256k")
}

fn is_kimi_reasoning_model(model: &str) -> bool {
    is_k3(model) || matches!(model, "kimi-for-coding" | "kimi-for-coding-highspeed")
}

fn is_fixed_sampling(model: &str) -> bool {
    is_kimi_reasoning_model(model)
}

fn kimi_k3_effort(effort: ReasoningEffort) -> &'static str {
    // Keep this exhaustive and aligned with Kimi's documented aliases for
    // defensive callers; normal validation handles Off before this mapping.
    match effort {
        ReasoningEffort::Off => "none",
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium | ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh | ReasoningEffort::Max => "max",
    }
}

fn strip_fixed_sampling(object: &mut serde_json::Map<String, serde_json::Value>) {
    object.remove("temperature");
    object.remove("top_p");
}

fn preserve_reasoning_content_for_tool_calls(
    object: &mut serde_json::Map<String, serde_json::Value>,
) {
    let Some(messages) = object
        .get_mut("messages")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };
    for message in messages {
        let Some(message) = message.as_object_mut() else {
            continue;
        };
        let is_assistant_tool_call = message.get("role").and_then(serde_json::Value::as_str)
            == Some("assistant")
            && message
                .get("tool_calls")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|tool_calls| !tool_calls.is_empty());
        if is_assistant_tool_call && !message.contains_key("reasoning_content") {
            message.insert("reasoning_content".into(), "".into());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepcode_core::types::Message;

    #[test]
    fn k3_uses_reasoning_effort_and_omits_fixed_sampling() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Medium),
            top_p: Some(0.9),
            unsupported_feature_policy: UnsupportedFeaturePolicy::AllowFallback,
            ..GenerateParams::default()
        };
        let body = KimiRequestBuilder
            .build_request("k3", &[Message::user("hello")], &[], None, &params)
            .unwrap();

        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("thinking").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
    }

    #[test]
    fn coding_highspeed_omits_thinking_controls() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::High),
            ..GenerateParams::default()
        };
        let body = KimiRequestBuilder
            .build_request(
                "kimi-for-coding-highspeed",
                &[Message::user("hello")],
                &[],
                None,
                &params,
            )
            .unwrap();

        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("thinking").is_none());
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn k3_uses_code_model_contract() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Medium),
            top_p: Some(0.9),
            unsupported_feature_policy: UnsupportedFeaturePolicy::AllowFallback,
            ..GenerateParams::default()
        };
        let body = KimiRequestBuilder
            .build_request("k3-256k", &[Message::user("hello")], &[], None, &params)
            .unwrap();
        let capabilities = KimiRequestBuilder.capabilities("k3-256k");

        assert_eq!(capabilities.provider, "kimi");
        assert!(!capabilities.parallel_tool_calls);
        assert!(!capabilities.seed);
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("thinking").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("parallel_tool_calls").is_none());
    }

    #[test]
    fn coding_model_rejects_required_tool_choice() {
        let params = GenerateParams {
            tool_choice: Some(ToolChoice::Required),
            ..GenerateParams::default()
        };
        let error = KimiRequestBuilder
            .build_request(
                "kimi-for-coding",
                &[Message::user("hello")],
                &[],
                None,
                &params,
            )
            .unwrap_err();

        assert!(error.to_string().contains("tool_choice.required"));
    }

    #[test]
    fn k3_supports_required_tool_choice() {
        let params = GenerateParams {
            tool_choice: Some(ToolChoice::Required),
            ..GenerateParams::default()
        };
        let tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            description: "Lookup a value".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        }];
        let body = KimiRequestBuilder
            .build_request("k3-256k", &[Message::user("hello")], &tools, None, &params)
            .unwrap();

        assert_eq!(body["tool_choice"], "required");
    }

    #[test]
    fn k3_supports_named_function_tool_choice() {
        let params = GenerateParams {
            tool_choice: Some(ToolChoice::Function("lookup".to_string())),
            ..GenerateParams::default()
        };
        let tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            description: "Lookup a value".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        }];
        let body = KimiRequestBuilder
            .build_request("k3", &[Message::user("hello")], &tools, None, &params)
            .unwrap();

        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "lookup");
    }

    #[test]
    fn coding_model_falls_back_from_required_to_auto() {
        let params = GenerateParams {
            tool_choice: Some(ToolChoice::Required),
            unsupported_feature_policy: UnsupportedFeaturePolicy::AllowFallback,
            ..GenerateParams::default()
        };
        let tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            description: "Lookup a value".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        }];
        let body = KimiRequestBuilder
            .build_request(
                "kimi-for-coding",
                &[Message::user("hello")],
                &tools,
                None,
                &params,
            )
            .unwrap();

        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn coding_model_falls_back_from_named_function_to_auto() {
        let params = GenerateParams {
            tool_choice: Some(ToolChoice::Function("lookup".to_string())),
            unsupported_feature_policy: UnsupportedFeaturePolicy::AllowFallback,
            ..GenerateParams::default()
        };
        let tools = vec![ToolDefinition {
            name: "lookup".to_string(),
            description: "Lookup a value".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
        }];
        let body = KimiRequestBuilder
            .build_request(
                "kimi-for-coding",
                &[Message::user("hello")],
                &tools,
                None,
                &params,
            )
            .unwrap();

        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn coding_model_rejects_named_function_tool_choice() {
        let params = GenerateParams {
            tool_choice: Some(ToolChoice::Function("lookup".to_string())),
            ..GenerateParams::default()
        };
        let error = KimiRequestBuilder
            .build_request(
                "kimi-for-coding",
                &[Message::user("hello")],
                &[],
                None,
                &params,
            )
            .unwrap_err();

        assert!(error.to_string().contains("tool_choice.function"));
    }

    #[test]
    fn unknown_model_accepts_conservative_off_profile() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Off),
            ..GenerateParams::default()
        };
        let body = KimiRequestBuilder
            .build_request(
                "future-kimi-model",
                &[Message::user("hello")],
                &[],
                None,
                &params,
            )
            .unwrap();

        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn tool_call_history_preserves_empty_reasoning_content() {
        let messages = vec![
            Message::user("inspect the project"),
            Message::assistant(vec![deepcode_core::types::ContentBlock::tool_use(
                "call-1",
                "read_file",
                serde_json::json!({"path": "README.md"}),
            )]),
            Message::tool_result("call-1", "contents", false),
        ];

        let body = KimiRequestBuilder
            .build_request("k3", &messages, &[], None, &GenerateParams::default())
            .unwrap();

        assert_eq!(body["messages"][1]["reasoning_content"], "");
    }
}
