use async_trait::async_trait;
use deepcode_core::config::ReasoningEffort;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::{
    GenerateParams, ProviderCapabilities, RequestBuilder, UnsupportedFeaturePolicy,
};
use deepcode_core::types::{Message, ToolDefinition};

use crate::openai_compat::{
    build_chat_completions_request, ChatCompletionsOptions, ChatMaxTokensField,
};

pub(crate) struct KimiRequestBuilder;

#[async_trait]
impl RequestBuilder for KimiRequestBuilder {
    fn capabilities(&self, model: &str) -> ProviderCapabilities {
        let fixed_sampling = matches!(
            model,
            "kimi-k3" | "kimi-k2.6" | "kimi-k2.7-code" | "kimi-k2.7-code-highspeed" | "kimi-k2.5"
        );
        ProviderCapabilities {
            provider: "kimi",
            temperature: !fixed_sampling,
            top_p: !fixed_sampling,
            stop_sequences: true,
            reasoning_effort: model.starts_with("kimi-k"),
            reasoning_efforts: kimi_efforts(model),
            reasoning_can_disable: matches!(model, "kimi-k2.6" | "kimi-k2.5"),
            response_format: true,
            tool_choice: true,
            parallel_tool_calls: true,
            strict_tools: true,
            prompt_cache_key: true,
            prediction: true,
            seed: true,
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
        let fixed_sampling = matches!(
            model,
            "kimi-k3" | "kimi-k2.6" | "kimi-k2.7-code" | "kimi-k2.7-code-highspeed" | "kimi-k2.5"
        );
        if fixed_sampling
            && (params.temperature.is_some() || params.top_p.is_some())
            && params.unsupported_feature_policy == UnsupportedFeaturePolicy::Error
        {
            return Err(deepcode_core::error::DeepCodeError::UnsupportedFeature {
                provider: "kimi".to_string(),
                feature: "fixed_sampling".to_string(),
            });
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
        match model {
            "kimi-k3" => {
                object.remove("thinking");
                strip_fixed_sampling(object);
                match params.reasoning_effort {
                    Some(ReasoningEffort::Off) => {
                        object.remove("reasoning_effort");
                    }
                    Some(effort) => {
                        object.insert("reasoning_effort".into(), kimi_k3_effort(effort).into());
                    }
                    None => {}
                }
            }
            "kimi-k2.6" => {
                let disabled = params
                    .reasoning_effort
                    .is_some_and(|effort| effort == ReasoningEffort::Off);
                object.remove("reasoning_effort");
                strip_fixed_sampling(object);
                object.insert(
                    "thinking".into(),
                    if disabled {
                        serde_json::json!({"type": "disabled"})
                    } else {
                        serde_json::json!({"type": "enabled", "keep": "all"})
                    },
                );
            }
            "kimi-k2.7-code" | "kimi-k2.7-code-highspeed" => {
                object.remove("reasoning_effort");
                object.remove("thinking");
                strip_fixed_sampling(object);
            }
            "kimi-k2.5" => {
                let disabled = params
                    .reasoning_effort
                    .is_some_and(|effort| effort == ReasoningEffort::Off);
                object.remove("reasoning_effort");
                strip_fixed_sampling(object);
                object.insert(
                    "thinking".into(),
                    serde_json::json!({"type": if disabled { "disabled" } else { "enabled" }}),
                );
            }
            _ => {}
        }
        Ok(body)
    }
}

fn kimi_efforts(model: &str) -> &'static [ReasoningEffort] {
    match model {
        "kimi-k3" => &[
            ReasoningEffort::Low,
            ReasoningEffort::High,
            ReasoningEffort::Max,
        ],
        "kimi-k2.7-code" | "kimi-k2.7-code-highspeed" => &[ReasoningEffort::High],
        "kimi-k2.6" | "kimi-k2.5" => &[ReasoningEffort::High],
        _ => &[],
    }
}

fn kimi_k3_effort(effort: ReasoningEffort) -> &'static str {
    match effort {
        ReasoningEffort::Off => "off",
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low",
        ReasoningEffort::Medium | ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh | ReasoningEffort::Max => "max",
    }
}

fn strip_fixed_sampling(object: &mut serde_json::Map<String, serde_json::Value>) {
    object.remove("temperature");
    object.remove("top_p");
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepcode_core::types::Message;

    #[test]
    fn k2_6_off_disables_thinking() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Off),
            top_p: Some(0.8),
            unsupported_feature_policy: UnsupportedFeaturePolicy::AllowFallback,
            ..GenerateParams::default()
        };
        let body = KimiRequestBuilder
            .build_request("kimi-k2.6", &[Message::user("hello")], &[], None, &params)
            .unwrap();
        assert_eq!(body["thinking"]["type"], "disabled");
        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert_eq!(body["max_completion_tokens"], 4096);
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn k2_6_preserves_thinking_when_enabled() {
        let body = KimiRequestBuilder
            .build_request(
                "kimi-k2.6",
                &[Message::user("hello")],
                &[],
                None,
                &GenerateParams::default(),
            )
            .unwrap();

        assert_eq!(body["thinking"]["type"], "enabled");
        assert_eq!(body["thinking"]["keep"], "all");
    }

    #[test]
    fn k3_uses_reasoning_effort_and_omits_fixed_sampling() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Medium),
            top_p: Some(0.9),
            unsupported_feature_policy: UnsupportedFeaturePolicy::AllowFallback,
            ..GenerateParams::default()
        };
        let body = KimiRequestBuilder
            .build_request("kimi-k3", &[Message::user("hello")], &[], None, &params)
            .unwrap();

        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("thinking").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
    }

    #[test]
    fn k2_7_code_highspeed_omits_thinking_controls() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::High),
            ..GenerateParams::default()
        };
        let body = KimiRequestBuilder
            .build_request(
                "kimi-k2.7-code-highspeed",
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
}
