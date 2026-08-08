use async_trait::async_trait;
use deepcode_core::config::ReasoningEffort;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::{
    GenerateParams, ProviderCapabilities, RequestBuilder, ToolChoice, UnsupportedFeaturePolicy,
};
use deepcode_core::types::{ContentBlock, MediaSource, Message, Role, ToolDefinition};

pub(crate) struct AnthropicRequestBuilder;

#[async_trait]
impl RequestBuilder for AnthropicRequestBuilder {
    fn capabilities(&self, model: &str) -> ProviderCapabilities {
        let supports_reasoning = !model.starts_with("claude-haiku-4-5");
        let fixed_sampling = uses_fixed_sampling(model);
        ProviderCapabilities {
            provider: "anthropic",
            temperature: !fixed_sampling,
            top_p: !fixed_sampling,
            stop_sequences: true,
            reasoning_effort: supports_reasoning,
            reasoning_efforts: anthropic_efforts(model),
            reasoning_can_disable: thinking_can_disable(model),
            reasoning_display: supports_reasoning,
            tool_choice: true,
            prompt_cache_key: true,
            prompt_cache_retention: true,
            image_input: true,
            file_input: true,
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
        self.capabilities(model)
            .validate_request(messages, params)?;
        let thinking_enabled = params
            .reasoning_effort
            .is_some_and(|effort| effort != ReasoningEffort::Off);
        if thinking_enabled
            && params.temperature.is_some()
            && params.unsupported_feature_policy == UnsupportedFeaturePolicy::Error
        {
            return Err(deepcode_core::error::DeepCodeError::UnsupportedFeature {
                provider: "anthropic".to_string(),
                feature: "temperature_with_adaptive_thinking".to_string(),
            });
        }
        if thinking_enabled
            && params
                .top_p
                .is_some_and(|top_p| !(0.95..=1.0).contains(&top_p))
            && params.unsupported_feature_policy == UnsupportedFeaturePolicy::Error
        {
            return Err(deepcode_core::error::DeepCodeError::UnsupportedFeature {
                provider: "anthropic".to_string(),
                feature: "top_p_with_adaptive_thinking".to_string(),
            });
        }
        if let Some(retention) = params.prompt_cache_retention.as_deref() {
            if !matches!(retention, "5m" | "1h")
                && params.unsupported_feature_policy == UnsupportedFeaturePolicy::Error
            {
                return Err(deepcode_core::error::DeepCodeError::UnsupportedFeature {
                    provider: "anthropic".to_string(),
                    feature: format!("prompt_cache_retention:{retention}"),
                });
            }
        }
        let mut body = serde_json::Map::new();
        body.insert("model".into(), model.into());
        body.insert(
            "max_tokens".into(),
            (params.max_tokens.unwrap_or(4096) as u64).into(),
        );

        // Anthropic puts system prompt at top level
        if let Some(sys) = system_prompt {
            body.insert("system".into(), sys.into());
        }

        // Convert messages to Anthropic format
        let mut anthropic_msgs: Vec<serde_json::Value> = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    // System messages go to top-level "system" field
                    if system_prompt.is_none() {
                        if let Some(ContentBlock::Text { text }) = msg.content.first() {
                            body.insert("system".into(), text.clone().into());
                        }
                    }
                }
                Role::User => {
                    let mut content = Vec::new();
                    for block in &msg.content {
                        match block {
                            ContentBlock::Text { text } => {
                                content.push(serde_json::json!({
                                    "type": "text",
                                    "text": text,
                                }));
                            }
                            ContentBlock::Image { source, .. } => {
                                content.push(anthropic_image(source));
                            }
                            ContentBlock::File { source, filename } => {
                                content.push(anthropic_file(source, filename.as_deref()))
                            }
                            ContentBlock::ToolResult {
                                tool_use_id,
                                content: result_text,
                                is_error,
                            } => {
                                content.push(serde_json::json!({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": result_text,
                                    "is_error": is_error,
                                }));
                            }
                            _ => {}
                        }
                    }
                    anthropic_msgs.push(serde_json::json!({
                        "role": "user",
                        "content": content,
                    }));
                }
                Role::Assistant => {
                    let mut content = Vec::new();
                    for block in &msg.content {
                        match block {
                            ContentBlock::Reasoning { text, metadata } => {
                                if let Some(block) =
                                    metadata.as_ref().and_then(|value| value.get("block"))
                                {
                                    let mut block = block.clone();
                                    if block["type"] == "thinking" {
                                        block["thinking"] = text.clone().into();
                                        if let Some(signature) = metadata
                                            .as_ref()
                                            .and_then(|value| value["signature"].as_str())
                                        {
                                            block["signature"] = signature.into();
                                        }
                                    }
                                    content.push(block);
                                } else if let Some(signature) = metadata
                                    .as_ref()
                                    .and_then(|value| value["signature"].as_str())
                                {
                                    content.push(serde_json::json!({
                                        "type": "thinking",
                                        "thinking": text,
                                        "signature": signature,
                                    }));
                                }
                            }
                            ContentBlock::ProviderItem { provider, value }
                                if provider == "anthropic" =>
                            {
                                content.push(value.clone());
                            }
                            ContentBlock::Text { text } => {
                                content.push(serde_json::json!({
                                    "type": "text",
                                    "text": text,
                                }));
                            }
                            ContentBlock::ToolUse { id, name, input } => {
                                content.push(serde_json::json!({
                                    "type": "tool_use",
                                    "id": id,
                                    "name": name,
                                    "input": input,
                                }));
                            }
                            _ => {}
                        }
                    }
                    anthropic_msgs.push(serde_json::json!({
                        "role": "assistant",
                        "content": content,
                    }));
                }
                Role::Tool => {
                    let mut content = Vec::new();
                    for block in &msg.content {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            content: result_text,
                            is_error,
                        } = block
                        {
                            content.push(serde_json::json!({
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": result_text,
                                "is_error": is_error,
                            }));
                        }
                    }
                    if !content.is_empty() {
                        anthropic_msgs.push(serde_json::json!({
                            "role": "user",
                            "content": content,
                        }));
                    }
                }
            }
        }

        body.insert("messages".into(), anthropic_msgs.into());

        // Convert tools to Anthropic format
        if !tools.is_empty() {
            let anthropic_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();
            body.insert("tools".into(), anthropic_tools.into());
            if let Some(choice) = params.tool_choice.as_ref() {
                body.insert("tool_choice".into(), anthropic_tool_choice(choice));
            }
        }

        // Params
        if let Some(effort) = params
            .reasoning_effort
            .filter(|effort| *effort != ReasoningEffort::Off)
        {
            let mapped = match effort {
                ReasoningEffort::Minimal => ReasoningEffort::Low,
                value => value,
            };
            body.insert(
                "output_config".into(),
                serde_json::json!({"effort": mapped.as_str()}),
            );
            let mut thinking = serde_json::json!({"type": "adaptive"});
            if let Some(display) = params.reasoning_display {
                thinking["display"] = display.as_str().into();
            }
            body.insert("thinking".into(), thinking);
        } else {
            if params.reasoning_effort == Some(ReasoningEffort::Off)
                && needs_explicit_thinking_disable(model)
            {
                body.insert("thinking".into(), serde_json::json!({"type": "disabled"}));
            }
            if !uses_fixed_sampling(model) {
                if let Some(temp) = params.temperature {
                    body.insert("temperature".into(), temp.into());
                }
            }
        }
        if !uses_fixed_sampling(model) {
            if let Some(top_p) = params
                .top_p
                .filter(|top_p| !thinking_enabled || (0.95..=1.0).contains(top_p))
            {
                body.insert("top_p".into(), top_p.into());
            }
        }
        if !params.stop_sequences.is_empty() {
            body.insert(
                "stop_sequences".into(),
                params.stop_sequences.clone().into(),
            );
        }
        if params.prompt_cache_key.is_some() || params.prompt_cache_retention.is_some() {
            // Anthropic automatic prompt caching has no client-provided key. The
            // shared key field acts as the provider-neutral opt-in signal.
            let mut cache_control = serde_json::json!({"type": "ephemeral"});
            if let Some(retention @ ("5m" | "1h")) = params.prompt_cache_retention.as_deref() {
                cache_control["ttl"] = retention.into();
            }
            body.insert("cache_control".into(), cache_control);
        }
        if let Some(options) = params
            .provider_options
            .get("anthropic")
            .and_then(|value| value.as_object())
        {
            for (key, value) in options {
                body.insert(key.clone(), value.clone());
            }
        }

        Ok(serde_json::Value::Object(body))
    }
}

fn anthropic_efforts(model: &str) -> &'static [ReasoningEffort] {
    if model.starts_with("claude-sonnet-4-6") || model.starts_with("claude-opus-4-6") {
        &[
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::Max,
        ]
    } else if model.starts_with("claude-haiku-4-5") {
        &[]
    } else {
        &[
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
            ReasoningEffort::Xhigh,
            ReasoningEffort::Max,
        ]
    }
}

fn uses_fixed_sampling(model: &str) -> bool {
    [
        "claude-fable-5",
        "claude-mythos-5",
        "claude-mythos-preview",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5",
    ]
    .iter()
    .any(|prefix| model.starts_with(prefix))
}

fn thinking_can_disable(model: &str) -> bool {
    needs_explicit_thinking_disable(model)
        || [
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        ]
        .iter()
        .any(|prefix| model.starts_with(prefix))
}

fn needs_explicit_thinking_disable(model: &str) -> bool {
    model.starts_with("claude-opus-5") || model.starts_with("claude-sonnet-5")
}

fn anthropic_tool_choice(choice: &ToolChoice) -> serde_json::Value {
    match choice {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::None => serde_json::json!({"type": "none"}),
        ToolChoice::Required => serde_json::json!({"type": "any"}),
        ToolChoice::Function(name) => serde_json::json!({"type": "tool", "name": name}),
    }
}

fn anthropic_image(source: &MediaSource) -> serde_json::Value {
    match source {
        MediaSource::Url { url } => serde_json::json!({
            "type": "image",
            "source": {"type": "url", "url": url},
        }),
        MediaSource::Base64 { media_type, data } => serde_json::json!({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }),
        MediaSource::FileId { file_id } => serde_json::json!({
            "type": "image",
            "source": {"type": "file", "file_id": file_id},
        }),
    }
}

fn anthropic_file(source: &MediaSource, filename: Option<&str>) -> serde_json::Value {
    let mut document = match source {
        MediaSource::Url { url } => serde_json::json!({
            "type": "document",
            "source": {"type": "url", "url": url},
        }),
        MediaSource::Base64 { media_type, data } => serde_json::json!({
            "type": "document",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }),
        MediaSource::FileId { file_id } => serde_json::json!({
            "type": "document",
            "source": {"type": "file", "file_id": file_id},
        }),
    };
    if let Some(filename) = filename {
        document["title"] = filename.into();
    }
    document
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepcode_core::types::{ContentBlock, Message, Role};

    #[test]
    fn tool_role_serializes_as_user_tool_result() {
        let builder = AnthropicRequestBuilder;
        let messages = vec![Message::tool_result("tool_1", "done", false)];

        let body = builder
            .build_request(
                "claude-test",
                &messages,
                &[],
                None,
                &GenerateParams::default(),
            )
            .unwrap();

        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "user");
        assert_eq!(msg["content"][0]["type"], "tool_result");
        assert_eq!(msg["content"][0]["tool_use_id"], "tool_1");
        assert_eq!(msg["content"][0]["content"], "done");
    }

    #[test]
    fn multiple_tool_results_serialize_in_one_user_message() {
        let builder = AnthropicRequestBuilder;
        let messages = vec![Message {
            role: Role::Tool,
            content: vec![
                ContentBlock::tool_result("tool_1", "one", false),
                ContentBlock::tool_result("tool_2", "two", true),
            ],
            id: None,
        }];

        let body = builder
            .build_request(
                "claude-test",
                &messages,
                &[],
                None,
                &GenerateParams::default(),
            )
            .unwrap();

        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "user");
        assert_eq!(msg["content"].as_array().unwrap().len(), 2);
        assert_eq!(msg["content"][0]["tool_use_id"], "tool_1");
        assert_eq!(msg["content"][0]["content"], "one");
        assert_eq!(msg["content"][0]["is_error"], false);
        assert_eq!(msg["content"][1]["tool_use_id"], "tool_2");
        assert_eq!(msg["content"][1]["content"], "two");
        assert_eq!(msg["content"][1]["is_error"], true);
    }

    #[test]
    fn maps_reasoning_effort_to_output_config() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Xhigh),
            ..GenerateParams::default()
        };
        let body = AnthropicRequestBuilder
            .build_request("claude-test", &[Message::user("hi")], &[], None, &params)
            .unwrap();
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["output_config"]["effort"], "xhigh");
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn serializes_stop_sequences_and_thinking_display() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::High),
            reasoning_display: Some(deepcode_core::provider::traits::ReasoningDisplay::Summarized),
            stop_sequences: vec!["DONE".to_string()],
            ..GenerateParams::default()
        };
        let body = AnthropicRequestBuilder
            .build_request(
                "claude-sonnet-4-6",
                &[Message::user("hi")],
                &[],
                None,
                &params,
            )
            .unwrap();
        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "summarized");
        assert_eq!(body["stop_sequences"][0], "DONE");
    }

    #[test]
    fn assistant_reasoning_serializes_as_signed_thinking_block() {
        let body = AnthropicRequestBuilder
            .build_request(
                "claude-test",
                &[Message::assistant(vec![
                    ContentBlock::reasoning_with_metadata(
                        "hidden",
                        serde_json::json!({"signature": "sig"}),
                    ),
                    ContentBlock::text("answer"),
                ])],
                &[],
                None,
                &GenerateParams::default(),
            )
            .unwrap();

        let content = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["thinking"], "hidden");
        assert_eq!(content[0]["signature"], "sig");
        assert_eq!(content[1]["type"], "text");
    }

    #[test]
    fn streaming_thinking_metadata_is_completed_before_replay() {
        let body = AnthropicRequestBuilder
            .build_request(
                "claude-sonnet-4-6",
                &[Message::assistant(vec![
                    ContentBlock::reasoning_with_metadata(
                        "complete thinking",
                        serde_json::json!({
                            "provider": "anthropic",
                            "signature": "sig",
                            "block": {"type": "thinking", "thinking": ""}
                        }),
                    ),
                    ContentBlock::text("answer"),
                ])],
                &[],
                None,
                &GenerateParams::default(),
            )
            .unwrap();
        let thinking = &body["messages"][0]["content"][0];
        assert_eq!(thinking["thinking"], "complete thinking");
        assert_eq!(thinking["signature"], "sig");
    }

    #[test]
    fn redacted_thinking_block_is_replayed_unchanged() {
        let redacted = serde_json::json!({
            "type": "redacted_thinking",
            "data": "opaque"
        });
        let body = AnthropicRequestBuilder
            .build_request(
                "claude-sonnet-4-6",
                &[Message::assistant(vec![
                    ContentBlock::reasoning_with_metadata(
                        "",
                        serde_json::json!({
                            "provider": "anthropic",
                            "block": redacted.clone()
                        }),
                    ),
                    ContentBlock::text("answer"),
                ])],
                &[],
                None,
                &GenerateParams::default(),
            )
            .unwrap();

        assert_eq!(body["messages"][0]["content"][0], redacted);
    }

    #[test]
    fn fixed_sampling_models_reject_sampling_parameters() {
        let capabilities = AnthropicRequestBuilder.capabilities("claude-opus-4-8");
        assert!(!capabilities.temperature);
        assert!(!capabilities.top_p);

        let params = GenerateParams {
            top_p: Some(0.9),
            ..GenerateParams::default()
        };
        assert!(AnthropicRequestBuilder
            .build_request(
                "claude-opus-4-8",
                &[Message::user("hi")],
                &[],
                None,
                &params,
            )
            .is_err());
    }

    #[test]
    fn fallback_drops_sampling_parameters_on_fixed_sampling_models() {
        let params = GenerateParams {
            temperature: Some(0.4),
            top_p: Some(0.8),
            unsupported_feature_policy: UnsupportedFeaturePolicy::AllowFallback,
            ..GenerateParams::default()
        };
        let body = AnthropicRequestBuilder
            .build_request(
                "claude-opus-4-8",
                &[Message::user("hi")],
                &[],
                None,
                &params,
            )
            .unwrap();
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
    }

    #[test]
    fn adaptive_thinking_only_accepts_supported_top_p_range() {
        let accepted = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::High),
            top_p: Some(0.95),
            ..GenerateParams::default()
        };
        let body = AnthropicRequestBuilder
            .build_request(
                "claude-sonnet-4-6",
                &[Message::user("hi")],
                &[],
                None,
                &accepted,
            )
            .unwrap();
        assert!(
            (body["top_p"].as_f64().unwrap() - 0.95).abs() < 1e-6,
            "serialized top_p should retain the configured value"
        );

        let rejected = GenerateParams {
            top_p: Some(0.9),
            ..accepted
        };
        assert!(AnthropicRequestBuilder
            .build_request(
                "claude-sonnet-4-6",
                &[Message::user("hi")],
                &[],
                None,
                &rejected,
            )
            .is_err());
    }

    #[test]
    fn opus_4_6_does_not_advertise_xhigh_effort() {
        let capabilities = AnthropicRequestBuilder.capabilities("claude-opus-4-6");
        assert!(!capabilities
            .reasoning_efforts
            .contains(&ReasoningEffort::Xhigh));
        assert!(capabilities
            .reasoning_efforts
            .contains(&ReasoningEffort::Max));
    }

    #[test]
    fn prompt_cache_opt_in_uses_anthropic_cache_control() {
        let params = GenerateParams {
            prompt_cache_key: Some("conversation-1".to_string()),
            prompt_cache_retention: Some("1h".to_string()),
            ..GenerateParams::default()
        };
        let body = AnthropicRequestBuilder
            .build_request(
                "claude-sonnet-4-6",
                &[Message::user("hi")],
                &[],
                None,
                &params,
            )
            .unwrap();
        assert_eq!(body["cache_control"]["type"], "ephemeral");
        assert_eq!(body["cache_control"]["ttl"], "1h");
        assert!(body.get("prompt_cache_key").is_none());
    }

    #[test]
    fn reasoning_off_uses_the_models_supported_disable_shape() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Off),
            ..GenerateParams::default()
        };
        let opus_5 = AnthropicRequestBuilder
            .build_request("claude-opus-5", &[Message::user("hi")], &[], None, &params)
            .unwrap();
        assert_eq!(opus_5["thinking"]["type"], "disabled");

        let opus_4_8 = AnthropicRequestBuilder
            .build_request(
                "claude-opus-4-8",
                &[Message::user("hi")],
                &[],
                None,
                &params,
            )
            .unwrap();
        assert!(opus_4_8.get("thinking").is_none());
    }
}
