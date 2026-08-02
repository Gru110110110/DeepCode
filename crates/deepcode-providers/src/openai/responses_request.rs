use async_trait::async_trait;
use deepcode_core::config::ReasoningEffort;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::{
    GenerateParams, ProviderCapabilities, RequestBuilder, ResponseFormat, ToolChoice,
};
use deepcode_core::types::{
    strict_json_schema, ContentBlock, ImageDetail, MediaSource, Message, Role, ToolDefinition,
};

use crate::openai_compat::merge_provider_options;

#[derive(Clone, Copy)]
pub(crate) struct ResponsesRequestBuilder;

#[async_trait]
impl RequestBuilder for ResponsesRequestBuilder {
    fn capabilities(&self, model: &str) -> ProviderCapabilities {
        let fixed_sampling = model.starts_with("gpt-5");
        ProviderCapabilities {
            provider: "openai",
            temperature: !fixed_sampling,
            top_p: !fixed_sampling,
            reasoning_effort: fixed_sampling,
            reasoning_efforts: if fixed_sampling {
                &[
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ]
            } else {
                &[]
            },
            reasoning_can_disable: fixed_sampling,
            reasoning_summary: fixed_sampling,
            reasoning_mode: fixed_sampling,
            reasoning_context: fixed_sampling,
            response_format: true,
            verbosity: true,
            tool_choice: true,
            parallel_tool_calls: true,
            strict_tools: true,
            prompt_cache_key: true,
            prompt_cache_retention: true,
            safety_identifier: true,
            metadata: true,
            store: true,
            previous_response_id: true,
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
        build_responses_request(
            model,
            messages,
            tools,
            system_prompt,
            params,
            ResponsesOptions {
                capabilities: self.capabilities(model),
                include_encrypted_reasoning: true,
                default_store: Some(false),
            },
        )
    }
}

#[derive(Clone, Copy)]
pub(crate) struct ResponsesOptions {
    pub capabilities: ProviderCapabilities,
    pub include_encrypted_reasoning: bool,
    pub default_store: Option<bool>,
}

pub(crate) fn build_responses_request(
    model: &str,
    messages: &[Message],
    tools: &[ToolDefinition],
    system_prompt: Option<&str>,
    params: &GenerateParams,
    options: ResponsesOptions,
) -> Result<serde_json::Value> {
    options.capabilities.validate_request(messages, params)?;
    let mut input = Vec::new();
    if let Some(system) = system_prompt {
        input.push(message_item("system", "input_text", system));
    }
    for message in messages {
        match message.role {
            Role::System | Role::User => {
                let role = if message.role == Role::System {
                    "system"
                } else {
                    "user"
                };
                let mut content = Vec::new();
                for block in &message.content {
                    match block {
                        ContentBlock::Text { text } => content.push(serde_json::json!({
                            "type": "input_text",
                            "text": text,
                        })),
                        ContentBlock::Image { source, detail } => {
                            content.push(responses_image(source, *detail));
                        }
                        ContentBlock::File { source, filename } => {
                            content.push(responses_file(source, filename.as_deref()))
                        }
                        _ => {}
                    }
                }
                if !content.is_empty() {
                    input.push(serde_json::json!({
                        "type": "message",
                        "role": role,
                        "content": content,
                    }));
                }
            }
            Role::Assistant => {
                let provider_items = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ProviderItem { provider, value }
                            if provider == options.capabilities.provider =>
                        {
                            Some(value.clone())
                        }
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                if !provider_items.is_empty() {
                    input.extend(provider_items);
                    continue;
                }
                for block in &message.content {
                    match block {
                        ContentBlock::Reasoning { metadata, .. } => {
                            if let Some(item) =
                                metadata.as_ref().and_then(|value| value.get("item"))
                            {
                                input.push(item.clone());
                            }
                        }
                        ContentBlock::Text { text } => {
                            input.push(message_item("assistant", "output_text", text));
                        }
                        ContentBlock::ToolUse {
                            id,
                            name,
                            input: arguments,
                        } => {
                            input.push(serde_json::json!({
                                "type": "function_call",
                                "call_id": id,
                                "name": name,
                                "arguments": serde_json::to_string(arguments).unwrap_or_default(),
                            }));
                        }
                        _ => {}
                    }
                }
            }
            Role::Tool => {
                for block in &message.content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } = block
                    {
                        input.push(serde_json::json!({
                            "type": "function_call_output",
                            "call_id": tool_use_id,
                            "output": content,
                        }));
                    }
                }
            }
        }
    }

    let mut body = serde_json::Map::new();
    body.insert("model".into(), model.into());
    body.insert("input".into(), input.into());
    if let Some(store) = params.store.or(options.default_store) {
        body.insert("store".into(), store.into());
    }
    if options.include_encrypted_reasoning {
        body.insert(
            "include".into(),
            serde_json::json!(["reasoning.encrypted_content"]),
        );
    }
    if !tools.is_empty() {
        body.insert(
            "tools".into(),
            tools
                .iter()
                .map(|tool| {
                    serde_json::json!({
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": if params.strict_tools == Some(true) {
                            strict_json_schema(&tool.input_schema)
                        } else {
                            tool.input_schema.clone()
                        },
                        "strict": params.strict_tools == Some(true),
                    })
                })
                .collect::<Vec<_>>()
                .into(),
        );
        body.insert(
            "tool_choice".into(),
            responses_tool_choice(params.tool_choice.as_ref()),
        );
    }
    if let Some(tokens) = params.max_tokens {
        body.insert("max_output_tokens".into(), (tokens as u64).into());
    }
    if (options.capabilities.reasoning_effort && params.reasoning_effort.is_some())
        || params.reasoning_summary.is_some()
        || params.reasoning_mode.is_some()
        || params.reasoning_context.is_some()
    {
        let mut reasoning = serde_json::Map::new();
        if let Some(effort) = params.reasoning_effort {
            reasoning.insert(
                "effort".into(),
                if effort == ReasoningEffort::Off {
                    "none".into()
                } else {
                    effort.as_str().into()
                },
            );
            if effort != ReasoningEffort::Off && params.reasoning_summary.is_none() {
                reasoning.insert("summary".into(), "auto".into());
            }
        }
        if let Some(summary) = params.reasoning_summary {
            reasoning.insert("summary".into(), summary.as_str().into());
        }
        if let Some(mode) = params.reasoning_mode {
            reasoning.insert("mode".into(), mode.as_str().into());
        }
        if let Some(context) = params.reasoning_context {
            reasoning.insert("context".into(), context.as_str().into());
        }
        body.insert("reasoning".into(), reasoning.into());
    }
    if options.capabilities.temperature {
        if let Some(temperature) = params.temperature {
            body.insert("temperature".into(), temperature.into());
        }
    }
    if options.capabilities.top_p {
        if let Some(top_p) = params.top_p {
            body.insert("top_p".into(), top_p.into());
        }
    }
    if let Some(parallel) = params.parallel_tool_calls {
        body.insert("parallel_tool_calls".into(), parallel.into());
    }
    if params.response_format.is_some() || params.verbosity.is_some() {
        let mut text = serde_json::Map::new();
        if let Some(format) = params.response_format.as_ref() {
            text.insert("format".into(), responses_format(format));
        }
        if let Some(verbosity) = params.verbosity {
            text.insert("verbosity".into(), verbosity.as_str().into());
        }
        body.insert("text".into(), text.into());
    }
    if let Some(key) = params.prompt_cache_key.as_ref() {
        body.insert("prompt_cache_key".into(), key.clone().into());
    }
    if let Some(retention) = params.prompt_cache_retention.as_ref() {
        body.insert("prompt_cache_retention".into(), retention.clone().into());
    }
    if let Some(identifier) = params.safety_identifier.as_ref() {
        body.insert("safety_identifier".into(), identifier.clone().into());
    }
    if !params.metadata.is_empty() {
        body.insert("metadata".into(), serde_json::json!(params.metadata));
    }
    if let Some(previous) = params.previous_response_id.as_ref() {
        body.insert("previous_response_id".into(), previous.clone().into());
    }
    merge_provider_options(&mut body, options.capabilities.provider, params);
    Ok(serde_json::Value::Object(body))
}

fn message_item(role: &str, content_type: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "message",
        "role": role,
        "content": [{"type": content_type, "text": text}],
    })
}

fn responses_tool_choice(choice: Option<&ToolChoice>) -> serde_json::Value {
    match choice {
        None | Some(ToolChoice::Auto) => "auto".into(),
        Some(ToolChoice::None) => "none".into(),
        Some(ToolChoice::Required) => "required".into(),
        Some(ToolChoice::Function(name)) => {
            serde_json::json!({"type": "function", "name": name})
        }
    }
}

fn responses_format(format: &ResponseFormat) -> serde_json::Value {
    match format {
        ResponseFormat::Text => serde_json::json!({"type": "text"}),
        ResponseFormat::JsonObject => serde_json::json!({"type": "json_object"}),
        ResponseFormat::JsonSchema {
            name,
            description,
            schema,
            strict,
        } => serde_json::json!({
            "type": "json_schema",
            "name": name,
            "description": description,
            "schema": schema,
            "strict": strict,
        }),
    }
}

fn responses_image(source: &MediaSource, detail: Option<ImageDetail>) -> serde_json::Value {
    let mut image = serde_json::json!({"type": "input_image"});
    match source {
        MediaSource::Url { url } => image["image_url"] = url.clone().into(),
        MediaSource::Base64 { media_type, data } => {
            image["image_url"] = format!("data:{};base64,{}", media_type, data).into();
        }
        MediaSource::FileId { file_id } => image["file_id"] = file_id.clone().into(),
    }
    if let Some(detail) = detail {
        image["detail"] = match detail {
            ImageDetail::Auto => "auto",
            ImageDetail::Low => "low",
            ImageDetail::High => "high",
            ImageDetail::Original => "original",
        }
        .into();
    }
    image
}

fn responses_file(source: &MediaSource, filename: Option<&str>) -> serde_json::Value {
    let mut file = serde_json::json!({"type": "input_file"});
    match source {
        MediaSource::Url { url } => file["file_url"] = url.clone().into(),
        MediaSource::Base64 { data, .. } => file["file_data"] = data.clone().into(),
        MediaSource::FileId { file_id } => file["file_id"] = file_id.clone().into(),
    }
    if let Some(filename) = filename {
        file["filename"] = filename.into();
    }
    file
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_responses_tools_and_effort() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::High),
            ..GenerateParams::default()
        };
        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read".to_string(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let body = ResponsesRequestBuilder
            .build_request("gpt-5.6", &[Message::user("hi")], &tools, None, &params)
            .unwrap();
        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
        assert_eq!(body["tools"][0]["name"], "read_file");
        assert_eq!(body["reasoning"]["effort"], "high");
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn off_maps_to_openai_none_effort() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Off),
            ..GenerateParams::default()
        };
        let body = ResponsesRequestBuilder
            .build_request("gpt-5.6", &[Message::user("hi")], &[], None, &params)
            .unwrap();

        assert_eq!(body["reasoning"]["effort"], "none");
        assert!(body["reasoning"].get("summary").is_none());
    }

    #[test]
    fn replays_raw_output_items_and_enables_strict_tools() {
        let raw = serde_json::json!({
            "id": "rs_1",
            "type": "reasoning",
            "encrypted_content": "ciphertext"
        });
        let messages = vec![Message::assistant(vec![
            ContentBlock::provider_item("openai", raw.clone()),
            ContentBlock::reasoning("summary"),
        ])];
        let tools = vec![ToolDefinition {
            name: "read".to_string(),
            description: "Read".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}},
                "required": ["path"]
            }),
        }];
        let params = GenerateParams {
            strict_tools: Some(true),
            prompt_cache_key: Some("session-1".to_string()),
            ..GenerateParams::default()
        };
        let body = ResponsesRequestBuilder
            .build_request("gpt-5.6", &messages, &tools, None, &params)
            .unwrap();
        assert_eq!(body["input"], serde_json::json!([raw]));
        assert_eq!(body["tools"][0]["strict"], true);
        assert_eq!(
            body["tools"][0]["parameters"]["additionalProperties"],
            false
        );
        assert_eq!(
            body["tools"][0]["parameters"]["required"],
            serde_json::json!(["offset", "path"])
        );
        assert_eq!(body["prompt_cache_key"], "session-1");
    }
}
