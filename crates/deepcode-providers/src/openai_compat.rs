use deepcode_core::error::Result;
use deepcode_core::provider::traits::{
    GenerateParams, ProviderCapabilities, ResponseFormat, ToolChoice,
};
use deepcode_core::types::{
    strict_json_schema, ContentBlock, ImageDetail, MediaSource, Message, Role, ToolDefinition,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct ChatCompletionsOptions {
    pub include_reasoning_content: bool,
    pub max_tokens_field: ChatMaxTokensField,
    pub capabilities: ProviderCapabilities,
}

impl Default for ChatCompletionsOptions {
    fn default() -> Self {
        Self {
            include_reasoning_content: false,
            max_tokens_field: ChatMaxTokensField::MaxTokens,
            capabilities: ProviderCapabilities::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) enum ChatMaxTokensField {
    #[default]
    MaxTokens,
    MaxCompletionTokens,
}

impl ChatMaxTokensField {
    fn key(self) -> &'static str {
        match self {
            Self::MaxTokens => "max_tokens",
            Self::MaxCompletionTokens => "max_completion_tokens",
        }
    }
}

pub(crate) fn build_chat_completions_request(
    model: &str,
    messages: &[Message],
    tools: &[ToolDefinition],
    system_prompt: Option<&str>,
    params: &GenerateParams,
    options: ChatCompletionsOptions,
) -> Result<serde_json::Value> {
    options.capabilities.validate_request(messages, params)?;
    let mut body = serde_json::Map::new();
    body.insert("model".into(), model.into());
    body.insert(
        "messages".into(),
        build_messages(messages, system_prompt, options)?.into(),
    );

    if !tools.is_empty() {
        body.insert(
            "tools".into(),
            build_tools(tools, params.strict_tools == Some(true)).into(),
        );
        body.insert(
            "tool_choice".into(),
            chat_tool_choice(params.tool_choice.as_ref()),
        );
    }

    if options.capabilities.temperature {
        if let Some(temp) = params.temperature {
            body.insert("temperature".into(), temp.into());
        }
    }
    if let Some(max_tok) = params.max_tokens {
        body.insert(
            options.max_tokens_field.key().into(),
            (max_tok as u64).into(),
        );
    }
    if options.capabilities.top_p {
        if let Some(top_p) = params.top_p {
            body.insert("top_p".into(), top_p.into());
        }
    }
    if options.capabilities.stop_sequences && !params.stop_sequences.is_empty() {
        body.insert("stop".into(), params.stop_sequences.clone().into());
    }
    if let Some(effort) = params.reasoning_effort {
        if effort != deepcode_core::config::ReasoningEffort::Off {
            body.insert("reasoning_effort".into(), effort.as_str().into());
        }
    }
    if options.capabilities.parallel_tool_calls {
        if let Some(parallel) = params.parallel_tool_calls {
            body.insert("parallel_tool_calls".into(), parallel.into());
        }
    }
    if options.capabilities.response_format {
        if let Some(format) = params.response_format.as_ref() {
            body.insert("response_format".into(), chat_response_format(format));
        }
    }
    if options.capabilities.verbosity {
        if let Some(verbosity) = params.verbosity {
            body.insert("verbosity".into(), verbosity.as_str().into());
        }
    }
    if options.capabilities.prediction {
        if let Some(content) = params.prediction.as_ref() {
            body.insert(
                "prediction".into(),
                serde_json::json!({"type": "content", "content": content}),
            );
        }
    }
    if options.capabilities.seed {
        if let Some(seed) = params.seed {
            body.insert("seed".into(), seed.into());
        }
    }
    if options.capabilities.logprobs {
        if let Some(logprobs) = params.logprobs {
            body.insert("logprobs".into(), logprobs.into());
        }
        if let Some(top_logprobs) = params.top_logprobs {
            body.insert("top_logprobs".into(), top_logprobs.into());
        }
    }
    if options.capabilities.prompt_cache_key {
        if let Some(key) = params.prompt_cache_key.as_ref() {
            body.insert("prompt_cache_key".into(), key.clone().into());
        }
    }
    if options.capabilities.safety_identifier {
        if let Some(identifier) = params.safety_identifier.as_ref() {
            body.insert("safety_identifier".into(), identifier.clone().into());
        }
    }
    if options.capabilities.store {
        if let Some(store) = params.store {
            body.insert("store".into(), store.into());
        }
    }
    merge_provider_options(&mut body, options.capabilities.provider, params);

    body.insert("stream".into(), false.into());
    Ok(serde_json::Value::Object(body))
}

fn build_messages(
    messages: &[Message],
    system_prompt: Option<&str>,
    options: ChatCompletionsOptions,
) -> Result<Vec<serde_json::Value>> {
    let mut msgs = Vec::new();

    if let Some(sys) = system_prompt {
        msgs.push(serde_json::json!({
            "role": "system",
            "content": sys,
        }));
    }

    for msg in messages {
        if msg.role == Role::Tool {
            push_tool_results(&mut msgs, msg);
        } else if msg.role == Role::Assistant {
            msgs.push(assistant_message(msg, options));
        } else {
            msgs.push(text_message(msg)?);
        }
    }

    Ok(msgs)
}

fn push_tool_results(msgs: &mut Vec<serde_json::Value>, msg: &Message) {
    for block in &msg.content {
        if let ContentBlock::ToolResult {
            tool_use_id,
            content,
            ..
        } = block
        {
            msgs.push(serde_json::json!({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": content,
            }));
        }
    }
}

fn assistant_message(msg: &Message, options: ChatCompletionsOptions) -> serde_json::Value {
    let mut text_parts = Vec::new();
    let mut reasoning_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &msg.content {
        match block {
            ContentBlock::Text { text } => text_parts.push(text.clone()),
            ContentBlock::Reasoning { text, .. } if options.include_reasoning_content => {
                reasoning_parts.push(text.clone());
            }
            ContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": serde_json::to_string(input).unwrap_or_default(),
                    },
                }));
            }
            _ => {}
        }
    }

    let content = text_parts.join("\n");
    let reasoning_content = reasoning_parts.join("\n");
    let mut assistant_msg = serde_json::Map::new();
    assistant_msg.insert("role".into(), "assistant".into());
    if content.is_empty() && !tool_calls.is_empty() {
        assistant_msg.insert("content".into(), serde_json::Value::Null);
    } else {
        assistant_msg.insert("content".into(), content.into());
    }
    if !reasoning_content.is_empty() {
        assistant_msg.insert("reasoning_content".into(), reasoning_content.into());
    }
    if !tool_calls.is_empty() {
        assistant_msg.insert("tool_calls".into(), tool_calls.into());
    }

    serde_json::Value::Object(assistant_msg)
}

fn text_message(msg: &Message) -> Result<serde_json::Value> {
    if msg.role == Role::User && msg.content.len() == 1 {
        if let Some(ContentBlock::Text { text }) = msg.content.first() {
            return Ok(serde_json::json!({
                "role": "user",
                "content": text,
            }));
        }
    }

    let content_parts: Vec<serde_json::Value> = msg
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => Ok(Some(serde_json::json!({
                "type": "text",
                "text": text,
            }))),
            ContentBlock::Image { source, detail } => Ok(Some(chat_image(source, *detail)?)),
            ContentBlock::Audio { source, format } => {
                Ok(Some(chat_audio(source, format.as_deref())?))
            }
            _ => Ok(None),
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    let role = role_name(&msg.role);

    if content_parts.is_empty() {
        Ok(serde_json::json!({
            "role": role,
            "content": "",
        }))
    } else {
        Ok(serde_json::json!({
            "role": role,
            "content": content_parts,
        }))
    }
}

fn chat_image(source: &MediaSource, detail: Option<ImageDetail>) -> Result<serde_json::Value> {
    let url = match source {
        MediaSource::Url { url } => url.clone(),
        MediaSource::Base64 { media_type, data } => {
            format!("data:{};base64,{}", media_type, data)
        }
        MediaSource::FileId { .. } => {
            return Err(deepcode_core::error::DeepCodeError::UnsupportedFeature {
                provider: "openai_compatible".to_string(),
                feature: "chat_image_file_id".to_string(),
            });
        }
    };
    let detail = match detail.unwrap_or(ImageDetail::Auto) {
        ImageDetail::Auto => "auto",
        ImageDetail::Low => "low",
        ImageDetail::High => "high",
        ImageDetail::Original => "original",
    };
    Ok(serde_json::json!({
        "type": "image_url",
        "image_url": {"url": url, "detail": detail},
    }))
}

fn chat_audio(source: &MediaSource, format: Option<&str>) -> Result<serde_json::Value> {
    let MediaSource::Base64 { data, .. } = source else {
        return Err(deepcode_core::error::DeepCodeError::UnsupportedFeature {
            provider: "openai_compatible".to_string(),
            feature: "chat_audio_non_base64".to_string(),
        });
    };
    Ok(serde_json::json!({
        "type": "input_audio",
        "input_audio": {"data": data, "format": format.unwrap_or("wav")},
    }))
}

fn build_tools(tools: &[ToolDefinition], strict: bool) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            let parameters = if strict {
                strict_json_schema(&tool.input_schema)
            } else {
                tool.input_schema.clone()
            };
            let mut function = serde_json::json!({
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            });
            if strict {
                function["strict"] = true.into();
            }
            serde_json::json!({
                "type": "function",
                "function": function,
            })
        })
        .collect()
}

fn chat_tool_choice(choice: Option<&ToolChoice>) -> serde_json::Value {
    match choice {
        None | Some(ToolChoice::Auto) => "auto".into(),
        Some(ToolChoice::None) => "none".into(),
        Some(ToolChoice::Required) => "required".into(),
        Some(ToolChoice::Function(name)) => serde_json::json!({
            "type": "function",
            "function": {"name": name},
        }),
    }
}

fn chat_response_format(format: &ResponseFormat) -> serde_json::Value {
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
            "json_schema": {
                "name": name,
                "description": description,
                "schema": schema,
                "strict": strict,
            }
        }),
    }
}

pub(crate) fn merge_provider_options(
    body: &mut serde_json::Map<String, serde_json::Value>,
    provider: &str,
    params: &GenerateParams,
) {
    if let Some(options) = params
        .provider_options
        .get(provider)
        .and_then(|value| value.as_object())
    {
        for (key, value) in options {
            body.insert(key.clone(), value.clone());
        }
    }
}

fn role_name(role: &Role) -> &'static str {
    match role {
        Role::System => "system",
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

#[cfg(test)]
mod parameter_tests {
    use super::*;

    #[test]
    fn serializes_reasoning_effort() {
        let params = GenerateParams {
            reasoning_effort: Some(deepcode_core::config::ReasoningEffort::High),
            ..GenerateParams::default()
        };
        let body = build_chat_completions_request(
            "reasoning-model",
            &[Message::user("hello")],
            &[],
            None,
            &params,
            ChatCompletionsOptions {
                capabilities: ProviderCapabilities {
                    provider: "openai",
                    reasoning_effort: true,
                    ..ProviderCapabilities::default()
                },
                ..ChatCompletionsOptions::default()
            },
        )
        .unwrap();
        assert_eq!(body["reasoning_effort"], "high");
    }
}
