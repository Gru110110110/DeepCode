use async_trait::async_trait;
use deepcode_core::config::ReasoningEffort;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::{
    GenerateParams, ProviderCapabilities, RequestBuilder, ResponseFormat,
};
use deepcode_core::types::{ContentBlock, Message, Role, ToolDefinition};
use std::collections::HashMap;

pub(crate) struct OllamaRequestBuilder;

#[async_trait]
impl RequestBuilder for OllamaRequestBuilder {
    fn capabilities(&self, _model: &str) -> ProviderCapabilities {
        ProviderCapabilities {
            provider: "ollama",
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
            seed: true,
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
        self.capabilities(model)
            .validate_request(messages, params)?;
        let mut body = serde_json::Map::new();
        body.insert("model".into(), model.into());
        body.insert(
            "messages".into(),
            build_messages(messages, system_prompt).into(),
        );

        if !tools.is_empty() {
            body.insert("tools".into(), build_tools(tools).into());
        }

        if let Some(effort) = params.reasoning_effort {
            body.insert("think".into(), ollama_think(effort));
        }

        let options = build_options(params);
        if !options.is_empty() {
            body.insert("options".into(), serde_json::Value::Object(options));
        }

        if let Some(format) = params.response_format.as_ref() {
            body.insert(
                "format".into(),
                match format {
                    ResponseFormat::Text => serde_json::Value::Null,
                    ResponseFormat::JsonObject => "json".into(),
                    ResponseFormat::JsonSchema { schema, .. } => schema.clone(),
                },
            );
        }
        if let Some(provider_options) = params
            .provider_options
            .get("ollama")
            .and_then(|value| value.as_object())
        {
            for (key, value) in provider_options {
                body.insert(key.clone(), value.clone());
            }
        }

        body.insert("stream".into(), false.into());
        Ok(serde_json::Value::Object(body))
    }
}

fn build_messages(messages: &[Message], system_prompt: Option<&str>) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    let mut tool_names_by_id = HashMap::new();

    if let Some(system) = system_prompt {
        out.push(message("system", system));
    }

    for msg in messages {
        match msg.role {
            Role::System => {
                for text in text_blocks(&msg.content) {
                    out.push(message("system", &text));
                }
            }
            Role::User => {
                out.push(message("user", &text_blocks(&msg.content).join("\n")));
            }
            Role::Assistant => {
                let mut object = serde_json::Map::new();
                object.insert("role".into(), "assistant".into());
                object.insert(
                    "content".into(),
                    text_blocks(&msg.content).join("\n").into(),
                );

                let thinking = reasoning_blocks(&msg.content).join("\n");
                if !thinking.is_empty() {
                    object.insert("thinking".into(), thinking.into());
                }

                let tool_calls = msg
                    .content
                    .iter()
                    .filter_map(|block| {
                        if let ContentBlock::ToolUse { id, name, input } = block {
                            tool_names_by_id.insert(id.clone(), name.clone());
                            Some(serde_json::json!({
                                "function": {
                                    "name": name,
                                    "arguments": input,
                                }
                            }))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>();

                if !tool_calls.is_empty() {
                    object.insert("tool_calls".into(), tool_calls.into());
                }

                out.push(serde_json::Value::Object(object));
            }
            Role::Tool => {
                for block in &msg.content {
                    if let ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } = block
                    {
                        let mut object = serde_json::Map::new();
                        object.insert("role".into(), "tool".into());
                        object.insert("content".into(), content.clone().into());
                        object.insert(
                            "tool_name".into(),
                            tool_names_by_id
                                .get(tool_use_id)
                                .cloned()
                                .unwrap_or_else(|| tool_use_id.clone())
                                .into(),
                        );
                        out.push(serde_json::Value::Object(object));
                    }
                }
            }
        }
    }

    out
}

fn message(role: &str, content: &str) -> serde_json::Value {
    serde_json::json!({
        "role": role,
        "content": content,
    })
}

fn text_blocks(blocks: &[ContentBlock]) -> Vec<String> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn reasoning_blocks(blocks: &[ContentBlock]) -> Vec<String> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Reasoning { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn build_tools(tools: &[ToolDefinition]) -> Vec<serde_json::Value> {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        })
        .collect()
}

fn build_options(params: &GenerateParams) -> serde_json::Map<String, serde_json::Value> {
    let mut options = serde_json::Map::new();
    if let Some(temperature) = params.temperature {
        options.insert("temperature".into(), temperature.into());
    }
    if let Some(top_p) = params.top_p {
        options.insert("top_p".into(), top_p.into());
    }
    if let Some(max_tokens) = params.max_tokens {
        options.insert("num_predict".into(), (max_tokens as u64).into());
    }
    if !params.stop_sequences.is_empty() {
        options.insert("stop".into(), params.stop_sequences.clone().into());
    }
    if let Some(seed) = params.seed {
        options.insert("seed".into(), seed.into());
    }
    options
}

fn ollama_think(effort: ReasoningEffort) -> serde_json::Value {
    match effort {
        ReasoningEffort::Off => false.into(),
        ReasoningEffort::Minimal | ReasoningEffort::Low => "low".into(),
        ReasoningEffort::Medium => "medium".into(),
        ReasoningEffort::High => "high".into(),
        ReasoningEffort::Xhigh | ReasoningEffort::Max => "max".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepcode_core::types::{ContentBlock, Message, ToolDefinition};

    #[test]
    fn builds_native_chat_options_and_think() {
        let params = GenerateParams {
            max_tokens: Some(2048),
            top_p: Some(0.9),
            reasoning_effort: Some(ReasoningEffort::High),
            stop_sequences: vec!["</done>".to_string()],
            ..GenerateParams::default()
        };
        let body = OllamaRequestBuilder
            .build_request("gpt-oss", &[Message::user("hi")], &[], Some("sys"), &params)
            .unwrap();

        assert_eq!(body["model"], "gpt-oss");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][1]["role"], "user");
        assert_eq!(body["think"], "high");
        assert_eq!(body["options"]["num_predict"], 2048);
        assert!((body["options"]["top_p"].as_f64().unwrap() - 0.9).abs() < 0.0001);
        assert_eq!(body["options"]["stop"][0], "</done>");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn off_disables_native_thinking() {
        let params = GenerateParams {
            reasoning_effort: Some(ReasoningEffort::Off),
            ..GenerateParams::default()
        };
        let body = OllamaRequestBuilder
            .build_request("model", &[Message::user("hi")], &[], None, &params)
            .unwrap();

        assert_eq!(body["think"], false);
    }

    #[test]
    fn assistant_tool_use_and_result_use_ollama_shape() {
        let messages = vec![
            Message::assistant(vec![
                ContentBlock::reasoning("thinking"),
                ContentBlock::tool_use("call_1", "read_file", serde_json::json!({"path":"x"})),
            ]),
            Message::tool_result("call_1", "contents", false),
        ];
        let tools = vec![ToolDefinition {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            input_schema: serde_json::json!({"type":"object"}),
        }];
        let body = OllamaRequestBuilder
            .build_request("model", &messages, &tools, None, &GenerateParams::default())
            .unwrap();

        assert_eq!(body["messages"][0]["role"], "assistant");
        assert_eq!(body["messages"][0]["thinking"], "thinking");
        assert_eq!(
            body["messages"][0]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        assert_eq!(
            body["messages"][0]["tool_calls"][0]["function"]["arguments"]["path"],
            "x"
        );
        assert_eq!(body["messages"][1]["role"], "tool");
        assert_eq!(body["messages"][1]["tool_name"], "read_file");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
    }
}
