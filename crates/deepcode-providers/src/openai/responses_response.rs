use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_core::provider::traits::{
    FinishReason, GenerateResponse, ResponseContentBlock, ResponseParser, StreamDelta, Usage,
};

#[derive(Clone, Copy)]
pub(crate) struct ResponsesResponseParser {
    provider: &'static str,
}

impl ResponsesResponseParser {
    pub(crate) const fn openai() -> Self {
        Self { provider: "openai" }
    }

    pub(crate) const fn deepseek() -> Self {
        Self {
            provider: "deepseek",
        }
    }
}

#[async_trait]
impl ResponseParser for ResponsesResponseParser {
    fn parse_response(&self, raw: &serde_json::Value) -> Result<GenerateResponse> {
        let mut content = Vec::new();
        for item in raw["output"].as_array().into_iter().flatten() {
            match item["type"].as_str() {
                Some("message") => {
                    for block in item["content"].as_array().into_iter().flatten() {
                        if block["type"] == "output_text" {
                            if let Some(text) = block["text"].as_str() {
                                content.push(ResponseContentBlock::Text(text.to_string()));
                            }
                        }
                    }
                }
                Some("reasoning") => {
                    let reasoning = reasoning_text(item);
                    content.push(ResponseContentBlock::Reasoning {
                        text: reasoning,
                        metadata: Some(serde_json::json!({
                            "provider": self.provider,
                            "item": item,
                        })),
                    });
                }
                Some("function_call") => {
                    let id = item["call_id"].as_str().unwrap_or("").to_string();
                    let name = item["name"].as_str().unwrap_or("").to_string();
                    let input = item["arguments"]
                        .as_str()
                        .and_then(|value| serde_json::from_str(value).ok())
                        .unwrap_or_else(|| serde_json::json!({}));
                    content.push(ResponseContentBlock::ToolUse { id, name, input });
                }
                _ => {}
            }
            content.push(ResponseContentBlock::ProviderItem {
                provider: self.provider.to_string(),
                value: item.clone(),
            });
        }
        let finish_reason = if content
            .iter()
            .any(|block| matches!(block, ResponseContentBlock::ToolUse { .. }))
        {
            FinishReason::ToolCalls
        } else if raw["status"] == "incomplete" {
            FinishReason::Length
        } else {
            FinishReason::Stop
        };
        Ok(GenerateResponse {
            content,
            usage: responses_usage(&raw["usage"]),
            finish_reason,
        })
    }

    fn parse_stream_chunk(&self, raw_line: &str) -> Result<Option<StreamDelta>> {
        let line = raw_line.trim();
        if line.is_empty() || !line.starts_with("data:") {
            return Ok(None);
        }
        let data = line.trim_start_matches("data:").trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(None);
        }
        let event: serde_json::Value =
            serde_json::from_str(data).map_err(|error| DeepCodeError::Parse(error.to_string()))?;
        Ok(match event["type"].as_str() {
            Some("response.output_text.delta") => event["delta"]
                .as_str()
                .map(|text| StreamDelta::TextDelta(text.to_string())),
            Some("response.reasoning_text.delta") => event["delta"]
                .as_str()
                .map(|text| StreamDelta::ReasoningDelta(text.to_string())),
            Some("response.reasoning_summary_text.delta") => event["delta"]
                .as_str()
                .map(|text| StreamDelta::ReasoningDelta(text.to_string())),
            Some("response.output_item.done") if event["item"]["type"] == "reasoning" => {
                Some(StreamDelta::Batch(vec![
                    StreamDelta::ReasoningMetadata(serde_json::json!({
                        "provider": self.provider,
                        "item": event["item"],
                    })),
                    StreamDelta::ProviderItem {
                        provider: self.provider.to_string(),
                        value: event["item"].clone(),
                    },
                ]))
            }
            Some("response.output_item.added") if event["item"]["type"] == "function_call" => {
                Some(StreamDelta::ToolUseStart {
                    id: event["item"]["call_id"].as_str().unwrap_or("").to_string(),
                    name: event["item"]["name"].as_str().unwrap_or("").to_string(),
                    index: event["output_index"].as_u64().map(|value| value as usize),
                    input_delta: None,
                })
            }
            Some("response.function_call_arguments.delta") => Some(StreamDelta::ToolUseInput {
                id: event["call_id"].as_str().unwrap_or("").to_string(),
                index: event["output_index"].as_u64().map(|value| value as usize),
                input_delta: event["delta"].as_str().unwrap_or("").to_string(),
            }),
            Some("response.output_item.done") if event["item"]["type"] == "function_call" => {
                Some(StreamDelta::Batch(vec![
                    StreamDelta::ToolUseEnd {
                        id: event["item"]["call_id"].as_str().unwrap_or("").to_string(),
                        index: event["output_index"].as_u64().map(|value| value as usize),
                    },
                    StreamDelta::ProviderItem {
                        provider: self.provider.to_string(),
                        value: event["item"].clone(),
                    },
                ]))
            }
            Some("response.output_item.done") => Some(StreamDelta::ProviderItem {
                provider: self.provider.to_string(),
                value: event["item"].clone(),
            }),
            Some("response.completed") => Some(response_done_delta(&event, FinishReason::Stop)),
            Some("response.incomplete") => Some(response_done_delta(&event, FinishReason::Length)),
            Some("response.failed") => {
                return Err(DeepCodeError::Provider(
                    response_error_message(&event)
                        .unwrap_or("Responses stream failed")
                        .to_string(),
                ));
            }
            Some("error") => {
                return Err(DeepCodeError::Provider(
                    event["message"]
                        .as_str()
                        .unwrap_or("OpenAI stream error")
                        .to_string(),
                ));
            }
            _ => None,
        })
    }
}

fn reasoning_text(item: &serde_json::Value) -> String {
    let mut parts = Vec::new();
    collect_text_parts(&item["summary"], &mut parts);
    collect_text_parts(&item["content"], &mut parts);
    parts.join("\n")
}

fn collect_text_parts(value: &serde_json::Value, parts: &mut Vec<String>) {
    match value {
        serde_json::Value::String(text) if !text.is_empty() => parts.push(text.clone()),
        serde_json::Value::Array(items) => {
            for item in items {
                collect_text_parts(item, parts);
            }
        }
        serde_json::Value::Object(object) => {
            for key in ["text", "content"] {
                if let Some(text) = object.get(key).and_then(|value| value.as_str()) {
                    if !text.is_empty() {
                        parts.push(text.to_string());
                    }
                }
            }
        }
        _ => {}
    }
}

fn response_done_delta(event: &serde_json::Value, finish: FinishReason) -> StreamDelta {
    let usage = responses_usage(&event["response"]["usage"]);
    StreamDelta::Batch(vec![
        StreamDelta::Usage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cached_input_tokens: usage.cached_input_tokens,
            cache_miss_input_tokens: usage.cache_miss_input_tokens,
            reasoning_output_tokens: usage.reasoning_output_tokens,
        },
        StreamDelta::Finished(finish),
    ])
}

fn responses_usage(usage: &serde_json::Value) -> Usage {
    let input_tokens = usage["input_tokens"].as_u64().unwrap_or(0) as usize;
    let output_tokens = usage["output_tokens"].as_u64().unwrap_or(0) as usize;
    let cached_input_tokens = usage["input_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0) as usize;
    let cache_miss_input_tokens = if usage["input_tokens_details"]["cached_tokens"].is_number() {
        input_tokens.saturating_sub(cached_input_tokens)
    } else {
        0
    };
    let reasoning_output_tokens = usage["output_tokens_details"]["reasoning_tokens"]
        .as_u64()
        .unwrap_or(0) as usize;

    Usage {
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_miss_input_tokens,
        reasoning_output_tokens,
    }
}

fn response_error_message(event: &serde_json::Value) -> Option<&str> {
    event["response"]["error"]["message"]
        .as_str()
        .or_else(|| event["error"]["message"].as_str())
        .or_else(|| event["message"].as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_response_stream_tool_events() {
        let parser = ResponsesResponseParser::openai();
        let start = r#"data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","call_id":"call_1","name":"shell"}}"#;
        assert!(matches!(
            parser.parse_stream_chunk(start).unwrap().unwrap(),
            StreamDelta::ToolUseStart { id, name, .. } if id == "call_1" && name == "shell"
        ));
    }

    #[test]
    fn function_arguments_delta_uses_index_when_call_id_is_absent() {
        let parser = ResponsesResponseParser::openai();
        let line = r#"data: {"type":"response.function_call_arguments.delta","output_index":1,"item_id":"fc_1","delta":"{\"command\":\"pwd\"}"}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(
            delta,
            StreamDelta::ToolUseInput { id, index: Some(1), input_delta }
                if id.is_empty() && input_delta == "{\"command\":\"pwd\"}"
        ));
    }

    #[test]
    fn parses_deepseek_reasoning_content() {
        let parser = ResponsesResponseParser::openai();
        let raw = serde_json::json!({
            "output": [{
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "think"}]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 2}
        });
        let response = parser.parse_response(&raw).unwrap();
        assert!(
            matches!(&response.content[0], ResponseContentBlock::Reasoning { text, .. } if text == "think")
        );

        let line = r#"data: {"type":"response.reasoning_text.delta","delta":"step"}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(delta, StreamDelta::ReasoningDelta(text) if text == "step"));
    }

    #[test]
    fn preserves_encrypted_reasoning_without_visible_text() {
        let parser = ResponsesResponseParser::openai();
        let raw = serde_json::json!({
            "output": [{
                "id": "rs_1",
                "type": "reasoning",
                "encrypted_content": "ciphertext",
                "summary": []
            }]
        });
        let response = parser.parse_response(&raw).unwrap();
        assert!(matches!(
            &response.content[0],
            ResponseContentBlock::Reasoning { text, .. } if text.is_empty()
        ));
        assert!(matches!(
            &response.content[1],
            ResponseContentBlock::ProviderItem { provider, value }
                if provider == "openai" && value["encrypted_content"] == "ciphertext"
        ));
    }

    #[test]
    fn parses_incomplete_responses_usage_before_length_finish() {
        let parser = ResponsesResponseParser::openai();
        let line = r#"data: {"type":"response.incomplete","response":{"usage":{"input_tokens":7,"output_tokens":5}}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(
            delta,
            StreamDelta::Batch(ref deltas)
                if matches!(deltas[0], StreamDelta::Usage { input_tokens: 7, output_tokens: 5, .. })
                    && matches!(deltas[1], StreamDelta::Finished(FinishReason::Length))
        ));
    }

    #[test]
    fn parses_responses_cache_usage_details() {
        let parser = ResponsesResponseParser::openai();
        let raw = serde_json::json!({
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}],
            "usage": {
                "input_tokens": 120,
                "output_tokens": 30,
                "input_tokens_details": {"cached_tokens": 80},
                "output_tokens_details": {"reasoning_tokens": 12}
            }
        });
        let response = parser.parse_response(&raw).unwrap();
        assert_eq!(response.usage.input_tokens, 120);
        assert_eq!(response.usage.output_tokens, 30);
        assert_eq!(response.usage.cached_input_tokens, 80);
        assert_eq!(response.usage.cache_miss_input_tokens, 40);
        assert_eq!(response.usage.reasoning_output_tokens, 12);

        let line = r#"data: {"type":"response.completed","response":{"usage":{"input_tokens":120,"output_tokens":30,"input_tokens_details":{"cached_tokens":80},"output_tokens_details":{"reasoning_tokens":12}}}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(
            delta,
            StreamDelta::Batch(ref deltas)
                if matches!(
                    deltas[0],
                    StreamDelta::Usage {
                        input_tokens: 120,
                        output_tokens: 30,
                        cached_input_tokens: 80,
                        cache_miss_input_tokens: 40,
                        reasoning_output_tokens: 12,
                    }
                )
        ));
    }
}
