use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_core::provider::traits::{
    FinishReason, GenerateResponse, ResponseContentBlock, ResponseParser, StreamDelta, Usage,
};

#[derive(Clone, Copy)]
pub(crate) struct AnthropicResponseParser;

#[async_trait]
impl ResponseParser for AnthropicResponseParser {
    fn parse_response(&self, raw: &serde_json::Value) -> Result<GenerateResponse> {
        let mut content = Vec::new();

        if let Some(blocks) = raw["content"].as_array() {
            for block in blocks {
                match block["type"].as_str() {
                    Some("thinking") => {
                        content.push(ResponseContentBlock::Reasoning {
                            text: block["thinking"].as_str().unwrap_or("").to_string(),
                            metadata: Some(serde_json::json!({
                                "provider": "anthropic",
                                "block": block,
                                "signature": block["signature"],
                            })),
                        });
                    }
                    Some("redacted_thinking") => {
                        content.push(ResponseContentBlock::Reasoning {
                            text: String::new(),
                            metadata: Some(serde_json::json!({
                                "provider": "anthropic",
                                "block": block,
                            })),
                        });
                    }
                    Some("text") => {
                        if let Some(text) = block["text"].as_str() {
                            content.push(ResponseContentBlock::Text(text.to_string()));
                        }
                    }
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or("").to_string();
                        let name = block["name"].as_str().unwrap_or("").to_string();
                        let input = block["input"].clone();
                        content.push(ResponseContentBlock::ToolUse { id, name, input });
                    }
                    _ => {}
                }
            }
        }

        let finish_reason = match raw["stop_reason"].as_str() {
            Some("end_turn") => FinishReason::Stop,
            Some("tool_use") => FinishReason::ToolCalls,
            Some("max_tokens") => FinishReason::Length,
            _ => FinishReason::Stop,
        };

        let usage = raw.get("usage").map_or(Usage::default(), anthropic_usage);

        Ok(GenerateResponse {
            content,
            usage,
            finish_reason,
        })
    }

    fn parse_stream_chunk(&self, raw_line: &str) -> Result<Option<StreamDelta>> {
        let line = raw_line.trim();
        if line.is_empty() || !line.starts_with("data:") {
            return Ok(None);
        }

        let data = line.strip_prefix("data:").unwrap().trim();
        if data.is_empty() {
            return Ok(None);
        }

        let event: serde_json::Value =
            serde_json::from_str(data).map_err(|e| DeepCodeError::Parse(e.to_string()))?;

        match event["type"].as_str() {
            Some("error") => Err(DeepCodeError::Provider(
                event["error"]["message"]
                    .as_str()
                    .unwrap_or("Unknown Anthropic stream error")
                    .to_string(),
            )),
            Some("message_start") => {
                if event["message"].get("usage").is_some() {
                    let usage = anthropic_usage(&event["message"]["usage"]);
                    return Ok(Some(StreamDelta::Usage {
                        input_tokens: usage.input_tokens,
                        output_tokens: 0,
                        cached_input_tokens: usage.cached_input_tokens,
                        cache_miss_input_tokens: usage.cache_miss_input_tokens,
                        reasoning_output_tokens: 0,
                    }));
                }
                Ok(None)
            }
            Some("content_block_start") => {
                let block = &event["content_block"];
                if block["type"] == "tool_use" {
                    let index = event["index"].as_u64().map(|value| value as usize);
                    let id = block["id"].as_str().unwrap_or("").to_string();
                    let name = block["name"].as_str().unwrap_or("").to_string();
                    Ok(Some(StreamDelta::ToolUseStart {
                        id,
                        name,
                        index,
                        input_delta: None,
                    }))
                } else if block["type"] == "thinking" || block["type"] == "redacted_thinking" {
                    Ok(Some(StreamDelta::ReasoningMetadata(serde_json::json!({
                        "provider": "anthropic",
                        "block": block,
                    }))))
                } else {
                    Ok(None)
                }
            }
            Some("content_block_delta") => {
                let delta = &event["delta"];
                match delta["type"].as_str() {
                    Some("text_delta") => {
                        if let Some(text) = delta["text"].as_str() {
                            Ok(Some(StreamDelta::TextDelta(text.to_string())))
                        } else {
                            Ok(None)
                        }
                    }
                    Some("thinking_delta") => Ok(delta["thinking"]
                        .as_str()
                        .map(|text| StreamDelta::ReasoningDelta(text.to_string()))),
                    Some("signature_delta") => Ok(delta["signature"].as_str().map(|signature| {
                        StreamDelta::ReasoningMetadata(serde_json::json!({
                            "provider": "anthropic",
                            "signature": signature,
                        }))
                    })),
                    Some("input_json_delta") => {
                        if let Some(json) = delta["partial_json"].as_str() {
                            let index = event["index"].as_u64().map(|value| value as usize);
                            Ok(Some(StreamDelta::ToolUseInput {
                                id: String::new(),
                                index,
                                input_delta: json.to_string(),
                            }))
                        } else {
                            Ok(None)
                        }
                    }
                    _ => Ok(None),
                }
            }
            Some("content_block_stop") => Ok(None),
            Some("message_delta") => {
                let mut deltas = Vec::new();
                if event.get("usage").is_some() {
                    let usage = anthropic_usage(&event["usage"]);
                    deltas.push(StreamDelta::Usage {
                        input_tokens: 0,
                        output_tokens: usage.output_tokens,
                        cached_input_tokens: 0,
                        cache_miss_input_tokens: 0,
                        reasoning_output_tokens: usage.reasoning_output_tokens,
                    });
                }
                let stop_reason = event["delta"]["stop_reason"].as_str();
                if stop_reason.is_some() {
                    deltas.push(StreamDelta::Finished(map_finish_reason(stop_reason)));
                }
                Ok(batch_or_single(deltas))
            }
            Some("message_stop") => Ok(Some(StreamDelta::Finished(FinishReason::Stop))),
            _ => Ok(None),
        }
    }
}

fn anthropic_usage(usage: &serde_json::Value) -> Usage {
    let uncached = usage["input_tokens"].as_u64().unwrap_or(0) as usize;
    let cache_creation = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0) as usize;
    let cached_input_tokens = usage["cache_read_input_tokens"].as_u64().unwrap_or(0) as usize;
    let output_tokens = usage["output_tokens"].as_u64().unwrap_or(0) as usize;
    let reasoning_output_tokens = usage["output_tokens_details"]["thinking_tokens"]
        .as_u64()
        .or_else(|| usage["thinking_tokens"].as_u64())
        .unwrap_or(0) as usize;

    Usage {
        input_tokens: uncached + cache_creation + cached_input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_miss_input_tokens: if cache_creation > 0 || cached_input_tokens > 0 {
            uncached + cache_creation
        } else {
            0
        },
        reasoning_output_tokens,
    }
}

fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("tool_use") => FinishReason::ToolCalls,
        Some("max_tokens") => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

fn batch_or_single(mut deltas: Vec<StreamDelta>) -> Option<StreamDelta> {
    match deltas.len() {
        0 => None,
        1 => deltas.pop(),
        _ => Some(StreamDelta::Batch(deltas)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_response_text_and_tool() {
        let parser = AnthropicResponseParser;
        let raw = serde_json::json!({
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {"path": "/tmp/x"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 12, "output_tokens": 8}
        });
        let resp = parser.parse_response(&raw).unwrap();
        assert_eq!(resp.content.len(), 2);
        assert!(
            matches!(&resp.content[0], ResponseContentBlock::Text(t) if t == "hello"
            )
        );
        assert!(matches!(
            &resp.content[1], ResponseContentBlock::ToolUse { id, name, .. } if id == "tu1" && name == "read_file"
        ));
        assert!(matches!(resp.finish_reason, FinishReason::ToolCalls));
        assert_eq!(resp.usage.input_tokens, 12);
        assert_eq!(resp.usage.output_tokens, 8);
    }

    #[test]
    fn preserves_redacted_thinking_as_replayable_metadata() {
        let parser = AnthropicResponseParser;
        let raw = serde_json::json!({
            "content": [{"type": "redacted_thinking", "data": "opaque"}],
            "stop_reason": "end_turn"
        });
        let response = parser.parse_response(&raw).unwrap();
        assert!(matches!(
            &response.content[0],
            ResponseContentBlock::Reasoning { text, metadata: Some(metadata) }
                if text.is_empty() && metadata["block"]["data"] == "opaque"
        ));
    }

    #[test]
    fn parse_stream_message_start_usage() {
        let parser = AnthropicResponseParser;
        let line = r#"data: {"type":"message_start","message":{"usage":{"input_tokens":15}}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(
            delta,
            StreamDelta::Usage {
                input_tokens: 15,
                output_tokens: 0,
                ..
            }
        ));
    }

    #[test]
    fn parse_stream_text_delta() {
        let parser = AnthropicResponseParser;
        let line =
            r#"data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"world"}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(delta, StreamDelta::TextDelta(t) if t == "world"));
    }

    #[test]
    fn parse_stream_tool_use_start() {
        let parser = AnthropicResponseParser;
        let line = r#"data: {"type":"content_block_start","content_block":{"type":"tool_use","id":"x1","name":"shell"}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(
            matches!(delta, StreamDelta::ToolUseStart { id, name, input_delta, .. } if id == "x1" && name == "shell" && input_delta.is_none())
        );
    }

    #[test]
    fn parse_stream_tool_use_input() {
        let parser = AnthropicResponseParser;
        let line = r#"data: {"type":"content_block_delta","delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(
            matches!(delta, StreamDelta::ToolUseInput { input_delta, .. } if input_delta == "{\"a\":1}")
        );
    }

    #[test]
    fn parse_stream_message_delta_usage() {
        let parser = AnthropicResponseParser;
        let line = r#"data: {"type":"message_delta","usage":{"output_tokens":9}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(
            delta,
            StreamDelta::Usage {
                input_tokens: 0,
                output_tokens: 9,
                ..
            }
        ));
    }

    #[test]
    fn parse_stream_message_delta_finish() {
        let parser = AnthropicResponseParser;
        let line = r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(delta, StreamDelta::Finished(FinishReason::Stop)));
    }

    #[test]
    fn parse_stream_message_delta_usage_and_finish() {
        let parser = AnthropicResponseParser;
        let line = r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9,"output_tokens_details":{"thinking_tokens":3}}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(
            delta,
            StreamDelta::Batch(ref deltas)
                if matches!(deltas.first(), Some(StreamDelta::Usage { output_tokens: 9, reasoning_output_tokens: 3, .. }))
                    && matches!(deltas.get(1), Some(StreamDelta::Finished(FinishReason::ToolCalls)))
        ));
    }

    #[test]
    fn parse_stream_content_block_stop_is_not_tool_end() {
        let parser = AnthropicResponseParser;
        let line = r#"data: {"type":"content_block_stop","index":0}"#;
        assert!(parser.parse_stream_chunk(line).unwrap().is_none());
    }

    #[test]
    fn parse_stream_message_stop() {
        let parser = AnthropicResponseParser;
        let line = r#"data: {"type":"message_stop"}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(delta, StreamDelta::Finished(FinishReason::Stop)));
    }

    #[test]
    fn parse_stream_skip_empty() {
        let parser = AnthropicResponseParser;
        assert!(parser.parse_stream_chunk("data: ").unwrap().is_none());
        assert!(parser.parse_stream_chunk("").unwrap().is_none());
    }

    #[test]
    fn parse_response_usage_includes_cache_tokens() {
        let parser = AnthropicResponseParser;
        let raw = serde_json::json!({
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "cache_creation_input_tokens": 3,
                "cache_read_input_tokens": 5,
                "output_tokens": 8,
                "output_tokens_details": {"thinking_tokens": 2}
            }
        });
        let resp = parser.parse_response(&raw).unwrap();
        assert_eq!(resp.usage.input_tokens, 20);
        assert_eq!(resp.usage.cached_input_tokens, 5);
        assert_eq!(resp.usage.cache_miss_input_tokens, 15);
        assert_eq!(resp.usage.reasoning_output_tokens, 2);
    }
}
