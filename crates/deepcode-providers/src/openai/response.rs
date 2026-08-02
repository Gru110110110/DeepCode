use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_core::provider::traits::{
    FinishReason, GenerateResponse, ResponseContentBlock, ResponseParser, StreamDelta, Usage,
};

#[derive(Clone, Copy)]
pub(crate) struct OpenAiResponseParser;

#[async_trait]
impl ResponseParser for OpenAiResponseParser {
    fn parse_response(&self, raw: &serde_json::Value) -> Result<GenerateResponse> {
        let choice = raw["choices"]
            .get(0)
            .ok_or_else(|| DeepCodeError::Parse("No choices in response".to_string()))?;

        let msg = &choice["message"];
        let mut content = Vec::new();

        // Parse hidden reasoning content used by reasoning-compatible providers.
        if let Some(reasoning) = msg["reasoning_content"].as_str() {
            if !reasoning.is_empty() {
                content.push(ResponseContentBlock::Reasoning {
                    text: reasoning.to_string(),
                    metadata: None,
                });
            }
        }

        // Parse text content
        if let Some(text) = msg["content"].as_str() {
            if !text.is_empty() {
                content.push(ResponseContentBlock::Text(text.to_string()));
            }
        }

        // Parse tool calls
        if let Some(tool_calls) = msg["tool_calls"].as_array() {
            for tc in tool_calls {
                let id = tc["id"].as_str().unwrap_or("").to_string();
                let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
                let input = parse_tool_arguments(&tc["function"]["arguments"]);
                content.push(ResponseContentBlock::ToolUse { id, name, input });
            }
        }

        let finish_reason = map_finish_reason(choice["finish_reason"].as_str());

        let usage = raw.get("usage").map_or(Usage::default(), chat_usage);

        Ok(GenerateResponse {
            content,
            usage,
            finish_reason,
        })
    }

    fn parse_stream_chunk(&self, raw_line: &str) -> Result<Option<StreamDelta>> {
        let line = raw_line.trim();

        // Skip empty lines and non-data lines
        if line.is_empty() || !line.starts_with("data:") {
            return Ok(None);
        }

        let data = line.strip_prefix("data:").unwrap().trim();

        // [DONE] marker
        if data == "[DONE]" {
            return Ok(Some(StreamDelta::Finished(FinishReason::Stop)));
        }

        // Parse JSON
        let chunk: serde_json::Value =
            serde_json::from_str(data).map_err(|e| DeepCodeError::Parse(e.to_string()))?;
        if let Some(error) = provider_error(&chunk) {
            return Err(error);
        }

        let usage_delta = stream_usage_delta(&chunk);

        // Usage can arrive either as a standalone chunk with empty choices or
        // attached to the final chunk, depending on the OpenAI-compatible API.
        if chunk["choices"]
            .as_array()
            .map(|a| a.is_empty())
            .unwrap_or(false)
        {
            return Ok(usage_delta);
        }

        let choice = match chunk["choices"].get(0) {
            Some(c) => c,
            None => return Ok(usage_delta),
        };

        // Check finish_reason
        if let Some(reason) = choice["finish_reason"].as_str() {
            let finish = map_finish_reason(Some(reason));
            return Ok(with_usage_delta(
                usage_delta,
                Some(StreamDelta::Finished(finish)),
            ));
        }

        let delta = &choice["delta"];

        // Reasoning delta (DeepSeek/OpenAI-compatible reasoning models)
        if let Some(reasoning) = delta["reasoning_content"].as_str() {
            if !reasoning.is_empty() {
                return Ok(with_usage_delta(
                    usage_delta,
                    Some(StreamDelta::ReasoningDelta(reasoning.to_string())),
                ));
            }
        }

        // Text delta
        if let Some(text) = delta["content"].as_str() {
            if !text.is_empty() {
                return Ok(with_usage_delta(
                    usage_delta,
                    Some(StreamDelta::TextDelta(text.to_string())),
                ));
            }
        }

        // Tool call deltas
        if let Some(tool_calls) = delta["tool_calls"].as_array() {
            let mut deltas = Vec::new();
            for tc in tool_calls {
                let index = tc["index"].as_u64().map(|i| i as usize);

                // New tool call starting
                if let (Some(id), Some(name)) = (tc["id"].as_str(), tc["function"]["name"].as_str())
                {
                    let input_delta = tool_arguments_delta(&tc["function"]["arguments"]);
                    deltas.push(StreamDelta::ToolUseStart {
                        id: id.to_string(),
                        name: name.to_string(),
                        index,
                        input_delta,
                    });
                    continue;
                }

                // Tool input delta
                if let Some(args) = tool_arguments_delta(&tc["function"]["arguments"]) {
                    let id = tc["id"].as_str().unwrap_or("").to_string();
                    deltas.push(StreamDelta::ToolUseInput {
                        id,
                        index,
                        input_delta: args,
                    });
                }
            }
            let delta = match deltas.len() {
                0 => None,
                1 => deltas.pop(),
                _ => Some(StreamDelta::Batch(deltas)),
            };
            return Ok(with_usage_delta(usage_delta, delta));
        }

        Ok(usage_delta)
    }
}

fn parse_tool_arguments(arguments: &serde_json::Value) -> serde_json::Value {
    if let Some(args) = arguments.as_str() {
        return serde_json::from_str(args).unwrap_or_default();
    }
    if arguments.is_object() || arguments.is_array() {
        return arguments.clone();
    }
    serde_json::Value::Object(serde_json::Map::new())
}

fn tool_arguments_delta(arguments: &serde_json::Value) -> Option<String> {
    if let Some(args) = arguments.as_str() {
        if args.is_empty() {
            None
        } else {
            Some(args.to_string())
        }
    } else if arguments.is_object() || arguments.is_array() {
        Some(arguments.to_string())
    } else {
        None
    }
}

fn provider_error(raw: &serde_json::Value) -> Option<DeepCodeError> {
    let error = raw.get("error")?;
    if let Some(message) = error.as_str() {
        return Some(DeepCodeError::Provider(message.to_string()));
    }
    Some(DeepCodeError::Provider(
        error["message"]
            .as_str()
            .or_else(|| raw["message"].as_str())
            .unwrap_or("Unknown provider error")
            .to_string(),
    ))
}

fn chat_usage(usage: &serde_json::Value) -> Usage {
    let input_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0) as usize;
    let output_tokens = usage["completion_tokens"].as_u64().unwrap_or(0) as usize;
    let cached_input_tokens = usage["prompt_cache_hit_tokens"]
        .as_u64()
        .or_else(|| usage["cached_tokens"].as_u64())
        .or_else(|| usage["prompt_tokens_details"]["cached_tokens"].as_u64())
        .unwrap_or(0) as usize;
    let cache_miss_input_tokens = usage["prompt_cache_miss_tokens"]
        .as_u64()
        .map(|value| value as usize)
        .unwrap_or_else(|| {
            if cached_input_tokens > 0 {
                input_tokens.saturating_sub(cached_input_tokens)
            } else {
                0
            }
        });
    let reasoning_output_tokens = usage["completion_tokens_details"]["reasoning_tokens"]
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

fn stream_usage_delta(chunk: &serde_json::Value) -> Option<StreamDelta> {
    let usage_value = chunk.get("usage")?;
    if usage_value.is_null() {
        return None;
    }

    let usage = chat_usage(usage_value);
    Some(StreamDelta::Usage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cached_input_tokens: usage.cached_input_tokens,
        cache_miss_input_tokens: usage.cache_miss_input_tokens,
        reasoning_output_tokens: usage.reasoning_output_tokens,
    })
}

fn with_usage_delta(
    usage_delta: Option<StreamDelta>,
    delta: Option<StreamDelta>,
) -> Option<StreamDelta> {
    match (usage_delta, delta) {
        (Some(usage), Some(StreamDelta::Batch(deltas))) => {
            let mut combined = vec![usage];
            combined.extend(deltas);
            Some(StreamDelta::Batch(combined))
        }
        (Some(usage), Some(delta)) => Some(StreamDelta::Batch(vec![usage, delta])),
        (Some(usage), None) => Some(usage),
        (None, delta) => delta,
    }
}

fn map_finish_reason(reason: Option<&str>) -> FinishReason {
    match reason {
        Some("tool_calls") => FinishReason::ToolCalls,
        Some("length") | Some("insufficient_system_resource") => FinishReason::Length,
        Some("content_filter") => FinishReason::ContentFilter,
        _ => FinishReason::Stop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_response_text_only() {
        let parser = OpenAiResponseParser;
        let raw = serde_json::json!({
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let resp = parser.parse_response(&raw).unwrap();
        assert_eq!(resp.content.len(), 1);
        assert!(
            matches!(&resp.content[0], ResponseContentBlock::Text(t) if t == "hello"
            )
        );
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 5);
        assert!(matches!(resp.finish_reason, FinishReason::Stop));
    }

    #[test]
    fn parse_response_reasoning_content() {
        let parser = OpenAiResponseParser;
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "reasoning_content": "think",
                    "content": "answer"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        });
        let resp = parser.parse_response(&raw).unwrap();
        assert_eq!(resp.content.len(), 2);
        assert!(
            matches!(&resp.content[0], ResponseContentBlock::Reasoning { text, .. } if text == "think")
        );
        assert!(matches!(&resp.content[1], ResponseContentBlock::Text(t) if t == "answer"));
    }

    #[test]
    fn parse_response_tool_calls() {
        let parser = OpenAiResponseParser;
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": "{\"path\":\"/tmp/a\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 15}
        });
        let resp = parser.parse_response(&raw).unwrap();
        assert_eq!(resp.content.len(), 1);
        assert!(matches!(
            &resp.content[0],
            ResponseContentBlock::ToolUse { id, name, .. } if id == "call_1" && name == "read_file"
        ));
        assert!(matches!(resp.finish_reason, FinishReason::ToolCalls));
    }

    #[test]
    fn parse_response_tool_calls_accepts_object_arguments() {
        let parser = OpenAiResponseParser;
        let raw = serde_json::json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "read_file", "arguments": {"path": "README.md"}}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let resp = parser.parse_response(&raw).unwrap();

        assert!(matches!(
            &resp.content[0],
            ResponseContentBlock::ToolUse { input, .. } if input["path"] == "README.md"
        ));
    }

    #[test]
    fn parse_stream_text_delta() {
        let parser = OpenAiResponseParser;
        let line = r#"data: {"choices":[{"delta":{"content":"hi"}}]}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(delta, StreamDelta::TextDelta(t) if t == "hi"));
    }

    #[test]
    fn parse_stream_tool_use_start() {
        let parser = OpenAiResponseParser;
        let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"shell"}}]}}]}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(
            matches!(delta, StreamDelta::ToolUseStart { id, name, input_delta, .. } if id == "c1" && name == "shell" && input_delta.is_none())
        );
    }

    #[test]
    fn parse_stream_tool_use_start_preserves_initial_arguments() {
        let parser = OpenAiResponseParser;
        let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\"path\":\"README.md\"}"}}]}}]}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(
            delta,
            StreamDelta::ToolUseStart { id, name, input_delta, .. }
                if id == "c1"
                    && name == "read_file"
                    && input_delta.as_deref() == Some("{\"path\":\"README.md\"}")
        ));
    }

    #[test]
    fn parse_stream_tool_use_start_accepts_object_arguments() {
        let parser = OpenAiResponseParser;
        let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":{"path":"README.md"}}}]}}]}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();

        assert!(matches!(
            delta,
            StreamDelta::ToolUseStart { id, name, input_delta, .. }
                if id == "c1"
                    && name == "read_file"
                    && input_delta.as_deref() == Some("{\"path\":\"README.md\"}")
        ));
    }

    #[test]
    fn parse_stream_preserves_multiple_tool_starts_in_one_chunk() {
        let parser = OpenAiResponseParser;
        let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":""}},{"index":1,"id":"c2","function":{"name":"grep","arguments":""}}]}}]}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();

        assert!(matches!(
            delta,
            StreamDelta::Batch(ref deltas)
                if matches!(&deltas[0], StreamDelta::ToolUseStart { id, index: Some(0), .. } if id == "c1")
                    && matches!(&deltas[1], StreamDelta::ToolUseStart { id, index: Some(1), .. } if id == "c2")
        ));
    }

    #[test]
    fn parse_stream_preserves_tool_index_on_argument_delta() {
        let parser = OpenAiResponseParser;
        let line = r#"data: {"choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"pattern\":"}}]}}]}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();

        assert!(matches!(
            delta,
            StreamDelta::ToolUseInput { id, index: Some(1), input_delta }
                if id.is_empty() && input_delta == "{\"pattern\":"
        ));
    }

    #[test]
    fn parse_stream_reasoning_delta() {
        let parser = OpenAiResponseParser;
        let line = r#"data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(delta, StreamDelta::ReasoningDelta(t) if t == "thinking"));
    }

    #[test]
    fn parse_stream_finished() {
        let parser = OpenAiResponseParser;
        let line = r#"data: {"choices":[{"finish_reason":"stop"}]}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(delta, StreamDelta::Finished(FinishReason::Stop)));
    }

    #[test]
    fn parse_deepseek_insufficient_resource_as_length() {
        let parser = OpenAiResponseParser;
        let raw = serde_json::json!({
            "choices": [{
                "message": {"content": "partial"},
                "finish_reason": "insufficient_system_resource"
            }]
        });
        let resp = parser.parse_response(&raw).unwrap();
        assert!(matches!(resp.finish_reason, FinishReason::Length));

        let line = r#"data: {"choices":[{"finish_reason":"insufficient_system_resource"}]}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(delta, StreamDelta::Finished(FinishReason::Length)));
    }

    #[test]
    fn parse_stream_usage() {
        let parser = OpenAiResponseParser;
        let line = r#"data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":4}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(
            delta,
            StreamDelta::Usage {
                input_tokens: 8,
                output_tokens: 4,
                ..
            }
        ));
    }

    #[test]
    fn parse_stream_usage_attached_to_finish_chunk() {
        let parser = OpenAiResponseParser;
        let line = r#"data: {"choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":4}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();

        assert!(matches!(
            delta,
            StreamDelta::Batch(ref deltas)
                if matches!(
                    deltas.first(),
                    Some(StreamDelta::Usage {
                        input_tokens: 8,
                        output_tokens: 4,
                        ..
                    })
                ) && matches!(
                    deltas.get(1),
                    Some(StreamDelta::Finished(FinishReason::Stop))
                )
        ));
    }

    #[test]
    fn parse_deepseek_cache_usage_details() {
        let parser = OpenAiResponseParser;
        let raw = serde_json::json!({
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "prompt_cache_hit_tokens": 80,
                "prompt_cache_miss_tokens": 40,
                "completion_tokens_details": {"reasoning_tokens": 12}
            }
        });
        let response = parser.parse_response(&raw).unwrap();
        assert_eq!(response.usage.input_tokens, 120);
        assert_eq!(response.usage.output_tokens, 30);
        assert_eq!(response.usage.cached_input_tokens, 80);
        assert_eq!(response.usage.cache_miss_input_tokens, 40);
        assert_eq!(response.usage.reasoning_output_tokens, 12);

        let line = r#"data: {"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":30,"prompt_cache_hit_tokens":80,"prompt_cache_miss_tokens":40,"completion_tokens_details":{"reasoning_tokens":12}}}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(
            delta,
            StreamDelta::Usage {
                input_tokens: 120,
                output_tokens: 30,
                cached_input_tokens: 80,
                cache_miss_input_tokens: 40,
                reasoning_output_tokens: 12,
            }
        ));
    }

    #[test]
    fn parse_kimi_top_level_cached_tokens() {
        let parser = OpenAiResponseParser;
        let raw = serde_json::json!({
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 25,
                "cached_tokens": 60
            }
        });
        let response = parser.parse_response(&raw).unwrap();
        assert_eq!(response.usage.input_tokens, 100);
        assert_eq!(response.usage.output_tokens, 25);
        assert_eq!(response.usage.cached_input_tokens, 60);
        assert_eq!(response.usage.cache_miss_input_tokens, 40);
    }

    #[test]
    fn parse_stream_done_marker() {
        let parser = OpenAiResponseParser;
        let delta = parser.parse_stream_chunk("data: [DONE]").unwrap().unwrap();
        assert!(matches!(delta, StreamDelta::Finished(FinishReason::Stop)));
    }

    #[test]
    fn parse_stream_reports_provider_error_event() {
        let parser = OpenAiResponseParser;
        let err = parser
            .parse_stream_chunk(r#"data: {"error":{"message":"bad request"}}"#)
            .unwrap_err();
        assert!(err.to_string().contains("bad request"));
    }

    #[test]
    fn parse_stream_skip_heartbeat() {
        let parser = OpenAiResponseParser;
        assert!(parser.parse_stream_chunk(": keep-alive").unwrap().is_none());
        assert!(parser.parse_stream_chunk("").unwrap().is_none());
    }
}
