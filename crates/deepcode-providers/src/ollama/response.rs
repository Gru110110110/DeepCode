use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_core::provider::traits::{
    FinishReason, GenerateResponse, ResponseContentBlock, ResponseParser, StreamDelta, Usage,
};

#[derive(Clone, Copy)]
pub(crate) struct OllamaResponseParser;

#[async_trait]
impl ResponseParser for OllamaResponseParser {
    fn parse_response(&self, raw: &serde_json::Value) -> Result<GenerateResponse> {
        if let Some(error) = provider_error(raw) {
            return Err(error);
        }

        let message = &raw["message"];
        let mut content = Vec::new();

        if let Some(thinking) = message["thinking"].as_str() {
            if !thinking.is_empty() {
                content.push(ResponseContentBlock::Reasoning {
                    text: thinking.to_string(),
                    metadata: None,
                });
            }
        }

        if let Some(text) = message["content"].as_str() {
            if !text.is_empty() {
                content.push(ResponseContentBlock::Text(text.to_string()));
            }
        }

        let tool_calls = parse_tool_calls(message);
        let has_tool_calls = !tool_calls.is_empty();
        content.extend(tool_calls);

        Ok(GenerateResponse {
            content,
            usage: ollama_usage(raw),
            finish_reason: finish_reason(raw["done_reason"].as_str(), has_tool_calls),
        })
    }

    fn parse_stream_chunk(&self, raw_line: &str) -> Result<Option<StreamDelta>> {
        let line = raw_line.trim();
        if line.is_empty() {
            return Ok(None);
        }

        let data = line.strip_prefix("data:").unwrap_or(line).trim();
        if data.is_empty() {
            return Ok(None);
        }
        if data == "[DONE]" {
            return Ok(Some(StreamDelta::Finished(FinishReason::Stop)));
        }

        let chunk: serde_json::Value =
            serde_json::from_str(data).map_err(|e| DeepCodeError::Parse(e.to_string()))?;
        if let Some(error) = provider_error(&chunk) {
            return Err(error);
        }

        let mut deltas = Vec::new();

        if chunk["done"].as_bool() == Some(true) {
            let usage = ollama_usage(&chunk);
            if usage.input_tokens > 0 || usage.output_tokens > 0 {
                deltas.push(StreamDelta::Usage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    cached_input_tokens: 0,
                    cache_miss_input_tokens: 0,
                    reasoning_output_tokens: 0,
                });
            }
        }

        let message = &chunk["message"];
        if let Some(thinking) = message["thinking"].as_str() {
            if !thinking.is_empty() {
                deltas.push(StreamDelta::ReasoningDelta(thinking.to_string()));
            }
        }
        if let Some(text) = message["content"].as_str() {
            if !text.is_empty() {
                deltas.push(StreamDelta::TextDelta(text.to_string()));
            }
        }

        let tool_calls = stream_tool_calls(message);
        let has_tool_calls = !tool_calls.is_empty();
        deltas.extend(tool_calls);

        if chunk["done"].as_bool() == Some(true) {
            deltas.push(StreamDelta::Finished(finish_reason(
                chunk["done_reason"].as_str(),
                has_tool_calls,
            )));
        }

        Ok(batch_or_single(deltas))
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
            .unwrap_or("Unknown error")
            .to_string(),
    ))
}

fn parse_tool_calls(message: &serde_json::Value) -> Vec<ResponseContentBlock> {
    message["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, call)| ResponseContentBlock::ToolUse {
            id: tool_call_id(call, index),
            name: call["function"]["name"].as_str().unwrap_or("").to_string(),
            input: parse_tool_arguments(&call["function"]["arguments"]),
        })
        .collect()
}

fn stream_tool_calls(message: &serde_json::Value) -> Vec<StreamDelta> {
    message["tool_calls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, call)| StreamDelta::ToolUseStart {
            id: tool_call_id(call, index),
            name: call["function"]["name"].as_str().unwrap_or("").to_string(),
            index: Some(index),
            input_delta: tool_arguments_delta(&call["function"]["arguments"]),
        })
        .collect()
}

fn tool_call_id(call: &serde_json::Value, index: usize) -> String {
    call["id"]
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| format!("call_{}", index))
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

fn ollama_usage(raw: &serde_json::Value) -> Usage {
    Usage {
        input_tokens: raw["prompt_eval_count"].as_u64().unwrap_or(0) as usize,
        output_tokens: raw["eval_count"].as_u64().unwrap_or(0) as usize,
        ..Usage::default()
    }
}

fn finish_reason(done_reason: Option<&str>, has_tool_calls: bool) -> FinishReason {
    if has_tool_calls {
        return FinishReason::ToolCalls;
    }
    match done_reason {
        Some("length") | Some("num_predict") => FinishReason::Length,
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
    fn parse_response_thinking_text_tool_and_usage() {
        let parser = OllamaResponseParser;
        let raw = serde_json::json!({
            "model": "gpt-oss",
            "message": {
                "role": "assistant",
                "thinking": "think",
                "content": "answer",
                "tool_calls": [{
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "README.md"}
                    }
                }]
            },
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5
        });

        let response = parser.parse_response(&raw).unwrap();
        assert_eq!(response.content.len(), 3);
        assert!(matches!(
            &response.content[0],
            ResponseContentBlock::Reasoning { text, .. } if text == "think"
        ));
        assert!(
            matches!(&response.content[1], ResponseContentBlock::Text(text) if text == "answer")
        );
        assert!(matches!(
            &response.content[2],
            ResponseContentBlock::ToolUse { id, name, input }
                if id == "call_0" && name == "read_file" && input["path"] == "README.md"
        ));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert!(matches!(response.finish_reason, FinishReason::ToolCalls));
    }

    #[test]
    fn parse_stream_raw_json_line() {
        let parser = OllamaResponseParser;
        let line = r#"{"message":{"content":"hello"},"done":false}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();
        assert!(matches!(delta, StreamDelta::TextDelta(text) if text == "hello"));
    }

    #[test]
    fn parse_stream_tool_and_done() {
        let parser = OllamaResponseParser;
        let line = r#"{"message":{"tool_calls":[{"function":{"name":"shell","arguments":{"cmd":"pwd"}}}]},"done":true,"done_reason":"stop","prompt_eval_count":8,"eval_count":4}"#;
        let delta = parser.parse_stream_chunk(line).unwrap().unwrap();

        assert!(matches!(
            delta,
            StreamDelta::Batch(ref deltas)
                if matches!(deltas.first(), Some(StreamDelta::Usage { input_tokens: 8, output_tokens: 4, .. }))
                    && matches!(deltas.get(1), Some(StreamDelta::ToolUseStart { id, name, input_delta, .. })
                        if id == "call_0" && name == "shell" && input_delta.as_deref() == Some("{\"cmd\":\"pwd\"}"))
                    && matches!(deltas.get(2), Some(StreamDelta::Finished(FinishReason::ToolCalls)))
        ));
    }

    #[test]
    fn parse_stream_done_without_usage_finishes() {
        let parser = OllamaResponseParser;
        let delta = parser
            .parse_stream_chunk(r#"{"done":true,"done_reason":"stop"}"#)
            .unwrap()
            .unwrap();
        assert!(matches!(delta, StreamDelta::Finished(FinishReason::Stop)));
    }
}
