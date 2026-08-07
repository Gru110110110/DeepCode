use deepcode_core::provider::traits::{ResponseContentBlock, StreamDelta};
use std::collections::{HashMap, VecDeque};

/// Accumulates streaming deltas into complete response content blocks.
pub(crate) struct StreamAccumulator {
    pub text_buffer: String,
    reasoning_buffer: String,
    reasoning_metadata: Option<serde_json::Value>,
    tool_calls: HashMap<String, ToolCallAccumulator>,
    tool_indices: HashMap<usize, String>,
    tool_order: VecDeque<String>,
    active_tool_id: Option<String>,
    pending_blocks: VecDeque<ResponseContentBlock>,
}

struct ToolCallAccumulator {
    name: String,
    input_json: String,
}

impl StreamAccumulator {
    pub(crate) fn new() -> Self {
        Self {
            text_buffer: String::new(),
            reasoning_buffer: String::new(),
            reasoning_metadata: None,
            tool_calls: HashMap::new(),
            tool_indices: HashMap::new(),
            tool_order: VecDeque::new(),
            active_tool_id: None,
            pending_blocks: VecDeque::new(),
        }
    }

    /// Process a single `StreamDelta`. Returns any completed `ResponseContentBlock`.
    pub(crate) fn process(&mut self, delta: StreamDelta) -> Option<ResponseContentBlock> {
        match delta {
            StreamDelta::TextDelta(text) => {
                self.text_buffer.push_str(&text);
                None
            }
            StreamDelta::ReasoningDelta(text) => {
                self.reasoning_buffer.push_str(&text);
                None
            }
            StreamDelta::ReasoningMetadata(metadata) => {
                merge_metadata(&mut self.reasoning_metadata, metadata);
                None
            }
            StreamDelta::ProviderItem { provider, value } => {
                // A completed provider item follows all deltas for that item.
                // Flush the derived content first so streaming and non-streaming
                // responses preserve the same content-block order.
                self.queue_buffered_content();
                self.pending_blocks
                    .push_back(ResponseContentBlock::ProviderItem { provider, value });
                self.pending_blocks.pop_front()
            }
            StreamDelta::ToolUseStart {
                id,
                name,
                index,
                input_delta,
            } => {
                // Flush any accumulated reasoning/text first.
                self.queue_buffered_content();
                self.active_tool_id = Some(id.clone());
                if let Some(index) = index {
                    self.tool_indices.insert(index, id.clone());
                }
                if !self.tool_calls.contains_key(&id) {
                    self.tool_order.push_back(id.clone());
                }
                self.tool_calls.insert(
                    id,
                    ToolCallAccumulator {
                        name,
                        input_json: input_delta.unwrap_or_default(),
                    },
                );
                self.pending_blocks.pop_front()
            }
            StreamDelta::ToolUseInput {
                id,
                index,
                input_delta,
            } => {
                let tool_id = self.resolve_tool_id(&id, index);
                if let Some(acc) = tool_id.and_then(|id| self.tool_calls.get_mut(&id)) {
                    acc.input_json.push_str(&input_delta);
                }
                None
            }
            StreamDelta::ToolUseEnd { id, index } => {
                if let Some(tool_id) = self.resolve_tool_id(&id, index) {
                    if let Some(block) = self.complete_tool_call(&tool_id) {
                        self.pending_blocks.push_back(block);
                    }
                    return self.pending_blocks.pop_front();
                }
                None
            }
            StreamDelta::Finished(_) => {
                self.queue_buffered_content();
                self.queue_open_tool_calls();
                self.pending_blocks.pop_front()
            }
            StreamDelta::Batch(deltas) => {
                let mut first = None;
                for delta in deltas {
                    if let Some(block) = self.process(delta) {
                        if first.is_none() {
                            first = Some(block);
                        } else {
                            self.pending_blocks.push_back(block);
                        }
                    }
                }
                first.or_else(|| self.pending_blocks.pop_front())
            }
            StreamDelta::Usage { .. } => None,
        }
    }

    fn queue_buffered_content(&mut self) {
        if !self.reasoning_buffer.is_empty() || self.reasoning_metadata.is_some() {
            let reasoning = std::mem::take(&mut self.reasoning_buffer);
            self.pending_blocks
                .push_back(ResponseContentBlock::Reasoning {
                    text: reasoning,
                    metadata: self.reasoning_metadata.take(),
                });
        }
        if !self.text_buffer.is_empty() {
            let text = std::mem::take(&mut self.text_buffer);
            self.pending_blocks
                .push_back(ResponseContentBlock::Text(text));
        }
    }

    /// Flush any remaining text after stream ends.
    pub(crate) fn flush(&mut self) -> Option<ResponseContentBlock> {
        if let Some(block) = self.pending_blocks.pop_front() {
            return Some(block);
        }
        self.queue_buffered_content();
        if let Some(block) = self.pending_blocks.pop_front() {
            return Some(block);
        }
        self.queue_open_tool_calls();
        self.pending_blocks.pop_front()
    }

    fn resolve_tool_id(&self, id: &str, index: Option<usize>) -> Option<String> {
        if !id.is_empty() {
            return Some(id.to_string());
        }
        if let Some(id) = index.and_then(|index| self.tool_indices.get(&index)) {
            return Some(id.clone());
        }
        self.active_tool_id.clone().or_else(|| {
            if self.tool_calls.len() == 1 {
                self.tool_calls.keys().next().cloned()
            } else {
                None
            }
        })
    }

    fn complete_tool_call(&mut self, tool_id: &str) -> Option<ResponseContentBlock> {
        if self.active_tool_id.as_deref() == Some(tool_id) {
            self.active_tool_id = None;
        }
        self.tool_indices.retain(|_, id| id != tool_id);
        self.tool_order.retain(|id| id != tool_id);
        self.tool_calls.remove(tool_id).map(|acc| {
            let input: serde_json::Value =
                serde_json::from_str(&acc.input_json).unwrap_or_default();
            ResponseContentBlock::ToolUse {
                id: tool_id.to_string(),
                name: acc.name,
                input,
            }
        })
    }

    fn queue_open_tool_calls(&mut self) {
        let open_ids: Vec<String> = self.tool_order.iter().cloned().collect();
        for tool_id in open_ids {
            if let Some(block) = self.complete_tool_call(&tool_id) {
                self.pending_blocks.push_back(block);
            }
        }
    }
}

fn merge_metadata(target: &mut Option<serde_json::Value>, mut incoming: serde_json::Value) {
    if let (Some(existing), Some(signature)) = (
        target.as_mut(),
        incoming
            .get("signature")
            .and_then(serde_json::Value::as_str),
    ) {
        let signature = format!(
            "{}{}",
            existing
                .get("signature")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(""),
            signature
        );
        if let Some(block) = existing
            .get_mut("block")
            .and_then(serde_json::Value::as_object_mut)
        {
            block.insert("signature".to_string(), signature.clone().into());
        }
        incoming["signature"] = signature.into();
    }
    match (target.as_mut(), incoming) {
        (Some(serde_json::Value::Object(existing)), serde_json::Value::Object(incoming)) => {
            existing.extend(incoming);
        }
        (_, value) => *target = Some(value),
    }
}

impl Default for StreamAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accumulate_text_only() {
        let mut acc = StreamAccumulator::new();
        assert!(acc
            .process(StreamDelta::TextDelta("hello ".into()))
            .is_none());
        assert!(acc
            .process(StreamDelta::TextDelta("world".into()))
            .is_none());
        let block = acc.flush().unwrap();
        assert!(matches!(block, ResponseContentBlock::Text(t) if t == "hello world"));
    }

    #[test]
    fn accumulate_reasoning() {
        let mut acc = StreamAccumulator::new();
        assert!(acc
            .process(StreamDelta::ReasoningDelta("thinking ".into()))
            .is_none());
        assert!(acc
            .process(StreamDelta::ReasoningDelta("done".into()))
            .is_none());
        let block = acc.flush().unwrap();
        assert!(
            matches!(block, ResponseContentBlock::Reasoning { text, .. } if text == "thinking done")
        );
    }

    #[test]
    fn preserves_reasoning_metadata_without_visible_text() {
        let mut acc = StreamAccumulator::new();
        acc.process(StreamDelta::ReasoningMetadata(serde_json::json!({
            "provider": "anthropic",
            "block": {"type": "thinking", "thinking": ""}
        })));
        acc.process(StreamDelta::ReasoningMetadata(serde_json::json!({
            "provider": "anthropic",
            "signature": "sig-1"
        })));
        acc.process(StreamDelta::ReasoningMetadata(serde_json::json!({
            "provider": "anthropic",
            "signature": "sig-2"
        })));
        let block = acc.flush().unwrap();
        assert!(matches!(
            block,
            ResponseContentBlock::Reasoning { text, metadata: Some(metadata) }
                if text.is_empty()
                    && metadata["signature"] == "sig-1sig-2"
                    && metadata["block"]["signature"] == "sig-1sig-2"
        ));
    }

    #[test]
    fn finished_flushes_text() {
        let mut acc = StreamAccumulator::new();
        acc.process(StreamDelta::TextDelta("done".into()));
        let block = acc
            .process(StreamDelta::Finished(
                deepcode_core::provider::traits::FinishReason::Stop,
            ))
            .unwrap();
        assert!(matches!(block, ResponseContentBlock::Text(t) if t == "done"));
    }

    #[test]
    fn tool_use_start_flushes_text() {
        let mut acc = StreamAccumulator::new();
        acc.process(StreamDelta::TextDelta("before ".into()));
        let block = acc
            .process(StreamDelta::ToolUseStart {
                id: "t1".into(),
                name: "shell".into(),
                index: None,
                input_delta: None,
            })
            .unwrap();
        assert!(matches!(block, ResponseContentBlock::Text(t) if t == "before "));
    }

    #[test]
    fn provider_item_follows_its_buffered_text() {
        let mut acc = StreamAccumulator::new();
        acc.process(StreamDelta::TextDelta("answer".into()));

        let first = acc
            .process(StreamDelta::ProviderItem {
                provider: "openai".into(),
                value: serde_json::json!({
                    "type": "message",
                    "content": [{"type": "output_text", "text": "answer"}]
                }),
            })
            .unwrap();
        let second = acc.flush().unwrap();

        assert!(matches!(first, ResponseContentBlock::Text(text) if text == "answer"));
        assert!(matches!(
            second,
            ResponseContentBlock::ProviderItem { provider, value }
                if provider == "openai" && value["type"] == "message"
        ));
    }

    #[test]
    fn tool_use_complete_cycle() {
        let mut acc = StreamAccumulator::new();
        acc.process(StreamDelta::ToolUseStart {
            id: "t1".into(),
            name: "read_file".into(),
            index: None,
            input_delta: None,
        });
        acc.process(StreamDelta::ToolUseInput {
            id: "t1".into(),
            index: None,
            input_delta: "{\"path\":\"/tmp/x\"}".into(),
        });
        let block = acc
            .process(StreamDelta::ToolUseEnd {
                id: "t1".into(),
                index: None,
            })
            .unwrap();
        assert!(
            matches!(block, ResponseContentBlock::ToolUse { id, name, .. } if id == "t1" && name == "read_file")
        );
    }

    #[test]
    fn mixed_text_and_tool() {
        let mut acc = StreamAccumulator::new();
        acc.process(StreamDelta::TextDelta("Let me check ".into()));
        let flushed = acc.process(StreamDelta::ToolUseStart {
            id: "t1".into(),
            name: "read_file".into(),
            index: None,
            input_delta: None,
        });
        assert!(matches!(flushed, Some(ResponseContentBlock::Text(t)) if t == "Let me check "));
        acc.process(StreamDelta::ToolUseInput {
            id: "t1".into(),
            index: None,
            input_delta: "{}".into(),
        });
        let tool = acc
            .process(StreamDelta::ToolUseEnd {
                id: "t1".into(),
                index: None,
            })
            .unwrap();
        assert!(matches!(tool, ResponseContentBlock::ToolUse { .. }));
        acc.process(StreamDelta::TextDelta(" and done".into()));
        let text = acc.flush().unwrap();
        assert!(matches!(text, ResponseContentBlock::Text(t) if t == " and done"));
    }

    #[test]
    fn pending_content_stays_before_tool() {
        let mut acc = StreamAccumulator::new();
        acc.process(StreamDelta::ReasoningDelta("think".into()));
        acc.process(StreamDelta::TextDelta("before".into()));
        let first = acc
            .process(StreamDelta::ToolUseStart {
                id: "t1".into(),
                name: "read_file".into(),
                index: None,
                input_delta: Some("{\"path\":\"README.md\"}".into()),
            })
            .unwrap();
        let second = acc
            .process(StreamDelta::ToolUseEnd {
                id: "t1".into(),
                index: None,
            })
            .unwrap();
        let third = acc.flush().unwrap();

        assert!(matches!(first, ResponseContentBlock::Reasoning { text, .. } if text == "think"));
        assert!(matches!(second, ResponseContentBlock::Text(t) if t == "before"));
        assert!(matches!(third, ResponseContentBlock::ToolUse { .. }));
    }

    #[test]
    fn usage_delta_ignored() {
        let mut acc = StreamAccumulator::new();
        assert!(acc
            .process(StreamDelta::Usage {
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 0,
                cache_miss_input_tokens: 0,
                reasoning_output_tokens: 0,
            })
            .is_none());
        assert!(acc.text_buffer.is_empty());
    }

    #[test]
    fn empty_tool_delta_ids_use_active_tool() {
        let mut acc = StreamAccumulator::new();
        acc.process(StreamDelta::ToolUseStart {
            id: "t1".into(),
            name: "read_file".into(),
            index: None,
            input_delta: None,
        });
        acc.process(StreamDelta::ToolUseInput {
            id: String::new(),
            index: None,
            input_delta: "{\"path\":\"README.md\"}".into(),
        });
        let block = acc
            .process(StreamDelta::ToolUseEnd {
                id: String::new(),
                index: None,
            })
            .unwrap();

        assert!(matches!(
            block,
            ResponseContentBlock::ToolUse { id, name, input }
                if id == "t1" && name == "read_file" && input["path"] == "README.md"
        ));
    }

    #[test]
    fn finished_flushes_open_tool_calls() {
        let mut acc = StreamAccumulator::new();
        acc.process(StreamDelta::ToolUseStart {
            id: "call_1".into(),
            name: "grep".into(),
            index: None,
            input_delta: None,
        });
        acc.process(StreamDelta::ToolUseInput {
            id: String::new(),
            index: None,
            input_delta: "{\"pattern\":\"DeepCode\"}".into(),
        });
        let block = acc
            .process(StreamDelta::Finished(
                deepcode_core::provider::traits::FinishReason::ToolCalls,
            ))
            .unwrap();

        assert!(matches!(
            block,
            ResponseContentBlock::ToolUse { id, name, input }
                if id == "call_1" && name == "grep" && input["pattern"] == "DeepCode"
        ));
    }

    #[test]
    fn tool_use_start_accepts_initial_input_delta() {
        let mut acc = StreamAccumulator::new();
        assert!(acc
            .process(StreamDelta::ToolUseStart {
                id: "call_1".into(),
                name: "read_file".into(),
                index: None,
                input_delta: Some("{\"path\":\"README.md\"}".into()),
            })
            .is_none());
        let block = acc
            .process(StreamDelta::Finished(
                deepcode_core::provider::traits::FinishReason::ToolCalls,
            ))
            .unwrap();

        assert!(matches!(
            block,
            ResponseContentBlock::ToolUse { id, name, input }
                if id == "call_1" && name == "read_file" && input["path"] == "README.md"
        ));
    }

    #[test]
    fn indexed_parallel_tool_inputs_are_associated_and_ordered() {
        let mut acc = StreamAccumulator::new();
        acc.process(StreamDelta::Batch(vec![
            StreamDelta::ToolUseStart {
                id: "call_1".into(),
                name: "read_file".into(),
                index: Some(0),
                input_delta: None,
            },
            StreamDelta::ToolUseStart {
                id: "call_2".into(),
                name: "grep".into(),
                index: Some(1),
                input_delta: None,
            },
        ]));
        acc.process(StreamDelta::ToolUseInput {
            id: String::new(),
            index: Some(1),
            input_delta: "{\"pattern\":\"DeepCode\"}".into(),
        });
        acc.process(StreamDelta::ToolUseInput {
            id: String::new(),
            index: Some(0),
            input_delta: "{\"path\":\"README.md\"}".into(),
        });
        let first = acc
            .process(StreamDelta::Finished(
                deepcode_core::provider::traits::FinishReason::ToolCalls,
            ))
            .unwrap();
        let second = acc.flush().unwrap();

        assert!(matches!(
            first,
            ResponseContentBlock::ToolUse { id, input, .. }
                if id == "call_1" && input["path"] == "README.md"
        ));
        assert!(matches!(
            second,
            ResponseContentBlock::ToolUse { id, input, .. }
                if id == "call_2" && input["pattern"] == "DeepCode"
        ));
    }
}
