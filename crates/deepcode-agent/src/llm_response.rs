use std::pin::Pin;

use deepcode_core::error::Result as CoreResult;
use deepcode_core::provider::traits::{FinishReason, ResponseContentBlock, StreamDelta};
use deepcode_core::types::ContentBlock;
use futures::{Stream, StreamExt};

use crate::event::{AgentEvent, CmdReceiver, EventSender};
use crate::r#loop::{handle_busy_command, BusyControl};
use crate::stream::StreamAccumulator;

pub(crate) const SESSION_TITLE_INSTRUCTION: &str = "For this response only, begin the final answer with exactly <session-title>concise title</session-title> on its own line, then continue with the normal answer. Use the user's language, describe the task rather than the answer, keep the title under 60 characters, and do not mention this metadata line.";
const SESSION_TITLE_OPEN: &str = "<session-title>";
const SESSION_TITLE_CLOSE: &str = "</session-title>";
const SESSION_TITLE_BUFFER_LIMIT: usize = 512;

pub(crate) type LlmStream = Pin<Box<dyn Stream<Item = CoreResult<StreamDelta>> + Send>>;
pub(crate) type ToolCall = (String, String, serde_json::Value);

pub(crate) struct StreamedResponse {
    pub response_blocks: Vec<ResponseContentBlock>,
    pub finish_reason: Option<FinishReason>,
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub cached_input_tokens: usize,
    pub cache_miss_input_tokens: usize,
    pub reasoning_output_tokens: usize,
    pub session_title_resolved: bool,
}

pub(crate) enum StreamResponseOutcome {
    Completed(StreamedResponse),
    Interrupted,
    Shutdown,
    Failed { error: String },
}

pub(crate) async fn collect_stream_response(
    mut stream: LlmStream,
    cmd_rx: &mut CmdReceiver,
    event_tx: &EventSender,
    capture_session_title: bool,
) -> StreamResponseOutcome {
    let mut accumulator = StreamAccumulator::new();
    let mut title_filter = SessionTitleFilter::new(capture_session_title);
    let mut response_blocks = Vec::new();
    let mut input_tokens = 0usize;
    let mut output_tokens = 0usize;
    let mut cached_input_tokens = 0usize;
    let mut cache_miss_input_tokens = 0usize;
    let mut reasoning_output_tokens = 0usize;
    let mut finish_reason = None;

    loop {
        let delta_result = tokio::select! {
            delta = stream.next() => delta,
            cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                BusyControl::Continue => continue,
                BusyControl::Interrupt => return StreamResponseOutcome::Interrupted,
                BusyControl::Shutdown => return StreamResponseOutcome::Shutdown,
            }
        };
        let Some(delta_result) = delta_result else {
            break;
        };

        match delta_result {
            Ok(delta) => {
                for delta in flatten_delta(delta) {
                    let delta = match delta {
                        StreamDelta::TextDelta(text) => {
                            let (visible, generated_title) = title_filter.push(&text);
                            if let Some(title) = generated_title {
                                let _ = event_tx.send(AgentEvent::SessionTitleGenerated {
                                    title: title.clone(),
                                });
                            }
                            let Some(visible) = visible else {
                                continue;
                            };
                            let _ = event_tx.send(AgentEvent::TextDelta(visible.clone()));
                            StreamDelta::TextDelta(visible)
                        }
                        StreamDelta::ReasoningDelta(text) => {
                            let _ = event_tx.send(AgentEvent::ReasoningDelta(text.clone()));
                            StreamDelta::ReasoningDelta(text)
                        }
                        StreamDelta::Usage {
                            input_tokens: it,
                            output_tokens: ot,
                            cached_input_tokens: cached,
                            cache_miss_input_tokens: missed,
                            reasoning_output_tokens: reasoning,
                        } => {
                            input_tokens += it;
                            output_tokens += ot;
                            cached_input_tokens += cached;
                            cache_miss_input_tokens += missed;
                            reasoning_output_tokens += reasoning;
                            continue;
                        }
                        other => other,
                    };
                    if let StreamDelta::Finished(reason) = &delta {
                        preserve_finish_reason(&mut finish_reason, reason.clone());
                    }
                    if let Some(block) = accumulator.process(delta) {
                        response_blocks.push(block);
                    }
                }
            }
            Err(e) => {
                let error = e.to_string();
                let _ = event_tx.send(AgentEvent::AgentError {
                    message: error.clone(),
                });
                return StreamResponseOutcome::Failed { error };
            }
        }
    }

    if let Some((visible, generated_title)) = title_filter.finish() {
        if let Some(title) = generated_title {
            let _ = event_tx.send(AgentEvent::SessionTitleGenerated {
                title: title.clone(),
            });
        }
        if let Some(visible) = visible {
            let _ = event_tx.send(AgentEvent::TextDelta(visible.clone()));
            let _ = accumulator.process(StreamDelta::TextDelta(visible));
        }
    }

    while let Some(block) = accumulator.flush() {
        response_blocks.push(block);
    }

    StreamResponseOutcome::Completed(StreamedResponse {
        response_blocks,
        finish_reason,
        input_tokens,
        output_tokens,
        cached_input_tokens,
        cache_miss_input_tokens,
        reasoning_output_tokens,
        session_title_resolved: !capture_session_title || title_filter.decided,
    })
}

fn preserve_finish_reason(current: &mut Option<FinishReason>, incoming: FinishReason) {
    if current
        .as_ref()
        .is_some_and(|reason| *reason != FinishReason::Stop && incoming == FinishReason::Stop)
    {
        return;
    }
    *current = Some(incoming);
}

fn flatten_delta(delta: StreamDelta) -> Vec<StreamDelta> {
    match delta {
        StreamDelta::Batch(deltas) => deltas.into_iter().flat_map(flatten_delta).collect(),
        other => vec![other],
    }
}

struct SessionTitleFilter {
    decided: bool,
    buffer: String,
    title: Option<String>,
}

impl SessionTitleFilter {
    fn new(enabled: bool) -> Self {
        Self {
            decided: !enabled,
            buffer: String::new(),
            title: None,
        }
    }

    fn push(&mut self, text: &str) -> (Option<String>, Option<String>) {
        if self.decided {
            return (nonempty(text.to_string()), None);
        }
        self.buffer.push_str(text);
        if !SESSION_TITLE_OPEN.starts_with(&self.buffer)
            && !self.buffer.starts_with(SESSION_TITLE_OPEN)
        {
            self.decided = true;
            return (nonempty(std::mem::take(&mut self.buffer)), None);
        }
        if let Some(newline) = self.buffer.find('\n') {
            let remainder = self.buffer[(newline + 1)..].to_string();
            let first_line = self.buffer[..newline].trim_end_matches('\r').to_string();
            self.buffer.clear();
            self.decided = true;
            if let Some(title) = parse_session_title(&first_line) {
                self.title = Some(title.clone());
                return (nonempty(remainder), Some(title));
            }
            return (nonempty(format!("{}\n{}", first_line, remainder)), None);
        }
        if self.buffer.len() > SESSION_TITLE_BUFFER_LIMIT {
            self.decided = true;
            return (nonempty(std::mem::take(&mut self.buffer)), None);
        }
        (None, None)
    }

    fn finish(&mut self) -> Option<(Option<String>, Option<String>)> {
        if self.decided || self.buffer.is_empty() {
            return None;
        }
        self.decided = true;
        let buffered = std::mem::take(&mut self.buffer);
        if let Some(title) = parse_session_title(buffered.trim_end_matches('\r')) {
            self.title = Some(title.clone());
            Some((None, Some(title)))
        } else {
            Some((Some(buffered), None))
        }
    }
}

fn parse_session_title(line: &str) -> Option<String> {
    let title = line
        .trim()
        .strip_prefix(SESSION_TITLE_OPEN)?
        .strip_suffix(SESSION_TITLE_CLOSE)?
        .trim();
    (!title.is_empty()).then(|| title.to_string())
}

fn nonempty(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

pub(crate) fn response_blocks_to_content(blocks: &[ResponseContentBlock]) -> Vec<ContentBlock> {
    blocks
        .iter()
        .map(|block| match block {
            ResponseContentBlock::Text(text) => ContentBlock::text(text),
            ResponseContentBlock::Reasoning { text, metadata } => metadata
                .as_ref()
                .map(|metadata| ContentBlock::reasoning_with_metadata(text, metadata.clone()))
                .unwrap_or_else(|| ContentBlock::reasoning(text)),
            ResponseContentBlock::ToolUse { id, name, input } => {
                ContentBlock::tool_use(id, name, input.clone())
            }
            ResponseContentBlock::ProviderItem { provider, value } => {
                ContentBlock::provider_item(provider, value.clone())
            }
        })
        .collect()
}

pub(crate) fn tool_calls_from_blocks(blocks: &[ResponseContentBlock]) -> Vec<ToolCall> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ResponseContentBlock::ToolUse { id, name, input } => {
                Some((id.clone(), name.clone(), input.clone()))
            }
            _ => None,
        })
        .collect()
}

pub(crate) fn final_text_from_blocks(blocks: &[ResponseContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            ResponseContentBlock::Text(text) => Some(text.clone()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event;

    #[tokio::test]
    async fn reasoning_deltas_are_forwarded_to_the_ui() {
        let stream: LlmStream = Box::pin(futures::stream::iter(vec![
            Ok(StreamDelta::ReasoningDelta("inspect".to_string())),
            Ok(StreamDelta::TextDelta("done".to_string())),
        ]));
        let (_cmd_tx, mut cmd_rx) = event::cmd_channel(1);
        let (event_tx, mut event_rx) = event::event_channel();

        let result = collect_stream_response(stream, &mut cmd_rx, &event_tx, false).await;

        assert!(matches!(result, StreamResponseOutcome::Completed(_)));
        assert!(matches!(
            event_rx.try_recv(),
            Ok(AgentEvent::ReasoningDelta(text)) if text == "inspect"
        ));
        assert!(matches!(
            event_rx.try_recv(),
            Ok(AgentEvent::TextDelta(text)) if text == "done"
        ));
    }

    #[tokio::test]
    async fn content_filter_finish_reason_survives_trailing_stop_event() {
        let stream: LlmStream = Box::pin(futures::stream::iter(vec![
            Ok(StreamDelta::TextDelta("refused".to_string())),
            Ok(StreamDelta::Finished(FinishReason::ContentFilter)),
            Ok(StreamDelta::Finished(FinishReason::Stop)),
        ]));
        let (_cmd_tx, mut cmd_rx) = event::cmd_channel(1);
        let (event_tx, _event_rx) = event::event_channel();

        let result = collect_stream_response(stream, &mut cmd_rx, &event_tx, false).await;
        let StreamResponseOutcome::Completed(response) = result else {
            panic!("expected completed response");
        };
        assert_eq!(response.finish_reason, Some(FinishReason::ContentFilter));
    }

    #[tokio::test]
    async fn session_title_is_captured_without_reaching_visible_text() {
        let stream: LlmStream = Box::pin(futures::stream::iter(vec![
            Ok(StreamDelta::TextDelta("<session-".to_string())),
            Ok(StreamDelta::TextDelta(
                "title>Inspect parser</session-title>\nHere is the result.".to_string(),
            )),
        ]));
        let (_cmd_tx, mut cmd_rx) = event::cmd_channel(1);
        let (event_tx, mut event_rx) = event::event_channel();

        let result = collect_stream_response(stream, &mut cmd_rx, &event_tx, true).await;

        let StreamResponseOutcome::Completed(response) = result else {
            panic!("stream should complete");
        };
        assert!(matches!(
            event_rx.try_recv(),
            Ok(AgentEvent::SessionTitleGenerated { title }) if title == "Inspect parser"
        ));
        assert!(matches!(
            event_rx.try_recv(),
            Ok(AgentEvent::TextDelta(text)) if text == "Here is the result."
        ));
        assert_eq!(
            final_text_from_blocks(&response.response_blocks),
            "Here is the result."
        );
    }

    #[tokio::test]
    async fn response_without_title_marker_is_forwarded_unchanged() {
        let stream: LlmStream = Box::pin(futures::stream::iter(vec![Ok(StreamDelta::TextDelta(
            "Normal answer without metadata.".to_string(),
        ))]));
        let (_cmd_tx, mut cmd_rx) = event::cmd_channel(1);
        let (event_tx, mut event_rx) = event::event_channel();

        let result = collect_stream_response(stream, &mut cmd_rx, &event_tx, true).await;

        assert!(matches!(result, StreamResponseOutcome::Completed(_)));
        assert!(matches!(
            event_rx.try_recv(),
            Ok(AgentEvent::TextDelta(text)) if text == "Normal answer without metadata."
        ));
    }

    #[tokio::test]
    async fn tool_only_response_keeps_title_capture_pending() {
        let stream: LlmStream = Box::pin(futures::stream::iter(vec![
            Ok(StreamDelta::ToolUseStart {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                index: None,
                input_delta: Some("{\"path\":\"README.md\"}".to_string()),
            }),
            Ok(StreamDelta::ToolUseEnd {
                id: "call_1".to_string(),
                index: None,
            }),
        ]));
        let (_cmd_tx, mut cmd_rx) = event::cmd_channel(1);
        let (event_tx, _) = event::event_channel();

        let result = collect_stream_response(stream, &mut cmd_rx, &event_tx, true).await;

        let StreamResponseOutcome::Completed(response) = result else {
            panic!("stream should complete");
        };
        assert!(!response.session_title_resolved);
    }
}
