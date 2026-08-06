use std::borrow::Cow;

use deepcode_core::types::{ContentBlock, Message, Role};

/// DeepSeek only requires reasoning replay for user turns that invoked a tool.
pub(crate) fn reasoning_replay_mask(messages: &[Message]) -> Vec<bool> {
    let mut replay = vec![false; messages.len()];
    let mut turn_start = 0;

    for turn_end in 1..=messages.len() {
        if turn_end == messages.len() || messages[turn_end].role == Role::User {
            let has_tool_call = messages[turn_start..turn_end].iter().any(|message| {
                message
                    .content
                    .iter()
                    .any(|block| matches!(block, ContentBlock::ToolUse { .. }))
            });
            replay[turn_start..turn_end].fill(has_tool_call);
            turn_start = turn_end;
        }
    }

    replay
}

pub(crate) fn chat_context_messages(messages: &[Message]) -> Cow<'_, [Message]> {
    let replay = reasoning_replay_mask(messages);
    let has_redundant_reasoning = messages.iter().zip(&replay).any(|(message, replay)| {
        !replay
            && message
                .content
                .iter()
                .any(|block| matches!(block, ContentBlock::Reasoning { .. }))
    });
    if !has_redundant_reasoning {
        return Cow::Borrowed(messages);
    }

    let mut filtered = messages.to_vec();
    remove_redundant_reasoning(&mut filtered, replay);
    Cow::Owned(filtered)
}

pub(crate) fn normalize_history(messages: &mut [Message]) {
    let replay = reasoning_replay_mask(messages);
    remove_redundant_reasoning(messages, replay);
}

fn remove_redundant_reasoning(messages: &mut [Message], replay: Vec<bool>) {
    for (message, replay) in messages.iter_mut().zip(replay) {
        if !replay {
            message
                .content
                .retain(|block| !matches!(block, ContentBlock::Reasoning { .. }));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_reasoning(message: &Message) -> bool {
        message
            .content
            .iter()
            .any(|block| matches!(block, ContentBlock::Reasoning { .. }))
    }

    #[test]
    fn removes_reasoning_from_turn_without_tool_calls_without_mutating_history() {
        let messages = vec![
            Message::user("question"),
            Message::assistant(vec![
                ContentBlock::reasoning("hidden chain"),
                ContentBlock::text("answer"),
            ]),
        ];

        let filtered = chat_context_messages(&messages);

        assert!(!has_reasoning(&filtered[1]));
        assert!(has_reasoning(&messages[1]));
    }

    #[test]
    fn keeps_all_reasoning_in_a_turn_that_used_tools() {
        let messages = vec![
            Message::user("question"),
            Message::assistant(vec![
                ContentBlock::reasoning("tool reasoning"),
                ContentBlock::tool_use("call_1", "read_file", serde_json::json!({})),
            ]),
            Message::tool_result("call_1", "result", false),
            Message::assistant(vec![
                ContentBlock::reasoning("final reasoning"),
                ContentBlock::text("answer"),
            ]),
            Message::user("next question"),
            Message::assistant(vec![
                ContentBlock::reasoning("redundant reasoning"),
                ContentBlock::text("next answer"),
            ]),
        ];

        let filtered = chat_context_messages(&messages);

        assert!(has_reasoning(&filtered[1]));
        assert!(has_reasoning(&filtered[3]));
        assert!(!has_reasoning(&filtered[5]));
    }

    #[test]
    fn normalizes_canonical_history_in_place() {
        let mut messages = vec![
            Message::user("question"),
            Message::assistant(vec![
                ContentBlock::reasoning("hidden chain"),
                ContentBlock::text("answer"),
            ]),
        ];

        normalize_history(&mut messages);

        assert!(!has_reasoning(&messages[1]));
    }
}
