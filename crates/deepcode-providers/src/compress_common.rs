use deepcode_core::error::Result;
use deepcode_core::types::Message;

/// How aggressive compression should be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum CompressionLevel {
    /// No compression needed.
    None = 0,
    /// Drop tool results beyond a size threshold.
    BudgetCutting = 1,
    /// Truncate individual tool outputs to a max character limit.
    Trimming = 2,
    /// Replace verbose tool outputs with short summaries.
    MicroCompression = 3,
    /// Fold consecutive tool-call/tool-result pairs into compact summaries.
    Folding = 4,
    /// Fold verbose tool output and remove complete oldest turns.
    Aggressive = 5,
}

/// Determine compression level based on how far over budget we are.
pub(crate) fn determine_level(current_tokens: usize, context_window: usize) -> CompressionLevel {
    let ratio = current_tokens as f64 / context_window as f64;
    if ratio < 0.80 {
        CompressionLevel::None
    } else if ratio < 0.85 {
        CompressionLevel::BudgetCutting
    } else if ratio < 0.90 {
        CompressionLevel::Trimming
    } else if ratio < 0.95 {
        CompressionLevel::MicroCompression
    } else if ratio < 1.00 {
        CompressionLevel::Folding
    } else {
        CompressionLevel::Aggressive
    }
}

pub(crate) fn context_window_from_target(target_tokens: usize) -> usize {
    target_tokens.saturating_mul(5) / 3
}

pub(crate) fn compress(
    messages: &[Message],
    current_tokens: usize,
    target_tokens: usize,
    estimate: impl Fn(&[Message]) -> usize,
) -> Result<(Vec<Message>, usize)> {
    let context_window = context_window_from_target(target_tokens);
    let level = determine_level(current_tokens, context_window);
    tracing::info!(?level, current_tokens, target_tokens, "Compressing context");

    let mut result = messages.to_vec();
    match level {
        CompressionLevel::None => return Ok((result, current_tokens)),
        CompressionLevel::BudgetCutting => result = budget_cut(&result),
        CompressionLevel::Trimming => {
            result = trim_tool_results(&budget_cut(&result));
        }
        CompressionLevel::MicroCompression => {
            result = micro_compress(&budget_cut(&result));
        }
        CompressionLevel::Folding => {
            result = fold_tool_pairs(&budget_cut(&result));
        }
        CompressionLevel::Aggressive => {
            result = micro_compress(&fold_tool_pairs(&result));
        }
    }

    fit_to_target(result, target_tokens, estimate)
}

pub(crate) fn fit_to_target(
    mut messages: Vec<deepcode_core::types::Message>,
    target_tokens: usize,
    estimate: impl Fn(&[deepcode_core::types::Message]) -> usize,
) -> deepcode_core::error::Result<(Vec<deepcode_core::types::Message>, usize)> {
    use deepcode_core::types::Role;

    let mut token_count = estimate(&messages);
    while token_count > target_tokens {
        let first = usize::from(messages.first().is_some_and(|m| m.role == Role::System));
        let Some(next_turn) = messages
            .iter()
            .enumerate()
            .skip(first + 1)
            .find_map(|(index, message)| (message.role == Role::User).then_some(index))
        else {
            return Err(deepcode_core::error::DeepCodeError::ContextLimitExceeded);
        };
        messages.drain(first..next_turn);
        token_count = estimate(&messages);
    }
    Ok((messages, token_count))
}

fn prefix_at_boundary(value: &str, max_bytes: usize) -> &str {
    if value.len() <= max_bytes {
        return value;
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

/// Maximum size (chars) for a single tool result before truncation.
pub(crate) const MAX_TOOL_RESULT_CHARS: usize = 50_000;

/// Apply budget cutting: drop old tool results that exceed the max size.
pub(crate) fn budget_cut(
    messages: &[deepcode_core::types::Message],
) -> Vec<deepcode_core::types::Message> {
    use deepcode_core::types::ContentBlock;

    messages
        .iter()
        .map(|m| {
            let mut msg = m.clone();
            for block in &mut msg.content {
                if let ContentBlock::ToolResult { content, .. } = block {
                    if content.len() > MAX_TOOL_RESULT_CHARS {
                        let truncated = prefix_at_boundary(content, MAX_TOOL_RESULT_CHARS);
                        *content = format!(
                            "{}\n\n... [truncated, {} -> {} chars]",
                            truncated,
                            content.len(),
                            MAX_TOOL_RESULT_CHARS
                        );
                    }
                }
            }
            msg
        })
        .collect()
}

/// Apply trimming: further reduce each tool result to a smaller cap.
pub(crate) fn trim_tool_results(
    messages: &[deepcode_core::types::Message],
) -> Vec<deepcode_core::types::Message> {
    use deepcode_core::types::ContentBlock;
    const TRIM_CHARS: usize = 20_000;

    messages
        .iter()
        .map(|m| {
            let mut msg = m.clone();
            for block in &mut msg.content {
                if let ContentBlock::ToolResult { content, .. } = block {
                    if content.len() > TRIM_CHARS {
                        *content = format!(
                            "{}\n\n... [trimmed, {} -> {} chars]",
                            prefix_at_boundary(content, TRIM_CHARS),
                            content.len(),
                            TRIM_CHARS
                        );
                    }
                }
            }
            msg
        })
        .collect()
}

/// Micro-compress: generate compact summaries for tool results.
pub(crate) fn micro_compress(
    messages: &[deepcode_core::types::Message],
) -> Vec<deepcode_core::types::Message> {
    use deepcode_core::types::ContentBlock;

    messages
        .iter()
        .map(|m| {
            let mut msg = m.clone();
            for block in &mut msg.content {
                if let ContentBlock::ToolResult { content, .. } = block {
                    if content.len() > 5000 {
                        let line_count = content.lines().count();
                        *content = format!(
                            "[Tool output: {} lines, {} chars]\nFirst 100 lines:\n{}\n... (truncated)",
                            line_count,
                            content.len(),
                            content.lines().take(100).collect::<Vec<_>>().join("\n")
                        );
                    }
                }
            }
            msg
        })
        .collect()
}

/// Fold consecutive tool-call/tool-result pairs into compact summaries.
pub(crate) fn fold_tool_pairs(
    messages: &[deepcode_core::types::Message],
) -> Vec<deepcode_core::types::Message> {
    use deepcode_core::types::{ContentBlock, Message, Role};

    let mut result: Vec<Message> = Vec::new();
    let mut i = 0;

    while i < messages.len() {
        let msg = &messages[i];

        // Look for assistant message with tool calls followed by tool results
        if msg.role == Role::Assistant
            && msg
                .content
                .iter()
                .any(|b| matches!(b, ContentBlock::ToolUse { .. }))
        {
            let mut folded = false;
            // Collect consecutive tool result messages after this
            if i + 1 < messages.len() && messages[i + 1].role == Role::Tool {
                let mut tool_results = Vec::new();
                let mut j = i + 1;
                while j < messages.len() && messages[j].role == Role::Tool {
                    for block in &messages[j].content {
                        if let ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } = block
                        {
                            let summary = if content.len() > 200 {
                                format!("{}...", prefix_at_boundary(content, 200))
                            } else {
                                content.clone()
                            };
                            tool_results.push((tool_use_id.clone(), summary));
                        }
                    }
                    j += 1;
                }

                // Create a folded message
                let summary_lines: Vec<String> = tool_results
                    .iter()
                    .map(|(id, summary)| format!("  [{}] {}", id, summary))
                    .collect();

                result.push(Message {
                    role: Role::Assistant,
                    content: vec![ContentBlock::text(format!(
                        "[Tool calls executed. Results:\n{}\n]",
                        summary_lines.join("\n")
                    ))],
                    id: None,
                });

                i = j;
                folded = true;
            }

            if folded {
                continue;
            }
        }

        result.push(msg.clone());
        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepcode_core::types::{ContentBlock, Message};

    #[test]
    fn truncation_preserves_utf8_boundaries() {
        let content = "中".repeat(20_000);
        let messages = vec![Message::tool_result("call", &content, false)];
        let compressed = budget_cut(&messages);
        assert!(matches!(
            &compressed[0].content[0],
            ContentBlock::ToolResult { content, .. } if content.contains("truncated")
        ));
    }

    #[test]
    fn fit_to_target_removes_complete_old_turns() {
        let messages = vec![
            Message::user("old"),
            Message::assistant(vec![ContentBlock::text("answer")]),
            Message::user("new"),
            Message::assistant(vec![ContentBlock::text("latest")]),
        ];
        let (messages, _) = fit_to_target(messages, 2, |items| items.len()).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, deepcode_core::types::Role::User);
    }
}
