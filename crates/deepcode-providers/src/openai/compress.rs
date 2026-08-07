use crate::compress_common;
use async_trait::async_trait;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::ContextCompressor;
use deepcode_core::types::{ContentBlock, Message, Role};

pub(crate) struct OpenAiContextCompressor;

#[async_trait]
impl ContextCompressor for OpenAiContextCompressor {
    fn needs_compression(&self, token_count: usize, context_window: usize) -> bool {
        token_count as f64 > context_window as f64 * 0.80
    }

    fn estimate_tokens(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| {
                // Responses history replays provider-owned output items verbatim.
                // When those items are present, the request builder intentionally
                // ignores the derived Text/Reasoning/ToolUse blocks in the same
                // assistant message, so do not double-count them here.
                if m.role == Role::Assistant {
                    let (has_provider_items, provider_tokens) = m
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::ProviderItem { provider, value }
                                if provider == "openai" =>
                            {
                                Some(value.to_string().len() / 4)
                            }
                            _ => None,
                        })
                        .fold((false, 0), |(_, total), tokens| (true, total + tokens));
                    if has_provider_items {
                        return provider_tokens;
                    }
                }

                m.content
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => text.len() / 4,
                        ContentBlock::Reasoning { text, .. } => text.len() / 4,
                        ContentBlock::ToolUse { input, .. } => {
                            50 + serde_json::to_string(input).unwrap_or_default().len() / 4
                        }
                        ContentBlock::ToolResult { content, .. } => content.len() / 4,
                        ContentBlock::Image { .. } => 1_600,
                        ContentBlock::Audio { source, .. } | ContentBlock::File { source, .. } => {
                            serde_json::to_string(source).map_or(0, |value| value.len() / 4)
                        }
                        ContentBlock::ProviderItem { value, .. } => value.to_string().len() / 4,
                    })
                    .sum::<usize>()
            })
            .sum()
    }

    async fn compress(
        &self,
        messages: &[Message],
        current_tokens: usize,
        target_tokens: usize,
    ) -> Result<(Vec<Message>, usize)> {
        compress_common::compress(messages, current_tokens, target_tokens, |messages| {
            self.estimate_tokens(messages)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_items_replace_derived_assistant_blocks_in_estimate() {
        let compressor = OpenAiContextCompressor;
        let provider_item = ContentBlock::provider_item(
            "openai",
            serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}]
            }),
        );
        let provider_only = vec![Message::assistant(vec![provider_item.clone()])];
        let with_derived = vec![Message::assistant(vec![
            ContentBlock::text("answer"),
            provider_item,
        ])];

        assert_eq!(
            compressor.estimate_tokens(&with_derived),
            compressor.estimate_tokens(&provider_only)
        );
    }

    #[test]
    fn ordinary_chat_assistant_blocks_are_still_counted() {
        let compressor = OpenAiContextCompressor;
        let empty = vec![Message::assistant(Vec::new())];
        let chat = vec![Message::assistant(vec![ContentBlock::text("a".repeat(40))])];

        assert_eq!(
            compressor.estimate_tokens(&chat),
            compressor.estimate_tokens(&empty) + 10
        );
    }
}
