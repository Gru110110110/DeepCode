use crate::compress_common;
use async_trait::async_trait;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::ContextCompressor;
use deepcode_core::types::{ContentBlock, Message};

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
