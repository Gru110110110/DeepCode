use crate::compress_common;
use async_trait::async_trait;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::ContextCompressor;
use deepcode_core::types::{ContentBlock, Message};

pub(crate) struct AnthropicContextCompressor;

#[async_trait]
impl ContextCompressor for AnthropicContextCompressor {
    fn needs_compression(&self, token_count: usize, context_window: usize) -> bool {
        token_count as f64 > context_window as f64 * 0.80
    }

    fn estimate_tokens(&self, messages: &[Message]) -> usize {
        // Anthropic uses a byte-pair encoding tokenizer (~3.5 chars/token)
        messages
            .iter()
            .map(|m| {
                m.content
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => {
                            // Approximate Anthropic tokenizer: ~3.5 chars/token
                            ((text.len() as f64) / 3.5) as usize
                        }
                        ContentBlock::Reasoning { text, .. } => {
                            ((text.len() as f64) / 3.5) as usize
                        }
                        ContentBlock::ToolUse { input, .. } => {
                            50 + serde_json::to_string(input).unwrap_or_default().len() / 3
                        }
                        ContentBlock::ToolResult { content, .. } => {
                            ((content.len() as f64) / 3.5) as usize
                        }
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
