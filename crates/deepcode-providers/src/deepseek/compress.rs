use crate::compress_common;
use async_trait::async_trait;
use deepcode_core::error::Result;
use deepcode_core::provider::traits::ContextCompressor;
use deepcode_core::types::{ContentBlock, Message};

pub(crate) struct DeepSeekContextCompressor;

#[async_trait]
impl ContextCompressor for DeepSeekContextCompressor {
    fn needs_compression(&self, token_count: usize, context_window: usize) -> bool {
        token_count as f64 > context_window as f64 * 0.80
    }

    fn estimate_tokens(&self, messages: &[Message]) -> usize {
        // DeepSeek uses a similar tokenizer to OpenAI but optimized for Chinese.
        // For Chinese text, ~2 chars per token; for English, ~4 chars per token.
        messages
            .iter()
            .map(|m| {
                m.content
                    .iter()
                    .map(|b| match b {
                        ContentBlock::Text { text } => {
                            let chars = text.chars().count();
                            let cjk_chars = text.chars().filter(|c| is_cjk(*c)).count();
                            let ascii_chars = chars - cjk_chars;
                            // CJK: ~2 chars/token, ASCII: ~4 chars/token
                            (cjk_chars / 2) + (ascii_chars / 4)
                        }
                        ContentBlock::Reasoning { text, .. } => {
                            let chars = text.chars().count();
                            let cjk_chars = text.chars().filter(|c| is_cjk(*c)).count();
                            let ascii_chars = chars - cjk_chars;
                            (cjk_chars / 2) + (ascii_chars / 4)
                        }
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

fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}'  // CJK Unified Ideographs Extension A
        | '\u{3000}'..='\u{303F}'  // CJK Symbols and Punctuation
        | '\u{FF00}'..='\u{FFEF}'  // Halfwidth and Fullwidth Forms
        | '\u{3040}'..='\u{309F}'  // Hiragana
        | '\u{30A0}'..='\u{30FF}'  // Katakana
        | '\u{AC00}'..='\u{D7AF}'  // Hangul Syllables
    )
}
