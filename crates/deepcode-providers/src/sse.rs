use deepcode_core::error::DeepCodeError;
use futures::stream::{self, Stream, StreamExt};

pub(crate) fn lines_from_byte_chunks<S>(
    chunks: S,
) -> impl Stream<Item = std::result::Result<String, DeepCodeError>>
where
    S: Stream<Item = std::result::Result<Vec<u8>, DeepCodeError>>,
{
    chunks
        .scan(Vec::new(), |buffer, chunk| {
            let mut out = Vec::new();
            match chunk {
                Ok(bytes) => {
                    buffer.extend_from_slice(&bytes);
                    while let Some(pos) = buffer.iter().position(|byte| *byte == b'\n') {
                        let mut line = buffer.drain(..=pos).collect::<Vec<_>>();
                        line.pop();
                        if line.last() == Some(&b'\r') {
                            line.pop();
                        }
                        out.push(String::from_utf8(line).map_err(|error| {
                            DeepCodeError::Parse(format!("Invalid UTF-8 in SSE stream: {}", error))
                        }));
                    }
                }
                Err(e) => out.push(Err(e)),
            }
            futures::future::ready(Some(stream::iter(out)))
        })
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;

    #[tokio::test]
    async fn lines_from_chunks_buffers_split_lines() {
        let chunks = stream::iter(vec![
            Ok(b"data: {\"a\"".to_vec()),
            Ok(b":1}\n\ndata: [DONE]\n".to_vec()),
        ]);

        let lines: Vec<String> = lines_from_byte_chunks(chunks)
            .filter_map(|item| async move { item.ok() })
            .collect()
            .await;

        assert_eq!(lines, vec!["data: {\"a\":1}", "", "data: [DONE]"]);
    }

    #[tokio::test]
    async fn lines_from_chunks_preserves_split_utf8() {
        let encoded = "data: 中文\n".as_bytes();
        let chunks = stream::iter(vec![Ok(encoded[..8].to_vec()), Ok(encoded[8..].to_vec())]);

        let lines: Vec<String> = lines_from_byte_chunks(chunks)
            .filter_map(|item| async move { item.ok() })
            .collect()
            .await;

        assert_eq!(lines, vec!["data: 中文"]);
    }
}
