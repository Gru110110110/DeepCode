use std::pin::Pin;

use deepcode_core::config::ProviderConfig;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_core::provider::traits::{ResponseParser, StreamDelta};
use futures::stream::Stream;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;

pub(crate) type Header = (&'static str, String);
pub(crate) type SseLineStream =
    Pin<Box<dyn Stream<Item = std::result::Result<String, DeepCodeError>> + Send>>;
pub(crate) type DeltaStream = Pin<Box<dyn Stream<Item = Result<StreamDelta>> + Send>>;

#[derive(Clone)]
pub(crate) struct RequestLimiter {
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl RequestLimiter {
    pub(crate) fn from_config(config: &ProviderConfig) -> Self {
        let limit = config
            .max_concurrent_requests
            .unwrap_or(tokio::sync::Semaphore::MAX_PERMITS)
            .clamp(1, tokio::sync::Semaphore::MAX_PERMITS);
        Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(limit)),
        }
    }

    pub(crate) async fn acquire(&self) -> Result<tokio::sync::OwnedSemaphorePermit> {
        self.semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| DeepCodeError::Http("Provider request limiter closed".to_string()))
    }
}

pub(crate) fn build_client(config: &ProviderConfig) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(
            config.request_timeout_secs.unwrap_or(300),
        ))
        .build()
        .map_err(|e| DeepCodeError::Http(format!("Failed to build HTTP client: {}", e)))
}

pub(crate) fn hold_permit(
    stream: DeltaStream,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> DeltaStream {
    Box::pin(stream.map(move |item| {
        let _permit = &permit;
        item
    }))
}

pub(crate) async fn send_json_request(
    client: &reqwest::Client,
    url: String,
    headers: Vec<Header>,
    body: &serde_json::Value,
) -> Result<serde_json::Value> {
    let resp = build_json_request(client, url, headers, body)
        .send()
        .await
        .map_err(|e| DeepCodeError::Http(e.to_string()))?;

    response_json(resp).await
}

pub(crate) async fn send_sse_request(
    client: &reqwest::Client,
    url: String,
    headers: Vec<Header>,
    body: &serde_json::Value,
) -> Result<SseLineStream> {
    let resp = build_json_request(client, url, headers, body)
        .send()
        .await
        .map_err(|e| DeepCodeError::Http(e.to_string()))?;

    if !resp.status().is_success() {
        return Err(provider_error(resp).await);
    }

    let stream = resp.bytes_stream().map(|chunk| {
        chunk
            .map(|bytes| bytes.to_vec())
            .map_err(|e| DeepCodeError::Http(e.to_string()))
    });

    Ok(Box::pin(crate::sse::lines_from_byte_chunks(stream)))
}

pub(crate) fn parse_sse_lines<P>(raw_stream: SseLineStream, parser: P) -> DeltaStream
where
    P: ResponseParser + Copy + Send + Sync + 'static,
{
    let parsed = raw_stream.filter_map(move |line_result| {
        let item = match line_result {
            Ok(line) => match parser.parse_stream_chunk(&line) {
                Ok(Some(delta)) => Some(Ok(delta)),
                Ok(None) => None,
                Err(e) => Some(Err(e)),
            },
            Err(e) => Some(Err(e)),
        };
        futures::future::ready(item)
    });

    Box::pin(parsed)
}

fn build_json_request(
    client: &reqwest::Client,
    url: String,
    headers: Vec<Header>,
    body: &serde_json::Value,
) -> reqwest::RequestBuilder {
    headers.into_iter().fold(
        client
            .post(url)
            .header("Content-Type", "application/json")
            .json(body),
        |request, (name, value)| request.header(name, value),
    )
}

async fn response_json(resp: reqwest::Response) -> Result<serde_json::Value> {
    let status = resp.status();
    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| DeepCodeError::Http(e.to_string()))?;

    if !status.is_success() {
        return Err(DeepCodeError::Provider(provider_error_message(&json)));
    }

    Ok(json)
}

async fn provider_error(resp: reqwest::Response) -> DeepCodeError {
    match resp.json::<serde_json::Value>().await {
        Ok(json) => DeepCodeError::Provider(provider_error_message(&json)),
        Err(e) => DeepCodeError::Http(e.to_string()),
    }
}

fn provider_error_message(json: &serde_json::Value) -> String {
    json["error"]["message"]
        .as_str()
        .or_else(|| json["error"].as_str())
        .or_else(|| json["message"].as_str())
        .unwrap_or("Unknown error")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_error_message_accepts_object_string_and_message_shapes() {
        assert_eq!(
            provider_error_message(&serde_json::json!({"error": {"message": "object"}})),
            "object"
        );
        assert_eq!(
            provider_error_message(&serde_json::json!({"error": "string"})),
            "string"
        );
        assert_eq!(
            provider_error_message(&serde_json::json!({"message": "top"})),
            "top"
        );
    }
}
