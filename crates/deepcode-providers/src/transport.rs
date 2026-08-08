use std::pin::Pin;

use deepcode_core::config::ProviderConfig;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_core::provider::traits::{ResponseParser, StreamDelta};
use futures::stream::Stream;
use futures::StreamExt;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

const DEFAULT_RETRY_BACKOFF: Duration = Duration::from_millis(500);
const MAX_RETRY_BACKOFF: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RetryPolicy {
    max_retries: usize,
}

impl RetryPolicy {
    pub(crate) const fn new(max_retries: usize) -> Self {
        Self { max_retries }
    }
}

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
    send_json_request_with_retry(client, url, headers, body, RetryPolicy::default()).await
}

pub(crate) async fn send_json_request_with_retry(
    client: &reqwest::Client,
    url: String,
    headers: Vec<Header>,
    body: &serde_json::Value,
    retry_policy: RetryPolicy,
) -> Result<serde_json::Value> {
    let mut retries = 0;
    loop {
        match build_json_request(client, &url, &headers, body)
            .send()
            .await
        {
            Ok(resp)
                if should_retry_status(resp.status()) && retries < retry_policy.max_retries =>
            {
                let delay = retry_delay(&resp, retries);
                retries += 1;
                tokio::time::sleep(delay).await;
            }
            Ok(resp) => return response_json(resp).await,
            Err(error)
                if is_retryable_request_error(&error) && retries < retry_policy.max_retries =>
            {
                let delay = exponential_backoff(retries);
                retries += 1;
                tokio::time::sleep(delay).await;
            }
            Err(error) => return Err(DeepCodeError::Http(error.to_string())),
        }
    }
}

pub(crate) async fn send_sse_request(
    client: &reqwest::Client,
    url: String,
    headers: Vec<Header>,
    body: &serde_json::Value,
) -> Result<SseLineStream> {
    send_sse_request_with_retry(client, url, headers, body, RetryPolicy::default()).await
}

pub(crate) async fn send_sse_request_with_retry(
    client: &reqwest::Client,
    url: String,
    headers: Vec<Header>,
    body: &serde_json::Value,
    retry_policy: RetryPolicy,
) -> Result<SseLineStream> {
    let mut retries = 0;
    let resp = loop {
        match build_json_request(client, &url, &headers, body)
            .send()
            .await
        {
            Ok(resp)
                if should_retry_status(resp.status()) && retries < retry_policy.max_retries =>
            {
                let delay = retry_delay(&resp, retries);
                retries += 1;
                tokio::time::sleep(delay).await;
            }
            Ok(resp) => break resp,
            Err(error)
                if is_retryable_request_error(&error) && retries < retry_policy.max_retries =>
            {
                let delay = exponential_backoff(retries);
                retries += 1;
                tokio::time::sleep(delay).await;
            }
            Err(error) => return Err(DeepCodeError::Http(error.to_string())),
        }
    };

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
    url: &str,
    headers: &[Header],
    body: &serde_json::Value,
) -> reqwest::RequestBuilder {
    headers.iter().fold(
        client
            .post(url)
            .header("Content-Type", "application/json")
            .json(body),
        |request, (name, value)| request.header(*name, value),
    )
}

fn should_retry_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn is_retryable_request_error(error: &reqwest::Error) -> bool {
    error.is_connect()
        || error.is_timeout()
        || (error.is_request() && !error.is_builder() && !error.is_redirect())
}

fn retry_delay(resp: &reqwest::Response, retry_index: usize) -> Duration {
    resp.headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| parse_retry_after(value, SystemTime::now()))
        .unwrap_or_else(|| exponential_backoff(retry_index))
}

fn parse_retry_after(value: &str, now: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let retry_at = chrono::DateTime::parse_from_rfc2822(value)
        .ok()?
        .with_timezone(&chrono::Utc);
    let now = chrono::DateTime::<chrono::Utc>::from(now);
    Some((retry_at - now).to_std().unwrap_or(Duration::ZERO))
}

fn exponential_backoff(retry_index: usize) -> Duration {
    let multiplier = 1_u32.checked_shl(retry_index as u32).unwrap_or(u32::MAX);
    DEFAULT_RETRY_BACKOFF
        .saturating_mul(multiplier)
        .min(MAX_RETRY_BACKOFF)
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

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

    #[test]
    fn anthropic_retry_statuses_are_transient() {
        assert!(should_retry_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(should_retry_status(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(should_retry_status(
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(should_retry_status(
            reqwest::StatusCode::from_u16(529).unwrap()
        ));
        assert!(!should_retry_status(reqwest::StatusCode::BAD_REQUEST));
        assert!(!should_retry_status(reqwest::StatusCode::UNAUTHORIZED));
    }

    #[test]
    fn retry_backoff_is_exponential() {
        assert_eq!(exponential_backoff(0), Duration::from_millis(500));
        assert_eq!(exponential_backoff(1), Duration::from_secs(1));
        assert_eq!(exponential_backoff(2), Duration::from_secs(2));
        assert_eq!(exponential_backoff(32), MAX_RETRY_BACKOFF);
        assert_eq!(exponential_backoff(usize::MAX), MAX_RETRY_BACKOFF);
    }

    #[test]
    fn retry_after_accepts_seconds_and_http_date() {
        let now = chrono::DateTime::parse_from_rfc2822("Wed, 21 Oct 2026 07:27:30 GMT")
            .unwrap()
            .with_timezone(&chrono::Utc);
        assert_eq!(
            parse_retry_after("12", SystemTime::from(now)),
            Some(Duration::from_secs(12))
        );
        assert_eq!(
            parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT", SystemTime::from(now)),
            Some(Duration::from_secs(30))
        );
        assert_eq!(
            parse_retry_after("Wed, 21 Oct 2026 07:27:00 GMT", SystemTime::from(now)),
            Some(Duration::ZERO)
        );
    }

    #[tokio::test]
    async fn json_request_retries_twice_and_honors_retry_after() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for attempt in 0..3 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = vec![0_u8; 4096];
                let _ = socket.read(&mut request).await.unwrap();
                let (status, body, retry_after) = if attempt < 2 {
                    (
                        "529 Overloaded",
                        r#"{"error":{"message":"busy"}}"#,
                        "Retry-After: 0\r\n",
                    )
                } else {
                    ("200 OK", r#"{"ok":true}"#, "")
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\n{retry_after}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let response = send_json_request_with_retry(
            &reqwest::Client::new(),
            format!("http://{address}/messages"),
            Vec::new(),
            &serde_json::json!({"messages": []}),
            RetryPolicy::new(2),
        )
        .await
        .unwrap();

        assert_eq!(response["ok"], true);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn sse_request_retries_before_streaming_starts() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = vec![0_u8; 4096];
                let _ = socket.read(&mut request).await.unwrap();
                let (status, content_type, body, retry_after) = if attempt == 0 {
                    (
                        "429 Too Many Requests",
                        "application/json",
                        r#"{"error":{"message":"slow down"}}"#,
                        "Retry-After: 0\r\n",
                    )
                } else {
                    (
                        "200 OK",
                        "text/event-stream",
                        "data: {\"type\":\"message_stop\"}\n\n",
                        "",
                    )
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\n{retry_after}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let lines = send_sse_request_with_retry(
            &reqwest::Client::new(),
            format!("http://{address}/messages"),
            Vec::new(),
            &serde_json::json!({"stream": true}),
            RetryPolicy::new(2),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

        assert_eq!(
            lines.into_iter().collect::<Result<Vec<_>>>().unwrap(),
            vec![r#"data: {"type":"message_stop"}"#, ""]
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn request_retries_when_connection_closes_before_response() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = vec![0_u8; 4096];
                let _ = socket.read(&mut request).await.unwrap();
                if attempt == 0 {
                    drop(socket);
                    continue;
                }
                let body = r#"{"ok":true}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let response = send_json_request_with_retry(
            &reqwest::Client::new(),
            format!("http://{address}/messages"),
            Vec::new(),
            &serde_json::json!({"messages": []}),
            RetryPolicy::new(1),
        )
        .await
        .unwrap();

        assert_eq!(response["ok"], true);
        server.await.unwrap();
    }
}
