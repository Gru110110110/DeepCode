use crate::tool::{Tool, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};
use std::time::Duration;

#[derive(Debug, Clone)]
pub(super) struct WebFetchTool;

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch content from a URL and return it as text. \
         Use for reading documentation, API responses, or web pages."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "description": "The URL to fetch"
                }
            },
            "required": ["url"]
        })
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::NETWORK
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let url = input["url"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'url' parameter".to_string(),
            })?;

        // Basic URL validation
        if !url.starts_with("http://") && !url.starts_with("https://") {
            return Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "URL must start with http:// or https://".to_string(),
            });
        }

        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(60))
            .build()
            .map_err(|e| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Failed to build HTTP client: {}", e),
            })?;
        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Failed to fetch URL: {}", e),
            })?;

        let status = resp.status();
        if status.is_redirection() {
            let location = resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|value| value.to_str().ok())
                .unwrap_or("<missing Location header>");
            return Ok(format!(
                "Status: {}\nRedirect not followed. Submit a new web_fetch request for: {}",
                status, location
            ));
        }

        let body = resp
            .text()
            .await
            .map_err(|e| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Failed to read response body: {}", e),
            })?;

        Ok(crate::execution::truncate_output(
            &format!("Status: {}\n\n{}", status, body),
            100_000,
        ))
    }
}
