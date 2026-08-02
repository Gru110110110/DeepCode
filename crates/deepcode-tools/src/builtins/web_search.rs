use crate::tool::{Tool, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};
use std::time::Duration;

#[derive(Debug, Clone)]
pub(super) struct WebSearchTool;

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the web using DuckDuckGo and return the top results. \
         Each result includes title, URL, and a short snippet."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5, max: 10)"
                }
            },
            "required": ["query"]
        })
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::NETWORK
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let query = input["query"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'query' parameter".to_string(),
            })?;
        let num_results = input["num_results"].as_u64().unwrap_or(5).clamp(1, 10) as usize;

        let url =
            reqwest::Url::parse_with_params("https://html.duckduckgo.com/html/", &[("q", query)])
                .map_err(|e| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Failed to build search URL: {}", e),
            })?;

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
            .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
            .send()
            .await
            .map_err(|e| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Failed to perform search: {}", e),
            })?;

        if !resp.status().is_success() {
            return Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Search returned HTTP status {}", resp.status()),
            });
        }

        let body = resp
            .text()
            .await
            .map_err(|e| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Failed to read search response: {}", e),
            })?;

        let results = parse_duckduckgo_results(&body, num_results);

        if results.is_empty() {
            Ok("No results found.".to_string())
        } else {
            Ok(results.join("\n\n"))
        }
    }
}

fn parse_duckduckgo_results(html: &str, limit: usize) -> Vec<String> {
    let mut results = Vec::new();
    let mut html = html;

    while let Some(result_start) = html.find("class=\"result\"") {
        html = &html[result_start..];

        // Extract title and URL
        let title = extract_between(html, "class=\"result__a\"", "</a>");
        let url = extract_href(html, "class=\"result__a\"");
        let snippet = extract_between(html, "class=\"result__snippet\"", "</a>");

        let title = strip_tags(title.unwrap_or("No title"));
        let snippet = strip_tags(snippet.unwrap_or("No description"));
        let url = url.unwrap_or("");

        // DuckDuckGo redirects URLs through their own domain; extract the real URL
        let real_url = extract_ddgo_url(url);

        results.push(format!(
            "Title: {}\nURL: {}\nSnippet: {}",
            title.trim(),
            real_url,
            snippet.trim()
        ));

        if results.len() >= limit {
            break;
        }

        // Move past this result
        if let Some(end) = html.find("class=\"result\"") {
            if end > 0 {
                html = &html[end + 1..];
            } else {
                break;
            }
        } else {
            break;
        }
    }

    results
}

fn extract_between<'a>(text: &'a str, start_marker: &str, end_marker: &str) -> Option<&'a str> {
    let start = text.find(start_marker)?;
    let rest = &text[start + start_marker.len()..];
    let gt = rest.find('>')?;
    let rest = &rest[gt + 1..];
    let end = rest.find(end_marker)?;
    Some(&rest[..end])
}

fn extract_href<'a>(text: &'a str, start_marker: &str) -> Option<&'a str> {
    let start = text.find(start_marker)?;
    let rest = &text[start..];
    let href_start = rest.find("href=\"")?;
    let rest = &rest[href_start + 6..];
    let href_end = rest.find('"')?;
    Some(&rest[..href_end])
}

fn strip_tags(html: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(ch);
        }
    }
    result
}

fn extract_ddgo_url(url: &str) -> String {
    if url.starts_with("//duckduckgo.com/l/?") {
        if let Some(ud) = url.find("uddg=") {
            let encoded = &url[ud + 5..];
            return percent_decode(encoded).unwrap_or_else(|| url.to_string());
        }
    }
    url.to_string()
}

fn percent_decode(s: &str) -> Option<String> {
    let mut result = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
            let byte = u8::from_str_radix(hex, 16).ok()?;
            result.push(byte);
            i += 3;
        } else if bytes[i] == b'+' {
            result.push(b' ');
            i += 1;
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(result).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percent_decode_preserves_utf8() {
        assert_eq!(
            percent_decode("%E4%B8%AD%E6%96%87").as_deref(),
            Some("中文")
        );
    }
}
