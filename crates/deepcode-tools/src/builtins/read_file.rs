use crate::execution::ToolExecutionConfig;
use crate::tool::{Tool, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};

#[derive(Debug, Clone)]
pub(super) struct ReadFileTool {
    execution: ToolExecutionConfig,
}

impl ReadFileTool {
    pub(super) fn new(execution: ToolExecutionConfig) -> Self {
        Self { execution }
    }
}

impl Default for ReadFileTool {
    fn default() -> Self {
        Self::new(ToolExecutionConfig::unrestricted())
    }
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file at the given path. \
         Supports optional offset and limit for pagination."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to read"
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from (1-based)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["path"]
        })
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::READ_ONLY
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let path = input["path"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'path' parameter".to_string(),
            })?;

        let resolved_path = self.execution.resolve_existing_path(self.name(), path)?;
        self.execution
            .ensure_existing_file_size(self.name(), &resolved_path)?;

        let content = tokio::fs::read_to_string(&resolved_path)
            .await
            .map_err(|e| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Cannot read {}: {}", resolved_path.display(), e),
            })?;

        let offset = input["offset"].as_u64().unwrap_or(1).saturating_sub(1) as usize;
        let limit = input["limit"].as_u64();

        let lines: Vec<&str> = content.lines().skip(offset).collect();
        let total_lines = lines.len();
        let truncated: Vec<&&str> = match limit {
            Some(max) => lines.iter().take(max as usize).collect(),
            None => lines.iter().collect(),
        };

        let mut result = String::new();
        let start_line = offset + 1;
        for (i, line) in truncated.iter().enumerate() {
            let line_num = start_line + i;
            result.push_str(&format!("{:>6}\t{}\n", line_num, line));
        }

        if let Some(max) = limit {
            if total_lines > max as usize {
                result.push_str(&format!(
                    "\n... ({} more lines, use offset={} to continue)",
                    total_lines - max as usize,
                    start_line + max as usize
                ));
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn read_file_basic() {
        let tmp = std::env::temp_dir().join(format!("deepcode_test_read_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "line1\nline2\nline3\n")
            .await
            .unwrap();

        let tool = ReadFileTool::default();
        let input = serde_json::json!({"path": tmp.to_str().unwrap()});
        let result = tool.execute(input).await.unwrap();

        assert!(result.contains("     1\tline1"));
        assert!(result.contains("     2\tline2"));
        assert!(result.contains("     3\tline3"));

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn read_file_with_offset_and_limit() {
        let tmp = std::env::temp_dir().join(format!("deepcode_test_read_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "a\nb\nc\nd\n").await.unwrap();

        let tool = ReadFileTool::default();
        let input = serde_json::json!({"path": tmp.to_str().unwrap(), "offset": 2, "limit": 2});
        let result = tool.execute(input).await.unwrap();

        assert!(result.contains("     2\tb"));
        assert!(result.contains("     3\tc"));
        assert!(!result.contains("     1\ta"));
        assert!(!result.contains("     4\td"));

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn read_file_missing_path() {
        let tool = ReadFileTool::default();
        let input = serde_json::json!({});
        let result = tool.execute(input).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn read_file_rejects_oversized_file() {
        let tmp = std::env::temp_dir().join(format!("deepcode_test_read_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "too large").await.unwrap();

        let tool = ReadFileTool::new(ToolExecutionConfig::unrestricted_with_max_size(Some(3)));
        let input = serde_json::json!({"path": tmp.to_str().unwrap()});
        let result = tool.execute(input).await;
        assert!(result.is_err());

        tokio::fs::remove_file(&tmp).await.unwrap();
    }
}
