use crate::execution::ToolExecutionConfig;
use crate::tool::{FileChangePreview, Tool, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};

#[derive(Debug, Clone)]
pub(super) struct WriteFileTool {
    execution: ToolExecutionConfig,
}

impl WriteFileTool {
    pub(super) fn new(execution: ToolExecutionConfig) -> Self {
        Self { execution }
    }
}

impl Default for WriteFileTool {
    fn default() -> Self {
        Self::new(ToolExecutionConfig::unrestricted())
    }
}

impl WriteFileTool {
    fn parse_input(&self, input: &serde_json::Value) -> Result<(String, String)> {
        let path = input["path"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'path' parameter".to_string(),
            })?
            .to_string();

        let content = input["content"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'content' parameter".to_string(),
            })?
            .to_string();

        Ok((path, content))
    }

    fn success_message(&self, preview: &FileChangePreview) -> String {
        let line_count = preview.after.lines().count();
        let byte_count = preview.after.len();
        format!(
            "Successfully wrote {} lines ({} bytes) to {}",
            line_count, byte_count, preview.path
        )
    }
}

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file, creating it if it doesn't exist. \
         Overwrites existing files."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        })
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::SAFE_MUTATION
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let Some(preview) = self.preview_change(input.clone()).await? else {
            return Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "write_file did not produce a preview".to_string(),
            });
        };
        self.execute_previewed(input, preview).await
    }

    async fn preview_change(&self, input: serde_json::Value) -> Result<Option<FileChangePreview>> {
        let (path, content) = self.parse_input(&input)?;
        let resolved_path = self.execution.resolve_writable_path(self.name(), &path)?;
        self.execution
            .ensure_content_size(self.name(), &resolved_path, &content)?;

        let before_exists = resolved_path.exists();
        let before = if before_exists {
            self.execution
                .ensure_existing_file_size(self.name(), &resolved_path)?;
            tokio::fs::read_to_string(&resolved_path)
                .await
                .map_err(|e| DeepCodeError::ToolExecution {
                    tool: self.name().to_string(),
                    message: format!("Cannot read {}: {}", resolved_path.display(), e),
                })?
        } else {
            String::new()
        };

        let display_path = resolved_path.display().to_string();
        let unified_diff =
            crate::diff::unified_diff(&display_path, &before, &content, before_exists);
        Ok(Some(FileChangePreview {
            path: display_path,
            before_exists,
            before,
            after: content,
            unified_diff,
        }))
    }

    async fn execute_previewed(
        &self,
        _input: serde_json::Value,
        preview: FileChangePreview,
    ) -> Result<String> {
        let success_message = self.success_message(&preview);
        crate::file_change::execute_previewed_file_change(
            self.name(),
            &self.execution,
            preview,
            success_message,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn write_file_creates_file() {
        let tmp =
            std::env::temp_dir().join(format!("deepcode_test_write_{}", uuid::Uuid::new_v4()));
        let tool = WriteFileTool::default();
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "content": "hello world"
        });
        let result = tool.execute(input).await.unwrap();
        assert!(result.contains("Successfully wrote"));

        let content = tokio::fs::read_to_string(&tmp).await.unwrap();
        assert_eq!(content, "hello world");

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn write_file_overwrites() {
        let tmp =
            std::env::temp_dir().join(format!("deepcode_test_write_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "old").await.unwrap();

        let tool = WriteFileTool::default();
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "content": "new"
        });
        tool.execute(input).await.unwrap();

        let content = tokio::fs::read_to_string(&tmp).await.unwrap();
        assert_eq!(content, "new");

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn write_file_missing_content() {
        let tool = WriteFileTool::default();
        let input = serde_json::json!({"path": "/tmp/x"});
        let result = tool.execute(input).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn write_file_rejects_oversized_content() {
        let tmp =
            std::env::temp_dir().join(format!("deepcode_test_write_{}", uuid::Uuid::new_v4()));
        let tool = WriteFileTool::new(ToolExecutionConfig::unrestricted_with_max_size(Some(4)));
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "content": "hello"
        });
        let result = tool.execute(input).await;
        assert!(result.is_err());
        assert!(!tmp.exists());
    }

    #[tokio::test]
    async fn write_file_preview_does_not_write_until_approved() {
        let tmp =
            std::env::temp_dir().join(format!("deepcode_test_write_{}", uuid::Uuid::new_v4()));
        let tool = WriteFileTool::default();
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "content": "previewed"
        });

        let preview = tool.preview_change(input.clone()).await.unwrap().unwrap();
        assert!(!tmp.exists());
        assert!(preview.unified_diff.contains("+previewed"));

        tool.execute_previewed(input, preview).await.unwrap();
        assert_eq!(tokio::fs::read_to_string(&tmp).await.unwrap(), "previewed");

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn write_file_preview_rejects_stale_file() {
        let tmp =
            std::env::temp_dir().join(format!("deepcode_test_write_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "old").await.unwrap();
        let tool = WriteFileTool::default();
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "content": "new"
        });

        let preview = tool.preview_change(input.clone()).await.unwrap().unwrap();
        tokio::fs::write(&tmp, "changed").await.unwrap();

        let result = tool.execute_previewed(input, preview).await;
        assert!(result.is_err());
        assert_eq!(tokio::fs::read_to_string(&tmp).await.unwrap(), "changed");

        tokio::fs::remove_file(&tmp).await.unwrap();
    }
}
