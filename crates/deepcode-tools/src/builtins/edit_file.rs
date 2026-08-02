use crate::execution::ToolExecutionConfig;
use crate::tool::{FileChangePreview, Tool, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};

#[derive(Debug, Clone)]
pub(super) struct EditFileTool {
    execution: ToolExecutionConfig,
}

impl EditFileTool {
    pub(super) fn new(execution: ToolExecutionConfig) -> Self {
        Self { execution }
    }
}

impl Default for EditFileTool {
    fn default() -> Self {
        Self::new(ToolExecutionConfig::unrestricted())
    }
}

impl EditFileTool {
    fn parse_common(&self, input: &serde_json::Value) -> Result<(String, String, String)> {
        let path = input["path"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'path' parameter".to_string(),
            })?
            .to_string();

        let mode = input["mode"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'mode' parameter".to_string(),
            })?
            .to_string();

        let new_string = input["new_string"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'new_string' parameter".to_string(),
            })?
            .to_string();

        Ok((path, mode, new_string))
    }

    fn apply_edit(
        &self,
        input: &serde_json::Value,
        path: &str,
        mode: &str,
        new_string: &str,
        content: &str,
    ) -> Result<String> {
        match mode {
            "replace" => {
                let old_string =
                    input["old_string"]
                        .as_str()
                        .ok_or_else(|| DeepCodeError::ToolExecution {
                            tool: self.name().to_string(),
                            message: "Missing 'old_string' parameter for replace mode".to_string(),
                        })?;

                if !content.contains(old_string) {
                    return Err(DeepCodeError::ToolExecution {
                        tool: self.name().to_string(),
                        message: format!(
                            "old_string not found in file '{}'. The file may have changed.",
                            path
                        ),
                    });
                }

                let count = content.matches(old_string).count();
                if count > 1 {
                    return Err(DeepCodeError::ToolExecution {
                        tool: self.name().to_string(),
                        message: format!(
                            "old_string appears {} times in file '{}'. Please provide a more unique string.",
                            count, path
                        ),
                    });
                }

                Ok(content.replacen(old_string, new_string, 1))
            }
            "insert" => {
                let line = input["line"]
                    .as_u64()
                    .ok_or_else(|| DeepCodeError::ToolExecution {
                        tool: self.name().to_string(),
                        message: "Missing or invalid 'line' parameter for insert mode".to_string(),
                    })? as usize;

                let lines: Vec<&str> = content.lines().collect();
                if line > lines.len() {
                    return Err(DeepCodeError::ToolExecution {
                        tool: self.name().to_string(),
                        message: format!(
                            "Line {} exceeds file length ({} lines)",
                            line,
                            lines.len()
                        ),
                    });
                }

                let mut new_lines: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
                new_lines.insert(line, new_string.to_string());
                Ok(new_lines.join("\n") + if content.ends_with('\n') { "\n" } else { "" })
            }
            _ => Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Unknown mode '{}'. Use 'replace' or 'insert'.", mode),
            }),
        }
    }
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Precisely edit a file. Supports two modes:\n\
         - 'replace': find old_string and replace with new_string.\n\
         - 'insert': insert new_string after a specific line number.\n\
         Use this instead of write_file when you only need to change part of a file."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "mode": {
                    "type": "string",
                    "enum": ["replace", "insert"],
                    "description": "Edit mode"
                },
                "old_string": {
                    "type": "string",
                    "description": "The existing text to find (required for replace mode)"
                },
                "new_string": {
                    "type": "string",
                    "description": "The new text to insert or replace with"
                },
                "line": {
                    "type": "integer",
                    "description": "Line number after which to insert (required for insert mode, 1-based)"
                }
            },
            "required": ["path", "mode", "new_string"]
        })
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::SAFE_MUTATION
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let Some(preview) = self.preview_change(input.clone()).await? else {
            return Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "edit_file did not produce a preview".to_string(),
            });
        };
        self.execute_previewed(input, preview).await
    }

    async fn preview_change(&self, input: serde_json::Value) -> Result<Option<FileChangePreview>> {
        let (path, mode, new_string) = self.parse_common(&input)?;
        let resolved_path = self.execution.resolve_existing_path(self.name(), &path)?;
        self.execution
            .ensure_existing_file_size(self.name(), &resolved_path)?;

        let content = tokio::fs::read_to_string(&resolved_path)
            .await
            .map_err(|e| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Failed to read file '{}': {}", resolved_path.display(), e),
            })?;

        let updated = self.apply_edit(&input, &path, &mode, &new_string, &content)?;
        self.execution
            .ensure_content_size(self.name(), &resolved_path, &updated)?;

        let display_path = resolved_path.display().to_string();
        let unified_diff = crate::diff::unified_diff(&display_path, &content, &updated, true);
        Ok(Some(FileChangePreview {
            path: display_path,
            before_exists: true,
            before: content,
            after: updated,
            unified_diff,
        }))
    }

    async fn execute_previewed(
        &self,
        _input: serde_json::Value,
        preview: FileChangePreview,
    ) -> Result<String> {
        let success_message = format!("File '{}' updated successfully.", preview.path);
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
    async fn edit_file_replace() {
        let tmp = std::env::temp_dir().join(format!("deepcode_test_edit_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "hello world\nfoo bar\n")
            .await
            .unwrap();

        let tool = EditFileTool::default();
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "mode": "replace",
            "old_string": "foo bar",
            "new_string": "baz qux"
        });
        let result = tool.execute(input).await.unwrap();
        assert!(result.contains("updated successfully"));

        let content = tokio::fs::read_to_string(&tmp).await.unwrap();
        assert!(content.contains("baz qux"));
        assert!(!content.contains("foo bar"));

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn edit_file_replace_not_found() {
        let tmp = std::env::temp_dir().join(format!("deepcode_test_edit_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "hello world\n").await.unwrap();

        let tool = EditFileTool::default();
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "mode": "replace",
            "old_string": "nonexistent",
            "new_string": "x"
        });
        let result = tool.execute(input).await;
        assert!(result.is_err());

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn edit_file_replace_ambiguous() {
        let tmp = std::env::temp_dir().join(format!("deepcode_test_edit_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "abc\nabc\n").await.unwrap();

        let tool = EditFileTool::default();
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "mode": "replace",
            "old_string": "abc",
            "new_string": "x"
        });
        let result = tool.execute(input).await;
        assert!(result.is_err());

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn edit_file_insert() {
        let tmp = std::env::temp_dir().join(format!("deepcode_test_edit_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "line1\nline2\nline3\n")
            .await
            .unwrap();

        let tool = EditFileTool::default();
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "mode": "insert",
            "line": 2,
            "new_string": "inserted"
        });
        let result = tool.execute(input).await.unwrap();
        assert!(result.contains("updated successfully"));

        let content = tokio::fs::read_to_string(&tmp).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines[0], "line1");
        assert_eq!(lines[1], "line2");
        assert_eq!(lines[2], "inserted");
        assert_eq!(lines[3], "line3");

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn edit_file_insert_out_of_range() {
        let tmp = std::env::temp_dir().join(format!("deepcode_test_edit_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "a\nb\n").await.unwrap();

        let tool = EditFileTool::default();
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "mode": "insert",
            "line": 10,
            "new_string": "x"
        });
        let result = tool.execute(input).await;
        assert!(result.is_err());

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn edit_file_rejects_oversized_update() {
        let tmp = std::env::temp_dir().join(format!("deepcode_test_edit_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "a").await.unwrap();

        let tool = EditFileTool::new(ToolExecutionConfig::unrestricted_with_max_size(Some(2)));
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "mode": "replace",
            "old_string": "a",
            "new_string": "abcd"
        });
        let result = tool.execute(input).await;
        assert!(result.is_err());
        assert_eq!(tokio::fs::read_to_string(&tmp).await.unwrap(), "a");

        tokio::fs::remove_file(&tmp).await.unwrap();
    }

    #[tokio::test]
    async fn edit_file_preview_rejects_stale_file() {
        let tmp = std::env::temp_dir().join(format!("deepcode_test_edit_{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, "old\n").await.unwrap();

        let tool = EditFileTool::default();
        let input = serde_json::json!({
            "path": tmp.to_str().unwrap(),
            "mode": "replace",
            "old_string": "old",
            "new_string": "new"
        });

        let preview = tool.preview_change(input.clone()).await.unwrap().unwrap();
        assert!(preview.unified_diff.contains("-old"));
        assert!(preview.unified_diff.contains("+new"));
        tokio::fs::write(&tmp, "changed\n").await.unwrap();

        let result = tool.execute_previewed(input, preview).await;
        assert!(result.is_err());
        assert_eq!(tokio::fs::read_to_string(&tmp).await.unwrap(), "changed\n");

        tokio::fs::remove_file(&tmp).await.unwrap();
    }
}
