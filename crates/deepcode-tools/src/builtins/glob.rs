use crate::execution::ToolExecutionConfig;
use crate::tool::{Tool, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};
use std::collections::HashSet;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;

#[derive(Debug, Clone)]
pub(super) struct GlobTool {
    execution: ToolExecutionConfig,
}

impl GlobTool {
    pub(super) fn new(execution: ToolExecutionConfig) -> Self {
        Self { execution }
    }
}

impl Default for GlobTool {
    fn default() -> Self {
        Self::new(ToolExecutionConfig::unrestricted())
    }
}

#[async_trait]
impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern (e.g., '**/*.rs', 'src/**/*.ts'). \
         Returns matching file paths sorted by modification time. \
         Automatically respects .gitignore rules."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match against file paths"
                },
                "path": {
                    "type": "string",
                    "description": "Base directory to search in (defaults to current directory)"
                }
            },
            "required": ["pattern"]
        })
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::READ_ONLY
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let pattern = input["pattern"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'pattern' parameter".to_string(),
            })?;

        let base_path = input["path"].as_str().unwrap_or(".");
        let base_path = self.execution.resolve_directory(self.name(), base_path)?;
        let full_pattern = base_path.join(pattern);
        let full_pattern_str = full_pattern.to_string_lossy().to_string();

        let entries = tokio::task::spawn_blocking(move || {
            let mut results = Vec::new();
            match glob::glob(&full_pattern_str) {
                Ok(paths) => {
                    for path in paths.flatten() {
                        if path.is_file() {
                            results.push(path);
                        }
                    }
                }
                Err(e) => {
                    return Err::<_, DeepCodeError>(DeepCodeError::ToolExecution {
                        tool: "glob".to_string(),
                        message: format!("Invalid glob pattern: {}", e),
                    });
                }
            }
            Ok(results)
        })
        .await
        .map_err(|e| DeepCodeError::ToolExecution {
            tool: self.name().to_string(),
            message: format!("Glob task panicked: {}", e),
        })?;

        let mut paths = entries?;
        filter_gitignored(&base_path, &mut paths).await;
        paths.sort_by_key(|p| p.metadata().and_then(|m| m.modified()).ok());
        paths.reverse(); // newest first

        let limit = 100;
        if paths.is_empty() {
            Ok(format!("No files matching '{}' found", pattern))
        } else {
            let lines: Vec<String> = paths
                .iter()
                .take(limit)
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            let mut result = lines.join("\n");
            if paths.len() > limit {
                result.push_str(&format!(
                    "\n... ({} total, showing first {})",
                    paths.len(),
                    limit
                ));
            }
            Ok(result)
        }
    }
}

async fn filter_gitignored(base_path: &std::path::Path, paths: &mut Vec<std::path::PathBuf>) {
    if paths.is_empty() {
        return;
    }
    let mut command = tokio::process::Command::new("git");
    command.kill_on_drop(true);
    command
        .arg("-C")
        .arg(base_path)
        .args(["check-ignore", "--no-index", "--stdin", "-z"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let Ok(mut child) = command.spawn() else {
        return;
    };
    let Some(mut stdin) = child.stdin.take() else {
        return;
    };
    let input = paths
        .iter()
        .filter_map(|path| path.strip_prefix(base_path).ok())
        .flat_map(|path| {
            let mut bytes = path.to_string_lossy().as_bytes().to_vec();
            bytes.push(0);
            bytes
        })
        .collect::<Vec<_>>();
    if stdin.write_all(&input).await.is_err() {
        return;
    }
    drop(stdin);
    let Ok(Ok(output)) =
        tokio::time::timeout(std::time::Duration::from_secs(10), child.wait_with_output()).await
    else {
        return;
    };
    if !output.status.success() && output.status.code() != Some(1) {
        return;
    }
    let ignored = output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| String::from_utf8_lossy(path).into_owned())
        .collect::<HashSet<_>>();
    paths.retain(|path| {
        path.strip_prefix(base_path)
            .map(|relative| !ignored.contains(&relative.to_string_lossy().into_owned()))
            .unwrap_or(true)
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn glob_finds_files() {
        let dir = std::env::temp_dir().join(format!("deepcode_test_glob_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("a.rs"), "").await.unwrap();
        tokio::fs::write(dir.join("b.rs"), "").await.unwrap();
        tokio::fs::write(dir.join("c.txt"), "").await.unwrap();

        let tool = GlobTool::default();
        let input = serde_json::json!({
            "pattern": "*.rs",
            "path": dir.to_str().unwrap()
        });
        let result = tool.execute(input).await.unwrap();
        assert!(result.contains("a.rs"));
        assert!(result.contains("b.rs"));
        assert!(!result.contains("c.txt"));

        tokio::fs::remove_dir_all(&dir).await.unwrap();
    }

    #[tokio::test]
    async fn glob_no_match() {
        let dir = std::env::temp_dir().join(format!("deepcode_test_glob_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();

        let tool = GlobTool::default();
        let input = serde_json::json!({
            "pattern": "*.nonexistent",
            "path": dir.to_str().unwrap()
        });
        let result = tool.execute(input).await.unwrap();
        assert!(result.contains("No files matching"));

        tokio::fs::remove_dir_all(&dir).await.unwrap();
    }

    #[tokio::test]
    async fn glob_respects_gitignore() {
        let dir = std::env::temp_dir().join(format!("deepcode_test_glob_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join(".gitignore"), "ignored.rs\n")
            .await
            .unwrap();
        tokio::fs::write(dir.join("ignored.rs"), "").await.unwrap();
        tokio::fs::write(dir.join("visible.rs"), "").await.unwrap();
        let status = tokio::process::Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(&dir)
            .status()
            .await
            .unwrap();
        assert!(status.success());

        let result = GlobTool::default()
            .execute(serde_json::json!({
                "pattern": "*.rs",
                "path": dir.to_str().unwrap()
            }))
            .await
            .unwrap();
        assert!(result.contains("visible.rs"));
        assert!(!result.contains("ignored.rs"));

        tokio::fs::remove_dir_all(&dir).await.unwrap();
    }

    #[tokio::test]
    async fn glob_missing_pattern() {
        let tool = GlobTool::default();
        let input = serde_json::json!({});
        let result = tool.execute(input).await;
        assert!(result.is_err());
    }
}
