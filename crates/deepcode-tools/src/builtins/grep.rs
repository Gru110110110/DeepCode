use crate::execution::{truncate_output, ToolExecutionConfig};
use crate::tool::{Tool, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};
use regex::Regex;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub(super) struct GrepTool {
    execution: ToolExecutionConfig,
}

impl GrepTool {
    pub(super) fn new(execution: ToolExecutionConfig) -> Self {
        Self { execution }
    }
}

impl Default for GrepTool {
    fn default() -> Self {
        Self::new(ToolExecutionConfig::unrestricted())
    }
}

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a regex pattern in files. \
         Returns matching lines with file paths and line numbers."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "The regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (defaults to current directory)"
                },
                "glob": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.rs')"
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

        let search_path = input["path"].as_str().unwrap_or(".");
        let search_path = self
            .execution
            .resolve_existing_path(self.name(), search_path)?;

        let regex = Regex::new(pattern).map_err(|error| DeepCodeError::ToolExecution {
            tool: self.name().to_string(),
            message: format!("Invalid regular expression: {error}"),
        })?;
        let file_glob = input["glob"]
            .as_str()
            .map(glob::Pattern::new)
            .transpose()
            .map_err(|error| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Invalid file glob: {error}"),
            })?;
        let max_file_size = self.execution.max_file_size_bytes;
        let matches = tokio::task::spawn_blocking(move || {
            search_files(&search_path, &regex, file_glob.as_ref(), max_file_size)
        })
        .await
        .map_err(|error| DeepCodeError::ToolExecution {
            tool: self.name().to_string(),
            message: format!("Search worker failed: {error}"),
        })??;

        let result = truncate_output(&matches.join("\n"), self.execution.max_output_bytes);
        if result.trim().is_empty() {
            Ok(format!("No matches for '{}' found", pattern))
        } else {
            Ok(result)
        }
    }
}

fn search_files(
    root: &Path,
    regex: &Regex,
    file_glob: Option<&glob::Pattern>,
    max_file_size: Option<usize>,
) -> Result<Vec<String>> {
    let mut pending = vec![root.to_path_buf()];
    let mut matches = Vec::new();
    while let Some(path) = pending.pop() {
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(_) => continue,
        };
        if metadata.file_type().is_symlink() {
            continue;
        }
        if metadata.is_dir() {
            let Ok(entries) = std::fs::read_dir(&path) else {
                continue;
            };
            let mut children = entries
                .filter_map(std::result::Result::ok)
                .map(|entry| entry.path())
                .collect::<Vec<PathBuf>>();
            children.sort();
            children.reverse();
            pending.extend(children);
            continue;
        }
        if !metadata.is_file()
            || max_file_size.is_some_and(|limit| metadata.len() > limit as u64)
            || file_glob.is_some_and(|pattern| {
                !pattern.matches_path(&path)
                    && !path
                        .file_name()
                        .is_some_and(|name| pattern.matches_path(Path::new(name)))
            })
        {
            continue;
        }
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        if bytes.contains(&0) {
            continue;
        }
        let content = String::from_utf8_lossy(&bytes);
        for (index, line) in content.lines().enumerate() {
            if regex.is_match(line) {
                matches.push(format!("{}:{}:{}", path.display(), index + 1, line));
                if matches.len() == 100 {
                    return Ok(matches);
                }
            }
        }
    }
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn grep_finds_text_without_shell_interpolation() {
        let dir = std::env::temp_dir().join(format!("deepcode_test_grep_{}", uuid::Uuid::new_v4()));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("a.txt"), "alpha\nbeta\n")
            .await
            .unwrap();

        let tool = GrepTool::default();
        let input = serde_json::json!({
            "pattern": "alpha",
            "path": dir.to_str().unwrap()
        });
        let result = tool.execute(input).await.unwrap();
        assert!(result.contains("alpha"));

        tokio::fs::remove_dir_all(&dir).await.unwrap();
    }

    #[tokio::test]
    async fn grep_pattern_is_not_executed_as_shell() {
        let dir = std::env::temp_dir().join(format!("deepcode_test_grep_{}", uuid::Uuid::new_v4()));
        let marker = std::env::temp_dir().join(format!(
            "deepcode_test_grep_marker_{}",
            uuid::Uuid::new_v4()
        ));
        tokio::fs::create_dir_all(&dir).await.unwrap();
        tokio::fs::write(dir.join("a.txt"), "plain text\n")
            .await
            .unwrap();

        let tool = GrepTool::default();
        let input = serde_json::json!({
            "pattern": format!("'; touch {}; '", marker.display()),
            "path": dir.to_str().unwrap()
        });
        let _ = tool.execute(input).await.unwrap();
        assert!(!marker.exists());

        tokio::fs::remove_dir_all(&dir).await.unwrap();
    }
}
