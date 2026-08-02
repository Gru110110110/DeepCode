use crate::execution::ToolExecutionConfig;
use crate::tool::{Tool, ToolExecutionContext, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};

#[derive(Debug, Clone)]
pub(super) struct GitDiffTool {
    execution: ToolExecutionConfig,
}

impl GitDiffTool {
    pub(super) fn new(execution: ToolExecutionConfig) -> Self {
        Self { execution }
    }

    async fn execute_inner(
        &self,
        input: serde_json::Value,
        context: ToolExecutionContext,
    ) -> Result<String> {
        let working_dir = input["working_dir"].as_str().unwrap_or(".");
        let working_dir = self.execution.resolve_directory(self.name(), working_dir)?;
        let file = input["file"].as_str();
        let max_lines = input["max_lines"].as_u64().unwrap_or(200) as usize;
        let policy = context
            .sandbox_policy
            .unwrap_or(deepcode_sandbox::SandboxPolicy::ReadOnly {
                network_access: false,
            });

        let mut args = vec![
            "--no-optional-locks".to_string(),
            "diff".to_string(),
            "--no-ext-diff".to_string(),
        ];
        if let Some(f) = file {
            args.push("--".to_string());
            args.push(f.to_string());
        }
        let output = self
            .execution
            .run_sandboxed_command(self.name(), "git", args, working_dir, policy)
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("git diff failed: {}", stderr),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = stdout.lines().collect();
        if lines.is_empty() {
            Ok("No diff (no changes)".to_string())
        } else if lines.len() > max_lines {
            let truncated: Vec<&str> = lines.into_iter().take(max_lines).collect();
            Ok(format!(
                "{}\n... (truncated, {} lines total)",
                truncated.join("\n"),
                stdout.lines().count()
            ))
        } else {
            Ok(stdout.to_string())
        }
    }
}

#[async_trait]
impl Tool for GitDiffTool {
    fn name(&self) -> &str {
        "git_diff"
    }

    fn description(&self) -> &str {
        "Get the git diff of the current repository. Can diff all changes or a specific file. \
         Optionally limit the number of lines returned."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file": {
                    "type": "string",
                    "description": "Specific file to diff (default: all changes)"
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: 200)"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the git command (default: current directory)"
                }
            }
        })
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::READ_ONLY
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        self.execute_inner(input, ToolExecutionContext::default())
            .await
    }

    async fn execute_with_context(
        &self,
        input: serde_json::Value,
        context: ToolExecutionContext,
    ) -> Result<String> {
        self.execute_inner(input, context).await
    }
}
