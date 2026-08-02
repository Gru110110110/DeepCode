use crate::execution::ToolExecutionConfig;
use crate::tool::{Tool, ToolExecutionContext, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};

#[derive(Debug, Clone)]
pub(super) struct GitStatusTool {
    execution: ToolExecutionConfig,
}

impl GitStatusTool {
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
        let policy = context
            .sandbox_policy
            .unwrap_or(deepcode_sandbox::SandboxPolicy::ReadOnly {
                network_access: false,
            });
        let output = self
            .execution
            .run_sandboxed_command(
                self.name(),
                "git",
                vec![
                    "--no-optional-locks".to_string(),
                    "status".to_string(),
                    "--porcelain".to_string(),
                    "-b".to_string(),
                ],
                working_dir,
                policy,
            )
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("git status failed: {}", stderr),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim().is_empty() {
            Ok("Working tree clean (no changes)".to_string())
        } else {
            Ok(stdout.to_string())
        }
    }
}

#[async_trait]
impl Tool for GitStatusTool {
    fn name(&self) -> &str {
        "git_status"
    }

    fn description(&self) -> &str {
        "Get the git status of the current repository. Returns modified, staged, and untracked files."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
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
