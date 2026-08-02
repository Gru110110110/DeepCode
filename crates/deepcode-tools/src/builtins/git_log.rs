use crate::execution::ToolExecutionConfig;
use crate::tool::{Tool, ToolExecutionContext, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};

#[derive(Debug, Clone)]
pub(super) struct GitLogTool {
    execution: ToolExecutionConfig,
}

impl GitLogTool {
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
        let n = input["n"].as_u64().unwrap_or(10) as usize;
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
                    "log".to_string(),
                    format!("-{}", n.min(1000)),
                    "--pretty=format:%h %ad | %s".to_string(),
                    "--date=short".to_string(),
                ],
                working_dir,
                policy,
            )
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("git log failed: {}", stderr),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.trim().is_empty() {
            Ok("No commits found".to_string())
        } else {
            Ok(stdout.to_string())
        }
    }
}

#[async_trait]
impl Tool for GitLogTool {
    fn name(&self) -> &str {
        "git_log"
    }

    fn description(&self) -> &str {
        "Get recent git commit history. Returns commit hash, author, date, and message."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Number of commits to return (default: 10)"
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
