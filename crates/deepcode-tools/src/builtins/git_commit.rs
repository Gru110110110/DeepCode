use crate::execution::ToolExecutionConfig;
use crate::tool::{Tool, ToolExecutionContext, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};

#[derive(Debug, Clone)]
pub(super) struct GitCommitTool {
    execution: ToolExecutionConfig,
}

impl GitCommitTool {
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
        let message = input["message"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'message' parameter".to_string(),
            })?;
        let allow_empty = input["allow_empty"].as_bool().unwrap_or(false);

        let mut args = vec!["commit".to_string(), "-m".to_string(), message.to_string()];
        if allow_empty {
            args.push("--allow-empty".to_string());
        }
        let policy =
            context
                .sandbox_policy
                .unwrap_or(deepcode_sandbox::SandboxPolicy::WorkspaceWrite {
                    writable_roots: vec![working_dir.clone()],
                    network_access: false,
                });
        let output = self
            .execution
            .run_sandboxed_command(self.name(), "git", args, working_dir, policy)
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("git commit failed: {}", stderr),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(format!("Commit created.\n{}", stdout))
    }
}

#[async_trait]
impl Tool for GitCommitTool {
    fn name(&self) -> &str {
        "git_commit"
    }

    fn description(&self) -> &str {
        "Create a git commit with the given message. Only commits staged changes."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Commit message"
                },
                "allow_empty": {
                    "type": "boolean",
                    "description": "Allow creating an empty commit (default: false)"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the git command (default: current directory)"
                }
            },
            "required": ["message"]
        })
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::SAFE_MUTATION
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
