use super::git_common::validate_branch_name;
use crate::execution::ToolExecutionConfig;
use crate::tool::{Tool, ToolExecutionContext, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};

#[derive(Debug, Clone)]
pub(super) struct GitCheckoutTool {
    execution: ToolExecutionConfig,
}

impl GitCheckoutTool {
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
        let create = input["create"].as_bool().unwrap_or(false);
        let policy =
            context
                .sandbox_policy
                .unwrap_or(deepcode_sandbox::SandboxPolicy::WorkspaceWrite {
                    writable_roots: vec![working_dir.clone()],
                    network_access: false,
                });

        if let Some(files) = input["files"].as_array() {
            let mut args = vec!["checkout".to_string(), "--".to_string()];
            for file in files {
                let path = file.as_str().ok_or_else(|| DeepCodeError::ToolExecution {
                    tool: self.name().to_string(),
                    message: "Every 'files' entry must be a string".to_string(),
                })?;
                args.push(path.to_string());
            }
            let output = self
                .execution
                .run_sandboxed_command(self.name(), "git", args, working_dir, policy)
                .await?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(DeepCodeError::ToolExecution {
                    tool: self.name().to_string(),
                    message: format!("git checkout failed: {}", stderr),
                });
            }
            return Ok("Files checked out successfully.".to_string());
        }

        let branch = input["branch"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'branch' or 'files' parameter".to_string(),
            })?;
        validate_branch_name(self.name(), branch)?;

        let mut args = vec!["checkout".to_string()];
        if create {
            args.push("-b".to_string());
        }
        args.push(branch.to_string());
        let output = self
            .execution
            .run_sandboxed_command(self.name(), "git", args, working_dir, policy)
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("git checkout failed: {}", stderr),
            });
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(format!("Switched to branch '{}'.\n{}", branch, stdout))
    }
}

#[async_trait]
impl Tool for GitCheckoutTool {
    fn name(&self) -> &str {
        "git_checkout"
    }

    fn description(&self) -> &str {
        "Checkout a branch, or create and checkout a new branch. Can also checkout specific files."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "branch": {
                    "type": "string",
                    "description": "Branch name to checkout"
                },
                "create": {
                    "type": "boolean",
                    "description": "Create a new branch before checking it out (default: false)"
                },
                "files": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Optional list of files to checkout (instead of a branch)"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the git command (default: current directory)"
                }
            }
        })
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::DESTRUCTIVE
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
