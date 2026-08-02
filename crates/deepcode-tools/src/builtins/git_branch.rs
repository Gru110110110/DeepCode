use super::git_common::validate_branch_name;
use crate::execution::ToolExecutionConfig;
use crate::tool::{Tool, ToolExecutionContext, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};

#[derive(Debug, Clone)]
pub(super) struct GitBranchTool {
    execution: ToolExecutionConfig,
}

impl GitBranchTool {
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
        let action = input["action"].as_str().unwrap_or("list");
        let context_policy = context.sandbox_policy.clone();
        let read_policy =
            context_policy
                .clone()
                .unwrap_or(deepcode_sandbox::SandboxPolicy::ReadOnly {
                    network_access: false,
                });
        let write_policy =
            context_policy.unwrap_or(deepcode_sandbox::SandboxPolicy::WorkspaceWrite {
                writable_roots: vec![working_dir.clone()],
                network_access: false,
            });

        match action {
            "list" => {
                let output = self
                    .execution
                    .run_sandboxed_command(
                        self.name(),
                        "git",
                        vec![
                            "--no-optional-locks".to_string(),
                            "branch".to_string(),
                            "-a".to_string(),
                        ],
                        working_dir,
                        read_policy,
                    )
                    .await?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(DeepCodeError::ToolExecution {
                        tool: self.name().to_string(),
                        message: format!("git branch failed: {}", stderr),
                    });
                }

                let stdout = String::from_utf8_lossy(&output.stdout);
                Ok(stdout.to_string())
            }
            "create" => {
                let branch =
                    input["branch"]
                        .as_str()
                        .ok_or_else(|| DeepCodeError::ToolExecution {
                            tool: self.name().to_string(),
                            message: "Missing 'branch' parameter for create".to_string(),
                        })?;
                validate_branch_name(self.name(), branch)?;

                let output = self
                    .execution
                    .run_sandboxed_command(
                        self.name(),
                        "git",
                        vec!["branch".to_string(), branch.to_string()],
                        working_dir,
                        write_policy,
                    )
                    .await?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(DeepCodeError::ToolExecution {
                        tool: self.name().to_string(),
                        message: format!("git branch failed: {}", stderr),
                    });
                }

                Ok(format!("Branch '{}' created.", branch))
            }
            "delete" => {
                let branch =
                    input["branch"]
                        .as_str()
                        .ok_or_else(|| DeepCodeError::ToolExecution {
                            tool: self.name().to_string(),
                            message: "Missing 'branch' parameter for delete".to_string(),
                        })?;
                validate_branch_name(self.name(), branch)?;

                let output = self
                    .execution
                    .run_sandboxed_command(
                        self.name(),
                        "git",
                        vec!["branch".to_string(), "-d".to_string(), branch.to_string()],
                        working_dir,
                        write_policy,
                    )
                    .await?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(DeepCodeError::ToolExecution {
                        tool: self.name().to_string(),
                        message: format!("git branch failed: {}", stderr),
                    });
                }

                Ok(format!("Branch '{}' deleted.", branch))
            }
            _ => Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: format!("Unknown action: {}", action),
            }),
        }
    }
}

#[async_trait]
impl Tool for GitBranchTool {
    fn name(&self) -> &str {
        "git_branch"
    }

    fn description(&self) -> &str {
        "List, create, or delete branches. Returns the current branch and all branches."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "create", "delete"],
                    "description": "Action to perform (default: list)"
                },
                "branch": {
                    "type": "string",
                    "description": "Branch name (required for create/delete)"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the git command (default: current directory)"
                }
            }
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
