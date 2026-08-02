use crate::execution::ToolExecutionConfig;
use crate::tool::{Tool, ToolExecutionContext, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};

#[derive(Debug, Clone)]
pub(super) struct GitAddTool {
    execution: ToolExecutionConfig,
}

impl GitAddTool {
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
        let files = input["files"]
            .as_array()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'files' parameter".to_string(),
            })?;

        let mut args = vec!["add".to_string(), "--".to_string()];
        for file in files {
            let path = file.as_str().ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Every 'files' entry must be a string".to_string(),
            })?;
            args.push(path.to_string());
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
                message: format!("git add failed: {}", stderr),
            });
        }

        Ok("Files staged successfully.".to_string())
    }
}

#[async_trait]
impl Tool for GitAddTool {
    fn name(&self) -> &str {
        "git_add"
    }

    fn description(&self) -> &str {
        "Stage files for commit. Accepts a list of file paths or '.' to stage all changes."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of file paths to stage. Use ['.'] to stage all changes."
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the git command (default: current directory)"
                }
            },
            "required": ["files"]
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
