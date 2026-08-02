use crate::execution::{truncate_output, ToolExecutionConfig};
use crate::tool::{Tool, ToolExecutionContext, ToolSafety};
use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_sandbox::{execute_prepared, CommandSpec, SandboxPolicy};

const SHELL_DESCRIPTION: &str = "Execute a shell command and return its output. \
Commands run in a sandboxed environment. \
Returns stdout and stderr combined.";

#[derive(Debug, Clone)]
pub(super) struct ShellTool {
    execution: ToolExecutionConfig,
}

impl ShellTool {
    pub(super) fn new(execution: ToolExecutionConfig) -> Self {
        Self { execution }
    }

    async fn execute_inner(
        &self,
        input: serde_json::Value,
        context: ToolExecutionContext,
    ) -> Result<String> {
        let command = input["command"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'command' parameter".to_string(),
            })?;

        let working_dir = input["working_dir"].as_str().unwrap_or(".");
        let working_dir = self.execution.resolve_directory(self.name(), working_dir)?;
        let policy = context
            .sandbox_policy
            .unwrap_or_else(|| SandboxPolicy::WorkspaceWrite {
                writable_roots: self
                    .execution
                    .workspace_root
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
                network_access: false,
            });

        let spec = CommandSpec::shell(command, working_dir.clone(), policy);
        let prepared = self.execution.sandbox_manager().prepare(spec)?;

        let output = execute_prepared(&prepared, self.execution.shell_timeout)
            .await
            .map_err(|error| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: error.to_string(),
            })?;

        let mut result = String::new();
        if !output.stdout.is_empty() {
            result.push_str(&String::from_utf8_lossy(&output.stdout));
        }
        if !output.stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str("[stderr]\n");
            result.push_str(&String::from_utf8_lossy(&output.stderr));
        }

        if result.is_empty() && output.status.success() {
            result = "(command completed successfully, no output)".to_string();
        }

        if !output.status.success() {
            result.push_str(&format!(
                "\n[exit code: {}]",
                output.status.code().unwrap_or(-1)
            ));
        }

        let sandbox_type = prepared
            .sandbox_type
            .map(|kind| format!("{:?}", kind))
            .unwrap_or_else(|| "none".to_string());
        result.push_str(&format!(
            "\n[sandboxed: {}, sandbox_type: {}, sandbox_policy: {}]",
            prepared.sandboxed,
            sandbox_type,
            prepared.sandbox_policy.label()
        ));
        #[cfg(target_os = "windows")]
        if matches!(
            prepared.sandbox_type,
            Some(deepcode_sandbox::SandboxType::WindowsRestrictedToken)
        ) {
            result.push_str(
                "\n[filesystem isolation: writes restricted; reads use current-user access]",
            );
            if !prepared.sandbox_policy.network_access() {
                result.push_str(
                    "\n[network isolation: advisory; use approval prompts for network commands]",
                );
            }
        }

        Ok(truncate_output(&result, self.execution.max_output_bytes))
    }
}

impl Default for ShellTool {
    fn default() -> Self {
        Self::new(ToolExecutionConfig::unrestricted())
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        SHELL_DESCRIPTION
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for the command"
                }
            },
            "required": ["command"]
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn shell_definition_keeps_policy_out_of_tool_schema() {
        let tool = ShellTool::default();
        let definition = tool.to_definition();

        assert_eq!(definition.description, SHELL_DESCRIPTION);
        let command_description = definition.input_schema["properties"]["command"]["description"]
            .as_str()
            .unwrap();
        assert_eq!(command_description, "The shell command to execute");
    }

    #[tokio::test]
    async fn shell_executes_echo() {
        let tool = ShellTool::default();
        let input = serde_json::json!({"command": "echo hello"});
        let result = tool
            .execute_with_context(
                input,
                ToolExecutionContext {
                    sandbox_policy: Some(SandboxPolicy::DangerFullAccess),
                },
            )
            .await
            .unwrap();
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn shell_returns_exit_code_on_failure() {
        let tool = ShellTool::default();
        let input = serde_json::json!({"command": "exit 42"});
        let result = tool
            .execute_with_context(
                input,
                ToolExecutionContext {
                    sandbox_policy: Some(SandboxPolicy::DangerFullAccess),
                },
            )
            .await
            .unwrap();
        assert!(result.contains("exit code: 42"));
    }

    #[tokio::test]
    async fn shell_timeout_stops_the_command_before_later_mutation() {
        let marker =
            std::env::temp_dir().join(format!("deepcode_timeout_marker_{}", uuid::Uuid::new_v4()));
        let tool = ShellTool::new(ToolExecutionConfig {
            workspace_root: None,
            max_file_size_bytes: None,
            shell_timeout: Duration::from_millis(20),
            max_output_bytes: 10_000,
        });
        #[cfg(not(target_os = "windows"))]
        let timeout_command = format!("sleep 0.2; touch {}", marker.display());
        #[cfg(target_os = "windows")]
        let timeout_command = format!(
            "Start-Sleep -Milliseconds 200; New-Item -ItemType File -Path '{}' | Out-Null",
            marker.display().to_string().replace('\'', "''")
        );
        let input = serde_json::json!({"command": timeout_command});

        let result = tool
            .execute_with_context(
                input,
                ToolExecutionContext {
                    sandbox_policy: Some(SandboxPolicy::DangerFullAccess),
                },
            )
            .await;
        assert!(result.is_err());

        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!marker.exists());
    }
}
