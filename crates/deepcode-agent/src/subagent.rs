use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_tools::registry::ToolRegistry;
use deepcode_tools::tool::{Tool, ToolSafety};
use tokio::sync::mpsc;

use crate::event::{AgentCommand, AgentEvent, EventSender};
use crate::r#loop;

/// A tool that spawns an isolated subagent to handle a specific subtask.
///
/// The subagent gets its own conversation context.
/// Permission requests that are not already covered by an existing rule are
/// denied because the parent command channel cannot safely relay nested prompts.
pub struct SubagentTool {
    llm: Arc<dyn deepcode_core::provider::traits::LlmProvider>,
    tools: Arc<ToolRegistry>,
    permissions: Arc<tokio::sync::Mutex<deepcode_permissions::pipeline::PermissionSystem>>,
    model: String,
    model_config: (usize, usize),
    event_tx: EventSender,
}

impl SubagentTool {
    pub fn new(
        llm: Arc<dyn deepcode_core::provider::traits::LlmProvider>,
        tools: Arc<ToolRegistry>,
        permissions: Arc<tokio::sync::Mutex<deepcode_permissions::pipeline::PermissionSystem>>,
        model: String,
        model_config: (usize, usize),
        event_tx: EventSender,
    ) -> Self {
        Self {
            llm,
            tools,
            permissions,
            model,
            model_config,
            event_tx,
        }
    }
}

#[async_trait]
impl Tool for SubagentTool {
    fn name(&self) -> &str {
        "agent"
    }

    fn description(&self) -> &str {
        "Spawn a subagent to handle a specific subtask independently. \
         The subagent has access to the same tools but starts with a fresh conversation context. \
         Provide a clear, self-contained description of the task. \
         The subagent will execute autonomously and return its final result."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A clear, self-contained description of the subtask for the subagent to perform."
                }
            },
            "required": ["task"]
        })
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::SAFE_MUTATION
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let task_description = input["task"]
            .as_str()
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: "agent".to_string(),
                message: "Missing 'task' parameter".to_string(),
            })?
            .to_string();

        let task_id = uuid::Uuid::new_v4().to_string();
        tracing::info!(
            task_id = %task_id,
            task_chars = task_description.chars().count(),
            "Subagent started"
        );

        let _ = self.event_tx.send(AgentEvent::SubagentStarted {
            task_id: task_id.clone(),
            task: task_description.clone(),
        });

        let (sub_cmd_tx, sub_cmd_rx) = mpsc::channel(32);
        let (sub_event_tx, mut sub_event_rx) = mpsc::unbounded_channel();

        let sub_llm = self.llm.clone();
        let sub_tools = self.tools.clone();
        let sub_permissions = self.permissions.clone();
        let sub_model = self.model.clone();
        let (sub_max_tokens, sub_context_window) = self.model_config;
        let sub_task = task_description.clone();

        let agent_handle = tokio::spawn(async move {
            let system_prompt = format!(
                "You are a specialized subagent. Your task: {}\n\
                 Work autonomously using the available tools. Return a concise final answer.",
                sub_task
            );
            r#loop::run(
                sub_llm,
                sub_tools,
                sub_permissions,
                sub_model,
                (sub_max_tokens, sub_context_window),
                None,
                Some(system_prompt),
                None,
                false,
                sub_cmd_rx,
                sub_event_tx,
            )
            .await
        });

        let _ = sub_cmd_tx
            .send(AgentCommand::Process {
                message: task_description.to_string(),
            })
            .await;

        let mut final_message = String::new();
        let mut subagent_error = None;

        while let Some(ev) = sub_event_rx.recv().await {
            match ev {
                AgentEvent::TextDelta(_) => {
                    // Do not forward text deltas to avoid polluting the parent's streaming output.
                }
                AgentEvent::ToolCallStarted { id, name, input } => {
                    let _ = self.event_tx.send(AgentEvent::SubagentEvent {
                        task_id: task_id.clone(),
                        event: std::sync::Arc::new(AgentEvent::ToolCallStarted { id, name, input }),
                    });
                }
                AgentEvent::ToolCallCompleted { id, name, result } => {
                    let _ = self.event_tx.send(AgentEvent::SubagentEvent {
                        task_id: task_id.clone(),
                        event: std::sync::Arc::new(AgentEvent::ToolCallCompleted {
                            id,
                            name,
                            result,
                        }),
                    });
                }
                AgentEvent::ToolCallFailed { id, name, error } => {
                    let _ = self.event_tx.send(AgentEvent::SubagentEvent {
                        task_id: task_id.clone(),
                        event: std::sync::Arc::new(AgentEvent::ToolCallFailed { id, name, error }),
                    });
                }
                AgentEvent::PermissionNeeded { request_id, .. } => {
                    let error = "Subagent requested an operation that needs additional permission; retry that operation in the parent agent".to_string();
                    tracing::warn!(
                        task_id = %task_id,
                        request_id = %request_id,
                        "Subagent stopped at nested permission request"
                    );
                    subagent_error = Some(error);
                    let _ = sub_cmd_tx.send(AgentCommand::Shutdown).await;
                    break;
                }
                AgentEvent::FileChangePreviewNeeded { request_id, .. } => {
                    // File writes require an explicit review in the parent session.
                    // Deny them here so the parent agent can retry the edit directly.
                    tracing::info!(
                        task_id = %task_id,
                        request_id = %request_id,
                        "Subagent file change preview rejected"
                    );
                    let _ = sub_cmd_tx
                        .send(AgentCommand::FileChangePreviewResponse {
                            request_id,
                            approved: false,
                        })
                        .await;
                }
                AgentEvent::AgentFinished { final_message: msg } => {
                    tracing::info!(
                        task_id = %task_id,
                        result_chars = msg.chars().count(),
                        "Subagent finished"
                    );
                    final_message = msg;
                    break;
                }
                AgentEvent::AgentError { message } => {
                    tracing::warn!(
                        task_id = %task_id,
                        error = %message,
                        "Subagent failed"
                    );
                    subagent_error = Some(message);
                    break;
                }
                _ => {}
            }
        }

        drop(sub_cmd_tx);
        match tokio::time::timeout(Duration::from_secs(5), agent_handle).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(e))) => {
                tracing::warn!(
                    task_id = %task_id,
                    error = %e,
                    "Subagent loop returned error during cleanup"
                );
            }
            Ok(Err(e)) => {
                tracing::warn!(
                    task_id = %task_id,
                    error = %e,
                    "Subagent task join failed during cleanup"
                );
            }
            Err(_) => {
                tracing::warn!(
                    task_id = %task_id,
                    "Subagent cleanup timed out"
                );
            }
        }

        let result = if let Some(err) = subagent_error {
            format!("Subagent failed: {}", err)
        } else {
            final_message
        };

        let _ = self.event_tx.send(AgentEvent::SubagentCompleted {
            task_id: task_id.clone(),
            result: result.clone(),
        });
        tracing::info!(
            task_id = %task_id,
            result_chars = result.chars().count(),
            "Subagent result emitted"
        );

        Ok(result)
    }
}
