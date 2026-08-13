use std::sync::Arc;

use deepcode_permissions::execpolicy::Decision;
use deepcode_permissions::pipeline::PermissionSystem;
use deepcode_permissions::policy::{ApprovalScope, PermissionEvaluation};

use crate::event::{AgentCommand, AgentEvent, CmdReceiver, EventSender};
use crate::state::AgentState;

pub(crate) enum PermissionOutcome {
    Approved(Option<ApprovalScope>),
    Denied,
    Interrupted,
    Shutdown,
}

pub(crate) enum FilePreviewOutcome {
    Approved,
    Denied,
    Interrupted,
    Shutdown,
}

pub(crate) enum PermissionResolution {
    Approved,
    DeniedByPolicy(String),
    DeniedByUser,
    Interrupted,
    Shutdown,
}

pub(crate) async fn wait_for_permission_response(
    cmd_rx: &mut CmdReceiver,
    request_id: &str,
    event_tx: &EventSender,
) -> PermissionOutcome {
    while let Some(cmd) = cmd_rx.recv().await {
        if crate::subagent::route_nested_command(&cmd) {
            continue;
        }
        match cmd {
            AgentCommand::PermissionResponse {
                request_id: response_id,
                approved,
                scope,
            } if response_id == request_id => {
                return if approved {
                    PermissionOutcome::Approved(Some(scope))
                } else {
                    PermissionOutcome::Denied
                };
            }
            AgentCommand::Interrupt => {
                return PermissionOutcome::Interrupted;
            }
            AgentCommand::Shutdown => return PermissionOutcome::Shutdown,
            AgentCommand::Process { .. }
            | AgentCommand::PlanProcess { .. }
            | AgentCommand::SetPlanMode { .. }
            | AgentCommand::SetModel { .. }
            | AgentCommand::SetAvailableModels { .. }
            | AgentCommand::SetReasoningEffort { .. }
            | AgentCommand::ClearSession
            | AgentCommand::PermissionsSnapshot => {
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: "Waiting for permission; answer the prompt first.".to_string(),
                });
            }
            AgentCommand::PermissionResponse { .. }
            | AgentCommand::FileChangePreviewResponse { .. }
            | AgentCommand::PlanResponse { .. } => {}
        }
    }

    PermissionOutcome::Shutdown
}

pub(crate) async fn wait_for_file_preview_response(
    cmd_rx: &mut CmdReceiver,
    request_id: &str,
    event_tx: &EventSender,
) -> FilePreviewOutcome {
    while let Some(cmd) = cmd_rx.recv().await {
        if crate::subagent::route_nested_command(&cmd) {
            continue;
        }
        match cmd {
            AgentCommand::FileChangePreviewResponse {
                request_id: response_id,
                approved,
            } if response_id == request_id => {
                return if approved {
                    FilePreviewOutcome::Approved
                } else {
                    FilePreviewOutcome::Denied
                };
            }
            AgentCommand::Interrupt => {
                return FilePreviewOutcome::Interrupted;
            }
            AgentCommand::Shutdown => return FilePreviewOutcome::Shutdown,
            AgentCommand::Process { .. }
            | AgentCommand::PlanProcess { .. }
            | AgentCommand::SetPlanMode { .. }
            | AgentCommand::SetModel { .. }
            | AgentCommand::SetAvailableModels { .. }
            | AgentCommand::SetReasoningEffort { .. }
            | AgentCommand::ClearSession
            | AgentCommand::PermissionsSnapshot => {
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: "Waiting for file change review; answer the prompt first.".to_string(),
                });
            }
            AgentCommand::PermissionResponse { .. }
            | AgentCommand::FileChangePreviewResponse { .. }
            | AgentCommand::PlanResponse { .. } => {}
        }
    }

    FilePreviewOutcome::Shutdown
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn resolve_permission(
    evaluation: PermissionEvaluation,
    tool_name: &str,
    tool_input: &serde_json::Value,
    tool_id: &str,
    permissions: &Arc<tokio::sync::Mutex<PermissionSystem>>,
    cmd_rx: &mut CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
    turn: usize,
) -> PermissionResolution {
    match evaluation.decision {
        Decision::Allow => PermissionResolution::Approved,
        Decision::Forbidden => {
            let reason = evaluation
                .justification
                .clone()
                .unwrap_or_else(|| evaluation.summary.clone());
            tracing::warn!(
                turn = turn + 1,
                tool = %tool_name,
                tool_id = %tool_id,
                reason = %reason,
                "Tool permission denied by policy"
            );
            let _ = event_tx.send(AgentEvent::ToolCallFailed {
                id: tool_id.to_string(),
                name: tool_name.to_string(),
                error: reason.clone(),
            });
            PermissionResolution::DeniedByPolicy(format!("Permission denied: {}", reason))
        }
        Decision::Prompt => {
            let req_id = uuid::Uuid::new_v4().to_string();
            tracing::info!(
                turn = turn + 1,
                tool = %tool_name,
                tool_id = %tool_id,
                request_id = %req_id,
                "Tool permission requested"
            );

            let _ = event_tx.send(AgentEvent::PermissionNeeded {
                request_id: req_id.clone(),
                tool_name: tool_name.to_string(),
                input: tool_input.clone(),
                evaluation: evaluation.clone(),
            });
            state.phase = crate::state::AgentPhase::WaitingForPermission;

            match wait_for_permission_response(cmd_rx, &req_id, event_tx).await {
                PermissionOutcome::Approved(scope) => {
                    tracing::info!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        scope = ?scope,
                        "Tool permission approved"
                    );
                    if let Some(scope) = scope {
                        let mut perm = permissions.lock().await;
                        let _ = perm.handle_response(tool_name, tool_input, true, scope);
                    }
                    PermissionResolution::Approved
                }
                PermissionOutcome::Denied => {
                    tracing::warn!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        "Tool permission denied by user"
                    );
                    let _ = event_tx.send(AgentEvent::ToolCallFailed {
                        id: tool_id.to_string(),
                        name: tool_name.to_string(),
                        error: "Permission denied by user".to_string(),
                    });
                    {
                        let mut perm = permissions.lock().await;
                        let _ =
                            perm.handle_response(tool_name, tool_input, false, ApprovalScope::Once);
                    }
                    PermissionResolution::DeniedByUser
                }
                PermissionOutcome::Interrupted => {
                    tracing::info!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        "Interrupted while waiting for permission"
                    );
                    PermissionResolution::Interrupted
                }
                PermissionOutcome::Shutdown => {
                    tracing::info!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        "Shutdown while waiting for permission"
                    );
                    PermissionResolution::Shutdown
                }
            }
        }
    }
}
