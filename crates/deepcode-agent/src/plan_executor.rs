use std::path::{Path, PathBuf};
use std::sync::Arc;

use deepcode_core::provider::traits::{
    FinishReason, GenerateParams, LlmProvider, ReasoningContext, ReasoningDisplay,
    ReasoningSummary, TextVerbosity,
};
use deepcode_tools::registry::ToolRegistry;

use crate::event::{AgentCommand, AgentEvent, CmdReceiver, EventSender};
use crate::llm_response::{
    collect_stream_response, final_text_from_blocks, response_blocks_to_content,
    tool_calls_from_blocks, StreamResponseOutcome,
};
use crate::r#loop::{
    handle_busy_command, wait_for_uncollected_subagents, BusyControl, SubagentBarrier,
};
use crate::state::{AgentPhase, AgentState};
use crate::validation::invalid_tool_input_message;

pub(crate) enum PlanDecision {
    Approved { plan: String },
    Rejected,
    Interrupted,
    Shutdown,
    Failed,
}

enum PlanResponseOutcome {
    Approved,
    Denied,
    ContinueDiscussing(String),
    Interrupted,
    Shutdown,
}

pub(crate) fn planning_instruction() -> String {
    "Plan-Act mode is active. First inspect the workspace only as needed, then produce a concise Markdown plan for the requested task. \
Do not modify files, run destructive commands, or use network tools while planning. You may spawn read-only explorer subagents and wait for their findings, but must not spawn workers. \
The plan must include ordered implementation steps and the checks you will run. \
After the plan, stop and wait for user approval before acting."
        .to_string()
}

pub(crate) fn approved_plan_instruction(plan: &str) -> String {
    format!(
        "The user approved this plan. Execute it now.\n\n{}\n\nAfter each step, check the result before continuing. If a step fails or the observed state contradicts the plan, stop and report the issue instead of continuing.",
        plan
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_plan_review(
    llm: &Arc<dyn LlmProvider>,
    tools: &Arc<ToolRegistry>,
    model: &str,
    model_config: (usize, usize),
    cmd_rx: &mut CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
    plan_directory: &Path,
    task_manager: Option<Arc<crate::subagent::AgentTaskManager>>,
) -> PlanDecision {
    state.push_user(&planning_instruction());
    let _ = event_tx.send(AgentEvent::SessionUpdated {
        messages: state.messages.clone(),
    });

    let plan_path = match prepare_plan_path(plan_directory).await {
        Ok(path) => path,
        Err(message) => {
            fail_plan_file(event_tx, state, message);
            return PlanDecision::Failed;
        }
    };

    let plan = match generate_plan(
        llm,
        tools,
        model,
        model_config,
        cmd_rx,
        event_tx,
        state,
        task_manager.clone(),
    )
    .await
    {
        PlanGeneration::Plan(plan) => plan,
        PlanGeneration::Interrupted => return PlanDecision::Interrupted,
        PlanGeneration::Shutdown => return PlanDecision::Shutdown,
        PlanGeneration::Failed => return PlanDecision::Failed,
    };
    if let Err(message) = write_plan_file(&plan_path, &plan).await {
        fail_plan_file(event_tx, state, message);
        return PlanDecision::Failed;
    }

    review_plan(
        llm,
        tools,
        model,
        model_config,
        cmd_rx,
        event_tx,
        state,
        plan_path,
        plan,
        false,
        task_manager,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn resume_plan_review(
    llm: &Arc<dyn LlmProvider>,
    tools: &Arc<ToolRegistry>,
    model: &str,
    model_config: (usize, usize),
    cmd_rx: &mut CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
    plan_directory: &Path,
    saved_plan_path: &str,
    task_manager: Option<Arc<crate::subagent::AgentTaskManager>>,
) -> PlanDecision {
    let plan_path =
        match resolve_resumed_plan_path(plan_directory, Path::new(saved_plan_path)).await {
            Ok(path) => path,
            Err(message) => {
                fail_plan_file(event_tx, state, message);
                return PlanDecision::Failed;
            }
        };
    let plan = match read_plan_file(&plan_path).await {
        Ok(plan) => plan,
        Err(message) => {
            fail_plan_file(event_tx, state, message);
            return PlanDecision::Failed;
        }
    };
    let _ = event_tx.send(AgentEvent::StatusUpdate {
        message: format!("Restoring plan review from {}.", plan_path.display()),
    });

    review_plan(
        llm,
        tools,
        model,
        model_config,
        cmd_rx,
        event_tx,
        state,
        plan_path,
        plan,
        true,
        task_manager,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn review_plan(
    llm: &Arc<dyn LlmProvider>,
    tools: &Arc<ToolRegistry>,
    model: &str,
    model_config: (usize, usize),
    cmd_rx: &mut CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
    plan_path: PathBuf,
    mut plan: String,
    mut restored: bool,
    task_manager: Option<Arc<crate::subagent::AgentTaskManager>>,
) -> PlanDecision {
    loop {
        let request_id = uuid::Uuid::new_v4().to_string();
        let _ = event_tx.send(AgentEvent::PlanApprovalNeeded {
            request_id: request_id.clone(),
            plan: plan.clone(),
            plan_path: plan_path.display().to_string(),
            restored,
        });
        state.phase = AgentPhase::WaitingForPlanApproval;

        match wait_for_plan_response(cmd_rx, &request_id, event_tx).await {
            PlanResponseOutcome::Approved => {
                let final_plan = match read_plan_file(&plan_path).await {
                    Ok(plan) => plan,
                    Err(message) => {
                        fail_plan_file(event_tx, state, message);
                        return PlanDecision::Failed;
                    }
                };
                if final_plan != plan {
                    let _ = event_tx.send(AgentEvent::StatusUpdate {
                        message: format!(
                            "Using externally edited plan from {}.",
                            plan_path.display()
                        ),
                    });
                }
                return PlanDecision::Approved { plan: final_plan };
            }
            PlanResponseOutcome::Denied => return PlanDecision::Rejected,
            PlanResponseOutcome::ContinueDiscussing(feedback) => {
                let current_plan = match read_plan_file(&plan_path).await {
                    Ok(plan) => plan,
                    Err(message) => {
                        fail_plan_file(event_tx, state, message);
                        return PlanDecision::Failed;
                    }
                };
                state.push_user(&format!(
                    "Continue discussing the proposed plan. Address the user's feedback or questions, then produce a complete updated plan for approval. Do not execute the plan yet. The current plan below was reloaded from disk and may contain edits made outside DeepCode.\n\nCurrent plan file: {}\n\nCurrent plan:\n{}\n\nUser feedback:\n{}",
                    plan_path.display(), current_plan, feedback
                ));
                let _ = event_tx.send(AgentEvent::SessionUpdated {
                    messages: state.messages.clone(),
                });
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: "Updating plan from feedback...".to_string(),
                });
                plan = match generate_plan(
                    llm,
                    tools,
                    model,
                    model_config,
                    cmd_rx,
                    event_tx,
                    state,
                    task_manager.clone(),
                )
                .await
                {
                    PlanGeneration::Plan(plan) => plan,
                    PlanGeneration::Interrupted => return PlanDecision::Interrupted,
                    PlanGeneration::Shutdown => return PlanDecision::Shutdown,
                    PlanGeneration::Failed => return PlanDecision::Failed,
                };
                if let Err(message) = write_plan_file(&plan_path, &plan).await {
                    fail_plan_file(event_tx, state, message);
                    return PlanDecision::Failed;
                }
                restored = false;
            }
            PlanResponseOutcome::Interrupted => return PlanDecision::Interrupted,
            PlanResponseOutcome::Shutdown => return PlanDecision::Shutdown,
        }
    }
}

async fn resolve_resumed_plan_path(
    plan_directory: &Path,
    saved_plan_path: &Path,
) -> Result<PathBuf, String> {
    if !saved_plan_path.is_absolute() {
        return Err(format!(
            "Persisted plan path {} is not absolute.",
            saved_plan_path.display()
        ));
    }
    reject_symlink(plan_directory).await?;
    reject_symlink(saved_plan_path).await?;
    let canonical_plan_dir = tokio::fs::canonicalize(plan_directory)
        .await
        .map_err(|error| format!("Cannot resolve {}: {error}", plan_directory.display()))?;
    let canonical_plan_path = tokio::fs::canonicalize(saved_plan_path)
        .await
        .map_err(|error| format!("Cannot resolve {}: {error}", saved_plan_path.display()))?;
    if canonical_plan_path.parent() != Some(canonical_plan_dir.as_path())
        || !is_managed_plan_filename(&canonical_plan_path)
    {
        return Err(format!(
            "Persisted plan file {} is outside the managed plan directory.",
            saved_plan_path.display()
        ));
    }
    Ok(canonical_plan_path)
}

fn is_managed_plan_filename(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("plan-"))
        .and_then(|name| name.strip_suffix(".md"))
        .is_some_and(|id| uuid::Uuid::parse_str(id).is_ok())
}

async fn prepare_plan_path(plan_directory: &Path) -> Result<PathBuf, String> {
    let parent = plan_directory.parent().ok_or_else(|| {
        format!(
            "Plan directory {} has no parent directory.",
            plan_directory.display()
        )
    })?;
    reject_symlink(parent).await?;
    tokio::fs::create_dir_all(parent)
        .await
        .map_err(|error| format!("Cannot create {}: {error}", parent.display()))?;
    reject_symlink(plan_directory).await?;
    tokio::fs::create_dir_all(plan_directory)
        .await
        .map_err(|error| format!("Cannot create {}: {error}", plan_directory.display()))?;
    restrict_plan_directory_permissions(plan_directory).await?;
    let canonical_plan_dir = tokio::fs::canonicalize(plan_directory)
        .await
        .map_err(|error| format!("Cannot resolve {}: {error}", plan_directory.display()))?;

    Ok(canonical_plan_dir.join(format!("plan-{}.md", uuid::Uuid::new_v4())))
}

async fn reject_symlink(path: &Path) -> Result<(), String> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(format!(
            "Refusing to use symlinked plan storage path {}.",
            path.display()
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("Cannot inspect {}: {error}", path.display())),
    }
}

async fn write_plan_file(path: &Path, plan: &str) -> Result<(), String> {
    reject_symlink(path).await?;
    tokio::fs::write(path, plan)
        .await
        .map_err(|error| format!("Cannot save plan to {}: {error}", path.display()))?;
    restrict_plan_file_permissions(path).await
}

#[cfg(unix)]
async fn restrict_plan_directory_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700))
        .await
        .map_err(|error| format!("Cannot secure plan directory {}: {error}", path.display()))
}

#[cfg(not(unix))]
async fn restrict_plan_directory_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
async fn restrict_plan_file_permissions(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;

    tokio::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .await
        .map_err(|error| format!("Cannot secure plan file {}: {error}", path.display()))
}

#[cfg(not(unix))]
async fn restrict_plan_file_permissions(_path: &Path) -> Result<(), String> {
    Ok(())
}

async fn read_plan_file(path: &Path) -> Result<String, String> {
    reject_symlink(path).await?;
    let plan = tokio::fs::read_to_string(path)
        .await
        .map_err(|error| format!("Cannot read plan from {}: {error}", path.display()))?;
    if plan.trim().is_empty() {
        return Err(format!(
            "Plan file {} is empty; add a plan before approving or continuing.",
            path.display()
        ));
    }
    Ok(plan)
}

fn fail_plan_file(event_tx: &EventSender, state: &mut AgentState, message: String) {
    let _ = event_tx.send(AgentEvent::AgentError { message });
    state.phase = AgentPhase::Error;
}

enum PlanGeneration {
    Plan(String),
    Interrupted,
    Shutdown,
    Failed,
}

#[allow(clippy::too_many_arguments)]
async fn generate_plan(
    llm: &Arc<dyn LlmProvider>,
    tools: &Arc<ToolRegistry>,
    model: &str,
    model_config: (usize, usize),
    cmd_rx: &mut CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
    task_manager: Option<Arc<crate::subagent::AgentTaskManager>>,
) -> PlanGeneration {
    let (max_tokens, context_window) = model_config;
    let reserved_output = max_tokens.min(context_window / 4);
    let input_budget = context_window.saturating_sub(reserved_output);

    let mut turn = 0usize;
    loop {
        state.turn_count = turn;
        state.phase = AgentPhase::Planning;

        match wait_for_uncollected_subagents(task_manager.as_ref(), cmd_rx, event_tx, |count| {
            format!("Waiting for {count} explorer task(s) before planning...")
        })
        .await
        {
            SubagentBarrier::Clear => {}
            SubagentBarrier::Completed(results) => {
                let rendered = serde_json::to_string(&results)
                    .unwrap_or_else(|error| format!("serialization failed: {error}"));
                state.push_user(&format!(
                    "Explorer tasks have finished. Use these structured findings before continuing the plan.\n\n{}",
                    rendered
                ));
                let _ = event_tx.send(AgentEvent::SessionUpdated {
                    messages: state.messages.clone(),
                });
            }
            SubagentBarrier::Interrupted => return PlanGeneration::Interrupted,
            SubagentBarrier::Shutdown => return PlanGeneration::Shutdown,
            SubagentBarrier::Failed(message) => {
                let _ = event_tx.send(AgentEvent::AgentError { message });
                state.phase = AgentPhase::Error;
                return PlanGeneration::Failed;
            }
        }

        let compressor = llm.context_compressor();
        let token_estimate = compressor.estimate_tokens(&state.messages);
        if compressor.needs_compression(token_estimate, input_budget) {
            let target = (input_budget as f64 * 0.60) as usize;
            let messages_to_compress = state.messages.clone();
            let compress_future =
                compressor.compress(&messages_to_compress, token_estimate, target);
            tokio::pin!(compress_future);
            let compressed_result = loop {
                tokio::select! {
                    result = &mut compress_future => break Some(result),
                    cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                        BusyControl::Continue => {}
                        BusyControl::Interrupt => break None,
                        BusyControl::Shutdown => return PlanGeneration::Shutdown,
                    }
                }
            };
            let Some(compressed_result) = compressed_result else {
                return PlanGeneration::Interrupted;
            };
            match compressed_result {
                Ok((messages, _)) => state.messages = messages,
                Err(error) => {
                    let _ = event_tx.send(AgentEvent::AgentError {
                        message: error.to_string(),
                    });
                    state.phase = AgentPhase::Error;
                    return PlanGeneration::Failed;
                }
            }
        }

        let capabilities = llm.capabilities(model);
        let current_tokens = llm.context_compressor().estimate_tokens(&state.messages);
        let safety_margin = (context_window / 100).clamp(1_024, 16_384);
        let request_max_tokens = max_tokens
            .min(context_window.saturating_sub(current_tokens.saturating_add(safety_margin)))
            .max(1);
        let reasoning_effort = state
            .reasoning_effort
            .as_deref()
            .and_then(|effort| effort.parse().ok());
        let enrich_reasoning = capabilities.reasoning_effort
            && reasoning_effort != Some(deepcode_core::config::ReasoningEffort::Off);
        let params = GenerateParams {
            max_tokens: Some(request_max_tokens),
            reasoning_effort,
            reasoning_summary: (enrich_reasoning && capabilities.reasoning_summary)
                .then_some(ReasoningSummary::Auto),
            reasoning_context: (enrich_reasoning && capabilities.reasoning_context)
                .then_some(ReasoningContext::AllTurns),
            reasoning_display: (enrich_reasoning && capabilities.reasoning_display)
                .then_some(ReasoningDisplay::Summarized),
            verbosity: capabilities.verbosity.then_some(TextVerbosity::High),
            parallel_tool_calls: capabilities.parallel_tool_calls.then_some(true),
            strict_tools: capabilities.strict_tools.then_some(true),
            prompt_cache_key: capabilities.prompt_cache_key.then(|| {
                state
                    .messages
                    .iter()
                    .find_map(|message| message.id.clone())
                    .unwrap_or_else(|| format!("deepcode-plan-{}", model))
            }),
            ..GenerateParams::default()
        };
        let tool_defs = tools.planning_definitions();
        let stream_result = {
            let stream_future =
                llm.generate_stream(model, &state.messages, &tool_defs, None, &params);
            tokio::pin!(stream_future);
            loop {
                tokio::select! {
                    result = &mut stream_future => break Some(result),
                    cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                        BusyControl::Continue => {}
                        BusyControl::Interrupt => break None,
                        BusyControl::Shutdown => return PlanGeneration::Shutdown,
                    }
                }
            }
        };
        let stream = match stream_result {
            Some(Ok(stream)) => stream,
            Some(Err(e)) => {
                let _ = event_tx.send(AgentEvent::AgentError {
                    message: e.to_string(),
                });
                state.phase = AgentPhase::Error;
                return PlanGeneration::Failed;
            }
            None => return PlanGeneration::Interrupted,
        };

        let streamed = match collect_stream_response(stream, cmd_rx, event_tx, false).await {
            StreamResponseOutcome::Completed(response) => response,
            StreamResponseOutcome::Interrupted => return PlanGeneration::Interrupted,
            StreamResponseOutcome::Shutdown => return PlanGeneration::Shutdown,
            StreamResponseOutcome::Failed { .. } => {
                state.phase = AgentPhase::Error;
                return PlanGeneration::Failed;
            }
        };

        let finish_reason = streamed.finish_reason.clone();
        let response_blocks = streamed.response_blocks;
        let assistant_blocks = response_blocks_to_content(&response_blocks);
        let tool_calls = tool_calls_from_blocks(&response_blocks);
        state.total_input_tokens += streamed.input_tokens;
        state.total_output_tokens += streamed.output_tokens;
        let _ = event_tx.send(AgentEvent::TurnComplete {
            input_tokens: streamed.input_tokens,
            output_tokens: streamed.output_tokens,
            cached_input_tokens: streamed.cached_input_tokens,
            cache_miss_input_tokens: streamed.cache_miss_input_tokens,
            reasoning_output_tokens: streamed.reasoning_output_tokens,
        });

        if finish_reason == Some(FinishReason::ContentFilter) {
            let _ = event_tx.send(AgentEvent::AgentError {
                message: "The model refused the request; try another model or revise the request."
                    .to_string(),
            });
            state.phase = AgentPhase::Error;
            return PlanGeneration::Failed;
        }

        state.push_assistant(assistant_blocks);
        llm.context_compressor()
            .normalize_history(&mut state.messages);
        let _ = event_tx.send(AgentEvent::SessionUpdated {
            messages: state.messages.clone(),
        });

        if tool_calls.is_empty() {
            let plan = final_text_from_blocks(&response_blocks);
            if plan.trim().is_empty() {
                let _ = event_tx.send(AgentEvent::AgentError {
                    message: "The model did not produce a plan.".to_string(),
                });
                state.phase = AgentPhase::Error;
                return PlanGeneration::Failed;
            }
            return PlanGeneration::Plan(plan);
        }

        let _ = event_tx.send(AgentEvent::StatusUpdate {
            message: format!(
                "Planning with {} read-only tool call(s)...",
                tool_calls.len()
            ),
        });

        for (tool_id, tool_name, tool_input) in tool_calls {
            if let Some(tool) = tools.get(&tool_name).cloned() {
                if tool_name == "spawn_agent"
                    && tool_input.get("role").and_then(serde_json::Value::as_str)
                        != Some("explorer")
                {
                    push_plan_tool_error(
                        event_tx,
                        state,
                        &tool_id,
                        &tool_name,
                        "Only explorer subagents are allowed during planning.",
                    );
                    continue;
                }
                let safety = tool.safety();
                if !safety.is_read_only || safety.requires_approval || safety.is_destructive {
                    push_plan_tool_error(
                        event_tx,
                        state,
                        &tool_id,
                        &tool_name,
                        "Tool is not allowed during planning; produce a plan without executing mutating or approval-required tools.",
                    );
                    continue;
                }

                if let Some(err) =
                    invalid_tool_input_message(&tool_name, tool.as_ref(), &tool_input)
                {
                    push_plan_tool_error(event_tx, state, &tool_id, &tool_name, &err);
                    continue;
                }

                let _ = event_tx.send(AgentEvent::ToolCallStarted {
                    id: tool_id.clone(),
                    name: tool_name.clone(),
                    input: tool_input.clone(),
                });
                let execute_future = tools.execute(&tool_name, tool_input);
                tokio::pin!(execute_future);
                let execute_result = loop {
                    tokio::select! {
                        result = &mut execute_future => break Some(result),
                        cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                            BusyControl::Continue => {}
                            BusyControl::Interrupt => break None,
                            BusyControl::Shutdown => return PlanGeneration::Shutdown,
                        }
                    }
                };
                let Some(execute_result) = execute_result else {
                    return PlanGeneration::Interrupted;
                };
                match execute_result {
                    Ok(result) => {
                        let _ = event_tx.send(AgentEvent::ToolCallCompleted {
                            id: tool_id.clone(),
                            name: tool_name.clone(),
                            result: result.clone(),
                        });
                        state.push_tool_result(&tool_id, &result, false);
                    }
                    Err(e) => {
                        let err = e.to_string();
                        let _ = event_tx.send(AgentEvent::ToolCallFailed {
                            id: tool_id.clone(),
                            name: tool_name.clone(),
                            error: err.clone(),
                        });
                        state.push_tool_result(&tool_id, &err, true);
                    }
                }
                let _ = event_tx.send(AgentEvent::SessionUpdated {
                    messages: state.messages.clone(),
                });
            } else {
                let err = format!("Tool '{}' not found", tool_name);
                push_plan_tool_error(event_tx, state, &tool_id, &tool_name, &err);
            }
        }

        turn = turn.saturating_add(1);
    }
}

async fn wait_for_plan_response(
    cmd_rx: &mut CmdReceiver,
    request_id: &str,
    event_tx: &EventSender,
) -> PlanResponseOutcome {
    while let Some(cmd) = cmd_rx.recv().await {
        if crate::subagent::route_nested_command(&cmd) {
            continue;
        }
        match cmd {
            AgentCommand::PlanResponse {
                request_id: response_id,
                approved,
            } if response_id == request_id => {
                return if approved {
                    PlanResponseOutcome::Approved
                } else {
                    PlanResponseOutcome::Denied
                };
            }
            AgentCommand::PlanFeedback {
                request_id: response_id,
                feedback,
            } if response_id == request_id && !feedback.trim().is_empty() => {
                return PlanResponseOutcome::ContinueDiscussing(feedback);
            }
            AgentCommand::Interrupt => {
                return PlanResponseOutcome::Interrupted;
            }
            AgentCommand::Shutdown => return PlanResponseOutcome::Shutdown,
            AgentCommand::Process { .. }
            | AgentCommand::PlanProcess { .. }
            | AgentCommand::ResumePlan { .. }
            | AgentCommand::SetPlanMode { .. }
            | AgentCommand::SetModel { .. }
            | AgentCommand::SetAvailableModels { .. }
            | AgentCommand::SetReasoningEffort { .. }
            | AgentCommand::ClearSession
            | AgentCommand::PermissionsSnapshot => {
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: "Waiting for plan approval; answer the prompt first.".to_string(),
                });
            }
            AgentCommand::PermissionResponse { .. }
            | AgentCommand::FileChangePreviewResponse { .. }
            | AgentCommand::PlanResponse { .. }
            | AgentCommand::PlanFeedback { .. } => {}
        }
    }

    PlanResponseOutcome::Shutdown
}

fn push_plan_tool_error(
    event_tx: &EventSender,
    state: &mut AgentState,
    tool_id: &str,
    tool_name: &str,
    error: &str,
) {
    let _ = event_tx.send(AgentEvent::ToolCallFailed {
        id: tool_id.to_string(),
        name: tool_name.to_string(),
        error: error.to_string(),
    });
    state.push_tool_result(tool_id, error, true);
    let _ = event_tx.send(AgentEvent::SessionUpdated {
        messages: state.messages.clone(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_root() -> PathBuf {
        std::env::temp_dir().join(format!("deepcode-plan-file-test-{}", uuid::Uuid::new_v4()))
    }

    fn cleanup(root: &Path) {
        let _ = std::fs::remove_dir_all(root);
    }

    #[tokio::test]
    async fn read_plan_file_rejects_missing_and_blank_files() {
        let root = test_root();
        tokio::fs::create_dir_all(&root).await.unwrap();
        let plan_path = root.join("plan.md");

        let missing = read_plan_file(&plan_path).await.unwrap_err();
        assert!(missing.contains("Cannot read plan"));

        tokio::fs::write(&plan_path, " \n\t").await.unwrap();
        let blank = read_plan_file(&plan_path).await.unwrap_err();
        assert!(blank.contains("is empty"));

        cleanup(&root);
    }

    #[tokio::test]
    async fn prepared_plan_path_uses_the_canonical_plan_directory() {
        let root = test_root();
        let plan_directory = root.join("plans");

        let plan_path = prepare_plan_path(&plan_directory).await.unwrap();

        assert_eq!(
            plan_path.parent().unwrap(),
            plan_directory.canonicalize().unwrap()
        );
        assert_eq!(
            plan_path.extension().and_then(|value| value.to_str()),
            Some("md")
        );
        assert!(plan_path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.starts_with("plan-")));
        assert!(!plan_path.exists());

        cleanup(&root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn plan_storage_is_private_to_the_current_user() {
        use std::os::unix::fs::PermissionsExt;

        let root = test_root();
        let plan_directory = root.join("plans");
        let plan_path = prepare_plan_path(&plan_directory).await.unwrap();
        write_plan_file(&plan_path, "# Private plan\n")
            .await
            .unwrap();

        let directory_mode = std::fs::metadata(&plan_directory)
            .unwrap()
            .permissions()
            .mode();
        let file_mode = std::fs::metadata(&plan_path).unwrap().permissions().mode();
        assert_eq!(directory_mode & 0o777, 0o700);
        assert_eq!(file_mode & 0o777, 0o600);

        cleanup(&root);
    }

    #[tokio::test]
    async fn resumed_plan_path_must_be_a_managed_plan_file() {
        let root = test_root();
        let plan_directory = root.join("plans");
        tokio::fs::create_dir_all(&plan_directory).await.unwrap();
        let outside_path = root.join(format!("plan-{}.md", uuid::Uuid::new_v4()));
        tokio::fs::write(&outside_path, "# Outside\n")
            .await
            .unwrap();
        let invalid_name = plan_directory.join("custom-plan.md");
        tokio::fs::write(&invalid_name, "# Invalid name\n")
            .await
            .unwrap();

        let outside = resolve_resumed_plan_path(&plan_directory, &outside_path)
            .await
            .unwrap_err();
        let invalid = resolve_resumed_plan_path(&plan_directory, &invalid_name)
            .await
            .unwrap_err();

        assert!(outside.contains("outside the managed plan directory"));
        assert!(invalid.contains("outside the managed plan directory"));
        cleanup(&root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn prepare_plan_path_rejects_symlinked_plan_directory() {
        let root = test_root();
        let target = root.join("target");
        let plan_directory = root.join("plans");
        tokio::fs::create_dir_all(&target).await.unwrap();
        std::os::unix::fs::symlink(&target, &plan_directory).unwrap();

        let error = prepare_plan_path(&plan_directory).await.unwrap_err();

        assert!(error.contains("symlinked plan storage path"));
        cleanup(&root);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn write_plan_file_does_not_follow_a_symlink() {
        let root = test_root();
        tokio::fs::create_dir_all(&root).await.unwrap();
        let target = root.join("target.md");
        let plan_path = root.join("plan.md");
        tokio::fs::write(&target, "unchanged").await.unwrap();
        std::os::unix::fs::symlink(&target, &plan_path).unwrap();

        let error = write_plan_file(&plan_path, "replacement")
            .await
            .unwrap_err();

        assert!(error.contains("symlinked plan storage path"));
        assert_eq!(
            tokio::fs::read_to_string(&target).await.unwrap(),
            "unchanged"
        );
        cleanup(&root);
    }
}
