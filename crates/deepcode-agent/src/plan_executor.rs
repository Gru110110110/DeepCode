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
use crate::r#loop::{handle_busy_command, BusyControl};
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
    Interrupted,
    Shutdown,
}

pub(crate) fn planning_instruction() -> String {
    "Plan-Act mode is active. First inspect the workspace only as needed, then produce a concise Markdown plan for the requested task. \
Do not modify files, run destructive commands, use network tools, or spawn subagents while planning. \
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

pub(crate) async fn execute_plan_review(
    llm: &Arc<dyn LlmProvider>,
    tools: &Arc<ToolRegistry>,
    model: &str,
    model_config: (usize, usize),
    cmd_rx: &mut CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
) -> PlanDecision {
    state.push_user(&planning_instruction());
    let _ = event_tx.send(AgentEvent::SessionUpdated {
        messages: state.messages.clone(),
    });

    let plan = match generate_plan(llm, tools, model, model_config, cmd_rx, event_tx, state).await {
        PlanGeneration::Plan(plan) => plan,
        PlanGeneration::Interrupted => return PlanDecision::Interrupted,
        PlanGeneration::Shutdown => return PlanDecision::Shutdown,
        PlanGeneration::Failed => return PlanDecision::Failed,
    };

    let request_id = uuid::Uuid::new_v4().to_string();
    let _ = event_tx.send(AgentEvent::PlanApprovalNeeded {
        request_id: request_id.clone(),
        plan: plan.clone(),
    });
    state.phase = AgentPhase::WaitingForPlanApproval;

    match wait_for_plan_response(cmd_rx, &request_id, event_tx).await {
        PlanResponseOutcome::Approved => PlanDecision::Approved { plan },
        PlanResponseOutcome::Denied => PlanDecision::Rejected,
        PlanResponseOutcome::Interrupted => PlanDecision::Interrupted,
        PlanResponseOutcome::Shutdown => PlanDecision::Shutdown,
    }
}

enum PlanGeneration {
    Plan(String),
    Interrupted,
    Shutdown,
    Failed,
}

async fn generate_plan(
    llm: &Arc<dyn LlmProvider>,
    tools: &Arc<ToolRegistry>,
    model: &str,
    model_config: (usize, usize),
    cmd_rx: &mut CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
) -> PlanGeneration {
    let (max_tokens, context_window) = model_config;
    let reserved_output = max_tokens.min(context_window / 4);
    let input_budget = context_window.saturating_sub(reserved_output);

    let mut turn = 0usize;
    loop {
        state.turn_count = turn;
        state.phase = AgentPhase::Planning;

        let compressor = llm.context_compressor();
        let token_estimate = compressor.estimate_tokens(&state.messages);
        if compressor.needs_compression(token_estimate, input_budget) {
            let target = (input_budget as f64 * 0.60) as usize;
            match compressor
                .compress(&state.messages, token_estimate, target)
                .await
            {
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
                match tools.execute(&tool_name, tool_input).await {
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
            AgentCommand::Interrupt => {
                let _ = event_tx.send(AgentEvent::Interrupted);
                return PlanResponseOutcome::Interrupted;
            }
            AgentCommand::Shutdown => return PlanResponseOutcome::Shutdown,
            AgentCommand::Process { .. }
            | AgentCommand::PlanProcess { .. }
            | AgentCommand::SetPlanMode { .. }
            | AgentCommand::SetModel { .. }
            | AgentCommand::SetReasoningEffort { .. }
            | AgentCommand::ClearSession
            | AgentCommand::PermissionsSnapshot => {
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: "Waiting for plan approval; answer the prompt first.".to_string(),
                });
            }
            AgentCommand::PermissionResponse { .. }
            | AgentCommand::FileChangePreviewResponse { .. }
            | AgentCommand::PlanResponse { .. } => {}
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
