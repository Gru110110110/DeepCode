use std::sync::Arc;

use deepcode_core::provider::traits::{
    FinishReason, GenerateParams, LlmProvider, ReasoningContext, ReasoningDisplay,
    ReasoningSummary, TextVerbosity,
};
use deepcode_core::types::{ContentBlock, Message, Role};
use deepcode_permissions::execpolicy::Decision;
use deepcode_permissions::pipeline::PermissionSystem;
use deepcode_tools::registry::ToolRegistry;
use deepcode_tools::tool::ToolExecutionContext;

use crate::event::{AgentEvent, CmdReceiver, EventSender};
use crate::llm_response::{
    collect_stream_response, final_text_from_blocks, response_blocks_to_content,
    tool_calls_from_blocks, StreamResponseOutcome, SESSION_TITLE_INSTRUCTION,
};
use crate::permission_handler::{
    resolve_permission, wait_for_file_preview_response, FilePreviewOutcome, PermissionResolution,
};
use crate::r#loop::{
    handle_busy_command, wait_for_uncollected_subagents, BusyControl, SubagentBarrier,
};
use crate::state::{AgentPhase, AgentState, ToolResultEntry};
use crate::validation::invalid_tool_input_message;

pub(crate) struct TurnResult {
    pub interrupted: bool,
    pub shutdown_requested: bool,
}

const REPEATED_FAILURE_LIMIT: usize = 2;
const REPEATED_BLOCKED_ROUND_LIMIT: usize = 2;
const REPEATED_FAILURE_INSTRUCTION: &str = "You have executed this exact tool call twice and it failed both times. Do not retry the same tool with the same input again. Change the input, gather new information first, use a materially different approach, or stop and explain the blocker to the user.";
const REPEATED_CALL_BLOCKED_MESSAGE: &str = "Repeated-failure guard blocked this tool call because the same tool and input already failed twice. The tool was not executed. Use a materially different approach or explain the blocker to the user.";

#[derive(Debug, Clone, PartialEq)]
struct ToolCallSignature {
    name: String,
    input: serde_json::Value,
}

impl ToolCallSignature {
    fn new(name: &str, input: &serde_json::Value) -> Self {
        Self {
            name: name.to_string(),
            input: input.clone(),
        }
    }

    fn matches(&self, name: &str, input: &serde_json::Value) -> bool {
        self.name == name && self.input == *input
    }
}

#[derive(Debug)]
struct FailureStreak {
    signature: ToolCallSignature,
    count: usize,
}

#[derive(Debug, Default)]
struct RepeatedFailureGuard {
    streak: Option<FailureStreak>,
}

impl RepeatedFailureGuard {
    fn record_failure(&mut self, name: &str, input: &serde_json::Value) -> bool {
        match self.streak.as_mut() {
            Some(streak) if streak.signature.matches(name, input) => {
                streak.count = streak.count.saturating_add(1);
            }
            _ => {
                self.streak = Some(FailureStreak {
                    signature: ToolCallSignature::new(name, input),
                    count: 1,
                });
            }
        }

        self.streak
            .as_ref()
            .is_some_and(|streak| streak.count == REPEATED_FAILURE_LIMIT)
    }

    fn should_block(&self, name: &str, input: &serde_json::Value) -> bool {
        self.streak.as_ref().is_some_and(|streak| {
            streak.count >= REPEATED_FAILURE_LIMIT && streak.signature.matches(name, input)
        })
    }

    fn reset(&mut self) {
        self.streak = None;
    }
}

fn messages_with_title_instruction(messages: &[Message]) -> Vec<Message> {
    let mut messages = messages.to_vec();
    if let Some(ContentBlock::Text { text }) = messages
        .iter_mut()
        .find(|message| message.role == Role::System)
        .and_then(|message| message.content.first_mut())
    {
        text.push_str("\n\n");
        text.push_str(SESSION_TITLE_INSTRUCTION);
    } else {
        messages.insert(
            0,
            Message {
                role: Role::System,
                content: vec![ContentBlock::text(SESSION_TITLE_INSTRUCTION)],
                id: None,
            },
        );
    }
    messages
}

#[derive(Debug)]
struct ParallelToolCall {
    id: String,
    name: String,
    input: serde_json::Value,
    context: ToolExecutionContext,
    call_index: usize,
}

#[derive(Debug)]
struct ToolExecutionResult {
    id: String,
    name: String,
    input: serde_json::Value,
    result: std::result::Result<String, String>,
}

fn send_session_updated(event_tx: &EventSender, state: &AgentState) {
    let _ = event_tx.send(AgentEvent::SessionUpdated {
        messages: state.messages.clone(),
    });
}

fn push_tool_result_and_update(
    state: &mut AgentState,
    event_tx: &EventSender,
    tool_id: &str,
    content: &str,
    is_error: bool,
) {
    state.push_tool_result(tool_id, content, is_error);
    send_session_updated(event_tx, state);
}

#[allow(clippy::too_many_arguments)]
fn tracked_tool_result_entry(
    failure_guard: &mut RepeatedFailureGuard,
    event_tx: &EventSender,
    turn: usize,
    tool_id: &str,
    tool_name: &str,
    tool_input: &serde_json::Value,
    content: &str,
    is_error: bool,
) -> ToolResultEntry {
    let content = if is_error {
        if failure_guard.record_failure(tool_name, tool_input) {
            tracing::warn!(
                turn = turn + 1,
                tool = %tool_name,
                tool_id = %tool_id,
                consecutive_failures = REPEATED_FAILURE_LIMIT,
                "Repeated tool failure detected"
            );
            let _ = event_tx.send(AgentEvent::StatusUpdate {
                message: "Changing strategy after a repeated tool failure...".to_string(),
            });
            format!("{}\n\n{}", content, REPEATED_FAILURE_INSTRUCTION)
        } else {
            content.to_string()
        }
    } else {
        failure_guard.reset();
        content.to_string()
    };

    ToolResultEntry::new(tool_id, content, is_error)
}

#[allow(clippy::too_many_arguments)]
fn push_tracked_tool_result_and_update(
    state: &mut AgentState,
    event_tx: &EventSender,
    failure_guard: &mut RepeatedFailureGuard,
    turn: usize,
    tool_id: &str,
    tool_name: &str,
    tool_input: &serde_json::Value,
    content: &str,
    is_error: bool,
) {
    let entry = tracked_tool_result_entry(
        failure_guard,
        event_tx,
        turn,
        tool_id,
        tool_name,
        tool_input,
        content,
        is_error,
    );
    state.push_tool_results(vec![entry]);
    send_session_updated(event_tx, state);
}

fn model_request_status(turn: usize) -> String {
    if turn == 0 {
        "Thinking...".to_string()
    } else {
        "Thinking after tools...".to_string()
    }
}

#[allow(clippy::too_many_arguments)]
async fn flush_parallel_tool_calls(
    batch: &mut Vec<ParallelToolCall>,
    tools: &Arc<ToolRegistry>,
    cmd_rx: &mut CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
    failure_guard: &mut RepeatedFailureGuard,
    turn: usize,
    interrupted: &mut bool,
    shutdown_requested: &mut bool,
) {
    if batch.is_empty() {
        return;
    }

    state.phase = AgentPhase::ExecutingTools;
    let calls = std::mem::take(batch);
    for call in &calls {
        tracing::info!(
            turn = turn + 1,
            tool = %call.name,
            tool_id = %call.id,
            tool_call_index = call.call_index,
            "Parallel tool call started"
        );
        let _ = event_tx.send(AgentEvent::ToolCallStarted {
            id: call.id.clone(),
            name: call.name.clone(),
            input: call.input.clone(),
        });
    }

    let execute_future = futures::future::join_all(calls.into_iter().map(|call| {
        let tools = Arc::clone(tools);
        async move {
            let input = call.input;
            let result = tools
                .execute_with_context(&call.name, input.clone(), call.context)
                .await
                .map_err(|e| e.to_string());
            ToolExecutionResult {
                id: call.id,
                name: call.name,
                input,
                result,
            }
        }
    }));
    tokio::pin!(execute_future);

    let execution_results = loop {
        tokio::select! {
            result = &mut execute_future => break Some(result),
            cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                BusyControl::Continue => {}
                BusyControl::Interrupt => {
                    *interrupted = true;
                    break None;
                }
                BusyControl::Shutdown => {
                    *shutdown_requested = true;
                    break None;
                }
            }
        }
    };

    let Some(execution_results) = execution_results else {
        tracing::info!(
            turn = turn + 1,
            interrupted = *interrupted,
            shutdown_requested = *shutdown_requested,
            "Parallel tool batch stopped before completion"
        );
        return;
    };

    let mut tool_results = Vec::with_capacity(execution_results.len());
    for execution_result in execution_results {
        match execution_result.result {
            Ok(result) => {
                tracing::info!(
                    turn = turn + 1,
                    tool = %execution_result.name,
                    tool_id = %execution_result.id,
                    result_chars = result.chars().count(),
                    "Parallel tool call completed"
                );
                let _ = event_tx.send(AgentEvent::ToolCallCompleted {
                    id: execution_result.id.clone(),
                    name: execution_result.name.clone(),
                    result: result.clone(),
                });
                tool_results.push(tracked_tool_result_entry(
                    failure_guard,
                    event_tx,
                    turn,
                    &execution_result.id,
                    &execution_result.name,
                    &execution_result.input,
                    &result,
                    false,
                ));
            }
            Err(error) => {
                tracing::warn!(
                    turn = turn + 1,
                    tool = %execution_result.name,
                    tool_id = %execution_result.id,
                    error = %error,
                    "Parallel tool call failed"
                );
                let _ = event_tx.send(AgentEvent::ToolCallFailed {
                    id: execution_result.id.clone(),
                    name: execution_result.name.clone(),
                    error: error.clone(),
                });
                tool_results.push(tracked_tool_result_entry(
                    failure_guard,
                    event_tx,
                    turn,
                    &execution_result.id,
                    &execution_result.name,
                    &execution_result.input,
                    &error,
                    true,
                ));
            }
        }
    }

    state.push_tool_results(tool_results);
    send_session_updated(event_tx, state);
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_turn(
    llm: &Arc<dyn LlmProvider>,
    tools: &Arc<ToolRegistry>,
    permissions: &Arc<tokio::sync::Mutex<PermissionSystem>>,
    model: &str,
    model_config: (usize, usize),
    cmd_rx: &mut CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
    task_manager: Option<Arc<crate::subagent::AgentTaskManager>>,
) -> TurnResult {
    let mut tool_call_count = 0usize;
    let mut interrupted = false;
    let mut shutdown_requested = false;
    let mut failure_guard = RepeatedFailureGuard::default();
    let mut force_text_only = false;
    let mut blocked_only_rounds = 0usize;
    let (max_tokens, context_window) = model_config;
    let reserved_output = max_tokens.min(context_window / 4);
    let input_budget = context_window.saturating_sub(reserved_output);

    let mut turn = 0usize;
    loop {
        state.turn_count = turn;
        tracing::info!(
            turn = turn + 1,
            messages = state.messages.len(),
            "Agent turn started"
        );

        match wait_for_uncollected_subagents(task_manager.as_ref(), cmd_rx, event_tx, |count| {
            format!("Waiting for {count} outstanding subagent task(s)...")
        })
        .await
        {
            SubagentBarrier::Clear => {}
            SubagentBarrier::Completed(results) => {
                let rendered = serde_json::to_string(&results)
                    .unwrap_or_else(|error| format!("serialization failed: {error}"));
                state.push_user(&format!(
                    "Subagents have finished. Their structured results are below. Review them before continuing; retry only when the reported error makes that useful.\n\n{}",
                    rendered
                ));
                send_session_updated(event_tx, state);
            }
            SubagentBarrier::Interrupted => {
                interrupted = true;
                break;
            }
            SubagentBarrier::Shutdown => {
                shutdown_requested = true;
                break;
            }
            SubagentBarrier::Failed(message) => {
                let _ = event_tx.send(AgentEvent::AgentError { message });
                state.phase = AgentPhase::Error;
                break;
            }
        }

        // 1. Context compression check
        let compressor = llm.context_compressor();
        let token_estimate = compressor.estimate_tokens(&state.messages);

        if compressor.needs_compression(token_estimate, input_budget) {
            state.phase = AgentPhase::CompressingContext;
            let _ = event_tx.send(AgentEvent::StatusUpdate {
                message: "Compressing context...".to_string(),
            });
            let target = (input_budget as f64 * 0.60) as usize;
            tracing::info!(
                turn = turn + 1,
                token_estimate,
                context_window,
                target_tokens = target,
                "Compressing context"
            );
            let messages_to_compress = state.messages.clone();
            let compress_future =
                compressor.compress(&messages_to_compress, token_estimate, target);
            tokio::pin!(compress_future);
            let compressed_result = loop {
                tokio::select! {
                    result = &mut compress_future => break Some(result),
                    cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                        BusyControl::Continue => {}
                        BusyControl::Interrupt => {
                            interrupted = true;
                            break None;
                        }
                        BusyControl::Shutdown => {
                            shutdown_requested = true;
                            break None;
                        }
                    }
                }
            };
            let Some(compressed_result) = compressed_result else {
                break;
            };
            match compressed_result {
                Ok((compressed, new_tokens)) => {
                    tracing::info!(
                        "Compressed context: {} -> {} tokens ({} messages -> {})",
                        token_estimate,
                        new_tokens,
                        state.messages.len(),
                        compressed.len()
                    );
                    state.messages = compressed;
                }
                Err(e) => {
                    tracing::warn!(
                        turn = turn + 1,
                        error = %e,
                        "Context compression failed"
                    );
                    let _ = event_tx.send(AgentEvent::AgentError {
                        message: e.to_string(),
                    });
                    state.phase = AgentPhase::Error;
                    break;
                }
            }
        }

        // 2. Call LLM with streaming
        state.phase = AgentPhase::Generating;
        let text_only_request = std::mem::take(&mut force_text_only);
        let tool_defs = if text_only_request {
            Vec::new()
        } else {
            tools.definitions()
        };
        let _ = event_tx.send(AgentEvent::StatusUpdate {
            message: model_request_status(turn),
        });

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
        let prompt_cache_key = capabilities.prompt_cache_key.then(|| {
            state
                .messages
                .iter()
                .find_map(|message| message.id.clone())
                .unwrap_or_else(|| format!("deepcode-{}", model))
        });
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
            prompt_cache_key,
            ..GenerateParams::default()
        };
        tracing::debug!(
            turn = turn + 1,
            tools = tool_defs.len(),
            "Requesting LLM stream"
        );
        let capture_session_title = state.session_title_pending;
        let title_messages =
            capture_session_title.then(|| messages_with_title_instruction(&state.messages));
        let request_messages = title_messages.as_deref().unwrap_or(&state.messages);
        let stream_result = {
            let stream_future =
                llm.generate_stream(model, request_messages, &tool_defs, None, &params);
            tokio::pin!(stream_future);
            loop {
                tokio::select! {
                    result = &mut stream_future => break Some(result),
                    cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                        BusyControl::Continue => {}
                        BusyControl::Interrupt => {
                            interrupted = true;
                            break None;
                        }
                        BusyControl::Shutdown => {
                            shutdown_requested = true;
                            break None;
                        }
                    }
                }
            }
        };
        let stream = match stream_result {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                tracing::warn!(
                    turn = turn + 1,
                    error = %e,
                    "LLM stream setup failed"
                );
                let _ = event_tx.send(AgentEvent::AgentError {
                    message: e.to_string(),
                });
                state.phase = AgentPhase::Error;
                break;
            }
            None => break,
        };

        // 3. Collect streaming deltas into response blocks
        let streamed =
            match collect_stream_response(stream, cmd_rx, event_tx, capture_session_title).await {
                StreamResponseOutcome::Completed(response) => response,
                StreamResponseOutcome::Interrupted => {
                    interrupted = true;
                    tracing::info!(
                        turn = turn + 1,
                        interrupted,
                        shutdown_requested,
                        "Agent turn stopped during streaming"
                    );
                    break;
                }
                StreamResponseOutcome::Shutdown => {
                    shutdown_requested = true;
                    tracing::info!(
                        turn = turn + 1,
                        interrupted,
                        shutdown_requested,
                        "Agent turn stopped during streaming"
                    );
                    break;
                }
                StreamResponseOutcome::Failed { error } => {
                    tracing::warn!(
                        turn = turn + 1,
                        error = %error,
                        "LLM stream failed"
                    );
                    state.phase = AgentPhase::Error;
                    break;
                }
            };

        // 4. Separate tool calls from text
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
            break;
        }

        if streamed.session_title_resolved {
            state.session_title_pending = false;
        }
        state.push_assistant(assistant_blocks);
        llm.context_compressor()
            .normalize_history(&mut state.messages);
        let _ = event_tx.send(AgentEvent::SessionUpdated {
            messages: state.messages.clone(),
        });

        // 5. If no tool calls, agent is done
        if tool_calls.is_empty() {
            let final_text = final_text_from_blocks(&response_blocks);

            tracing::info!(
                turn = turn + 1,
                input_tokens = streamed.input_tokens,
                output_tokens = streamed.output_tokens,
                final_message_chars = final_text.chars().count(),
                "Agent finished without tool calls"
            );
            let _ = event_tx.send(AgentEvent::AgentFinished {
                final_message: final_text,
            });
            state.phase = AgentPhase::Idle;
            break;
        }

        if text_only_request {
            let final_message = "Stopped because the model retried an identical tool call after it had already failed twice. The repeated call was not executed.".to_string();
            let mut blocked_results = Vec::with_capacity(tool_calls.len());
            for (tool_id, tool_name, _) in &tool_calls {
                let _ = event_tx.send(AgentEvent::ToolCallFailed {
                    id: tool_id.clone(),
                    name: tool_name.clone(),
                    error: REPEATED_CALL_BLOCKED_MESSAGE.to_string(),
                });
                blocked_results.push(ToolResultEntry::new(
                    tool_id,
                    REPEATED_CALL_BLOCKED_MESSAGE,
                    true,
                ));
            }
            state.push_tool_results(blocked_results);
            state.push_assistant(vec![ContentBlock::text(&final_message)]);
            send_session_updated(event_tx, state);
            let _ = event_tx.send(AgentEvent::AgentFinished { final_message });
            state.phase = AgentPhase::Idle;
            break;
        }

        // 6. Execute tool calls with permission checks
        state.phase = AgentPhase::ParsingToolCalls;
        tracing::info!(
            turn = turn + 1,
            tool_calls = tool_calls.len(),
            "Agent requested tool calls"
        );
        let _ = event_tx.send(AgentEvent::StatusUpdate {
            message: format!("Preparing {} tool call(s)...", tool_calls.len()),
        });
        let mut parallel_batch = Vec::new();
        let mut blocked_repeat_calls = 0usize;
        for (tool_id, tool_name, tool_input) in &tool_calls {
            tool_call_count += 1;
            let same_signature_pending = parallel_batch.iter().any(|call: &ParallelToolCall| {
                call.name == *tool_name && call.input == *tool_input
            });
            if failure_guard.should_block(tool_name, tool_input) || same_signature_pending {
                flush_parallel_tool_calls(
                    &mut parallel_batch,
                    tools,
                    cmd_rx,
                    event_tx,
                    state,
                    &mut failure_guard,
                    turn,
                    &mut interrupted,
                    &mut shutdown_requested,
                )
                .await;
                if interrupted || shutdown_requested {
                    break;
                }
            }

            if failure_guard.should_block(tool_name, tool_input) {
                tracing::warn!(
                    turn = turn + 1,
                    tool = %tool_name,
                    tool_id = %tool_id,
                    tool_call_index = tool_call_count,
                    "Repeated tool call blocked before execution"
                );
                let _ = event_tx.send(AgentEvent::ToolCallStarted {
                    id: tool_id.clone(),
                    name: tool_name.clone(),
                    input: tool_input.clone(),
                });
                let _ = event_tx.send(AgentEvent::ToolCallFailed {
                    id: tool_id.clone(),
                    name: tool_name.clone(),
                    error: REPEATED_CALL_BLOCKED_MESSAGE.to_string(),
                });
                push_tool_result_and_update(
                    state,
                    event_tx,
                    tool_id,
                    REPEATED_CALL_BLOCKED_MESSAGE,
                    true,
                );
                blocked_repeat_calls = blocked_repeat_calls.saturating_add(1);
                continue;
            }

            let tool = match tools.get(tool_name) {
                Some(t) => t.clone(),
                None => {
                    flush_parallel_tool_calls(
                        &mut parallel_batch,
                        tools,
                        cmd_rx,
                        event_tx,
                        state,
                        &mut failure_guard,
                        turn,
                        &mut interrupted,
                        &mut shutdown_requested,
                    )
                    .await;
                    if interrupted || shutdown_requested {
                        break;
                    }
                    let err = format!("Tool '{}' not found", tool_name);
                    tracing::warn!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        "Tool not found"
                    );
                    let _ = event_tx.send(AgentEvent::StatusUpdate {
                        message: "Adjusting tool selection...".to_string(),
                    });
                    push_tracked_tool_result_and_update(
                        state,
                        event_tx,
                        &mut failure_guard,
                        turn,
                        tool_id,
                        tool_name,
                        tool_input,
                        &err,
                        true,
                    );
                    continue;
                }
            };

            if let Some(err) = invalid_tool_input_message(tool_name, tool.as_ref(), tool_input) {
                flush_parallel_tool_calls(
                    &mut parallel_batch,
                    tools,
                    cmd_rx,
                    event_tx,
                    state,
                    &mut failure_guard,
                    turn,
                    &mut interrupted,
                    &mut shutdown_requested,
                )
                .await;
                if interrupted || shutdown_requested {
                    break;
                }
                tracing::warn!(
                    turn = turn + 1,
                    tool = %tool_name,
                    tool_id = %tool_id,
                    input = %tool_input,
                    error = %err,
                    "Tool input rejected before execution"
                );
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: "Adjusting tool arguments...".to_string(),
                });
                push_tracked_tool_result_and_update(
                    state,
                    event_tx,
                    &mut failure_guard,
                    turn,
                    tool_id,
                    tool_name,
                    tool_input,
                    &err,
                    true,
                );
                continue;
            }

            if let Err(err) = tool.preflight(tool_input) {
                flush_parallel_tool_calls(
                    &mut parallel_batch,
                    tools,
                    cmd_rx,
                    event_tx,
                    state,
                    &mut failure_guard,
                    turn,
                    &mut interrupted,
                    &mut shutdown_requested,
                )
                .await;
                if interrupted || shutdown_requested {
                    break;
                }
                tracing::warn!(
                    turn = turn + 1,
                    tool = %tool_name,
                    tool_id = %tool_id,
                    input = %tool_input,
                    error = %err,
                    "Tool input rejected by hard policy before permission"
                );
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: "Adjusting tool command...".to_string(),
                });
                push_tracked_tool_result_and_update(
                    state,
                    event_tx,
                    &mut failure_guard,
                    turn,
                    tool_id,
                    tool_name,
                    tool_input,
                    &err.to_string(),
                    true,
                );
                continue;
            }

            let safety = tool.safety();
            let can_run_in_parallel = safety.is_read_only && safety.is_concurrency_safe;
            if !can_run_in_parallel {
                flush_parallel_tool_calls(
                    &mut parallel_batch,
                    tools,
                    cmd_rx,
                    event_tx,
                    state,
                    &mut failure_guard,
                    turn,
                    &mut interrupted,
                    &mut shutdown_requested,
                )
                .await;
                if interrupted || shutdown_requested {
                    break;
                }
            }

            if !safety.is_read_only {
                if let Some(manager) = &task_manager {
                    let worker_barrier = manager.wait_for_workers_idle();
                    tokio::pin!(worker_barrier);
                    let workers_idle = loop {
                        tokio::select! {
                            () = &mut worker_barrier => break true,
                            cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                                BusyControl::Continue => {}
                                BusyControl::Interrupt => {
                                    interrupted = true;
                                    break false;
                                }
                                BusyControl::Shutdown => {
                                    shutdown_requested = true;
                                    break false;
                                }
                            }
                        }
                    };
                    if !workers_idle {
                        break;
                    }
                }
            }

            let mut started = false;
            if !can_run_in_parallel {
                tracing::info!(
                    turn = turn + 1,
                    tool = %tool_name,
                    tool_id = %tool_id,
                    tool_call_index = tool_call_count,
                    "Tool call started"
                );
                let _ = event_tx.send(AgentEvent::ToolCallStarted {
                    id: tool_id.clone(),
                    name: tool_name.clone(),
                    input: tool_input.clone(),
                });
                started = true;
            }

            // Permission check
            let evaluation_result = {
                let mut perm = permissions.lock().await;
                perm.check(tool.as_ref(), tool_input).await
            };

            let evaluation = match evaluation_result {
                Ok(evaluation) => evaluation,
                Err(e) => {
                    flush_parallel_tool_calls(
                        &mut parallel_batch,
                        tools,
                        cmd_rx,
                        event_tx,
                        state,
                        &mut failure_guard,
                        turn,
                        &mut interrupted,
                        &mut shutdown_requested,
                    )
                    .await;
                    if interrupted || shutdown_requested {
                        break;
                    }
                    if !started {
                        tracing::info!(
                            turn = turn + 1,
                            tool = %tool_name,
                            tool_id = %tool_id,
                            tool_call_index = tool_call_count,
                            "Tool call started"
                        );
                        let _ = event_tx.send(AgentEvent::ToolCallStarted {
                            id: tool_id.clone(),
                            name: tool_name.clone(),
                            input: tool_input.clone(),
                        });
                    }
                    tracing::warn!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        error = %e,
                        "Tool permission check failed"
                    );
                    let _ = event_tx.send(AgentEvent::ToolCallFailed {
                        id: tool_id.clone(),
                        name: tool_name.clone(),
                        error: e.to_string(),
                    });
                    push_tracked_tool_result_and_update(
                        state,
                        event_tx,
                        &mut failure_guard,
                        turn,
                        tool_id,
                        tool_name,
                        tool_input,
                        &e.to_string(),
                        true,
                    );
                    continue;
                }
            };

            if matches!(evaluation.decision, Decision::Forbidden) {
                flush_parallel_tool_calls(
                    &mut parallel_batch,
                    tools,
                    cmd_rx,
                    event_tx,
                    state,
                    &mut failure_guard,
                    turn,
                    &mut interrupted,
                    &mut shutdown_requested,
                )
                .await;
                if interrupted || shutdown_requested {
                    break;
                }
                if !started {
                    tracing::info!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        tool_call_index = tool_call_count,
                        "Tool call started"
                    );
                    let _ = event_tx.send(AgentEvent::ToolCallStarted {
                        id: tool_id.clone(),
                        name: tool_name.clone(),
                        input: tool_input.clone(),
                    });
                }
                let resolution = resolve_permission(
                    evaluation,
                    tool_name,
                    tool_input,
                    tool_id,
                    permissions,
                    cmd_rx,
                    event_tx,
                    state,
                    turn,
                )
                .await;
                match resolution {
                    PermissionResolution::DeniedByPolicy(error) => {
                        push_tracked_tool_result_and_update(
                            state,
                            event_tx,
                            &mut failure_guard,
                            turn,
                            tool_id,
                            tool_name,
                            tool_input,
                            &error,
                            true,
                        );
                    }
                    PermissionResolution::Interrupted => {
                        interrupted = true;
                        break;
                    }
                    PermissionResolution::Shutdown => {
                        shutdown_requested = true;
                        break;
                    }
                    PermissionResolution::Approved | PermissionResolution::DeniedByUser => {
                        debug_assert!(false, "forbidden policy produced an unexpected resolution");
                        failure_guard.reset();
                    }
                }
                continue;
            }

            let preview_future = tools.preview_change(tool_name, tool_input.clone());
            tokio::pin!(preview_future);
            let preview_result = loop {
                tokio::select! {
                    result = &mut preview_future => break Some(result),
                    cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                        BusyControl::Continue => {}
                        BusyControl::Interrupt => {
                            interrupted = true;
                            break None;
                        }
                        BusyControl::Shutdown => {
                            shutdown_requested = true;
                            break None;
                        }
                    }
                }
            };
            let Some(preview_result) = preview_result else {
                break;
            };
            let preview = match preview_result {
                Ok(preview) => preview,
                Err(e) => {
                    flush_parallel_tool_calls(
                        &mut parallel_batch,
                        tools,
                        cmd_rx,
                        event_tx,
                        state,
                        &mut failure_guard,
                        turn,
                        &mut interrupted,
                        &mut shutdown_requested,
                    )
                    .await;
                    if interrupted || shutdown_requested {
                        break;
                    }
                    if !started {
                        tracing::info!(
                            turn = turn + 1,
                            tool = %tool_name,
                            tool_id = %tool_id,
                            tool_call_index = tool_call_count,
                            "Tool call started"
                        );
                        let _ = event_tx.send(AgentEvent::ToolCallStarted {
                            id: tool_id.clone(),
                            name: tool_name.clone(),
                            input: tool_input.clone(),
                        });
                    }
                    tracing::warn!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        error = %e,
                        "Tool change preview failed"
                    );
                    let _ = event_tx.send(AgentEvent::ToolCallFailed {
                        id: tool_id.clone(),
                        name: tool_name.clone(),
                        error: e.to_string(),
                    });
                    push_tracked_tool_result_and_update(
                        state,
                        event_tx,
                        &mut failure_guard,
                        turn,
                        tool_id,
                        tool_name,
                        tool_input,
                        &e.to_string(),
                        true,
                    );
                    continue;
                }
            };

            if let Some(preview) = preview {
                flush_parallel_tool_calls(
                    &mut parallel_batch,
                    tools,
                    cmd_rx,
                    event_tx,
                    state,
                    &mut failure_guard,
                    turn,
                    &mut interrupted,
                    &mut shutdown_requested,
                )
                .await;
                if interrupted || shutdown_requested {
                    break;
                }
                if !started {
                    tracing::info!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        tool_call_index = tool_call_count,
                        "Tool call started"
                    );
                    let _ = event_tx.send(AgentEvent::ToolCallStarted {
                        id: tool_id.clone(),
                        name: tool_name.clone(),
                        input: tool_input.clone(),
                    });
                }
                let approved =
                    if preview.is_noop() || matches!(evaluation.decision, Decision::Allow) {
                        true
                    } else {
                        let req_id = uuid::Uuid::new_v4().to_string();
                        tracing::info!(
                            turn = turn + 1,
                            tool = %tool_name,
                            tool_id = %tool_id,
                            request_id = %req_id,
                            path = %preview.path,
                            "File change preview requested"
                        );
                        let _ = event_tx.send(AgentEvent::FileChangePreviewNeeded {
                            request_id: req_id.clone(),
                            tool_name: tool_name.clone(),
                            input: tool_input.clone(),
                            preview: preview.clone(),
                        });
                        state.phase = AgentPhase::WaitingForPermission;

                        match wait_for_file_preview_response(cmd_rx, &req_id, event_tx).await {
                            FilePreviewOutcome::Approved => true,
                            FilePreviewOutcome::Denied => {
                                let err = "File change rejected by user".to_string();
                                failure_guard.reset();
                                tracing::warn!(
                                    turn = turn + 1,
                                    tool = %tool_name,
                                    tool_id = %tool_id,
                                    "File change preview rejected"
                                );
                                let _ = event_tx.send(AgentEvent::ToolCallFailed {
                                    id: tool_id.clone(),
                                    name: tool_name.clone(),
                                    error: err.clone(),
                                });
                                push_tool_result_and_update(state, event_tx, tool_id, &err, true);
                                false
                            }
                            FilePreviewOutcome::Interrupted => {
                                interrupted = true;
                                false
                            }
                            FilePreviewOutcome::Shutdown => {
                                shutdown_requested = true;
                                false
                            }
                        }
                    };

                if !approved {
                    if interrupted || shutdown_requested {
                        break;
                    }
                    continue;
                }

                state.phase = AgentPhase::ExecutingTools;
                let preview_execute_future =
                    tools.execute_previewed(tool_name, tool_input.clone(), preview);
                tokio::pin!(preview_execute_future);
                let preview_execute_result = loop {
                    tokio::select! {
                        result = &mut preview_execute_future => break Some(result),
                        cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                            BusyControl::Continue => {}
                            BusyControl::Interrupt => {
                                interrupted = true;
                                break None;
                            }
                            BusyControl::Shutdown => {
                                shutdown_requested = true;
                                break None;
                            }
                        }
                    }
                };
                let Some(preview_execute_result) = preview_execute_result else {
                    break;
                };
                match preview_execute_result {
                    Ok(result) => {
                        tracing::info!(
                            turn = turn + 1,
                            tool = %tool_name,
                            tool_id = %tool_id,
                            result_chars = result.chars().count(),
                            "Previewed tool call completed"
                        );
                        let _ = event_tx.send(AgentEvent::ToolCallCompleted {
                            id: tool_id.clone(),
                            name: tool_name.clone(),
                            result: result.clone(),
                        });
                        push_tracked_tool_result_and_update(
                            state,
                            event_tx,
                            &mut failure_guard,
                            turn,
                            tool_id,
                            tool_name,
                            tool_input,
                            &result,
                            false,
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            turn = turn + 1,
                            tool = %tool_name,
                            tool_id = %tool_id,
                            error = %e,
                            "Previewed tool call failed"
                        );
                        let _ = event_tx.send(AgentEvent::ToolCallFailed {
                            id: tool_id.clone(),
                            name: tool_name.clone(),
                            error: e.to_string(),
                        });
                        push_tracked_tool_result_and_update(
                            state,
                            event_tx,
                            &mut failure_guard,
                            turn,
                            tool_id,
                            tool_name,
                            tool_input,
                            &e.to_string(),
                            true,
                        );
                    }
                }
                continue;
            }

            if can_run_in_parallel && matches!(evaluation.decision, Decision::Allow) {
                parallel_batch.push(ParallelToolCall {
                    id: tool_id.clone(),
                    name: tool_name.clone(),
                    input: tool_input.clone(),
                    context: ToolExecutionContext {
                        sandbox_policy: Some(evaluation.sandbox_policy.clone()),
                    },
                    call_index: tool_call_count,
                });
                continue;
            }

            flush_parallel_tool_calls(
                &mut parallel_batch,
                tools,
                cmd_rx,
                event_tx,
                state,
                &mut failure_guard,
                turn,
                &mut interrupted,
                &mut shutdown_requested,
            )
            .await;
            if interrupted || shutdown_requested {
                break;
            }
            if !started {
                tracing::info!(
                    turn = turn + 1,
                    tool = %tool_name,
                    tool_id = %tool_id,
                    tool_call_index = tool_call_count,
                    "Tool call started"
                );
                let _ = event_tx.send(AgentEvent::ToolCallStarted {
                    id: tool_id.clone(),
                    name: tool_name.clone(),
                    input: tool_input.clone(),
                });
            }

            let execution_context = ToolExecutionContext {
                sandbox_policy: Some(evaluation.sandbox_policy.clone()),
            };
            let resolution = resolve_permission(
                evaluation,
                tool_name,
                tool_input,
                tool_id,
                permissions,
                cmd_rx,
                event_tx,
                state,
                turn,
            )
            .await;

            match resolution {
                PermissionResolution::Approved => {}
                PermissionResolution::DeniedByPolicy(error) => {
                    push_tracked_tool_result_and_update(
                        state,
                        event_tx,
                        &mut failure_guard,
                        turn,
                        tool_id,
                        tool_name,
                        tool_input,
                        &error,
                        true,
                    );
                    continue;
                }
                PermissionResolution::DeniedByUser => {
                    failure_guard.reset();
                    push_tool_result_and_update(
                        state,
                        event_tx,
                        tool_id,
                        "Permission denied by user",
                        true,
                    );
                    continue;
                }
                PermissionResolution::Interrupted => {
                    interrupted = true;
                    break;
                }
                PermissionResolution::Shutdown => {
                    shutdown_requested = true;
                    break;
                }
            }

            state.phase = AgentPhase::ExecutingTools;
            let execute_future =
                tools.execute_with_context(tool_name, tool_input.clone(), execution_context);
            tokio::pin!(execute_future);
            let execute_result = loop {
                tokio::select! {
                    result = &mut execute_future => break Some(result),
                    cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                        BusyControl::Continue => {}
                        BusyControl::Interrupt => {
                            interrupted = true;
                            break None;
                        }
                        BusyControl::Shutdown => {
                            shutdown_requested = true;
                            break None;
                        }
                    }
                }
            };
            let Some(execute_result) = execute_result else {
                tracing::info!(
                    turn = turn + 1,
                    tool = %tool_name,
                    tool_id = %tool_id,
                    interrupted,
                    shutdown_requested,
                    "Tool call stopped before completion"
                );
                break;
            };
            match execute_result {
                Ok(result) => {
                    tracing::info!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        result_chars = result.chars().count(),
                        "Tool call completed"
                    );
                    let _ = event_tx.send(AgentEvent::ToolCallCompleted {
                        id: tool_id.clone(),
                        name: tool_name.clone(),
                        result: result.clone(),
                    });
                    push_tracked_tool_result_and_update(
                        state,
                        event_tx,
                        &mut failure_guard,
                        turn,
                        tool_id,
                        tool_name,
                        tool_input,
                        &result,
                        false,
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        turn = turn + 1,
                        tool = %tool_name,
                        tool_id = %tool_id,
                        error = %e,
                        "Tool call failed"
                    );
                    let _ = event_tx.send(AgentEvent::ToolCallFailed {
                        id: tool_id.clone(),
                        name: tool_name.clone(),
                        error: e.to_string(),
                    });
                    push_tracked_tool_result_and_update(
                        state,
                        event_tx,
                        &mut failure_guard,
                        turn,
                        tool_id,
                        tool_name,
                        tool_input,
                        &e.to_string(),
                        true,
                    );
                }
            }
        }

        if !interrupted && !shutdown_requested {
            flush_parallel_tool_calls(
                &mut parallel_batch,
                tools,
                cmd_rx,
                event_tx,
                state,
                &mut failure_guard,
                turn,
                &mut interrupted,
                &mut shutdown_requested,
            )
            .await;
        }

        if interrupted || shutdown_requested {
            tracing::info!(
                turn = turn + 1,
                interrupted,
                shutdown_requested,
                "Agent turn stopped after tool execution"
            );
            break;
        }

        if blocked_repeat_calls == tool_calls.len() {
            blocked_only_rounds = blocked_only_rounds.saturating_add(1);
            // Keep tools available after the first block so the model can recover. A second
            // blocked-only round proves it ignored that opportunity and may be safely stopped.
            if blocked_only_rounds >= REPEATED_BLOCKED_ROUND_LIMIT {
                force_text_only = true;
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: "Stopping repeated tool retries and preparing an explanation..."
                        .to_string(),
                });
            } else {
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: "Repeated call blocked; trying a different approach...".to_string(),
                });
            }
        } else {
            blocked_only_rounds = 0;
        }

        // Loop back to LLM call with tool results
        turn = turn.saturating_add(1);
    }

    TurnResult {
        interrupted,
        shutdown_requested,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_failure_guard_warns_after_two_and_blocks_the_third_attempt() {
        let mut guard = RepeatedFailureGuard::default();
        let first_input = serde_json::json!({"path": "a.rs", "old": "x"});
        let reordered_input = serde_json::json!({"old": "x", "path": "a.rs"});

        assert!(!guard.record_failure("edit_file", &first_input));
        assert!(guard.record_failure("edit_file", &reordered_input));
        assert!(guard.should_block("edit_file", &first_input));
    }

    #[test]
    fn repeated_failure_guard_ignores_error_text_but_resets_on_changed_input_or_success() {
        let mut guard = RepeatedFailureGuard::default();
        let input = serde_json::json!({"path": "a.rs"});
        let changed_input = serde_json::json!({"path": "b.rs"});

        assert!(!guard.record_failure("edit_file", &input));
        assert!(guard.record_failure("edit_file", &input));
        assert!(guard.should_block("edit_file", &input));

        assert!(!guard.record_failure("edit_file", &changed_input));
        assert!(!guard.should_block("edit_file", &input));

        assert!(guard.record_failure("edit_file", &changed_input));
        assert!(guard.should_block("edit_file", &changed_input));
        guard.reset();
        assert!(!guard.should_block("edit_file", &changed_input));
    }
}
