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
    resolve_permission, wait_for_file_preview_response, FilePreviewOutcome,
};
use crate::r#loop::{handle_busy_command, BusyControl};
use crate::state::{AgentPhase, AgentState, ToolResultEntry};
use crate::validation::invalid_tool_input_message;

pub(crate) struct TurnResult {
    pub interrupted: bool,
    pub shutdown_requested: bool,
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
            let result = tools
                .execute_with_context(&call.name, call.input, call.context)
                .await
                .map_err(|e| e.to_string());
            ToolExecutionResult {
                id: call.id,
                name: call.name,
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
                    name: execution_result.name,
                    result: result.clone(),
                });
                tool_results.push(ToolResultEntry::new(execution_result.id, result, false));
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
                    name: execution_result.name,
                    error: error.clone(),
                });
                tool_results.push(ToolResultEntry::new(execution_result.id, error, true));
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
) -> TurnResult {
    let mut tool_call_count = 0usize;
    let mut interrupted = false;
    let mut shutdown_requested = false;
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
            match compressor
                .compress(&state.messages, token_estimate, target)
                .await
            {
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
        let tool_defs = tools.definitions();
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
        for (tool_id, tool_name, tool_input) in &tool_calls {
            let tool = match tools.get(tool_name) {
                Some(t) => t.clone(),
                None => {
                    flush_parallel_tool_calls(
                        &mut parallel_batch,
                        tools,
                        cmd_rx,
                        event_tx,
                        state,
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
                    push_tool_result_and_update(state, event_tx, tool_id, &err, true);
                    continue;
                }
            };

            tool_call_count += 1;

            if let Some(err) = invalid_tool_input_message(tool_name, tool.as_ref(), tool_input) {
                flush_parallel_tool_calls(
                    &mut parallel_batch,
                    tools,
                    cmd_rx,
                    event_tx,
                    state,
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
                push_tool_result_and_update(state, event_tx, tool_id, &err, true);
                continue;
            }

            if let Err(err) = tool.preflight(tool_input) {
                flush_parallel_tool_calls(
                    &mut parallel_batch,
                    tools,
                    cmd_rx,
                    event_tx,
                    state,
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
                push_tool_result_and_update(state, event_tx, tool_id, &err.to_string(), true);
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
                    turn,
                    &mut interrupted,
                    &mut shutdown_requested,
                )
                .await;
                if interrupted || shutdown_requested {
                    break;
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
                    push_tool_result_and_update(state, event_tx, tool_id, &e.to_string(), true);
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
                resolve_permission(
                    evaluation,
                    tool_name,
                    tool_input,
                    tool_id,
                    permissions,
                    cmd_rx,
                    event_tx,
                    state,
                    &mut interrupted,
                    &mut shutdown_requested,
                    turn,
                )
                .await;
                if interrupted || shutdown_requested {
                    break;
                }
                continue;
            }

            let preview = match tools.preview_change(tool_name, tool_input.clone()).await {
                Ok(preview) => preview,
                Err(e) => {
                    flush_parallel_tool_calls(
                        &mut parallel_batch,
                        tools,
                        cmd_rx,
                        event_tx,
                        state,
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
                    push_tool_result_and_update(state, event_tx, tool_id, &e.to_string(), true);
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
                match tools
                    .execute_previewed(tool_name, tool_input.clone(), preview)
                    .await
                {
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
                        push_tool_result_and_update(state, event_tx, tool_id, &result, false);
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
                        push_tool_result_and_update(state, event_tx, tool_id, &e.to_string(), true);
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
            let approved = resolve_permission(
                evaluation,
                tool_name,
                tool_input,
                tool_id,
                permissions,
                cmd_rx,
                event_tx,
                state,
                &mut interrupted,
                &mut shutdown_requested,
                turn,
            )
            .await;

            if !approved {
                if interrupted || shutdown_requested {
                    break;
                }
                continue;
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
                    push_tool_result_and_update(state, event_tx, tool_id, &result, false);
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
                    push_tool_result_and_update(state, event_tx, tool_id, &e.to_string(), true);
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

        // Loop back to LLM call with tool results
        turn = turn.saturating_add(1);
    }

    TurnResult {
        interrupted,
        shutdown_requested,
    }
}
