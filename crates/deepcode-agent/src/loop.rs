use std::path::PathBuf;
use std::sync::Arc;

use deepcode_core::error::DeepCodeError;
use deepcode_core::provider::traits::LlmProvider;
use deepcode_permissions::pipeline::PermissionSystem;
use deepcode_tools::registry::ToolRegistry;

use crate::event::{AgentCommand, AgentEvent, EventSender};
use crate::plan_executor::{
    approved_plan_instruction, execute_plan_review, resume_plan_review, PlanDecision,
};
use crate::state::{AgentPhase, AgentState};
use crate::turn_executor::execute_turn;

pub(crate) enum BusyControl {
    Continue,
    Interrupt,
    Shutdown,
}

pub(crate) enum SubagentBarrier {
    Clear,
    Completed(Vec<crate::subagent::AgentTaskResult>),
    Interrupted,
    Shutdown,
    Failed(String),
}

pub(crate) async fn wait_for_uncollected_subagents(
    manager: Option<&Arc<crate::subagent::AgentTaskManager>>,
    cmd_rx: &mut crate::event::CmdReceiver,
    event_tx: &EventSender,
    status: impl FnOnce(usize) -> String,
) -> SubagentBarrier {
    let Some(manager) = manager else {
        return SubagentBarrier::Clear;
    };
    let task_ids = manager.uncollected_task_ids().await;
    if task_ids.is_empty() {
        return SubagentBarrier::Clear;
    }
    let _ = event_tx.send(AgentEvent::StatusUpdate {
        message: status(task_ids.len()),
    });
    let wait_future = manager.wait(&task_ids);
    tokio::pin!(wait_future);
    loop {
        tokio::select! {
            result = &mut wait_future => {
                return match result {
                    Ok(results) => SubagentBarrier::Completed(results),
                    Err(error) => SubagentBarrier::Failed(error.to_string()),
                };
            }
            cmd = cmd_rx.recv() => match handle_busy_command(cmd, event_tx) {
                BusyControl::Continue => {}
                BusyControl::Interrupt => return SubagentBarrier::Interrupted,
                BusyControl::Shutdown => return SubagentBarrier::Shutdown,
            }
        }
    }
}

pub(crate) fn handle_busy_command(
    cmd: Option<AgentCommand>,
    event_tx: &EventSender,
) -> BusyControl {
    if cmd
        .as_ref()
        .is_some_and(crate::subagent::route_nested_command)
    {
        return BusyControl::Continue;
    }
    match cmd {
        Some(AgentCommand::Interrupt) => BusyControl::Interrupt,
        Some(AgentCommand::Shutdown) | None => BusyControl::Shutdown,
        Some(AgentCommand::Process { .. })
        | Some(AgentCommand::PlanProcess { .. })
        | Some(AgentCommand::ResumePlan { .. })
        | Some(AgentCommand::SetPlanMode { .. })
        | Some(AgentCommand::SetModel { .. })
        | Some(AgentCommand::SetAvailableModels { .. })
        | Some(AgentCommand::SetReasoningEffort { .. })
        | Some(AgentCommand::ClearSession)
        | Some(AgentCommand::PermissionsSnapshot) => {
            let _ = event_tx.send(AgentEvent::StatusUpdate {
                message: "Already working; press Esc to interrupt.".to_string(),
            });
            BusyControl::Continue
        }
        Some(AgentCommand::PermissionResponse { .. })
        | Some(AgentCommand::FileChangePreviewResponse { .. })
        | Some(AgentCommand::PlanResponse { .. })
        | Some(AgentCommand::PlanFeedback { .. }) => BusyControl::Continue,
    }
}

#[allow(clippy::too_many_arguments)]
async fn process_user_message(
    llm: &Arc<dyn LlmProvider>,
    tools: &Arc<ToolRegistry>,
    permissions: &Arc<tokio::sync::Mutex<PermissionSystem>>,
    model: &str,
    model_config: (usize, usize),
    cmd_rx: &mut crate::event::CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
    message: String,
    plan_first: bool,
    plan_directory: &std::path::Path,
    task_manager: Option<&Arc<crate::subagent::AgentTaskManager>>,
) -> bool {
    if let Some(manager) = task_manager {
        manager.reset_for_parent_turn().await;
    }
    tracing::info!(
        message_chars = message.chars().count(),
        messages = state.messages.len(),
        plan_first,
        "Agent process started"
    );
    state.push_user(&message);
    let _ = event_tx.send(AgentEvent::SessionUpdated {
        messages: state.messages.clone(),
    });

    if plan_first {
        let decision = execute_plan_review(
            llm,
            tools,
            model,
            model_config,
            cmd_rx,
            event_tx,
            state,
            plan_directory,
            task_manager.cloned(),
        )
        .await;
        if let Err(exit) = handle_plan_decision(decision, event_tx, state, task_manager).await {
            return exit;
        }
    }

    execute_current_turn(
        llm,
        tools,
        permissions,
        model,
        model_config,
        cmd_rx,
        event_tx,
        state,
        task_manager,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn process_resumed_plan(
    llm: &Arc<dyn LlmProvider>,
    tools: &Arc<ToolRegistry>,
    permissions: &Arc<tokio::sync::Mutex<PermissionSystem>>,
    model: &str,
    model_config: (usize, usize),
    cmd_rx: &mut crate::event::CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
    plan_directory: &std::path::Path,
    plan_path: &str,
    task_manager: Option<&Arc<crate::subagent::AgentTaskManager>>,
) -> bool {
    if let Some(manager) = task_manager {
        manager.reset_for_parent_turn().await;
    }
    tracing::info!(plan_path, "Restoring persisted plan review");
    let decision = resume_plan_review(
        llm,
        tools,
        model,
        model_config,
        cmd_rx,
        event_tx,
        state,
        plan_directory,
        plan_path,
        task_manager.cloned(),
    )
    .await;
    if let Err(exit) = handle_plan_decision(decision, event_tx, state, task_manager).await {
        return exit;
    }

    execute_current_turn(
        llm,
        tools,
        permissions,
        model,
        model_config,
        cmd_rx,
        event_tx,
        state,
        task_manager,
    )
    .await
}

async fn handle_plan_decision(
    decision: PlanDecision,
    event_tx: &EventSender,
    state: &mut AgentState,
    task_manager: Option<&Arc<crate::subagent::AgentTaskManager>>,
) -> Result<(), bool> {
    match decision {
        PlanDecision::Approved { plan } => {
            let message = if state.plan_mode_enabled {
                "Executing approved plan. Plan mode remains enabled for future tasks."
            } else {
                "Executing approved plan."
            };
            let _ = event_tx.send(AgentEvent::StatusUpdate {
                message: message.to_string(),
            });
            state.push_user(&approved_plan_instruction(&plan));
            let _ = event_tx.send(AgentEvent::SessionUpdated {
                messages: state.messages.clone(),
            });
            Ok(())
        }
        PlanDecision::Rejected => {
            tracing::info!("Plan rejected by user");
            if let Some(manager) = task_manager {
                manager.cancel_all().await;
            }
            state.phase = AgentPhase::Idle;
            let message = "Plan rejected.".to_string();
            let _ = event_tx.send(AgentEvent::StatusUpdate {
                message: message.clone(),
            });
            let _ = event_tx.send(AgentEvent::AgentFinished {
                final_message: message,
            });
            Err(false)
        }
        PlanDecision::Interrupted => {
            tracing::info!("Agent plan interrupted");
            if let Some(manager) = task_manager {
                manager.interrupt_all().await;
            }
            let _ = event_tx.send(AgentEvent::Interrupted);
            state.phase = AgentPhase::Idle;
            Err(false)
        }
        PlanDecision::Shutdown => {
            tracing::info!("Agent plan shutdown requested");
            if let Some(manager) = task_manager {
                manager.cancel_all().await;
            }
            state.phase = AgentPhase::Finished;
            Err(true)
        }
        PlanDecision::Failed => {
            tracing::info!("Agent plan failed");
            if let Some(manager) = task_manager {
                manager.cancel_all().await;
            }
            if state.phase != AgentPhase::Error {
                state.phase = AgentPhase::Idle;
            }
            Err(false)
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn execute_current_turn(
    llm: &Arc<dyn LlmProvider>,
    tools: &Arc<ToolRegistry>,
    permissions: &Arc<tokio::sync::Mutex<PermissionSystem>>,
    model: &str,
    model_config: (usize, usize),
    cmd_rx: &mut crate::event::CmdReceiver,
    event_tx: &EventSender,
    state: &mut AgentState,
    task_manager: Option<&Arc<crate::subagent::AgentTaskManager>>,
) -> bool {
    let result = execute_turn(
        llm,
        tools,
        permissions,
        model,
        model_config,
        cmd_rx,
        event_tx,
        state,
        task_manager.cloned(),
    )
    .await;

    if result.interrupted {
        tracing::info!("Agent process interrupted");
        if let Some(manager) = task_manager {
            manager.interrupt_all().await;
        }
        let _ = event_tx.send(AgentEvent::Interrupted);
        state.phase = AgentPhase::Idle;
    }
    if result.shutdown_requested {
        tracing::info!("Agent process shutdown requested");
        if let Some(manager) = task_manager {
            manager.cancel_all().await;
        }
        state.phase = AgentPhase::Finished;
        return true;
    }
    if state.phase == AgentPhase::Error {
        if let Some(manager) = task_manager {
            manager.cancel_all().await;
        }
    }
    false
}

/// Run the agent loop.
///
/// This is the main runtime that orchestrates LLM calls, tool execution,
/// and permission checks. It communicates with the UI via mpsc channels.
#[allow(clippy::too_many_arguments)]
pub async fn run(
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    permissions: Arc<tokio::sync::Mutex<PermissionSystem>>,
    model: String,
    model_config: (usize, usize),
    reasoning_effort: Option<String>,
    system_prompt: Option<String>,
    initial_messages: Option<Vec<deepcode_core::types::Message>>,
    plan_directory: PathBuf,
    session_title_enabled: bool,
    cmd_rx: crate::event::CmdReceiver,
    event_tx: EventSender,
) -> Result<(), DeepCodeError> {
    run_internal(
        llm,
        tools,
        permissions,
        model,
        model_config,
        reasoning_effort,
        system_prompt,
        initial_messages,
        plan_directory,
        session_title_enabled,
        cmd_rx,
        event_tx,
        None,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn run_managed(
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    permissions: Arc<tokio::sync::Mutex<PermissionSystem>>,
    model: String,
    model_config: (usize, usize),
    reasoning_effort: Option<String>,
    system_prompt: Option<String>,
    initial_messages: Option<Vec<deepcode_core::types::Message>>,
    plan_directory: PathBuf,
    session_title_enabled: bool,
    cmd_rx: crate::event::CmdReceiver,
    event_tx: EventSender,
    task_manager: Arc<crate::subagent::AgentTaskManager>,
) -> Result<(), DeepCodeError> {
    run_internal(
        llm,
        tools,
        permissions,
        model,
        model_config,
        reasoning_effort,
        system_prompt,
        initial_messages,
        plan_directory,
        session_title_enabled,
        cmd_rx,
        event_tx,
        Some(task_manager),
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn run_internal(
    llm: Arc<dyn LlmProvider>,
    tools: Arc<ToolRegistry>,
    permissions: Arc<tokio::sync::Mutex<PermissionSystem>>,
    model: String,
    model_config: (usize, usize),
    reasoning_effort: Option<String>,
    system_prompt: Option<String>,
    initial_messages: Option<Vec<deepcode_core::types::Message>>,
    plan_directory: PathBuf,
    session_title_enabled: bool,
    cmd_rx: crate::event::CmdReceiver,
    event_tx: EventSender,
    task_manager: Option<Arc<crate::subagent::AgentTaskManager>>,
) -> Result<(), DeepCodeError> {
    let mut model = model;
    let mut model_config = model_config;
    let mut cmd_rx = cmd_rx;
    let mut state = AgentState::new(system_prompt, initial_messages);
    llm.context_compressor()
        .normalize_history(&mut state.messages);
    if !session_title_enabled {
        state.session_title_pending = false;
    }
    state.reasoning_effort = reasoning_effort;
    let (max_tokens, context_window) = model_config;
    tracing::info!(
        model = %model,
        max_tokens,
        context_window,
        "Agent loop started"
    );

    while let Some(cmd) = cmd_rx.recv().await {
        match cmd {
            AgentCommand::Shutdown => {
                tracing::info!("Agent shutdown requested");
                if let Some(manager) = &task_manager {
                    manager.cancel_all().await;
                }
                state.phase = AgentPhase::Finished;
                break;
            }
            AgentCommand::Interrupt => {
                tracing::info!("Agent interrupted while idle");
                if let Some(manager) = &task_manager {
                    manager.interrupt_all().await;
                }
                let _ = event_tx.send(AgentEvent::Interrupted);
                state.phase = AgentPhase::Idle;
            }
            AgentCommand::PermissionResponse {
                request_id,
                approved,
                scope,
            } => {
                let command = AgentCommand::PermissionResponse {
                    request_id,
                    approved,
                    scope,
                };
                let _ = crate::subagent::route_nested_command(&command);
            }
            AgentCommand::PlanResponse {
                request_id,
                approved,
            } => {
                let command = AgentCommand::PlanResponse {
                    request_id,
                    approved,
                };
                let _ = crate::subagent::route_nested_command(&command);
            }
            AgentCommand::PlanFeedback {
                request_id,
                feedback,
            } => {
                let command = AgentCommand::PlanFeedback {
                    request_id,
                    feedback,
                };
                let _ = crate::subagent::route_nested_command(&command);
            }
            AgentCommand::FileChangePreviewResponse {
                request_id,
                approved,
            } => {
                let command = AgentCommand::FileChangePreviewResponse {
                    request_id,
                    approved,
                };
                let _ = crate::subagent::route_nested_command(&command);
            }
            AgentCommand::SetPlanMode { enabled } => {
                state.plan_mode_enabled = enabled;
                let status = if enabled {
                    "Plan mode enabled. New tasks will require plan approval before execution."
                } else {
                    "Agent mode enabled. New tasks will execute directly."
                };
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: status.to_string(),
                });
            }
            AgentCommand::SetModel {
                model: new_model,
                max_tokens,
                context_window,
            } => {
                model = new_model;
                model_config = (max_tokens, context_window);
                if let Some(manager) = &task_manager {
                    manager.update_main_model(model.clone()).await;
                }
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: format!("Model switched to {}.", model),
                });
            }
            AgentCommand::SetAvailableModels { models } => {
                if let Some(manager) = &task_manager {
                    manager.update_models(models).await;
                }
            }
            AgentCommand::SetReasoningEffort { effort } => {
                state.reasoning_effort = effort;
                if let Some(manager) = &task_manager {
                    manager
                        .update_main_reasoning_effort(state.reasoning_effort.clone())
                        .await;
                }
                let label = state.reasoning_effort.as_deref().unwrap_or("off");
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: format!("Reasoning effort set to {}.", label),
                });
            }
            AgentCommand::ClearSession => {
                state.clear_conversation();
                let _ = event_tx.send(AgentEvent::SessionUpdated {
                    messages: state.messages.clone(),
                });
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: "Conversation cleared.".to_string(),
                });
            }
            AgentCommand::PermissionsSnapshot => {
                let lines = permissions.lock().await.snapshot_lines();
                let _ = event_tx.send(AgentEvent::PermissionsSnapshot { lines });
            }
            AgentCommand::ResumePlan { plan_path } => {
                if process_resumed_plan(
                    &llm,
                    &tools,
                    &permissions,
                    &model,
                    model_config,
                    &mut cmd_rx,
                    &event_tx,
                    &mut state,
                    &plan_directory,
                    &plan_path,
                    task_manager.as_ref(),
                )
                .await
                {
                    break;
                }
            }
            AgentCommand::PlanProcess { message } => {
                if process_user_message(
                    &llm,
                    &tools,
                    &permissions,
                    &model,
                    model_config,
                    &mut cmd_rx,
                    &event_tx,
                    &mut state,
                    message,
                    true,
                    &plan_directory,
                    task_manager.as_ref(),
                )
                .await
                {
                    break;
                }
            }
            AgentCommand::Process { message } => {
                let plan_first = state.plan_mode_enabled;
                if process_user_message(
                    &llm,
                    &tools,
                    &permissions,
                    &model,
                    model_config,
                    &mut cmd_rx,
                    &event_tx,
                    &mut state,
                    message,
                    plan_first,
                    &plan_directory,
                    task_manager.as_ref(),
                )
                .await
                {
                    break;
                }
            }
        }
    }

    if let Some(manager) = &task_manager {
        manager.cancel_all().await;
    }
    tracing::info!("Agent loop exited");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use deepcode_core::error::{DeepCodeError, Result as CoreResult};
    use deepcode_core::provider::traits::{
        ContextCompressor, FinishReason, GenerateParams, GenerateResponse, RequestBuilder,
        ResponseParser, StreamDelta, Usage,
    };
    use deepcode_core::types::{ContentBlock, Message, Role, ToolDefinition};
    use deepcode_permissions::pipeline::PermissionSystem;
    use deepcode_tools::registry::ToolRegistry;
    use deepcode_tools::tool::{FileChangePreview, Tool, ToolSafety};
    use futures::stream::{self, Stream};
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn test_plan_directory() -> PathBuf {
        std::env::temp_dir()
            .join(format!("deepcode-plan-test-{}", uuid::Uuid::new_v4()))
            .join("plans")
    }

    fn cleanup_test_plan_directory(plan_directory: &std::path::Path) {
        if let Some(root) = plan_directory.parent() {
            let _ = std::fs::remove_dir_all(root);
        }
    }

    struct TestRequestBuilder;

    #[async_trait]
    impl RequestBuilder for TestRequestBuilder {
        fn build_request(
            &self,
            _model: &str,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _system_prompt: Option<&str>,
            _params: &GenerateParams,
        ) -> CoreResult<serde_json::Value> {
            Ok(serde_json::json!({}))
        }
    }

    struct TestResponseParser;

    #[async_trait]
    impl ResponseParser for TestResponseParser {
        fn parse_response(&self, _raw_body: &serde_json::Value) -> CoreResult<GenerateResponse> {
            Ok(GenerateResponse {
                content: vec![],
                usage: Usage::default(),
                finish_reason: FinishReason::Stop,
            })
        }

        fn parse_stream_chunk(&self, _raw_line: &str) -> CoreResult<Option<StreamDelta>> {
            Ok(None)
        }
    }

    struct TestCompressor;

    #[async_trait]
    impl ContextCompressor for TestCompressor {
        fn needs_compression(&self, _token_count: usize, _context_window: usize) -> bool {
            false
        }

        fn estimate_tokens(&self, _messages: &[Message]) -> usize {
            1
        }

        async fn compress(
            &self,
            messages: &[Message],
            _current_tokens: usize,
            _target_tokens: usize,
        ) -> CoreResult<(Vec<Message>, usize)> {
            Ok((messages.to_vec(), 1))
        }
    }

    struct ToolLoopProvider {
        calls: AtomicUsize,
        pending_stream: bool,
        refusal: bool,
        tool_calls: Vec<(String, String, String)>,
        request_builder: TestRequestBuilder,
        response_parser: TestResponseParser,
        compressor: TestCompressor,
    }

    struct RepeatingFailureProvider {
        calls: AtomicUsize,
        recover_after_block: bool,
        tool_name: String,
        tool_input: String,
        request_builder: TestRequestBuilder,
        response_parser: TestResponseParser,
        compressor: TestCompressor,
    }

    struct SubagentBarrierProvider {
        parent_calls: AtomicUsize,
        child_finished: Arc<std::sync::atomic::AtomicBool>,
        child_delay: std::time::Duration,
        request_builder: TestRequestBuilder,
        response_parser: TestResponseParser,
        compressor: TestCompressor,
    }

    impl RepeatingFailureProvider {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
                recover_after_block: false,
                tool_name: "always_fail".to_string(),
                tool_input: "{\"value\":\"same\"}".to_string(),
                request_builder: TestRequestBuilder,
                response_parser: TestResponseParser,
                compressor: TestCompressor,
            }
        }

        fn recovering() -> Self {
            Self {
                recover_after_block: true,
                ..Self::new()
            }
        }

        fn repeating(tool_name: &str, tool_input: &str) -> Self {
            Self {
                tool_name: tool_name.to_string(),
                tool_input: tool_input.to_string(),
                ..Self::new()
            }
        }
    }

    impl ToolLoopProvider {
        fn new() -> Self {
            Self::with_tool_input("{\"value\":\"ok\"}")
        }

        fn missing_required_tool_input() -> Self {
            Self::with_tool_input("{}")
        }

        fn with_tool_call(tool_name: &str, tool_input_delta: &str) -> Self {
            Self::with_tool_calls(vec![("call_1", tool_name, tool_input_delta)])
        }

        fn with_tool_calls(tool_calls: Vec<(&str, &str, &str)>) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                pending_stream: false,
                refusal: false,
                tool_calls: tool_calls
                    .into_iter()
                    .map(|(id, name, input_delta)| {
                        (id.to_string(), name.to_string(), input_delta.to_string())
                    })
                    .collect(),
                request_builder: TestRequestBuilder,
                response_parser: TestResponseParser,
                compressor: TestCompressor,
            }
        }

        fn pending() -> Self {
            let mut provider = Self::new();
            provider.pending_stream = true;
            provider
        }

        fn refusal() -> Self {
            let mut provider = Self::new();
            provider.refusal = true;
            provider
        }

        fn with_tool_input(tool_input_delta: &str) -> Self {
            Self::with_tool_call("echo_tool", tool_input_delta)
        }
    }

    #[async_trait]
    impl LlmProvider for SubagentBarrierProvider {
        fn name(&self) -> &str {
            "subagent-barrier-test"
        }

        fn request_builder(&self) -> &dyn RequestBuilder {
            &self.request_builder
        }

        fn response_parser(&self) -> &dyn ResponseParser {
            &self.response_parser
        }

        fn context_compressor(&self) -> &dyn ContextCompressor {
            &self.compressor
        }

        async fn generate_stream(
            &self,
            _model: &str,
            messages: &[Message],
            _tools: &[ToolDefinition],
            _system_prompt: Option<&str>,
            _params: &GenerateParams,
        ) -> CoreResult<Pin<Box<dyn Stream<Item = CoreResult<StreamDelta>> + Send>>> {
            let is_child = messages.iter().any(|message| {
                message.role == Role::System
                    && message.content.iter().any(|block| {
                        matches!(
                            block,
                            ContentBlock::Text { text }
                                if text.contains("read-only code explorer")
                        )
                    })
            });
            if is_child {
                let child_finished = Arc::clone(&self.child_finished);
                let child_delay = self.child_delay;
                return Ok(Box::pin(stream::once(async move {
                    tokio::time::sleep(child_delay).await;
                    child_finished.store(true, Ordering::SeqCst);
                    Ok(StreamDelta::Batch(vec![
                        StreamDelta::TextDelta("child evidence".to_string()),
                        StreamDelta::Finished(FinishReason::Stop),
                    ]))
                })));
            }

            let call = self.parent_calls.fetch_add(1, Ordering::SeqCst);
            if call == 0 {
                return Ok(Box::pin(stream::iter(
                    vec![
                        StreamDelta::ToolUseStart {
                            id: "spawn_1".to_string(),
                            name: "spawn_agent".to_string(),
                            index: None,
                            input_delta: None,
                        },
                        StreamDelta::ToolUseInput {
                            id: "spawn_1".to_string(),
                            index: None,
                            input_delta: "{\"task\":\"inspect\",\"role\":\"explorer\"}".to_string(),
                        },
                        StreamDelta::ToolUseEnd {
                            id: "spawn_1".to_string(),
                            index: None,
                        },
                        StreamDelta::Finished(FinishReason::ToolCalls),
                    ]
                    .into_iter()
                    .map(Ok),
                )));
            }

            assert!(
                self.child_finished.load(Ordering::SeqCst),
                "parent requested a conclusion before the child finished"
            );
            assert!(messages.iter().any(|message| {
                message.role == Role::User
                    && message.content.iter().any(|block| {
                        matches!(
                            block,
                            ContentBlock::Text { text }
                                if text.contains("Subagents have finished")
                                    && text.contains("child evidence")
                        )
                    })
            }));
            Ok(Box::pin(stream::iter(
                vec![
                    StreamDelta::TextDelta("final after child".to_string()),
                    StreamDelta::Finished(FinishReason::Stop),
                ]
                .into_iter()
                .map(Ok),
            )))
        }

        async fn send_request(&self, _body: &serde_json::Value) -> CoreResult<serde_json::Value> {
            Ok(serde_json::json!({}))
        }
    }

    #[async_trait]
    impl LlmProvider for ToolLoopProvider {
        fn name(&self) -> &str {
            "test"
        }

        fn request_builder(&self) -> &dyn RequestBuilder {
            &self.request_builder
        }

        fn response_parser(&self) -> &dyn ResponseParser {
            &self.response_parser
        }

        fn context_compressor(&self) -> &dyn ContextCompressor {
            &self.compressor
        }

        async fn generate_stream(
            &self,
            _model: &str,
            messages: &[Message],
            _tools: &[ToolDefinition],
            _system_prompt: Option<&str>,
            _params: &GenerateParams,
        ) -> CoreResult<Pin<Box<dyn Stream<Item = CoreResult<StreamDelta>> + Send>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if self.pending_stream {
                return Ok(Box::pin(stream::pending()));
            }
            if self.refusal {
                return Ok(Box::pin(stream::iter(
                    vec![
                        StreamDelta::TextDelta("I cannot help with that.".to_string()),
                        StreamDelta::Usage {
                            input_tokens: 10,
                            output_tokens: 5,
                            cached_input_tokens: 0,
                            cache_miss_input_tokens: 0,
                            reasoning_output_tokens: 0,
                        },
                        StreamDelta::Finished(FinishReason::ContentFilter),
                        StreamDelta::Finished(FinishReason::Stop),
                    ]
                    .into_iter()
                    .map(Ok),
                )));
            }
            let has_tool_result = messages.iter().any(|m| matches!(m.role, Role::Tool));
            let deltas = if has_tool_result {
                vec![
                    StreamDelta::TextDelta("done".to_string()),
                    StreamDelta::Usage {
                        input_tokens: 10,
                        output_tokens: 5,
                        cached_input_tokens: 0,
                        cache_miss_input_tokens: 0,
                        reasoning_output_tokens: 0,
                    },
                    StreamDelta::Finished(FinishReason::Stop),
                ]
            } else {
                let mut deltas = Vec::new();
                for (id, name, input_delta) in &self.tool_calls {
                    deltas.push(StreamDelta::ToolUseStart {
                        id: id.clone(),
                        name: name.clone(),
                        index: None,
                        input_delta: None,
                    });
                    deltas.push(StreamDelta::ToolUseInput {
                        id: id.clone(),
                        index: None,
                        input_delta: input_delta.clone(),
                    });
                    deltas.push(StreamDelta::ToolUseEnd {
                        id: id.clone(),
                        index: None,
                    });
                }
                deltas.push(StreamDelta::Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cached_input_tokens: 0,
                    cache_miss_input_tokens: 0,
                    reasoning_output_tokens: 0,
                });
                deltas.push(StreamDelta::Finished(FinishReason::ToolCalls));
                deltas
            };
            Ok(Box::pin(stream::iter(deltas.into_iter().map(Ok))))
        }

        async fn send_request(&self, _body: &serde_json::Value) -> CoreResult<serde_json::Value> {
            Ok(serde_json::json!({}))
        }
    }

    #[async_trait]
    impl LlmProvider for RepeatingFailureProvider {
        fn name(&self) -> &str {
            "repeating-failure-test"
        }

        fn request_builder(&self) -> &dyn RequestBuilder {
            &self.request_builder
        }

        fn response_parser(&self) -> &dyn ResponseParser {
            &self.response_parser
        }

        fn context_compressor(&self) -> &dyn ContextCompressor {
            &self.compressor
        }

        async fn generate_stream(
            &self,
            _model: &str,
            _messages: &[Message],
            tools: &[ToolDefinition],
            _system_prompt: Option<&str>,
            _params: &GenerateParams,
        ) -> CoreResult<Pin<Box<dyn Stream<Item = CoreResult<StreamDelta>> + Send>>> {
            let request_index = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            let deltas = if tools.is_empty() {
                vec![
                    StreamDelta::TextDelta(
                        "I could not complete the repeated operation.".to_string(),
                    ),
                    StreamDelta::Finished(FinishReason::Stop),
                ]
            } else if self.recover_after_block && request_index == 4 {
                vec![
                    StreamDelta::ToolUseStart {
                        id: "recovery_call".to_string(),
                        name: "echo_tool".to_string(),
                        index: None,
                        input_delta: None,
                    },
                    StreamDelta::ToolUseInput {
                        id: "recovery_call".to_string(),
                        index: None,
                        input_delta: "{\"value\":\"recovered\"}".to_string(),
                    },
                    StreamDelta::ToolUseEnd {
                        id: "recovery_call".to_string(),
                        index: None,
                    },
                    StreamDelta::Finished(FinishReason::ToolCalls),
                ]
            } else if self.recover_after_block && request_index >= 5 {
                vec![
                    StreamDelta::TextDelta("Recovered with a different approach.".to_string()),
                    StreamDelta::Finished(FinishReason::Stop),
                ]
            } else {
                vec![
                    StreamDelta::ToolUseStart {
                        id: format!("repeat_call_{request_index}"),
                        name: self.tool_name.clone(),
                        index: None,
                        input_delta: None,
                    },
                    StreamDelta::ToolUseInput {
                        id: format!("repeat_call_{request_index}"),
                        index: None,
                        input_delta: self.tool_input.clone(),
                    },
                    StreamDelta::ToolUseEnd {
                        id: format!("repeat_call_{request_index}"),
                        index: None,
                    },
                    StreamDelta::Finished(FinishReason::ToolCalls),
                ]
            };
            Ok(Box::pin(stream::iter(deltas.into_iter().map(Ok))))
        }

        async fn send_request(&self, _body: &serde_json::Value) -> CoreResult<serde_json::Value> {
            Ok(serde_json::json!({}))
        }
    }

    struct EchoTool;

    struct AlwaysFailTool {
        executions: Arc<AtomicUsize>,
    }

    struct TestShellTool {
        executions: Arc<AtomicUsize>,
    }

    struct PreflightRejectTool {
        executions: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo_tool"
        }

        fn description(&self) -> &str {
            "Echo test input"
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            })
        }

        fn safety(&self) -> ToolSafety {
            ToolSafety::READ_ONLY
        }

        async fn execute(&self, input: serde_json::Value) -> CoreResult<String> {
            Ok(input["value"].as_str().unwrap_or("").to_string())
        }
    }

    #[async_trait]
    impl Tool for AlwaysFailTool {
        fn name(&self) -> &str {
            "always_fail"
        }

        fn description(&self) -> &str {
            "Always returns the same failure"
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            })
        }

        fn safety(&self) -> ToolSafety {
            ToolSafety::READ_ONLY
        }

        async fn execute(&self, _input: serde_json::Value) -> CoreResult<String> {
            self.executions.fetch_add(1, Ordering::SeqCst);
            Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "deterministic failure".to_string(),
            })
        }
    }

    #[async_trait]
    impl Tool for TestShellTool {
        fn name(&self) -> &str {
            "shell"
        }

        fn description(&self) -> &str {
            "Test shell"
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            })
        }

        fn safety(&self) -> ToolSafety {
            ToolSafety::DESTRUCTIVE
        }

        async fn execute(&self, _input: serde_json::Value) -> CoreResult<String> {
            self.executions.fetch_add(1, Ordering::SeqCst);
            Ok("executed".to_string())
        }
    }

    #[async_trait]
    impl Tool for PreflightRejectTool {
        fn name(&self) -> &str {
            "preflight_reject_tool"
        }

        fn description(&self) -> &str {
            "Rejects test input before permission"
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            })
        }

        fn safety(&self) -> ToolSafety {
            ToolSafety::DESTRUCTIVE
        }

        fn preflight(&self, _input: &serde_json::Value) -> CoreResult<()> {
            Err(DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "blocked by hard test policy".to_string(),
            })
        }

        async fn execute(&self, _input: serde_json::Value) -> CoreResult<String> {
            self.executions.fetch_add(1, Ordering::SeqCst);
            Ok("should not execute".to_string())
        }
    }

    struct PreviewTool {
        executions: Arc<AtomicUsize>,
    }

    struct DelayedEchoTool {
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
    }

    impl DelayedEchoTool {
        fn new(active: Arc<AtomicUsize>, max_active: Arc<AtomicUsize>) -> Self {
            Self { active, max_active }
        }

        fn update_max_active(&self, active: usize) {
            let mut current = self.max_active.load(Ordering::SeqCst);
            while active > current {
                match self.max_active.compare_exchange(
                    current,
                    active,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(next) => current = next,
                }
            }
        }
    }

    #[async_trait]
    impl Tool for DelayedEchoTool {
        fn name(&self) -> &str {
            "delayed_echo"
        }

        fn description(&self) -> &str {
            "Echo after a delay"
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "delay_ms": {"type": "integer"}
                },
                "required": ["value"]
            })
        }

        fn safety(&self) -> ToolSafety {
            ToolSafety::READ_ONLY
        }

        async fn execute(&self, input: serde_json::Value) -> CoreResult<String> {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.update_max_active(active);
            let delay_ms = input["delay_ms"].as_u64().unwrap_or(0);
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            self.active.fetch_sub(1, Ordering::SeqCst);
            Ok(input["value"].as_str().unwrap_or("").to_string())
        }
    }

    #[async_trait]
    impl Tool for PreviewTool {
        fn name(&self) -> &str {
            "preview_tool"
        }

        fn description(&self) -> &str {
            "Preview test tool"
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"]
            })
        }

        fn safety(&self) -> ToolSafety {
            ToolSafety::SAFE_MUTATION
        }

        async fn execute(&self, _input: serde_json::Value) -> CoreResult<String> {
            self.executions.fetch_add(1, Ordering::SeqCst);
            Ok("executed without preview".to_string())
        }

        async fn preview_change(
            &self,
            _input: serde_json::Value,
        ) -> CoreResult<Option<FileChangePreview>> {
            Ok(Some(FileChangePreview {
                path: "test.txt".to_string(),
                before_exists: true,
                before: "old\n".to_string(),
                after: "new\n".to_string(),
                unified_diff: "--- a/test.txt\n+++ b/test.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n"
                    .to_string(),
            }))
        }

        async fn execute_previewed(
            &self,
            _input: serde_json::Value,
            _preview: FileChangePreview,
        ) -> CoreResult<String> {
            self.executions.fetch_add(1, Ordering::SeqCst);
            Ok("applied".to_string())
        }
    }

    async fn run_test_agent_with_provider(
        llm: Arc<dyn LlmProvider>,
        interrupt: bool,
    ) -> Vec<AgentEvent> {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let tools = Arc::new(registry);
        let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
            deepcode_permissions::policy::PermissionSystemConfig::default(),
        )));
        let (cmd_tx, cmd_rx) = crate::event::cmd_channel(8);
        let (event_tx, mut event_rx) = crate::event::event_channel();

        let handle = tokio::spawn(run(
            llm,
            tools,
            permissions,
            "test-model".to_string(),
            (128, 4096),
            None,
            None,
            None,
            test_plan_directory(),
            false,
            cmd_rx,
            event_tx,
        ));

        cmd_tx
            .send(AgentCommand::Process {
                message: "go".to_string(),
            })
            .await
            .unwrap();
        if interrupt {
            cmd_tx.send(AgentCommand::Interrupt).await.unwrap();
        }

        let mut events = Vec::new();
        while let Some(event) = event_rx.recv().await {
            let done = matches!(
                event,
                AgentEvent::AgentFinished { .. } | AgentEvent::AgentError { .. }
            ) || (interrupt && matches!(event, AgentEvent::Interrupted));
            events.push(event);
            if done {
                break;
            }
        }

        drop(cmd_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        events
    }

    async fn run_test_agent() -> Vec<AgentEvent> {
        run_test_agent_with_provider(Arc::new(ToolLoopProvider::new()), false).await
    }

    #[tokio::test]
    async fn refusal_is_reported_without_persisting_assistant_content() {
        let events =
            run_test_agent_with_provider(Arc::new(ToolLoopProvider::refusal()), false).await;

        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::AgentError { .. })));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::AgentFinished { .. })));
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::TextDelta(text) if text.contains("cannot help"))
        ));

        let latest_messages = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::SessionUpdated { messages } => Some(messages),
                _ => None,
            })
            .next_back()
            .expect("the user message should update the session");
        assert!(latest_messages
            .iter()
            .all(|message| message.role != Role::Assistant));
    }

    async fn run_agent_with_registry(
        llm: Arc<dyn LlmProvider>,
        registry: ToolRegistry,
        auto_approve_file_previews: bool,
    ) -> Vec<AgentEvent> {
        let tools = Arc::new(registry);
        let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
            deepcode_permissions::policy::PermissionSystemConfig::default(),
        )));
        let (cmd_tx, cmd_rx) = crate::event::cmd_channel(16);
        let (event_tx, mut event_rx) = crate::event::event_channel();

        let handle = tokio::spawn(run(
            llm,
            tools,
            permissions,
            "test-model".to_string(),
            (128, 4096),
            None,
            None,
            None,
            test_plan_directory(),
            false,
            cmd_rx,
            event_tx,
        ));

        cmd_tx
            .send(AgentCommand::Process {
                message: "go".to_string(),
            })
            .await
            .unwrap();

        let mut events = Vec::new();
        while let Some(event) = event_rx.recv().await {
            if auto_approve_file_previews {
                if let AgentEvent::FileChangePreviewNeeded { request_id, .. } = &event {
                    cmd_tx
                        .send(AgentCommand::FileChangePreviewResponse {
                            request_id: request_id.clone(),
                            approved: true,
                        })
                        .await
                        .unwrap();
                }
            }
            let done = matches!(
                event,
                AgentEvent::AgentFinished { .. } | AgentEvent::AgentError { .. }
            );
            events.push(event);
            if done {
                break;
            }
        }

        drop(cmd_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        events
    }

    async fn run_preview_agent(approved: bool) -> (Vec<AgentEvent>, usize) {
        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(PreviewTool {
            executions: executions.clone(),
        }));
        let tools = Arc::new(registry);
        let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
            deepcode_permissions::policy::PermissionSystemConfig::default(),
        )));
        let (cmd_tx, cmd_rx) = crate::event::cmd_channel(8);
        let (event_tx, mut event_rx) = crate::event::event_channel();

        let handle = tokio::spawn(run(
            Arc::new(ToolLoopProvider::with_tool_call(
                "preview_tool",
                "{\"value\":\"ok\"}",
            )),
            tools,
            permissions,
            "test-model".to_string(),
            (128, 4096),
            None,
            None,
            None,
            test_plan_directory(),
            false,
            cmd_rx,
            event_tx,
        ));

        cmd_tx
            .send(AgentCommand::Process {
                message: "go".to_string(),
            })
            .await
            .unwrap();

        let mut events = Vec::new();
        while let Some(event) = event_rx.recv().await {
            if let AgentEvent::FileChangePreviewNeeded { request_id, .. } = &event {
                assert_eq!(executions.load(Ordering::SeqCst), 0);
                cmd_tx
                    .send(AgentCommand::FileChangePreviewResponse {
                        request_id: request_id.clone(),
                        approved,
                    })
                    .await
                    .unwrap();
            }
            let done = matches!(
                event,
                AgentEvent::AgentFinished { .. } | AgentEvent::AgentError { .. }
            );
            events.push(event);
            if done {
                break;
            }
        }

        drop(cmd_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        (events, executions.load(Ordering::SeqCst))
    }

    async fn run_permission_agent(approved: bool) -> (Vec<AgentEvent>, usize) {
        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(TestShellTool {
            executions: executions.clone(),
        }));
        let tools = Arc::new(registry);
        let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
            deepcode_permissions::policy::PermissionSystemConfig::default(),
        )));
        let (cmd_tx, cmd_rx) = crate::event::cmd_channel(8);
        let (event_tx, mut event_rx) = crate::event::event_channel();

        let handle = tokio::spawn(run(
            Arc::new(ToolLoopProvider::with_tool_call(
                "shell",
                "{\"command\":\"cargo check\"}",
            )),
            tools,
            permissions,
            "test-model".to_string(),
            (128, 4096),
            None,
            None,
            None,
            test_plan_directory(),
            false,
            cmd_rx,
            event_tx,
        ));

        cmd_tx
            .send(AgentCommand::Process {
                message: "go".to_string(),
            })
            .await
            .unwrap();

        let mut events = Vec::new();
        while let Some(event) = event_rx.recv().await {
            if let AgentEvent::PermissionNeeded { request_id, .. } = &event {
                cmd_tx
                    .send(AgentCommand::PermissionResponse {
                        request_id: request_id.clone(),
                        approved,
                        scope: deepcode_permissions::policy::ApprovalScope::Once,
                    })
                    .await
                    .unwrap();
            }
            let done = matches!(
                event,
                AgentEvent::AgentFinished { .. } | AgentEvent::AgentError { .. }
            );
            events.push(event);
            if done {
                break;
            }
        }

        drop(cmd_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        (events, executions.load(Ordering::SeqCst))
    }

    async fn run_plan_agent_with_registry(
        llm: Arc<dyn LlmProvider>,
        registry: ToolRegistry,
        approved: bool,
        persistent_mode: bool,
        executions: Option<Arc<AtomicUsize>>,
    ) -> (Vec<AgentEvent>, usize) {
        let tools = Arc::new(registry);
        let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
            deepcode_permissions::policy::PermissionSystemConfig::default(),
        )));
        let (cmd_tx, cmd_rx) = crate::event::cmd_channel(8);
        let (event_tx, mut event_rx) = crate::event::event_channel();
        let plan_directory = test_plan_directory();

        let handle = tokio::spawn(run(
            llm,
            tools,
            permissions,
            "test-model".to_string(),
            (128, 4096),
            None,
            None,
            None,
            plan_directory.clone(),
            false,
            cmd_rx,
            event_tx,
        ));

        if persistent_mode {
            cmd_tx
                .send(AgentCommand::SetPlanMode { enabled: true })
                .await
                .unwrap();
            cmd_tx
                .send(AgentCommand::Process {
                    message: "go".to_string(),
                })
                .await
                .unwrap();
        } else {
            cmd_tx
                .send(AgentCommand::PlanProcess {
                    message: "go".to_string(),
                })
                .await
                .unwrap();
        }

        let mut events = Vec::new();
        while let Some(event) = event_rx.recv().await {
            if let AgentEvent::PlanApprovalNeeded {
                request_id,
                plan,
                plan_path,
                ..
            } = &event
            {
                assert_eq!(
                    std::fs::read_to_string(plan_path).unwrap().trim(),
                    plan.trim()
                );
                cmd_tx
                    .send(AgentCommand::PlanResponse {
                        request_id: request_id.clone(),
                        approved,
                    })
                    .await
                    .unwrap();
            }
            let done = matches!(
                event,
                AgentEvent::AgentFinished { .. } | AgentEvent::AgentError { .. }
            );
            events.push(event);
            if done {
                break;
            }
        }

        drop(cmd_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        let executions = executions
            .as_ref()
            .map(|count| count.load(Ordering::SeqCst))
            .unwrap_or(0);
        cleanup_test_plan_directory(&plan_directory);
        (events, executions)
    }

    async fn run_plan_agent(approved: bool, persistent_mode: bool) -> Vec<AgentEvent> {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let (events, _) = run_plan_agent_with_registry(
            Arc::new(ToolLoopProvider::new()),
            registry,
            approved,
            persistent_mode,
            None,
        )
        .await;
        events
    }

    #[derive(Clone, Copy)]
    enum InvalidPlanFile {
        Missing,
        Empty,
        #[cfg(unix)]
        Symlink,
    }

    async fn run_agent_with_invalid_plan_file(mutation: InvalidPlanFile) -> Vec<AgentEvent> {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let tools = Arc::new(registry);
        let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
            deepcode_permissions::policy::PermissionSystemConfig::default(),
        )));
        let (cmd_tx, cmd_rx) = crate::event::cmd_channel(8);
        let (event_tx, mut event_rx) = crate::event::event_channel();
        let plan_directory = test_plan_directory();

        let handle = tokio::spawn(run(
            Arc::new(ToolLoopProvider::new()),
            tools,
            permissions,
            "test-model".to_string(),
            (128, 4096),
            None,
            None,
            None,
            plan_directory.clone(),
            false,
            cmd_rx,
            event_tx,
        ));
        cmd_tx
            .send(AgentCommand::PlanProcess {
                message: "go".to_string(),
            })
            .await
            .unwrap();

        let mut events = Vec::new();
        while let Some(event) = event_rx.recv().await {
            if let AgentEvent::PlanApprovalNeeded {
                request_id,
                plan_path,
                ..
            } = &event
            {
                match mutation {
                    InvalidPlanFile::Missing => tokio::fs::remove_file(plan_path).await.unwrap(),
                    InvalidPlanFile::Empty => tokio::fs::write(plan_path, " \n").await.unwrap(),
                    #[cfg(unix)]
                    InvalidPlanFile::Symlink => {
                        let target = plan_directory.parent().unwrap().join("replacement-plan.md");
                        tokio::fs::write(&target, "# Replacement plan\n")
                            .await
                            .unwrap();
                        tokio::fs::remove_file(plan_path).await.unwrap();
                        std::os::unix::fs::symlink(target, plan_path).unwrap();
                    }
                }
                cmd_tx
                    .send(AgentCommand::PlanResponse {
                        request_id: request_id.clone(),
                        approved: true,
                    })
                    .await
                    .unwrap();
            }
            let done = matches!(&event, AgentEvent::AgentError { .. });
            events.push(event);
            if done {
                break;
            }
        }

        drop(cmd_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        cleanup_test_plan_directory(&plan_directory);
        events
    }

    fn session_contains_approved_plan(events: &[AgentEvent]) -> bool {
        events.iter().any(|event| match event {
            AgentEvent::SessionUpdated { messages } => messages.iter().any(|message| {
                message.content.iter().any(|block| {
                    matches!(
                        block,
                        ContentBlock::Text { text }
                            if text.contains("The user approved this plan")
                    )
                })
            }),
            _ => false,
        })
    }

    #[tokio::test]
    async fn agent_executes_tool_and_finishes_with_result_context() {
        let events = run_test_agent().await;

        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::ToolCallCompleted { result, .. } if result == "ok")
        ));
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::AgentFinished { final_message } if final_message == "done")
        ));
    }

    #[tokio::test]
    async fn repeated_failure_guard_allows_one_recovery_round_before_stopping() {
        let executions = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(RepeatingFailureProvider::new());
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(AlwaysFailTool {
            executions: executions.clone(),
        }));

        let events = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            run_agent_with_registry(provider.clone(), registry, false),
        )
        .await
        .expect("repeated failure guard should terminate the tool loop");

        assert_eq!(executions.load(Ordering::SeqCst), 2);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 5);
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::StatusUpdate { message }
                    if message == "Changing strategy after a repeated tool failure..."
            )
        }));
        assert!(events.iter().any(|event| match event {
            AgentEvent::SessionUpdated { messages } => messages.iter().any(|message| {
                message.content.iter().any(|block| {
                    matches!(
                        block,
                        ContentBlock::ToolResult { content, .. }
                            if content.contains("Do not retry the same tool with the same input again")
                    )
                })
            }),
            _ => false,
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::ToolCallFailed { error, .. }
                    if error.contains("Repeated-failure guard blocked")
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::AgentFinished { final_message }
                    if final_message == "I could not complete the repeated operation."
            )
        }));
    }

    #[tokio::test]
    async fn repeated_failure_guard_keeps_tools_available_for_a_different_approach() {
        let executions = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(RepeatingFailureProvider::recovering());
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(AlwaysFailTool {
            executions: executions.clone(),
        }));
        registry.register(Arc::new(EchoTool));

        let events = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            run_agent_with_registry(provider.clone(), registry, false),
        )
        .await
        .expect("the model should be able to recover with another tool");

        assert_eq!(executions.load(Ordering::SeqCst), 2);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 5);
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::StatusUpdate { message }
                    if message == "Repeated call blocked; trying a different approach..."
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::ToolCallCompleted { name, result, .. }
                    if name == "echo_tool" && result == "recovered"
            )
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::AgentFinished { final_message }
                    if final_message == "Recovered with a different approach."
            )
        }));
    }

    #[tokio::test]
    async fn repeated_failure_guard_counts_deterministic_policy_denials() {
        let executions = Arc::new(AtomicUsize::new(0));
        let provider = Arc::new(RepeatingFailureProvider::repeating(
            "shell",
            "{\"command\":\"rm -rf /\"}",
        ));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(TestShellTool {
            executions: executions.clone(),
        }));

        let events = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            run_agent_with_registry(provider.clone(), registry, false),
        )
        .await
        .expect("repeated policy denials should terminate the tool loop");

        assert_eq!(executions.load(Ordering::SeqCst), 0);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 5);
        assert!(events.iter().any(|event| match event {
            AgentEvent::SessionUpdated { messages } => messages.iter().any(|message| {
                message.content.iter().any(|block| {
                    matches!(
                        block,
                        ContentBlock::ToolResult { content, .. }
                            if content.contains("Permission denied")
                                && content.contains("Do not retry the same tool")
                    )
                })
            }),
            _ => false,
        }));
        assert!(events.iter().any(|event| {
            matches!(
                event,
                AgentEvent::ToolCallFailed { error, .. }
                    if error.contains("Repeated-failure guard blocked")
            )
        }));
    }

    #[tokio::test]
    async fn tool_and_final_responses_both_report_usage() {
        let events = run_test_agent().await;
        let usage: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::TurnComplete {
                    input_tokens,
                    output_tokens,
                    ..
                } => Some((*input_tokens, *output_tokens)),
                _ => None,
            })
            .collect();

        assert_eq!(usage, vec![(10, 5), (10, 5)]);
    }

    #[tokio::test]
    async fn agent_reports_when_it_resumes_thinking_after_tools() {
        let events = run_test_agent().await;
        let tool_completed = events
            .iter()
            .position(|event| matches!(event, AgentEvent::ToolCallCompleted { .. }))
            .expect("expected tool completion");
        let thinking_after_tools = events
            .iter()
            .position(|event| {
                matches!(
                    event,
                    AgentEvent::StatusUpdate { message } if message == "Thinking after tools..."
                )
            })
            .expect("expected thinking-after-tools status");

        assert!(tool_completed < thinking_after_tools);
    }

    #[tokio::test]
    async fn interrupt_stops_a_pending_turn() {
        let events =
            run_test_agent_with_provider(Arc::new(ToolLoopProvider::pending()), true).await;

        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::Interrupted)));
    }

    #[tokio::test]
    async fn agent_executes_read_only_tool_calls_in_parallel() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DelayedEchoTool::new(
            active.clone(),
            max_active.clone(),
        )));
        let provider = ToolLoopProvider::with_tool_calls(vec![
            (
                "call_1",
                "delayed_echo",
                "{\"value\":\"one\",\"delay_ms\":120}",
            ),
            (
                "call_2",
                "delayed_echo",
                "{\"value\":\"two\",\"delay_ms\":120}",
            ),
            (
                "call_3",
                "delayed_echo",
                "{\"value\":\"three\",\"delay_ms\":120}",
            ),
        ]);

        let events = run_agent_with_registry(Arc::new(provider), registry, false).await;

        assert!(
            max_active.load(Ordering::SeqCst) > 1,
            "expected overlapping tool executions"
        );
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, AgentEvent::ToolCallCompleted { .. }))
                .count(),
            3
        );
    }

    #[tokio::test]
    async fn parallel_tool_results_are_recorded_in_original_order() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DelayedEchoTool::new(active, max_active)));
        let provider = ToolLoopProvider::with_tool_calls(vec![
            (
                "call_1",
                "delayed_echo",
                "{\"value\":\"slow\",\"delay_ms\":120}",
            ),
            (
                "call_2",
                "delayed_echo",
                "{\"value\":\"fast\",\"delay_ms\":10}",
            ),
            (
                "call_3",
                "delayed_echo",
                "{\"value\":\"middle\",\"delay_ms\":60}",
            ),
        ]);

        let events = run_agent_with_registry(Arc::new(provider), registry, false).await;
        let tool_message = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::SessionUpdated { messages } => messages.last(),
                _ => None,
            })
            .find(|message| message.role == Role::Tool && message.content.len() == 3)
            .expect("expected batched tool result message");
        let results: Vec<_> = tool_message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    ..
                } => Some((tool_use_id.as_str(), content.as_str())),
                _ => None,
            })
            .collect();

        assert_eq!(
            results,
            vec![("call_1", "slow"), ("call_2", "fast"), ("call_3", "middle")]
        );
    }

    #[tokio::test]
    async fn mutating_tool_call_flushes_parallel_batch_as_ordering_barrier() {
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(DelayedEchoTool::new(active, max_active)));
        registry.register(Arc::new(PreviewTool {
            executions: executions.clone(),
        }));
        let provider = ToolLoopProvider::with_tool_calls(vec![
            (
                "call_1",
                "delayed_echo",
                "{\"value\":\"one\",\"delay_ms\":60}",
            ),
            (
                "call_2",
                "delayed_echo",
                "{\"value\":\"two\",\"delay_ms\":60}",
            ),
            ("call_3", "preview_tool", "{\"value\":\"write\"}"),
            (
                "call_4",
                "delayed_echo",
                "{\"value\":\"four\",\"delay_ms\":0}",
            ),
        ]);

        let events = run_agent_with_registry(Arc::new(provider), registry, true).await;
        assert_eq!(executions.load(Ordering::SeqCst), 1);

        let completed_ids: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ToolCallCompleted { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(completed_ids, vec!["call_1", "call_2", "call_3", "call_4"]);

        let preview_started = events
            .iter()
            .position(
                |event| matches!(event, AgentEvent::ToolCallStarted { id, .. } if id == "call_3"),
            )
            .expect("expected preview tool start");
        let first_batch_done = events
            .iter()
            .position(
                |event| matches!(event, AgentEvent::ToolCallCompleted { id, .. } if id == "call_2"),
            )
            .expect("expected first parallel batch completion");
        assert!(first_batch_done < preview_started);
    }

    #[tokio::test]
    async fn agent_waits_for_file_preview_before_execution() {
        let (events, executions) = run_preview_agent(true).await;

        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::FileChangePreviewNeeded { preview, .. } if preview.path == "test.txt")
        ));
        assert_eq!(executions, 1);
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::ToolCallCompleted { result, .. } if result == "applied")
        ));
    }

    #[tokio::test]
    async fn agent_rejects_file_preview_without_execution() {
        let (events, executions) = run_preview_agent(false).await;

        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::FileChangePreviewNeeded { .. })));
        assert_eq!(executions, 0);
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::ToolCallFailed { error, .. } if error.contains("rejected"))
        ));
    }

    #[tokio::test]
    async fn user_permission_denial_is_preserved_without_counting_as_a_tool_failure() {
        let (events, executions) = run_permission_agent(false).await;

        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::PermissionNeeded { .. })));
        assert_eq!(executions, 0);
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::ToolCallFailed { error, .. } if error == "Permission denied by user")
        ));
        assert!(events.iter().any(|event| match event {
            AgentEvent::SessionUpdated { messages } => messages.iter().any(|message| {
                message.content.iter().any(|block| {
                    matches!(
                        block,
                        ContentBlock::ToolResult { content, .. }
                            if content == "Permission denied by user"
                    )
                })
            }),
            _ => false,
        }));
        assert!(!events.iter().any(|event| match event {
            AgentEvent::SessionUpdated { messages } => messages.iter().any(|message| {
                message.content.iter().any(|block| {
                    matches!(
                        block,
                        ContentBlock::ToolResult { content, .. }
                            if content.contains("Do not retry the same tool")
                    )
                })
            }),
            _ => false,
        }));
    }

    #[tokio::test]
    async fn user_permission_approval_still_executes_the_tool() {
        let (events, executions) = run_permission_agent(true).await;

        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::PermissionNeeded { .. })));
        assert_eq!(executions, 1);
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::ToolCallCompleted { result, .. } if result == "executed")
        ));
    }

    #[tokio::test]
    async fn agent_plan_process_waits_for_approval_then_executes() {
        let events = run_plan_agent(true, false).await;

        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::PlanApprovalNeeded { plan, .. } if plan == "done")
        ));
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::AgentFinished { final_message } if final_message == "done")
        ));
    }

    #[tokio::test]
    async fn resumed_plan_review_reloads_and_executes_the_latest_disk_plan() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let tools = Arc::new(registry);
        let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
            deepcode_permissions::policy::PermissionSystemConfig::default(),
        )));
        let (cmd_tx, cmd_rx) = crate::event::cmd_channel(8);
        let (event_tx, mut event_rx) = crate::event::event_channel();
        let plan_directory = test_plan_directory();
        tokio::fs::create_dir_all(&plan_directory).await.unwrap();
        let plan_path = plan_directory.join(format!("plan-{}.md", uuid::Uuid::new_v4()));
        tokio::fs::write(&plan_path, "# Persisted plan\n\n1. Inspect\n")
            .await
            .unwrap();

        let handle = tokio::spawn(run(
            Arc::new(ToolLoopProvider::new()),
            tools,
            permissions,
            "test-model".to_string(),
            (128, 4096),
            None,
            None,
            Some(vec![Message::user("original task")]),
            plan_directory.clone(),
            false,
            cmd_rx,
            event_tx,
        ));
        cmd_tx
            .send(AgentCommand::ResumePlan {
                plan_path: plan_path.display().to_string(),
            })
            .await
            .unwrap();

        let mut restored_prompt_seen = false;
        let mut latest_plan_executed = false;
        let mut finished = false;
        while let Some(event) = event_rx.recv().await {
            match &event {
                AgentEvent::PlanApprovalNeeded {
                    request_id,
                    plan,
                    plan_path: event_path,
                    restored,
                } => {
                    assert!(*restored);
                    assert_eq!(plan, "# Persisted plan\n\n1. Inspect\n");
                    assert_eq!(
                        std::path::Path::new(event_path),
                        plan_path.canonicalize().unwrap()
                    );
                    restored_prompt_seen = true;
                    tokio::fs::write(
                        &plan_path,
                        "# Edited after restart\n\n1. Execute the latest file\n",
                    )
                    .await
                    .unwrap();
                    cmd_tx
                        .send(AgentCommand::PlanResponse {
                            request_id: request_id.clone(),
                            approved: true,
                        })
                        .await
                        .unwrap();
                }
                AgentEvent::SessionUpdated { messages } => {
                    latest_plan_executed |= messages.iter().any(|message| {
                        message.role == Role::User
                            && message.content.iter().any(|block| {
                                matches!(
                                    block,
                                    ContentBlock::Text { text }
                                        if text.contains("The user approved this plan")
                                            && text.contains("# Edited after restart")
                                            && text.contains("Execute the latest file")
                                )
                            })
                    });
                }
                AgentEvent::AgentFinished { final_message } => {
                    assert_eq!(final_message, "done");
                    finished = true;
                    break;
                }
                AgentEvent::AgentError { message } => panic!("unexpected agent error: {message}"),
                _ => {}
            }
        }

        assert!(restored_prompt_seen);
        assert!(latest_plan_executed);
        assert!(finished);
        drop(cmd_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        cleanup_test_plan_directory(&plan_directory);
    }

    #[tokio::test]
    async fn plan_feedback_revises_in_the_same_turn_before_execution() {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool));
        let tools = Arc::new(registry);
        let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
            deepcode_permissions::policy::PermissionSystemConfig::default(),
        )));
        let (cmd_tx, cmd_rx) = crate::event::cmd_channel(8);
        let (event_tx, mut event_rx) = crate::event::event_channel();
        let plan_directory = test_plan_directory();

        let handle = tokio::spawn(run(
            Arc::new(ToolLoopProvider::new()),
            tools,
            permissions,
            "test-model".to_string(),
            (128, 4096),
            None,
            None,
            None,
            plan_directory.clone(),
            false,
            cmd_rx,
            event_tx,
        ));
        cmd_tx
            .send(AgentCommand::PlanProcess {
                message: "go".to_string(),
            })
            .await
            .unwrap();

        let mut approval_count = 0;
        let mut feedback_in_history = false;
        let mut disk_edit_seen_in_discussion = false;
        let mut externally_edited_plan_used = false;
        let mut first_plan_path = None;
        let mut finished = false;
        while let Some(event) = event_rx.recv().await {
            match &event {
                AgentEvent::PlanApprovalNeeded {
                    request_id,
                    plan_path,
                    ..
                } => {
                    approval_count += 1;
                    if let Some(first_plan_path) = &first_plan_path {
                        assert_eq!(plan_path, first_plan_path);
                    } else {
                        first_plan_path = Some(plan_path.clone());
                    }
                    if approval_count == 1 {
                        tokio::fs::write(
                            plan_path,
                            "# Externally edited draft\n\n1. Preserve this disk edit\n",
                        )
                        .await
                        .unwrap();
                        cmd_tx
                            .send(AgentCommand::PlanFeedback {
                                request_id: request_id.clone(),
                                feedback: "Add an integration-test step".to_string(),
                            })
                            .await
                            .unwrap();
                    } else {
                        tokio::fs::write(
                            plan_path,
                            "# Externally edited plan\n\n1. Execute the edited file\n",
                        )
                        .await
                        .unwrap();
                        cmd_tx
                            .send(AgentCommand::PlanResponse {
                                request_id: request_id.clone(),
                                approved: true,
                            })
                            .await
                            .unwrap();
                    }
                }
                AgentEvent::SessionUpdated { messages } => {
                    feedback_in_history |= messages.iter().any(|message| {
                        message.role == Role::User
                            && message.content.iter().any(|block| {
                                matches!(
                                    block,
                                    ContentBlock::Text { text }
                                        if text.contains("Add an integration-test step")
                                )
                            })
                    });
                    disk_edit_seen_in_discussion |= messages.iter().any(|message| {
                        message.role == Role::User
                            && message.content.iter().any(|block| {
                                matches!(
                                    block,
                                    ContentBlock::Text { text }
                                        if text.contains("Continue discussing the proposed plan")
                                            && text.contains("# Externally edited draft")
                                            && text.contains("Preserve this disk edit")
                                )
                            })
                    });
                    externally_edited_plan_used |= messages.iter().any(|message| {
                        message.role == Role::User
                            && message.content.iter().any(|block| {
                                matches!(
                                    block,
                                    ContentBlock::Text { text }
                                        if text.contains("The user approved this plan")
                                            && text.contains("# Externally edited plan")
                                            && text.contains("Execute the edited file")
                                )
                            })
                    });
                }
                AgentEvent::AgentFinished { final_message } => {
                    assert_eq!(final_message, "done");
                    finished = true;
                    break;
                }
                AgentEvent::AgentError { message } => panic!("unexpected agent error: {message}"),
                _ => {}
            }
        }

        assert_eq!(approval_count, 2);
        assert!(feedback_in_history);
        assert!(disk_edit_seen_in_discussion);
        assert!(externally_edited_plan_used);
        assert!(finished);
        drop(cmd_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), handle).await;
        cleanup_test_plan_directory(&plan_directory);
    }

    #[tokio::test]
    async fn missing_plan_file_is_rejected_before_execution() {
        let events = run_agent_with_invalid_plan_file(InvalidPlanFile::Missing).await;

        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::AgentError { message } if message.contains("Cannot read plan"))
        ));
        assert!(!session_contains_approved_plan(&events));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::AgentFinished { .. })));
    }

    #[tokio::test]
    async fn empty_plan_file_is_rejected_before_execution() {
        let events = run_agent_with_invalid_plan_file(InvalidPlanFile::Empty).await;

        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::AgentError { message } if message.contains("is empty"))
        ));
        assert!(!session_contains_approved_plan(&events));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::AgentFinished { .. })));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlinked_plan_file_is_rejected_before_execution() {
        let events = run_agent_with_invalid_plan_file(InvalidPlanFile::Symlink).await;

        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::AgentError { message } if message.contains("symlinked plan storage path"))
        ));
        assert!(!session_contains_approved_plan(&events));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::AgentFinished { .. })));
    }

    #[tokio::test]
    async fn agent_rejects_plan_without_execution() {
        let events = run_plan_agent(false, false).await;

        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::PlanApprovalNeeded { .. })));
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::AgentFinished { final_message } if final_message == "Plan rejected.")
        ));
    }

    #[tokio::test]
    async fn persistent_plan_mode_routes_process_through_plan_review() {
        let events = run_plan_agent(true, true).await;

        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::PlanApprovalNeeded { .. })));
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::AgentFinished { final_message } if final_message == "done")
        ));
    }

    #[tokio::test]
    async fn planning_does_not_execute_mutating_tools_before_approval() {
        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(PreviewTool {
            executions: executions.clone(),
        }));

        let (events, execution_count) = run_plan_agent_with_registry(
            Arc::new(ToolLoopProvider::with_tool_call(
                "preview_tool",
                "{\"value\":\"ok\"}",
            )),
            registry,
            false,
            false,
            Some(executions),
        )
        .await;

        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::ToolCallFailed { name, error, .. } if name == "preview_tool" && error.contains("not allowed during planning"))
        ));
        assert!(events
            .iter()
            .any(|event| matches!(event, AgentEvent::PlanApprovalNeeded { .. })));
        assert_eq!(execution_count, 0);
    }

    #[tokio::test]
    async fn agent_handles_invalid_tool_arguments_as_internal_recovery() {
        let events = run_test_agent_with_provider(
            Arc::new(ToolLoopProvider::missing_required_tool_input()),
            false,
        )
        .await;

        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::StatusUpdate { message } if message == "Adjusting tool arguments...")
        ));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallStarted { .. })));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::ToolCallFailed { .. })));
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::AgentFinished { final_message } if final_message == "done")
        ));
    }

    #[tokio::test]
    async fn agent_handles_hard_policy_rejection_before_permission_prompt() {
        let executions = Arc::new(AtomicUsize::new(0));
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(PreflightRejectTool {
            executions: executions.clone(),
        }));

        let events = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            run_agent_with_registry(
                Arc::new(ToolLoopProvider::with_tool_call(
                    "preflight_reject_tool",
                    "{\"value\":\"blocked\"}",
                )),
                registry,
                false,
            ),
        )
        .await
        .expect("hard policy rejection should not wait for user permission");

        assert_eq!(executions.load(Ordering::SeqCst), 0);
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::StatusUpdate { message } if message == "Adjusting tool command...")
        ));
        assert!(!events
            .iter()
            .any(|event| matches!(event, AgentEvent::PermissionNeeded { .. })));
        assert!(events.iter().any(
            |event| matches!(event, AgentEvent::AgentFinished { final_message } if final_message == "done")
        ));
    }

    #[tokio::test]
    async fn managed_agent_waits_for_child_results_before_requesting_conclusion() {
        let provider = Arc::new(SubagentBarrierProvider {
            parent_calls: AtomicUsize::new(0),
            child_finished: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            child_delay: std::time::Duration::from_millis(40),
            request_builder: TestRequestBuilder,
            response_parser: TestResponseParser,
            compressor: TestCompressor,
        });
        let llm: Arc<dyn LlmProvider> = provider.clone();
        let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
            deepcode_permissions::policy::PermissionSystemConfig::default(),
        )));
        let (cmd_tx, cmd_rx) = crate::event::cmd_channel(32);
        let (event_tx, mut event_rx) = crate::event::event_channel();
        let model = deepcode_core::config::ModelProfile {
            id: "test-model".to_string(),
            provider: "test".to_string(),
            display_name: None,
            context_window: 16_384,
            max_output_tokens: 1_024,
            reasoning_efforts: vec![deepcode_core::config::ReasoningEffort::Off],
        };
        let base_tools = Arc::new(ToolRegistry::new());
        let manager = crate::subagent::AgentTaskManager::new(
            Arc::clone(&llm),
            Arc::clone(&base_tools),
            Arc::clone(&permissions),
            Arc::new(tokio::sync::RwLock::new(
                crate::subagent::AgentRuntimeSettings {
                    model: model.id.clone(),
                    reasoning_effort: Some("off".to_string()),
                    default_subagent_model: None,
                    default_subagent_reasoning_effort: None,
                },
            )),
            vec![model],
            event_tx.clone(),
            2,
        );
        let mut tools = (*base_tools).clone();
        tools.register(Arc::new(crate::subagent::SpawnAgentTool::new(Arc::clone(
            &manager,
        ))));
        let handle = tokio::spawn(run_managed(
            llm,
            Arc::new(tools),
            permissions,
            "test-model".to_string(),
            (1_024, 16_384),
            Some("off".to_string()),
            Some("parent agent".to_string()),
            None,
            test_plan_directory(),
            false,
            cmd_rx,
            event_tx,
            manager,
        ));
        cmd_tx
            .send(AgentCommand::Process {
                message: "delegate".to_string(),
            })
            .await
            .unwrap();

        let final_message = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if let Some(AgentEvent::AgentFinished { final_message }) = event_rx.recv().await {
                    break final_message;
                }
            }
        })
        .await
        .expect("managed agent should finish");

        assert_eq!(final_message, "final after child");
        assert_eq!(provider.parent_calls.load(Ordering::SeqCst), 2);
        cmd_tx.send(AgentCommand::Shutdown).await.unwrap();
        handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn managed_interrupt_is_emitted_after_children_stop() {
        let provider = Arc::new(SubagentBarrierProvider {
            parent_calls: AtomicUsize::new(0),
            child_finished: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            child_delay: std::time::Duration::from_secs(5),
            request_builder: TestRequestBuilder,
            response_parser: TestResponseParser,
            compressor: TestCompressor,
        });
        let llm: Arc<dyn LlmProvider> = provider;
        let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
            deepcode_permissions::policy::PermissionSystemConfig::default(),
        )));
        let (cmd_tx, cmd_rx) = crate::event::cmd_channel(32);
        let (event_tx, mut event_rx) = crate::event::event_channel();
        let model = deepcode_core::config::ModelProfile {
            id: "test-model".to_string(),
            provider: "test".to_string(),
            display_name: None,
            context_window: 16_384,
            max_output_tokens: 1_024,
            reasoning_efforts: vec![deepcode_core::config::ReasoningEffort::Off],
        };
        let base_tools = Arc::new(ToolRegistry::new());
        let manager = crate::subagent::AgentTaskManager::new(
            Arc::clone(&llm),
            Arc::clone(&base_tools),
            Arc::clone(&permissions),
            Arc::new(tokio::sync::RwLock::new(
                crate::subagent::AgentRuntimeSettings {
                    model: model.id.clone(),
                    reasoning_effort: Some("off".to_string()),
                    default_subagent_model: None,
                    default_subagent_reasoning_effort: None,
                },
            )),
            vec![model],
            event_tx.clone(),
            2,
        );
        let mut tools = (*base_tools).clone();
        tools.register(Arc::new(crate::subagent::SpawnAgentTool::new(Arc::clone(
            &manager,
        ))));
        let handle = tokio::spawn(run_managed(
            llm,
            Arc::new(tools),
            permissions,
            "test-model".to_string(),
            (1_024, 16_384),
            Some("off".to_string()),
            Some("parent agent".to_string()),
            None,
            test_plan_directory(),
            false,
            cmd_rx,
            event_tx,
            Arc::clone(&manager),
        ));
        cmd_tx
            .send(AgentCommand::Process {
                message: "delegate".to_string(),
            })
            .await
            .unwrap();

        let task_id = tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if let Some(AgentEvent::SubagentStarted { task_id, .. }) = event_rx.recv().await {
                    break task_id;
                }
            }
        })
        .await
        .expect("child should start");
        cmd_tx.send(AgentCommand::Interrupt).await.unwrap();

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                if matches!(event_rx.recv().await, Some(AgentEvent::Interrupted)) {
                    break;
                }
            }
        })
        .await
        .expect("parent should report interruption");
        let results = manager.wait(&[task_id]).await.unwrap();
        assert_eq!(results[0].status, crate::subagent::TaskStatus::Interrupted);
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(25), async {
                while let Some(event) = event_rx.recv().await {
                    if matches!(
                        event,
                        AgentEvent::SubagentEvent { .. }
                            | AgentEvent::SubagentCompleted { .. }
                            | AgentEvent::SubagentStarted { .. }
                    ) {
                        return;
                    }
                }
            })
            .await
            .is_err()
        );

        cmd_tx.send(AgentCommand::Shutdown).await.unwrap();
        handle.await.unwrap().unwrap();
    }
}
