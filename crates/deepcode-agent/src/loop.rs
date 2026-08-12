use std::sync::Arc;

use deepcode_core::error::DeepCodeError;
use deepcode_core::provider::traits::LlmProvider;
use deepcode_permissions::pipeline::PermissionSystem;
use deepcode_tools::registry::ToolRegistry;

use crate::event::{AgentCommand, AgentEvent, EventSender};
use crate::plan_executor::{approved_plan_instruction, execute_plan_review, PlanDecision};
use crate::state::{AgentPhase, AgentState};
use crate::turn_executor::execute_turn;

pub(crate) enum BusyControl {
    Continue,
    Interrupt,
    Shutdown,
}

pub(crate) fn handle_busy_command(
    cmd: Option<AgentCommand>,
    event_tx: &EventSender,
) -> BusyControl {
    match cmd {
        Some(AgentCommand::Interrupt) => {
            let _ = event_tx.send(AgentEvent::Interrupted);
            BusyControl::Interrupt
        }
        Some(AgentCommand::Shutdown) | None => BusyControl::Shutdown,
        Some(AgentCommand::Process { .. })
        | Some(AgentCommand::PlanProcess { .. })
        | Some(AgentCommand::SetPlanMode { .. })
        | Some(AgentCommand::SetModel { .. })
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
        | Some(AgentCommand::PlanResponse { .. }) => BusyControl::Continue,
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
) -> bool {
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
        match execute_plan_review(llm, tools, model, model_config, cmd_rx, event_tx, state).await {
            PlanDecision::Approved { plan } => {
                state.push_user(&approved_plan_instruction(&plan));
                let _ = event_tx.send(AgentEvent::SessionUpdated {
                    messages: state.messages.clone(),
                });
            }
            PlanDecision::Rejected => {
                tracing::info!("Plan rejected by user");
                state.phase = AgentPhase::Idle;
                let message = "Plan rejected.".to_string();
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: message.clone(),
                });
                let _ = event_tx.send(AgentEvent::AgentFinished {
                    final_message: message,
                });
                return false;
            }
            PlanDecision::Interrupted => {
                tracing::info!("Agent plan interrupted");
                state.phase = AgentPhase::Idle;
                return false;
            }
            PlanDecision::Shutdown => {
                tracing::info!("Agent plan shutdown requested");
                state.phase = AgentPhase::Finished;
                return true;
            }
            PlanDecision::Failed => {
                tracing::info!("Agent plan failed");
                if state.phase != AgentPhase::Error {
                    state.phase = AgentPhase::Idle;
                }
                return false;
            }
        }
    }

    let result = execute_turn(
        llm,
        tools,
        permissions,
        model,
        model_config,
        cmd_rx,
        event_tx,
        state,
    )
    .await;

    if result.interrupted {
        tracing::info!("Agent process interrupted");
        state.phase = AgentPhase::Idle;
    }
    if result.shutdown_requested {
        tracing::info!("Agent process shutdown requested");
        state.phase = AgentPhase::Finished;
        return true;
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
    session_title_enabled: bool,
    cmd_rx: crate::event::CmdReceiver,
    event_tx: EventSender,
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
                state.phase = AgentPhase::Finished;
                break;
            }
            AgentCommand::Interrupt => {
                tracing::info!("Agent interrupted while idle");
                let _ = event_tx.send(AgentEvent::Interrupted);
                state.phase = AgentPhase::Idle;
            }
            AgentCommand::PermissionResponse {
                request_id,
                approved,
                scope,
            } => {
                let _ = (request_id, approved, scope);
            }
            AgentCommand::PlanResponse {
                request_id,
                approved,
            } => {
                let _ = (request_id, approved);
            }
            AgentCommand::FileChangePreviewResponse {
                request_id,
                approved,
            } => {
                let _ = (request_id, approved);
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
                let _ = event_tx.send(AgentEvent::StatusUpdate {
                    message: format!("Model switched to {}.", model),
                });
            }
            AgentCommand::SetReasoningEffort { effort } => {
                state.reasoning_effort = effort;
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
                )
                .await
                {
                    break;
                }
            }
        }
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

    struct EchoTool;

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

        let handle = tokio::spawn(run(
            llm,
            tools,
            permissions,
            "test-model".to_string(),
            (128, 4096),
            None,
            None,
            None,
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
            if let AgentEvent::PlanApprovalNeeded { request_id, .. } = &event {
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
}
