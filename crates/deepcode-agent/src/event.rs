use tokio::sync::mpsc;

/// Commands sent from the UI/cli to the agent loop.
#[derive(Debug, Clone)]
pub enum AgentCommand {
    /// Start processing a user message.
    Process { message: String },
    /// Start processing a user message in plan-first mode.
    PlanProcess { message: String },
    /// Enable or disable persistent plan-first mode.
    SetPlanMode { enabled: bool },
    /// Change the model used by subsequent turns.
    SetModel {
        model: String,
        max_tokens: usize,
        context_window: usize,
    },
    /// Replace the live model catalog used to validate subagent overrides.
    SetAvailableModels {
        models: Vec<deepcode_core::config::ModelProfile>,
    },
    /// Change reasoning depth used by subsequent requests. `None` disables it.
    SetReasoningEffort { effort: Option<String> },
    /// Clear conversation state while preserving the system prompt.
    ClearSession,
    /// Request a read-only snapshot of the active permissions policy.
    PermissionsSnapshot,
    /// Respond to a plan approval request.
    PlanResponse { request_id: String, approved: bool },
    /// Interrupt the current agent turn and return to idle.
    Interrupt,
    /// Respond to a permission request.
    PermissionResponse {
        request_id: String,
        approved: bool,
        scope: deepcode_permissions::policy::ApprovalScope,
    },
    /// Respond to a file change preview request.
    FileChangePreviewResponse { request_id: String, approved: bool },
    /// Shutdown the agent gracefully.
    Shutdown,
}

/// Events emitted from the agent loop to the UI/cli.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A user-facing status update for the current agent activity.
    StatusUpdate { message: String },
    /// A streaming text delta.
    TextDelta(String),
    /// A streaming reasoning delta.
    ReasoningDelta(String),
    /// A tool call is about to be executed.
    ToolCallStarted {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A tool call has completed successfully.
    ToolCallCompleted {
        id: String,
        name: String,
        result: String,
    },
    /// A tool call failed.
    ToolCallFailed {
        id: String,
        name: String,
        error: String,
    },
    /// Permission is needed for a tool call.
    PermissionNeeded {
        request_id: String,
        tool_name: String,
        input: serde_json::Value,
        evaluation: deepcode_permissions::policy::PermissionEvaluation,
    },
    /// A file-content change must be reviewed before it is written.
    FileChangePreviewNeeded {
        request_id: String,
        tool_name: String,
        input: serde_json::Value,
        preview: deepcode_tools::tool::FileChangePreview,
    },
    /// The agent has produced a plan and needs user approval before acting.
    PlanApprovalNeeded { request_id: String, plan: String },
    /// A turn (one LLM call + tool execution cycle) has completed.
    TurnComplete {
        input_tokens: usize,
        output_tokens: usize,
        cached_input_tokens: usize,
        cache_miss_input_tokens: usize,
        reasoning_output_tokens: usize,
    },
    /// The current agent turn was interrupted.
    Interrupted,
    /// The agent run has finished.
    AgentFinished { final_message: String },
    /// The full canonical conversation state changed.
    SessionUpdated {
        messages: Vec<deepcode_core::types::Message>,
    },
    /// A concise title captured from the first response of a new session.
    SessionTitleGenerated { title: String },
    /// The agent encountered an error.
    AgentError { message: String },
    /// A read-only snapshot of the active permissions policy.
    PermissionsSnapshot { lines: Vec<String> },

    /// A subagent has started working on a task.
    SubagentStarted { task_id: String, task: String },

    /// A subagent has completed its task.
    SubagentCompleted { task_id: String, result: String },

    /// An event emitted by a subagent, forwarded to the parent UI.
    SubagentEvent {
        task_id: String,
        event: std::sync::Arc<AgentEvent>,
    },
}

/// Convenience channel types.
pub type CmdSender = mpsc::Sender<AgentCommand>;
pub type CmdReceiver = mpsc::Receiver<AgentCommand>;
pub type EventSender = mpsc::UnboundedSender<AgentEvent>;
pub type EventReceiver = mpsc::UnboundedReceiver<AgentEvent>;

/// Create a paired command channel.
pub fn cmd_channel(buffer: usize) -> (CmdSender, CmdReceiver) {
    mpsc::channel(buffer)
}

/// Create a paired event channel.
pub fn event_channel() -> (EventSender, EventReceiver) {
    mpsc::unbounded_channel()
}
