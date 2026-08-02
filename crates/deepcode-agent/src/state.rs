use deepcode_core::types::{ContentBlock, Message, Role};

/// A tool execution result ready to append to the canonical conversation.
#[derive(Debug, Clone)]
pub(crate) struct ToolResultEntry {
    pub tool_use_id: String,
    pub content: String,
    pub is_error: bool,
}

impl ToolResultEntry {
    pub(crate) fn new(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error,
        }
    }
}

/// Tracks which phase of the agent loop we're in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AgentPhase {
    /// Waiting for user input.
    Idle,
    /// Sending request to LLM, collecting streaming response.
    Generating,
    /// Parsing tool calls from the LLM response.
    ParsingToolCalls,
    /// Executing tools (possibly in parallel).
    ExecutingTools,
    /// Waiting for user approval on a tool execution.
    WaitingForPermission,
    /// Generating a plan before execution.
    Planning,
    /// Waiting for user approval on a generated plan.
    WaitingForPlanApproval,
    /// Compressing context before the next LLM call.
    CompressingContext,
    /// Terminal: agent run complete.
    Finished,
    /// Terminal: error encountered.
    Error,
}

/// Holds all mutable state for an agent session.
pub(crate) struct AgentState {
    pub phase: AgentPhase,
    pub messages: Vec<Message>,
    pub turn_count: usize,
    pub total_input_tokens: usize,
    pub total_output_tokens: usize,
    pub plan_mode_enabled: bool,
    pub reasoning_effort: Option<String>,
    pub session_title_pending: bool,
}

impl AgentState {
    pub(crate) fn new(
        system_prompt: Option<String>,
        initial_messages: Option<Vec<Message>>,
    ) -> Self {
        let mut messages = Vec::new();
        let initial_messages = initial_messages.unwrap_or_default();
        let has_initial_conversation = initial_messages
            .iter()
            .any(|message| message.role == Role::User);
        let has_initial_system = initial_messages
            .first()
            .is_some_and(|msg| msg.role == deepcode_core::types::Role::System);
        if let Some(sys) = system_prompt.filter(|_| !has_initial_system) {
            messages.push(Message {
                role: deepcode_core::types::Role::System,
                content: vec![deepcode_core::types::ContentBlock::text(sys)],
                id: None,
            });
        }
        messages.extend(initial_messages);
        Self {
            phase: AgentPhase::Idle,
            messages,
            turn_count: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            plan_mode_enabled: false,
            reasoning_effort: None,
            session_title_pending: !has_initial_conversation,
        }
    }

    /// Add a user message to the conversation.
    pub(crate) fn push_user(&mut self, text: &str) {
        self.messages.push(Message::user(text));
    }

    pub(crate) fn clear_conversation(&mut self) {
        self.messages.retain(|message| message.role == Role::System);
        self.turn_count = 0;
        self.total_input_tokens = 0;
        self.total_output_tokens = 0;
        self.phase = AgentPhase::Idle;
        self.session_title_pending = true;
    }

    /// Add an assistant response to the conversation.
    pub(crate) fn push_assistant(&mut self, blocks: Vec<deepcode_core::types::ContentBlock>) {
        self.messages.push(Message::assistant(blocks));
    }

    /// Add a tool result to the conversation.
    pub(crate) fn push_tool_result(&mut self, tool_use_id: &str, content: &str, is_error: bool) {
        self.push_tool_results(vec![ToolResultEntry::new(tool_use_id, content, is_error)]);
    }

    /// Add one tool message containing multiple tool results.
    pub(crate) fn push_tool_results(&mut self, results: Vec<ToolResultEntry>) {
        if results.is_empty() {
            return;
        }

        self.messages.push(Message {
            role: Role::Tool,
            content: results
                .into_iter()
                .map(|result| {
                    ContentBlock::tool_result(&result.tool_use_id, &result.content, result.is_error)
                })
                .collect(),
            id: None,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_conversation_keeps_only_system_prompt() {
        let mut state = AgentState::new(Some("system".to_string()), None);
        state.push_user("question");
        state.push_assistant(vec![ContentBlock::text("answer")]);
        state.push_tool_result("call", "result", false);
        state.total_input_tokens = 10;
        state.total_output_tokens = 20;

        state.clear_conversation();

        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].role, Role::System);
        assert_eq!(state.total_input_tokens, 0);
        assert_eq!(state.total_output_tokens, 0);
        assert_eq!(state.phase, AgentPhase::Idle);
    }
}
