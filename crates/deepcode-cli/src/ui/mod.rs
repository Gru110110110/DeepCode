// Terminal UI for interactive chat mode.
// Uses an alternate-screen full-screen renderer with internal transcript scroll.

use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::session::{SavedPendingPlan, SavedSession, SessionStore, SessionSummary};
use crossterm::cursor::SetCursorStyle;
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event,
};
use crossterm::terminal::{
    DisableLineWrap, EnableLineWrap, EnterAlternateScreen, LeaveAlternateScreen,
};
use deepcode_agent::event::{self as agent_event};
use deepcode_permissions::policy::{ApprovalScope, PermissionEvaluation};

pub(crate) mod banner;
pub(crate) mod diff;
pub(crate) mod events;
pub(crate) mod input;
pub(crate) mod markdown;
pub(crate) mod renderer;
pub(crate) mod scrolling;

pub(crate) use events::{tool_activity, tool_issue_status};

pub(crate) const STARTUP_BANNER_WIDTH: usize = 68;
pub(crate) const INPUT_HELP_TEXT: &str =
    "Agent mode · Shift+Tab to switch · Enter to send · Ctrl+C to exit";
pub(crate) const MESSAGE_GAP_LINES: usize = 1;
const ENABLE_ALT_SCROLL_MODE: &[u8] = b"\x1b[?1007h";
const DISABLE_ALT_SCROLL_MODE: &[u8] = b"\x1b[?1007l";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PermissionChoice {
    AllowOnce,
    AllowSession,
    AllowAlways,
    Deny,
    Quit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FilePreviewChoice {
    Apply,
    Reject,
    Quit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PlanChoice {
    Approve,
    ContinueDiscussing,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TuiAction {
    Exit,
    Resume(String),
}

#[derive(Clone)]
pub(crate) struct PendingSessionPicker {
    pub sessions: Vec<SessionSummary>,
    pub selected: usize,
    pub show_all: bool,
}

impl PendingSessionPicker {
    pub(crate) fn selected_session(&self) -> Option<&SessionSummary> {
        self.sessions.get(self.selected)
    }
}

impl PlanChoice {
    pub(crate) const ALL: [Self; 2] = [Self::Approve, Self::ContinueDiscussing];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Approve => "Approve",
            Self::ContinueDiscussing => "Continue Discussing",
        }
    }

    pub(crate) fn next(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|choice| *choice == self)
            .unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }

    pub(crate) fn previous(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|choice| *choice == self)
            .unwrap_or(0);
        Self::ALL[(idx + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

impl FilePreviewChoice {
    pub(crate) const ALL: [Self; 3] = [Self::Apply, Self::Reject, Self::Quit];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Apply => "Apply",
            Self::Reject => "Reject",
            Self::Quit => "Quit",
        }
    }

    pub(crate) fn next(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|choice| *choice == self)
            .unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }

    pub(crate) fn previous(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|choice| *choice == self)
            .unwrap_or(0);
        Self::ALL[(idx + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    pub(crate) fn response(self) -> Option<bool> {
        match self {
            Self::Apply => Some(true),
            Self::Reject => Some(false),
            Self::Quit => None,
        }
    }
}

impl PermissionChoice {
    pub(crate) const ALL: [Self; 5] = [
        Self::AllowOnce,
        Self::AllowSession,
        Self::AllowAlways,
        Self::Deny,
        Self::Quit,
    ];

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::AllowOnce => "Allow once",
            Self::AllowSession => "Allow session",
            Self::AllowAlways => "Always allow",
            Self::Deny => "Deny",
            Self::Quit => "Quit",
        }
    }

    pub(crate) fn next(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|choice| *choice == self)
            .unwrap_or(0);
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }

    pub(crate) fn previous(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|choice| *choice == self)
            .unwrap_or(0);
        Self::ALL[(idx + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    pub(crate) fn response(self) -> Option<(bool, ApprovalScope)> {
        match self {
            Self::AllowOnce => Some((true, ApprovalScope::Once)),
            Self::AllowSession => Some((true, ApprovalScope::Session)),
            Self::AllowAlways => Some((true, ApprovalScope::Persistent)),
            Self::Deny => Some((false, ApprovalScope::Once)),
            Self::Quit => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StartupHeader {
    pub model: String,
    pub directory: String,
}

impl StartupHeader {
    pub(crate) fn current(model: &str, reasoning_effort: Option<&str>) -> Self {
        let directory = std::env::current_dir()
            .map(|path| path.display().to_string())
            .unwrap_or_else(|_| ".".to_string());

        Self {
            model: format!("{} {}", model, reasoning_effort.unwrap_or("off")),
            directory,
        }
    }
}

/// A pending permission request from the agent.
#[derive(Clone)]
pub(crate) struct PendingPermission {
    pub request_id: String,
    pub tool_name: String,
    pub input: serde_json::Value,
    pub evaluation: Option<PermissionEvaluation>,
    pub selected: PermissionChoice,
}

#[derive(Clone)]
pub(crate) struct PendingFilePreview {
    pub request_id: String,
    pub preview: deepcode_tools::tool::FileChangePreview,
    pub selected: FilePreviewChoice,
}

#[derive(Clone)]
pub(crate) enum DeferredApproval {
    Permission(PendingPermission),
    FilePreview {
        tool_name: String,
        pending: PendingFilePreview,
    },
}

#[derive(Clone)]
pub(crate) struct PendingPlanApproval {
    pub request_id: String,
    pub plan: String,
    pub plan_path: String,
    pub selected: PlanChoice,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PastedInputBlock {
    pub(crate) id: usize,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) line_count: usize,
    pub(crate) char_count: usize,
}

impl PastedInputBlock {
    pub(crate) fn label(&self) -> String {
        if self.line_count > 1 {
            format!("[Pasted text #{} +{} lines]", self.id, self.line_count)
        } else {
            format!("[Pasted text #{} +{} chars]", self.id, self.char_count)
        }
    }
}

/// Shared state between the UI thread and agent.
pub(crate) struct AppState {
    pub startup_header: Option<StartupHeader>,
    pub input: String,
    pub cursor_pos: usize,
    pub pasted_input_blocks: Vec<PastedInputBlock>,
    pub next_pasted_input_id: usize,
    pub viewport: renderer::ViewportState,
    pub messages: Vec<ChatMessage>,
    pub core_messages: Vec<deepcode_core::types::Message>,
    pub streaming_text: String,
    pub reasoning_text: String,
    pub reasoning_started_at: Option<Instant>,
    pub reasoning_expanded: bool,
    pub status: String,
    pub running: bool,
    pub plan_mode_enabled: bool,
    pub available_models: Vec<deepcode_core::config::ModelProfile>,
    pub model_catalog: Option<ModelCatalogContext>,
    pub current_model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub session_store: Option<SessionStore>,
    pub session: Option<SavedSession>,
    pub pending_sessions: Option<PendingSessionPicker>,
    pub exit_action: TuiAction,
    pub pending_plan: Option<PendingPlanApproval>,
    pub plan_feedback_request_id: Option<String>,
    pub active_plan_path: Option<String>,
    pub pending_permission: Option<PendingPermission>,
    pub pending_file_preview: Option<PendingFilePreview>,
    pub deferred_approvals: VecDeque<DeferredApproval>,
    pub working_since: Option<Instant>,
    pub interrupt_requested: bool,
    pub last_usage: Option<TurnUsage>,
}

pub(crate) type TurnUsage = deepcode_core::provider::traits::Usage;

#[derive(Clone)]
pub(crate) struct ModelCatalogContext {
    pub provider: String,
    pub config: deepcode_core::config::ProviderConfig,
    pub data_root: std::path::PathBuf,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl AppState {
    pub(crate) fn new() -> Self {
        Self {
            startup_header: None,
            input: String::new(),
            cursor_pos: 0,
            pasted_input_blocks: Vec::new(),
            next_pasted_input_id: 1,
            viewport: renderer::ViewportState::default(),
            messages: Vec::new(),
            core_messages: Vec::new(),
            streaming_text: String::new(),
            reasoning_text: String::new(),
            reasoning_started_at: None,
            reasoning_expanded: false,
            status: "Ready. Type /exit to quit, Ctrl+C to cancel.".to_string(),
            running: true,
            plan_mode_enabled: false,
            available_models: Vec::new(),
            model_catalog: None,
            current_model: None,
            reasoning_effort: None,
            session_store: None,
            session: None,
            pending_sessions: None,
            exit_action: TuiAction::Exit,
            pending_plan: None,
            plan_feedback_request_id: None,
            active_plan_path: None,
            pending_permission: None,
            pending_file_preview: None,
            deferred_approvals: VecDeque::new(),
            working_since: None,
            interrupt_requested: false,
            last_usage: None,
        }
    }

    pub(crate) fn with_messages(messages: Vec<ChatMessage>) -> Self {
        let mut state = Self::new();
        state.messages = messages;
        state
    }

    pub(crate) fn with_session(
        messages: Vec<ChatMessage>,
        core_messages: Vec<deepcode_core::types::Message>,
    ) -> Self {
        let mut state = Self::with_messages(messages);
        state.core_messages = core_messages;
        state
    }

    pub(crate) fn restore_workflow(&mut self, session: &SavedSession) {
        self.plan_mode_enabled = session.plan_mode_enabled;
        if let Some(pending) = &session.pending_plan {
            self.active_plan_path = Some(pending.plan_path.clone());
            self.working_since = Some(Instant::now());
            self.status = "Restoring plan review...".to_string();
        }
    }

    pub(crate) fn scroll_up(&mut self, amount: usize) {
        let delta = i32::try_from(amount).unwrap_or(i32::MAX);
        self.viewport.pending_scroll_delta =
            self.viewport.pending_scroll_delta.saturating_sub(delta);
    }

    pub(crate) fn scroll_down(&mut self, amount: usize) {
        let delta = i32::try_from(amount).unwrap_or(i32::MAX);
        self.viewport.pending_scroll_delta =
            self.viewport.pending_scroll_delta.saturating_add(delta);
    }

    pub(crate) fn scroll_to_bottom(&mut self) {
        self.viewport.transcript_scroll = scrolling::TranscriptScroll::to_bottom();
        self.viewport.pending_scroll_delta = 0;
        self.viewport.transcript_scrollbar_dragging = false;
    }

    pub(crate) fn clear_input_blocks(&mut self) {
        self.pasted_input_blocks.clear();
        self.next_pasted_input_id = 1;
    }

    pub(crate) fn reasoning_elapsed_secs(&self) -> Option<u64> {
        self.reasoning_started_at
            .map(|started| started.elapsed().as_secs())
    }

    pub(crate) fn finish_reasoning(&mut self) {
        let Some(started) = self.reasoning_started_at.take() else {
            return;
        };
        let text = std::mem::take(&mut self.reasoning_text);
        if text.is_empty() {
            return;
        }
        let elapsed = started.elapsed().as_secs().max(1);
        self.messages.push(ChatMessage {
            role: format!("reasoning:{elapsed}"),
            content: text,
        });
    }

    pub(crate) fn persist_session(&mut self) -> anyhow::Result<()> {
        self.sync_session_snapshot();
        let (Some(store), Some(session)) = (&self.session_store, &mut self.session) else {
            return Ok(());
        };
        store.save(session)
    }

    pub(crate) fn apply_generated_session_title(&mut self, title: &str) -> anyhow::Result<bool> {
        let (Some(store), Some(session)) = (&self.session_store, &mut self.session) else {
            return Ok(false);
        };
        if session.title_generated || !session.set_generated_title(title) {
            return Ok(false);
        }
        store.save(session)?;
        Ok(true)
    }

    pub(crate) fn save_committed_session(&mut self) -> anyhow::Result<()> {
        self.sync_session_snapshot();
        let (Some(store), Some(session)) = (&self.session_store, &mut self.session) else {
            return Ok(());
        };
        store.save(session)
    }

    fn sync_session_snapshot(&mut self) {
        let Some(session) = &mut self.session else {
            return;
        };
        session.ui_messages = self.messages.clone();
        session.core_messages = self.core_messages.clone();
        if let Some(model) = &self.current_model {
            session.model = model.clone();
        }
        session.reasoning_effort = self
            .reasoning_effort
            .clone()
            .unwrap_or_else(|| "off".to_string());
        session.plan_mode_enabled = self.plan_mode_enabled;
        session.pending_plan = self
            .active_plan_path
            .clone()
            .map(|plan_path| SavedPendingPlan { plan_path });
    }
}

/// Run the TUI in a dedicated OS thread (crossterm raw mode requires it).
pub(crate) fn run_tui(
    cmd_tx: agent_event::CmdSender,
    mut event_rx: agent_event::EventReceiver,
    state: Arc<Mutex<AppState>>,
) -> anyhow::Result<TuiAction> {
    crossterm::terminal::enable_raw_mode()?;
    let _guard = TerminalCleanup;
    let mut stdout = std::io::stdout();
    stdout.write_all(ENABLE_ALT_SCROLL_MODE)?;
    crossterm::execute!(
        stdout,
        EnterAlternateScreen,
        DisableLineWrap,
        EnableBracketedPaste,
        EnableMouseCapture,
        SetCursorStyle::BlinkingBar
    )?;

    let mut renderer = renderer::FullscreenRenderer::new(stdout)?;
    {
        let mut s = state.lock().unwrap();
        renderer.draw(&mut s)?;
    }

    fullscreen_loop(&mut renderer, &cmd_tx, &mut event_rx, &state)?;
    renderer.finish()?;
    Ok(state.lock().unwrap().exit_action.clone())
}

struct TerminalCleanup;

impl Drop for TerminalCleanup {
    fn drop(&mut self) {
        let _ = crossterm::terminal::disable_raw_mode();
        let mut stdout = std::io::stdout();
        let _ = stdout.write_all(DISABLE_ALT_SCROLL_MODE);
        let _ = crossterm::execute!(
            stdout,
            DisableBracketedPaste,
            DisableMouseCapture,
            EnableLineWrap,
            LeaveAlternateScreen,
            SetCursorStyle::DefaultUserShape,
            crossterm::style::ResetColor
        );
    }
}

fn fullscreen_loop(
    renderer: &mut renderer::FullscreenRenderer,
    cmd_tx: &agent_event::CmdSender,
    event_rx: &mut agent_event::EventReceiver,
    state: &Arc<Mutex<AppState>>,
) -> anyhow::Result<()> {
    let mut rendered_reasoning_second = None;
    loop {
        while let Ok(event) = event_rx.try_recv() {
            events::handle_agent_event(event, state);
            let mut s = state.lock().unwrap();
            renderer.draw(&mut s)?;
            rendered_reasoning_second = s.reasoning_elapsed_secs();
        }

        if !state.lock().unwrap().running {
            break;
        }

        if event::poll(std::time::Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) => {
                    if !input::handle_key(key, cmd_tx, state)? {
                        break;
                    }
                    let mut s = state.lock().unwrap();
                    renderer.draw(&mut s)?;
                }
                Event::Paste(text) => {
                    input::handle_paste(text, state)?;
                    let mut s = state.lock().unwrap();
                    renderer.draw(&mut s)?;
                }
                Event::Mouse(mouse) => {
                    let mut s = state.lock().unwrap();
                    if let Some(text) = renderer::handle_mouse(mouse, &mut s) {
                        match renderer.copy_to_clipboard(&text) {
                            Ok(()) => {
                                let chars = text.chars().count();
                                s.status = format!("Copied selection ({} chars).", chars);
                            }
                            Err(err) => {
                                s.status = format!("Could not copy selection: {}", err);
                            }
                        }
                    }
                    renderer.draw(&mut s)?;
                }
                Event::Resize(_, _) => {
                    let mut s = state.lock().unwrap();
                    renderer.draw(&mut s)?;
                }
                _ => {}
            }
        } else {
            let mut s = state.lock().unwrap();
            let current_second = s.reasoning_elapsed_secs();
            if current_second != rendered_reasoning_second {
                renderer.draw(&mut s)?;
                rendered_reasoning_second = current_second;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_state_initializes_with_default_status() {
        let state = AppState::new();
        assert!(state.status.contains("/exit"));
        assert!(state.messages.is_empty());
    }

    #[test]
    fn pending_plan_checkpoint_persists_and_restores_workflow_state() {
        let root = std::env::temp_dir().join(format!(
            "deepcode_pending_plan_session_{}",
            uuid::Uuid::new_v4()
        ));
        let store = SessionStore::at(root.clone());
        let session = SavedSession::new(
            "/workspace".to_string(),
            "provider".to_string(),
            "model".to_string(),
            "high".to_string(),
        );
        let session_id = session.id.clone();
        let mut state = AppState::with_session(
            vec![ChatMessage {
                role: "user".to_string(),
                content: "Implement recovery".to_string(),
            }],
            vec![deepcode_core::types::Message::user("Implement recovery")],
        );
        state.session_store = Some(store.clone());
        state.session = Some(session);
        state.plan_mode_enabled = true;
        state.active_plan_path =
            Some("/tmp/plan-00000000-0000-0000-0000-000000000001.md".to_string());

        state.save_committed_session().unwrap();
        let saved = store.load(&session_id).unwrap();
        let mut restored =
            AppState::with_session(saved.ui_messages.clone(), saved.core_messages.clone());
        restored.restore_workflow(&saved);

        assert!(saved.plan_mode_enabled);
        assert_eq!(
            saved
                .pending_plan
                .as_ref()
                .map(|plan| plan.plan_path.as_str()),
            Some("/tmp/plan-00000000-0000-0000-0000-000000000001.md")
        );
        assert!(restored.plan_mode_enabled);
        assert_eq!(
            restored.active_plan_path.as_deref(),
            Some("/tmp/plan-00000000-0000-0000-0000-000000000001.md")
        );
        assert!(restored.working_since.is_some());
        assert_eq!(restored.status, "Restoring plan review...");
        let _ = std::fs::remove_dir_all(root);
    }
}
