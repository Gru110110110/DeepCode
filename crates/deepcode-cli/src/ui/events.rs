use std::sync::{Arc, Mutex};
use std::time::Instant;

use deepcode_agent::event::{self as agent_event, AgentEvent};

use super::{
    AppState, ChatMessage, FilePreviewChoice, PendingFilePreview, PendingPermission,
    PendingPlanApproval, PendingSessionPicker, PermissionChoice, PlanChoice, TuiAction, TurnUsage,
    INPUT_HELP_TEXT,
};

pub(crate) fn input_status_text(state: &AppState) -> String {
    if let Some(picker) = &state.pending_sessions {
        let scope = if picker.show_all {
            "all workspaces"
        } else {
            "this workspace"
        };
        return format!(
            "Sessions ({}, {}) - Up/Down select, Enter resume, a toggle, Esc close",
            picker.sessions.len(),
            scope
        );
    }
    if let Some(plan) = &state.pending_plan {
        return plan_title(plan);
    }
    if let Some(preview) = &state.pending_file_preview {
        return file_preview_title(preview);
    }
    if let Some(permission) = &state.pending_permission {
        return permission_title(permission);
    }
    if state.status == "Ready." || state.status.starts_with("Ready. ") {
        if state.plan_mode_enabled {
            return append_usage_summary(
                "Plan mode · Shift+Tab to switch · Enter to plan · Ctrl+C to exit",
                state.last_usage.as_ref(),
            );
        }
        append_usage_summary(INPUT_HELP_TEXT, state.last_usage.as_ref())
    } else {
        state.status.clone()
    }
}

pub(crate) fn set_plan_mode(state: &mut AppState, cmd_tx: &agent_event::CmdSender, enabled: bool) {
    if state.working_since.is_some() {
        state.status = "Already working; press Esc to interrupt before switching mode.".to_string();
        return;
    }

    let previous = state.plan_mode_enabled;
    state.plan_mode_enabled = enabled;
    state.status = if enabled {
        "Switched to Plan mode. Press Shift+Tab for Agent mode.".to_string()
    } else {
        "Switched to Agent mode. Press Shift+Tab for Plan mode.".to_string()
    };

    if cmd_tx
        .blocking_send(agent_event::AgentCommand::SetPlanMode { enabled })
        .is_err()
    {
        state.plan_mode_enabled = previous;
        state.status = "Agent channel closed; mode was not changed.".to_string();
    }
}

fn append_usage_summary(base: &str, usage: Option<&TurnUsage>) -> String {
    let Some(usage) = usage else {
        return base.to_string();
    };
    let mut summary = format!("last in/out {}/{}", usage.input_tokens, usage.output_tokens);
    let cache_total = usage.cached_input_tokens + usage.cache_miss_input_tokens;
    if let Some(hit_rate) = usage
        .cached_input_tokens
        .saturating_mul(100)
        .checked_div(cache_total)
    {
        summary.push_str(&format!(
            ", cache hit {}% ({}/{})",
            hit_rate, usage.cached_input_tokens, cache_total
        ));
    }
    if usage.reasoning_output_tokens > 0 {
        summary.push_str(&format!(", reasoning {}", usage.reasoning_output_tokens));
    }
    format!("{base} · {summary}")
}

pub(crate) fn should_hide_empty_input_prompt(state: &AppState) -> bool {
    state.pending_plan.is_none()
        && state.pending_permission.is_none()
        && state.pending_file_preview.is_none()
        && state.working_since.is_some()
        && state.input.is_empty()
}

pub(crate) fn plan_title(plan: &PendingPlanApproval) -> String {
    let first_line = plan
        .plan
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("Plan ready");
    format!("Review plan: {}", truncate_chars(first_line, 104))
}

pub(crate) fn plan_options_line(plan: &PendingPlanApproval) -> String {
    let options = PlanChoice::ALL
        .iter()
        .map(|choice| {
            if *choice == plan.selected {
                format!("[{}]", choice.label())
            } else {
                format!(" {} ", choice.label())
            }
        })
        .collect::<Vec<_>>()
        .join("  ");
    format!("←/→ {}  Enter", options)
}

pub(crate) fn file_preview_title(preview: &PendingFilePreview) -> String {
    format!(
        "Review changes: {}",
        truncate_chars(&preview.preview.path, 120)
    )
}

pub(crate) fn file_preview_options_line(preview: &PendingFilePreview) -> String {
    let options = FilePreviewChoice::ALL
        .iter()
        .map(|choice| {
            if *choice == preview.selected {
                format!("[{}]", choice.label())
            } else {
                format!(" {} ", choice.label())
            }
        })
        .collect::<Vec<_>>()
        .join("  ");
    format!("←/→ {}  Enter", options)
}

pub(crate) fn permission_title(permission: &PendingPermission) -> String {
    let tool_name = human_tool_name(&permission.tool_name);
    let (_, detail) = tool_activity(&permission.tool_name, &permission.input);
    let detail = truncate_chars(&detail, 72);
    let risk = permission
        .evaluation
        .as_ref()
        .map(|eval| format!("{} / {}", eval.risk.as_str(), eval.sandbox_policy.label()))
        .unwrap_or_else(|| "approval".to_string());
    if detail.is_empty() {
        format!("Permission required: {} ({})", tool_name, risk)
    } else {
        format!("Permission required: {} ({}) - {}", tool_name, risk, detail)
    }
}

pub(crate) fn permission_options_line(permission: &PendingPermission) -> String {
    let options = PermissionChoice::ALL
        .iter()
        .map(|choice| {
            if *choice == permission.selected {
                format!("[{}]", choice.label())
            } else {
                format!(" {} ", choice.label())
            }
        })
        .collect::<Vec<_>>()
        .join("  ");
    format!("←/→ {}  Enter", options)
}

pub(crate) fn flush_streaming_text(state: &mut AppState) {
    let text = std::mem::take(&mut state.streaming_text);
    if !text.is_empty() {
        state.messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: text,
        });
    }
}

fn preview_json(input: &serde_json::Value, max_len: usize) -> String {
    let preview = serde_json::to_string(input).unwrap_or_default();
    truncate_chars(&preview, max_len)
}

fn preview_lines(value: &str, max_lines: usize, max_len: usize) -> String {
    truncate_chars(
        &value.lines().take(max_lines).collect::<Vec<_>>().join("\n"),
        max_len,
    )
}

fn truncate_chars(value: &str, max_len: usize) -> String {
    let count = value.chars().count();
    if count <= max_len {
        return value.to_string();
    }
    if max_len <= 3 {
        return "...".to_string();
    }
    let kept: String = value.chars().take(max_len - 3).collect();
    format!("{}...", kept)
}

fn single_line(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_single_line(value: &str, max_len: usize) -> String {
    truncate_chars(&single_line(value), max_len)
}

fn json_str<'a>(input: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(serde_json::Value::as_str)
}

fn display_path(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
        .to_string()
}

fn human_tool_name(name: &str) -> String {
    match name {
        "read_file" => "read file".to_string(),
        "grep" => "search".to_string(),
        "glob" => "file listing".to_string(),
        "git_status" => "git status".to_string(),
        "git_diff" => "git diff".to_string(),
        "git_log" => "git log".to_string(),
        "shell" => "command".to_string(),
        "write_file" => "file write".to_string(),
        "edit_file" => "file edit".to_string(),
        "git_add" => "git stage".to_string(),
        "git_commit" => "git commit".to_string(),
        "git_checkout" => "git checkout".to_string(),
        "git_branch" => "git branch".to_string(),
        "web_fetch" => "web fetch".to_string(),
        "web_search" => "web search".to_string(),
        "agent" => "subagent".to_string(),
        _ => name.replace('_', " "),
    }
}

fn clean_tool_error(name: &str, error: &str) -> String {
    let tool_prefix = format!("Tool execution error: {} -- ", name);
    if let Some(stripped) = error.strip_prefix(&tool_prefix) {
        return stripped.to_string();
    }
    if let Some(stripped) = error.strip_prefix("Tool execution error: ") {
        return stripped.to_string();
    }
    error.to_string()
}

fn is_argument_error(error: &str) -> bool {
    error.contains("Missing '")
        || error.contains("Missing or invalid '")
        || error.contains("must be a")
        || error.contains("Tool input rejected before execution")
}

pub(crate) fn tool_issue_status(name: &str, error: &str) -> String {
    if is_argument_error(error) {
        return format!("Adjusting {} arguments...", human_tool_name(name));
    }

    truncate_chars(
        &format!(
            "Could not use {}: {}",
            human_tool_name(name),
            clean_tool_error(name, error)
        ),
        180,
    )
}

pub(crate) fn tool_activity(name: &str, input: &serde_json::Value) -> (&'static str, String) {
    match name {
        "read_file" => (
            "Explored",
            json_str(input, "path")
                .map(|path| format!("Read {}", display_path(path)))
                .unwrap_or_else(|| "Read file".to_string()),
        ),
        "grep" => {
            let pattern = json_str(input, "pattern");
            let path = json_str(input, "path")
                .or_else(|| json_str(input, "directory"))
                .map(display_path);
            let detail = match (pattern, path) {
                (Some(pattern), Some(path)) if !path.is_empty() => {
                    format!("Search {} in {}", pattern, path)
                }
                (Some(pattern), _) => format!("Search {}", pattern),
                (None, Some(path)) if !path.is_empty() => format!("Search in {}", path),
                _ => "Search files".to_string(),
            };
            ("Explored", detail)
        }
        "glob" => (
            "Explored",
            json_str(input, "pattern")
                .map(|pattern| format!("List {}", pattern))
                .unwrap_or_else(|| "List files".to_string()),
        ),
        "git_status" => ("Explored", "Check git status".to_string()),
        "git_diff" => ("Explored", "Inspect git diff".to_string()),
        "git_log" => ("Explored", "Inspect git log".to_string()),
        "shell" => (
            "Ran",
            json_str(input, "command")
                .map(|command| truncate_single_line(command, 96))
                .unwrap_or_else(|| "Run command".to_string()),
        ),
        "write_file" => (
            "Edited",
            json_str(input, "path")
                .map(|path| format!("Write {}", display_path(path)))
                .unwrap_or_else(|| "Write file".to_string()),
        ),
        "edit_file" => (
            "Edited",
            json_str(input, "path")
                .map(|path| format!("Edit {}", display_path(path)))
                .unwrap_or_else(|| "Edit file".to_string()),
        ),
        "git_add" => ("Edited", "Stage files".to_string()),
        "git_commit" => ("Edited", "Commit changes".to_string()),
        "git_checkout" => (
            "Edited",
            format!("Checkout {}", json_str(input, "branch").unwrap_or("branch")),
        ),
        "git_branch" => (
            "Edited",
            format!("Update branch {}", json_str(input, "branch").unwrap_or("")),
        ),
        "web_fetch" => (
            "Browsed",
            truncate_single_line(json_str(input, "url").unwrap_or("Fetch URL"), 96),
        ),
        "web_search" => (
            "Browsed",
            truncate_single_line(json_str(input, "query").unwrap_or("Search web"), 96),
        ),
        "agent" => (
            "Delegated",
            truncate_single_line(json_str(input, "task").unwrap_or("Run subagent"), 96),
        ),
        _ => ("Used Tool", format!("{} {}", name, preview_json(input, 96))),
    }
}

pub(crate) fn tool_status(name: &str, input: &serde_json::Value) -> String {
    let (group, detail) = tool_activity(name, input);
    format!("{}: {}", group, detail)
}

pub(crate) fn handle_agent_event(event: AgentEvent, state: &Arc<Mutex<AppState>>) {
    let mut s = state.lock().unwrap();

    match event {
        AgentEvent::StatusUpdate { message } => {
            s.interrupt_requested = false;
            if message.starts_with("Thinking") {
                s.reasoning_started_at.get_or_insert_with(Instant::now);
            }
            s.status = message.clone();
        }
        AgentEvent::ReasoningDelta(text) => {
            s.reasoning_started_at.get_or_insert_with(Instant::now);
            s.reasoning_text.push_str(&text);
        }
        AgentEvent::TextDelta(text) => {
            s.finish_reasoning();
            s.streaming_text.push_str(&text);
        }
        AgentEvent::ToolCallStarted { name, input, .. } => {
            s.finish_reasoning();
            if name == "agent" {
                flush_streaming_text(&mut s);
                if let Err(error) = s.persist_session() {
                    s.status = format!("Could not save session: {}", error);
                }
                return;
            }
            flush_streaming_text(&mut s);
            s.status = tool_status(&name, &input);
            if let Err(error) = s.persist_session() {
                s.status = format!("Could not save session: {}", error);
            }
        }
        AgentEvent::ToolCallCompleted { name, .. } => {
            s.status = format!("Completed {}.", name);
        }
        AgentEvent::ToolCallFailed { name, error, .. } => {
            s.status = tool_issue_status(&name, &error);
        }
        AgentEvent::PermissionNeeded {
            request_id,
            tool_name,
            input,
            evaluation,
        } => {
            s.finish_reasoning();
            flush_streaming_text(&mut s);
            s.pending_permission = Some(PendingPermission {
                request_id,
                tool_name: tool_name.clone(),
                input,
                evaluation: Some(evaluation),
                selected: PermissionChoice::AllowOnce,
            });
            s.status = format!("Permission required: {}", tool_name);
            if let Err(error) = s.persist_session() {
                s.status = format!("Could not save session: {}", error);
            }
        }
        AgentEvent::FileChangePreviewNeeded {
            request_id,
            tool_name,
            input: _,
            preview,
        } => {
            s.finish_reasoning();
            flush_streaming_text(&mut s);
            let operation = if preview.before_exists {
                "Update"
            } else {
                "Create"
            };
            let content = format!("{operation}: {}\n{}", preview.path, preview.unified_diff);
            s.messages.push(ChatMessage {
                role: "diff".to_string(),
                content,
            });
            s.pending_file_preview = Some(PendingFilePreview {
                request_id,
                preview,
                selected: FilePreviewChoice::Apply,
            });
            s.status = format!("Review changes: {}", tool_name);
            if let Err(error) = s.persist_session() {
                s.status = format!("Could not save session: {}", error);
            }
        }
        AgentEvent::PlanApprovalNeeded { request_id, plan } => {
            s.finish_reasoning();
            let had_streaming_plan = !s.streaming_text.is_empty();
            flush_streaming_text(&mut s);
            if !had_streaming_plan && !plan.trim().is_empty() {
                s.messages.push(ChatMessage {
                    role: "assistant".to_string(),
                    content: plan.clone(),
                });
            }
            s.pending_plan = Some(PendingPlanApproval {
                request_id,
                plan,
                selected: PlanChoice::Approve,
            });
            s.status = "Review plan.".to_string();
            if let Err(error) = s.persist_session() {
                s.status = format!("Could not save session: {}", error);
            }
        }
        AgentEvent::TurnComplete {
            input_tokens,
            output_tokens,
            cached_input_tokens,
            cache_miss_input_tokens,
            reasoning_output_tokens,
        } => {
            let usage = TurnUsage {
                input_tokens,
                output_tokens,
                cached_input_tokens,
                cache_miss_input_tokens,
                reasoning_output_tokens,
            };
            if usage.has_reported_tokens() {
                s.last_usage = Some(usage);
            }
        }
        AgentEvent::AgentFinished { .. } => {
            s.finish_reasoning();
            flush_streaming_text(&mut s);
            s.working_since = None;
            s.interrupt_requested = false;
            s.pending_plan = None;
            s.pending_permission = None;
            s.pending_file_preview = None;
            s.status = "Ready.".to_string();
            if let Err(error) = s.persist_session() {
                s.status = format!("Could not save session: {}", error);
            }
        }
        AgentEvent::Interrupted => {
            s.finish_reasoning();
            flush_streaming_text(&mut s);
            s.working_since = None;
            s.interrupt_requested = false;
            s.pending_plan = None;
            s.pending_permission = None;
            s.pending_file_preview = None;
            s.status = "Interrupted.".to_string();
            if let Err(error) = s.persist_session() {
                s.status = format!("Could not save session: {}", error);
            }
        }
        AgentEvent::AgentError { message } => {
            s.finish_reasoning();
            flush_streaming_text(&mut s);
            s.working_since = None;
            s.interrupt_requested = false;
            s.pending_plan = None;
            s.pending_permission = None;
            s.pending_file_preview = None;
            s.messages.push(ChatMessage {
                role: "error".to_string(),
                content: format!("Error: {}", message),
            });
            s.status = format!("Error: {}", message);
            if let Err(error) = s.persist_session() {
                s.status = format!("Could not save session: {}", error);
            }
        }
        AgentEvent::SessionUpdated { messages } => {
            s.core_messages = messages;
            if let Err(error) = s.persist_session() {
                s.status = format!("Could not save session: {}", error);
            }
        }
        AgentEvent::SessionTitleGenerated { title } => {
            if let Err(error) = s.apply_generated_session_title(&title) {
                s.status = format!("Could not save session title: {}", error);
            }
        }
        AgentEvent::PermissionsSnapshot { lines } => {
            s.finish_reasoning();
            flush_streaming_text(&mut s);
            s.messages.push(ChatMessage {
                role: "system".to_string(),
                content: lines.join("\n"),
            });
            s.status = "Permissions snapshot shown.".to_string();
            if let Err(error) = s.persist_session() {
                s.status = format!("Could not save session: {}", error);
            }
        }
        AgentEvent::SubagentStarted { task_id: _, task } => {
            s.finish_reasoning();
            flush_streaming_text(&mut s);
            s.status = format!("Subagent running: {}...", truncate_single_line(&task, 96));
        }
        AgentEvent::SubagentCompleted { task_id: _, result } => {
            let preview = preview_lines(&result, 3, 240);
            if !preview.is_empty() {
                s.status = format!("Subagent completed: {}", preview);
            }
        }
        AgentEvent::SubagentEvent { task_id: _, event } => match event.as_ref() {
            AgentEvent::ToolCallStarted { name, input, .. } => {
                s.status = format!("Subagent {}", tool_status(name, input));
            }
            AgentEvent::ToolCallCompleted { name, result, .. } => {
                let preview = preview_lines(result, 2, 180);
                if !preview.is_empty() {
                    s.status = format!("Subagent completed {}.", name);
                }
            }
            AgentEvent::ToolCallFailed { name, error, .. } => {
                s.status = format!("Subagent {}", tool_issue_status(name, error));
            }
            AgentEvent::AgentError { message } => {
                s.status = truncate_chars(&format!("Subagent issue: {}", message), 180);
            }
            AgentEvent::PermissionsSnapshot { .. } => {}
            _ => {}
        },
    }
}

/// Handle slash commands (e.g., /clear, /help, /exit).
/// Returns `true` to continue, `false` to quit.
pub(crate) fn handle_slash_command(
    input: String,
    state: &mut AppState,
    cmd_tx: &agent_event::CmdSender,
) -> bool {
    let parts: Vec<&str> = input.split_whitespace().collect();
    let cmd = parts.first().copied().unwrap_or("");

    match cmd {
        "/exit" | "/quit" => {
            state.running = false;
            return false;
        }
        "/clear" => {
            if state.working_since.is_some()
                || state.pending_plan.is_some()
                || state.pending_permission.is_some()
                || state.pending_file_preview.is_some()
            {
                state.status =
                    "Finish or interrupt the current operation before clearing.".to_string();
            } else if let Some(previous) = state.session.clone() {
                if let Err(error) = state.persist_session() {
                    state.status = format!("Could not save current session: {}", error);
                } else {
                    let next = crate::session::SavedSession::new(
                        previous.workspace_root,
                        previous.provider,
                        state.current_model.clone().unwrap_or(previous.model),
                        state
                            .reasoning_effort
                            .clone()
                            .unwrap_or(previous.reasoning_effort),
                    );
                    state.messages.clear();
                    state.core_messages.clear();
                    state.streaming_text.clear();
                    state.last_usage = None;
                    state.session = Some(next);
                    if cmd_tx
                        .blocking_send(agent_event::AgentCommand::ClearSession)
                        .is_err()
                    {
                        state.status = "Agent channel closed.".to_string();
                    } else {
                        state.status = "Conversation cleared. New session started.".to_string();
                    }
                }
            } else {
                state.status = "Session storage is unavailable.".to_string();
            }
        }
        "/sessions" => open_session_picker(state),
        "/permissions" => {
            if cmd_tx
                .blocking_send(agent_event::AgentCommand::PermissionsSnapshot)
                .is_err()
            {
                state.status = "Agent channel closed.".to_string();
            } else {
                state.status = "Loading permissions snapshot...".to_string();
            }
        }
        "/resume" => {
            if parts.len() == 1 {
                open_session_picker(state);
            } else if parts.len() > 2 {
                state.status = "Usage: /resume [session-id]".to_string();
            } else if state.working_since.is_some()
                || state.pending_plan.is_some()
                || state.pending_permission.is_some()
                || state.pending_file_preview.is_some()
            {
                state.status =
                    "Finish or interrupt the current operation before resuming.".to_string();
            } else if let Some(store) = &state.session_store {
                match store.load(parts[1]) {
                    Ok(_) => {
                        if let Err(error) = state.persist_session() {
                            state.status = format!("Could not save current session: {}", error);
                        } else {
                            state.exit_action = TuiAction::Resume(parts[1].to_string());
                            state.running = false;
                            return false;
                        }
                    }
                    Err(error) => state.status = error.to_string(),
                }
            }
        }
        "/model" => {
            if parts.len() == 1 {
                let current = state.current_model.as_deref().unwrap_or("unknown");
                let choices = state
                    .available_models
                    .iter()
                    .map(|model| model.id.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                state.status = format!("Current model: {}. Available: {}", current, choices);
            } else if parts.len() == 2 && parts[1] == "refresh" {
                refresh_models(state);
            } else if parts.len() != 2 {
                state.status = "Usage: /model <name|refresh>".to_string();
            } else if state.working_since.is_some() {
                state.status =
                    "Finish or interrupt the current operation before switching model.".to_string();
            } else if let Some(model) = state
                .available_models
                .iter()
                .find(|model| model.id == parts[1])
                .cloned()
            {
                let name = model.id.clone();
                let effort = state.reasoning_effort.as_deref().unwrap_or("off");
                if !model.supports_effort_str(effort) {
                    state.status = format!(
                        "Model '{}' does not support effort '{}'. Supported: {}",
                        name,
                        effort,
                        model.effort_names().join(", ")
                    );
                    return true;
                }
                if cmd_tx
                    .blocking_send(agent_event::AgentCommand::SetModel {
                        model: name.clone(),
                        max_tokens: model.max_output_tokens,
                        context_window: model.context_window,
                    })
                    .is_err()
                {
                    state.status = "Agent channel closed.".to_string();
                } else {
                    state.current_model = Some(name.clone());
                    if let Some(header) = state.startup_header.as_mut() {
                        let effort = state.reasoning_effort.as_deref().unwrap_or("off");
                        header.model = format!("{} {}", name, effort);
                    }
                    state.status = format!("Model switched to {}.", name);
                    if let Err(error) = state.persist_session() {
                        state.status =
                            format!("Model switched, but session save failed: {}", error);
                    }
                }
            } else {
                state.status = format!("Unknown model: {}. Use /model to list models.", parts[1]);
            }
        }
        "/effort" => {
            let supported = state
                .current_model
                .as_deref()
                .and_then(|id| state.available_models.iter().find(|model| model.id == id))
                .map(|model| {
                    model
                        .effort_names()
                        .into_iter()
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_else(|| vec!["off".to_string()]);
            if parts.len() == 1 {
                let current = state.reasoning_effort.as_deref().unwrap_or("off");
                state.status = format!(
                    "Current reasoning effort: {}. Available: {}",
                    current,
                    supported.join(", ")
                );
            } else if parts.len() != 2 {
                state.status = "Usage: /effort <off|minimal|low|medium|high|max|xhigh>".to_string();
            } else if state.working_since.is_some() {
                state.status =
                    "Finish or interrupt the current operation before changing effort.".to_string();
            } else {
                let effort = parts[1].to_ascii_lowercase();
                if !supported.iter().any(|candidate| candidate == &effort) {
                    state.status = format!(
                        "Unsupported reasoning effort: {}. Available: {}",
                        parts[1],
                        supported.join(", ")
                    );
                } else {
                    let effective = Some(effort.clone());
                    if cmd_tx
                        .blocking_send(agent_event::AgentCommand::SetReasoningEffort {
                            effort: effective.clone(),
                        })
                        .is_err()
                    {
                        state.status = "Agent channel closed.".to_string();
                    } else {
                        state.reasoning_effort = effective;
                        if let Some(header) = state.startup_header.as_mut() {
                            let model = state.current_model.as_deref().unwrap_or("unknown");
                            header.model = format!("{} {}", model, effort);
                        }
                        state.status = format!("Reasoning effort set to {}.", effort);
                        if let Err(error) = state.persist_session() {
                            state.status =
                                format!("Effort changed, but session save failed: {}", error);
                        }
                    }
                }
            }
        }
        "/plan" => {
            if parts.len() == 2 && parts[1].eq_ignore_ascii_case("off") {
                set_plan_mode(state, cmd_tx, false);
            } else {
                let task = input
                    .strip_prefix("/plan")
                    .map(str::trim)
                    .unwrap_or_default();
                if task.is_empty() {
                    set_plan_mode(state, cmd_tx, true);
                } else {
                    state.messages.push(ChatMessage {
                        role: "user".to_string(),
                        content: task.to_string(),
                    });
                    state
                        .core_messages
                        .push(deepcode_core::types::Message::user(task));
                    state.streaming_text.clear();
                    state.working_since = Some(std::time::Instant::now());
                    state.interrupt_requested = false;
                    state.status = "Planning...".to_string();

                    if cmd_tx
                        .blocking_send(agent_event::AgentCommand::PlanProcess {
                            message: task.to_string(),
                        })
                        .is_err()
                    {
                        state.status = "Agent channel closed.".to_string();
                        state.working_since = None;
                    } else if let Err(error) = state.persist_session() {
                        state.status = format!("Message sent, but session save failed: {}", error);
                    }
                }
            }
        }
        "/act" => {
            set_plan_mode(state, cmd_tx, false);
        }
        "/help" => {
            state.status =
                "Commands: /model /effort /permissions /sessions /resume /plan /act /clear /help /exit"
                    .to_string();
            state.messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: "Available commands:\n\n  /model         - Show current and available models\n  /model <name>  - Switch model\n  /model refresh - Refresh the model catalog\n  /effort        - Show current reasoning effort\n  /effort <tier> - Set reasoning effort\n  /permissions   - Show the active permission policy\n  /resume        - Browse this workspace's sessions\n  /resume <id>   - Resume a saved session directly\n  /sessions      - Browse saved sessions\n  /plan          - Enable plan mode\n  /plan off      - Disable plan mode\n  /plan <task>   - Plan one task before execution\n  /act           - Disable plan mode\n  /clear         - Save this session and start a new one\n  /help          - Show this help\n  /exit          - Exit DeepCode".to_string(),
            });
        }
        _ => {
            state.status = format!(
                "Unknown command: {}. Type /help for available commands.",
                cmd
            );
        }
    }
    true
}

fn refresh_models(state: &mut AppState) {
    if state.working_since.is_some()
        || state.pending_plan.is_some()
        || state.pending_permission.is_some()
        || state.pending_file_preview.is_some()
    {
        state.status =
            "Finish or interrupt the current operation before refreshing models.".to_string();
        return;
    }
    let Some(context) = state.model_catalog.clone() else {
        state.status = "Model catalog refresh is unavailable for this session.".to_string();
        return;
    };
    let current = state.current_model.clone();
    let previous = current.as_deref().and_then(|id| {
        state
            .available_models
            .iter()
            .find(|model| model.id == id)
            .cloned()
    });
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            state.status = format!("Could not start model refresh: {}", error);
            return;
        }
    };
    match runtime.block_on(deepcode_providers::catalog::resolve_model_catalog(
        &context.provider,
        &context.config,
        &context.data_root,
        true,
    )) {
        Ok(mut catalog) => {
            if let Some(previous) = previous {
                if !catalog.models.iter().any(|model| model.id == previous.id) {
                    catalog.models.insert(0, previous);
                }
            }
            let count = catalog.models.len();
            state.available_models = catalog.models;
            state.status = if catalog.status.stale {
                format!("Model refresh used stale data; {} models available.", count)
            } else {
                format!("Model catalog refreshed; {} models available.", count)
            };
        }
        Err(error) => {
            tracing::warn!(provider = %context.provider, error = %error, "Manual model refresh failed");
            state.status =
                "Model refresh failed; the current catalog is still available.".to_string();
        }
    }
}

fn open_session_picker(state: &mut AppState) {
    if state.working_since.is_some()
        || state.pending_plan.is_some()
        || state.pending_permission.is_some()
        || state.pending_file_preview.is_some()
    {
        state.status =
            "Finish or interrupt the current operation before opening sessions.".to_string();
        return;
    }

    let (Some(store), Some(session)) = (&state.session_store, &state.session) else {
        state.status = "Session storage is unavailable.".to_string();
        return;
    };

    match store.list(Some(&session.workspace_root), None) {
        Ok(sessions) => {
            state.pending_sessions = Some(PendingSessionPicker {
                sessions,
                selected: 0,
                show_all: false,
            });
            state.status = "Session history.".to_string();
        }
        Err(error) => state.status = format!("Could not list sessions: {}", error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepcode_permissions::approval_key::ApprovalKey;
    use deepcode_permissions::policy::{PermissionEvaluation, RiskLevel, ToolCategory};
    use deepcode_sandbox::SandboxPolicy;

    fn sample_permission_evaluation() -> PermissionEvaluation {
        PermissionEvaluation::allow(
            ToolCategory::Network,
            RiskLevel::Routine,
            ApprovalKey("approval".to_string()),
            ApprovalKey("grouping".to_string()),
            SandboxPolicy::ReadOnly {
                network_access: true,
            },
            "web search".to_string(),
        )
    }

    #[test]
    fn status_updates_do_not_create_history_messages() {
        let state = Arc::new(Mutex::new(AppState::new()));

        handle_agent_event(
            AgentEvent::StatusUpdate {
                message: "Working...".to_string(),
            },
            &state,
        );

        let state = state.lock().unwrap();
        assert!(state.messages.is_empty());
        assert!(state.working_since.is_none());
        assert_eq!(state.status, "Working...");
    }

    #[test]
    fn turn_complete_records_usage_without_changing_active_status() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.status = "Working...".to_string();
        }

        handle_agent_event(
            AgentEvent::TurnComplete {
                input_tokens: 123,
                output_tokens: 456,
                cached_input_tokens: 100,
                cache_miss_input_tokens: 23,
                reasoning_output_tokens: 42,
            },
            &state,
        );

        let state = state.lock().unwrap();
        assert!(state.messages.is_empty());
        assert_eq!(state.status, "Working...");
        assert_eq!(input_status_text(&state), "Working...");
        assert_eq!(
            state.last_usage,
            Some(TurnUsage {
                input_tokens: 123,
                output_tokens: 456,
                cached_input_tokens: 100,
                cache_miss_input_tokens: 23,
                reasoning_output_tokens: 42,
            })
        );
    }

    #[test]
    fn turn_complete_ignores_missing_zero_usage() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.last_usage = Some(TurnUsage {
                input_tokens: 123,
                output_tokens: 456,
                cached_input_tokens: 100,
                cache_miss_input_tokens: 23,
                reasoning_output_tokens: 42,
            });
        }

        handle_agent_event(
            AgentEvent::TurnComplete {
                input_tokens: 0,
                output_tokens: 0,
                cached_input_tokens: 0,
                cache_miss_input_tokens: 0,
                reasoning_output_tokens: 0,
            },
            &state,
        );

        let state = state.lock().unwrap();
        assert_eq!(
            state.last_usage,
            Some(TurnUsage {
                input_tokens: 123,
                output_tokens: 456,
                cached_input_tokens: 100,
                cache_miss_input_tokens: 23,
                reasoning_output_tokens: 42,
            })
        );
    }

    #[test]
    fn ready_status_shows_last_cache_hit_rate() {
        let mut state = AppState::new();
        state.status = "Ready.".to_string();
        state.last_usage = Some(TurnUsage {
            input_tokens: 123,
            output_tokens: 456,
            cached_input_tokens: 100,
            cache_miss_input_tokens: 23,
            reasoning_output_tokens: 42,
        });

        let status = input_status_text(&state);

        assert!(status.contains("last in/out 123/456"));
        assert!(status.contains("cache hit 81% (100/123)"));
        assert!(status.contains("reasoning 42"));
    }

    #[test]
    fn ready_status_names_mode_and_switch_shortcut() {
        let mut state = AppState::new();
        state.status = "Ready.".to_string();

        let agent_status = input_status_text(&state);
        assert!(agent_status.contains("Agent mode"));
        assert!(agent_status.contains("Shift+Tab"));

        state.plan_mode_enabled = true;
        let plan_status = input_status_text(&state);
        assert!(plan_status.contains("Plan mode"));
        assert!(plan_status.contains("Shift+Tab"));
    }

    #[test]
    fn reasoning_is_finished_before_answer_text() {
        let state = Arc::new(Mutex::new(AppState::new()));

        handle_agent_event(
            AgentEvent::StatusUpdate {
                message: "Working...".to_string(),
            },
            &state,
        );
        handle_agent_event(
            AgentEvent::ReasoningDelta("inspect parser".to_string()),
            &state,
        );
        handle_agent_event(AgentEvent::TextDelta("done".to_string()), &state);

        let state = state.lock().unwrap();
        assert_eq!(state.messages.len(), 1);
        assert!(state.messages[0].role.starts_with("reasoning:"));
        assert_eq!(state.messages[0].content, "inspect parser");
        assert_eq!(state.streaming_text, "done");
        assert!(state.reasoning_started_at.is_none());
    }

    #[test]
    fn working_state_hides_empty_input_prompt() {
        let mut state = AppState::new();
        state.working_since = Some(Instant::now());

        assert!(should_hide_empty_input_prompt(&state));

        state.input = "draft".to_string();
        assert!(!should_hide_empty_input_prompt(&state));
    }

    #[test]
    fn permission_prompt_is_not_hidden_while_working() {
        let mut state = AppState::new();
        state.working_since = Some(Instant::now());
        state.pending_permission = Some(PendingPermission {
            request_id: "perm_1".to_string(),
            tool_name: "web_search".to_string(),
            input: serde_json::json!({"query": "DeepSeek V4"}),
            evaluation: None,
            selected: PermissionChoice::AllowOnce,
        });

        assert!(!should_hide_empty_input_prompt(&state));
    }

    #[test]
    fn tool_events_update_status_without_history_noise() {
        let state = Arc::new(Mutex::new(AppState::new()));

        handle_agent_event(
            AgentEvent::ToolCallStarted {
                id: "call_1".to_string(),
                name: "web_search".to_string(),
                input: serde_json::json!({"query": "DeepSeek V4"}),
            },
            &state,
        );
        handle_agent_event(
            AgentEvent::ToolCallFailed {
                id: "call_1".to_string(),
                name: "web_search".to_string(),
                error: "Missing 'query' parameter".to_string(),
            },
            &state,
        );

        let state = state.lock().unwrap();
        assert!(state.messages.is_empty());
        assert_eq!(state.status, "Adjusting web search arguments...");
    }

    #[test]
    fn tool_activity_never_displays_missing_placeholders() {
        let cases = [
            ("read_file", serde_json::json!({})),
            ("grep", serde_json::json!({})),
            ("glob", serde_json::json!({})),
            ("shell", serde_json::json!({})),
            ("write_file", serde_json::json!({})),
            ("edit_file", serde_json::json!({})),
        ];

        for (name, input) in cases {
            let (_, detail) = tool_activity(name, &input);
            assert!(!detail.contains("<missing"), "{}: {}", name, detail);
        }
    }

    #[test]
    fn agent_tool_activity_is_single_line() {
        let (_, detail) = tool_activity(
            "agent",
            &serde_json::json!({
                "task": "Analyze the project.\nLook at permissions.\nReport findings."
            }),
        );

        assert_eq!(
            detail,
            "Analyze the project. Look at permissions. Report findings."
        );
    }

    #[test]
    fn permission_prompt_stays_in_input_state_not_history() {
        let state = Arc::new(Mutex::new(AppState::new()));

        handle_agent_event(
            AgentEvent::PermissionNeeded {
                request_id: "perm_1".to_string(),
                tool_name: "web_search".to_string(),
                input: serde_json::json!({"query": "DeepSeek V4"}),
                evaluation: sample_permission_evaluation(),
            },
            &state,
        );

        let state = state.lock().unwrap();
        let permission = state.pending_permission.as_ref().unwrap();
        assert!(state.messages.is_empty());
        assert_eq!(permission.selected, PermissionChoice::AllowOnce);
        assert!(input_status_text(&state).contains("Permission required: web search"));
        assert!(permission_options_line(permission).contains("[Allow once]"));
    }

    #[test]
    fn file_preview_event_creates_review_state_and_diff_message() {
        let state = Arc::new(Mutex::new(AppState::new()));

        handle_agent_event(
            AgentEvent::FileChangePreviewNeeded {
                request_id: "preview_1".to_string(),
                tool_name: "write_file".to_string(),
                input: serde_json::json!({"path": "a.txt", "content": "new"}),
                preview: deepcode_tools::tool::FileChangePreview {
                    path: "a.txt".to_string(),
                    before_exists: true,
                    before: "old\n".to_string(),
                    after: "new\n".to_string(),
                    unified_diff: "--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-old\n+new\n"
                        .to_string(),
                },
            },
            &state,
        );

        let state = state.lock().unwrap();
        assert!(state.pending_file_preview.is_some());
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].role, "diff");
        assert!(state.messages[0].content.contains("Update: a.txt"));
        assert!(input_status_text(&state).contains("Review changes: a.txt"));
        assert!(
            file_preview_options_line(state.pending_file_preview.as_ref().unwrap())
                .contains("[Apply]")
        );
    }

    #[test]
    fn new_file_preview_uses_create_title() {
        let state = Arc::new(Mutex::new(AppState::new()));

        handle_agent_event(
            AgentEvent::FileChangePreviewNeeded {
                request_id: "preview_1".to_string(),
                tool_name: "write_file".to_string(),
                input: serde_json::json!({"path": "new.txt", "content": "new"}),
                preview: deepcode_tools::tool::FileChangePreview {
                    path: "new.txt".to_string(),
                    before_exists: false,
                    before: String::new(),
                    after: "new\n".to_string(),
                    unified_diff: "--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1,1 @@\n+new\n"
                        .to_string(),
                },
            },
            &state,
        );

        let state = state.lock().unwrap();
        assert!(state.messages[0].content.contains("Create: new.txt"));
    }

    #[test]
    fn plan_approval_event_creates_review_state() {
        let state = Arc::new(Mutex::new(AppState::new()));

        handle_agent_event(AgentEvent::TextDelta("1. Inspect\n".to_string()), &state);
        handle_agent_event(
            AgentEvent::PlanApprovalNeeded {
                request_id: "plan_1".to_string(),
                plan: "1. Inspect\n2. Edit".to_string(),
            },
            &state,
        );

        let state = state.lock().unwrap();
        let plan = state.pending_plan.as_ref().unwrap();
        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.messages[0].role, "assistant");
        assert!(state.messages[0].content.contains("1. Inspect"));
        assert_eq!(plan.selected, PlanChoice::Approve);
        assert!(input_status_text(&state).contains("Review plan: 1. Inspect"));
        assert!(plan_options_line(plan).contains("[Approve]"));
    }

    #[test]
    fn slash_plan_enables_persistent_plan_mode() {
        let mut state = AppState::new();
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        assert!(handle_slash_command(
            "/plan".to_string(),
            &mut state,
            &cmd_tx
        ));

        assert!(state.plan_mode_enabled);
        match cmd_rx.try_recv().unwrap() {
            agent_event::AgentCommand::SetPlanMode { enabled } => assert!(enabled),
            other => panic!("unexpected command: {:?}", other),
        }
    }

    #[test]
    fn slash_plan_with_task_sends_one_shot_plan_process() {
        let mut state = AppState::new();
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        assert!(handle_slash_command(
            "/plan update parser".to_string(),
            &mut state,
            &cmd_tx
        ));

        assert!(!state.plan_mode_enabled);
        assert_eq!(state.messages[0].content, "update parser");
        match cmd_rx.try_recv().unwrap() {
            agent_event::AgentCommand::PlanProcess { message } => {
                assert_eq!(message, "update parser");
            }
            other => panic!("unexpected command: {:?}", other),
        }
    }

    #[test]
    fn slash_model_switches_to_configured_model() {
        let mut state = AppState::new();
        state.available_models = vec![deepcode_core::config::ModelProfile {
            id: "fast-model".to_string(),
            provider: "test".to_string(),
            display_name: Some("Fast Model".to_string()),
            max_output_tokens: 8192,
            context_window: 64_000,
            reasoning_efforts: vec![deepcode_core::config::ReasoningEffort::Off],
        }];
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        assert!(handle_slash_command(
            "/model fast-model".to_string(),
            &mut state,
            &cmd_tx
        ));
        assert_eq!(state.current_model.as_deref(), Some("fast-model"));
        match cmd_rx.try_recv().unwrap() {
            agent_event::AgentCommand::SetModel {
                model,
                max_tokens,
                context_window,
            } => {
                assert_eq!(model, "fast-model");
                assert_eq!(max_tokens, 8192);
                assert_eq!(context_window, 64_000);
            }
            other => panic!("unexpected command: {:?}", other),
        }
    }

    #[test]
    fn slash_effort_sets_and_disables_reasoning_effort() {
        let mut state = AppState::new();
        state.current_model = Some("reasoning-model".to_string());
        state.available_models = vec![deepcode_core::config::ModelProfile {
            id: "reasoning-model".to_string(),
            provider: "test".to_string(),
            display_name: None,
            max_output_tokens: 8192,
            context_window: 64_000,
            reasoning_efforts: vec![
                deepcode_core::config::ReasoningEffort::Off,
                deepcode_core::config::ReasoningEffort::High,
            ],
        }];
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_slash_command("/effort high".to_string(), &mut state, &cmd_tx);
        assert_eq!(state.reasoning_effort.as_deref(), Some("high"));
        assert!(matches!(
            cmd_rx.try_recv().unwrap(),
            agent_event::AgentCommand::SetReasoningEffort { effort: Some(value) }
                if value == "high"
        ));

        handle_slash_command("/effort off".to_string(), &mut state, &cmd_tx);
        assert_eq!(state.reasoning_effort.as_deref(), Some("off"));
        assert!(matches!(
            cmd_rx.try_recv().unwrap(),
            agent_event::AgentCommand::SetReasoningEffort { effort: Some(value) }
                if value == "off"
        ));
    }

    #[test]
    fn model_status_update_does_not_block_effort_change() {
        let mut state = AppState::new();
        state.available_models = vec![deepcode_core::config::ModelProfile {
            id: "reasoning-model".to_string(),
            provider: "test".to_string(),
            display_name: None,
            max_output_tokens: 8192,
            context_window: 64_000,
            reasoning_efforts: vec![
                deepcode_core::config::ReasoningEffort::Off,
                deepcode_core::config::ReasoningEffort::High,
            ],
        }];
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        assert!(handle_slash_command(
            "/model reasoning-model".to_string(),
            &mut state,
            &cmd_tx
        ));
        assert!(matches!(
            cmd_rx.try_recv().unwrap(),
            agent_event::AgentCommand::SetModel { .. }
        ));

        let state = Arc::new(Mutex::new(state));
        handle_agent_event(
            AgentEvent::StatusUpdate {
                message: "Model switched to reasoning-model.".to_string(),
            },
            &state,
        );

        let mut state = state.lock().unwrap();
        assert!(state.working_since.is_none());
        assert!(handle_slash_command(
            "/effort high".to_string(),
            &mut state,
            &cmd_tx
        ));
        assert_eq!(state.reasoning_effort.as_deref(), Some("high"));
        assert!(matches!(
            cmd_rx.try_recv().unwrap(),
            agent_event::AgentCommand::SetReasoningEffort { effort: Some(value) }
                if value == "high"
        ));
    }

    #[test]
    fn slash_resume_opens_current_workspace_session_picker() {
        let root = std::env::temp_dir().join(format!("deepcode_resume_{}", uuid::Uuid::new_v4()));
        let store = crate::session::SessionStore::at(root.clone());
        let mut current = crate::session::SavedSession::new(
            "/workspace".to_string(),
            "deepseek".to_string(),
            "model".to_string(),
            "high".to_string(),
        );
        let mut previous = crate::session::SavedSession::new(
            "/workspace".to_string(),
            "deepseek".to_string(),
            "model".to_string(),
            "high".to_string(),
        );
        let mut other_workspace = crate::session::SavedSession::new(
            "/other-workspace".to_string(),
            "deepseek".to_string(),
            "model".to_string(),
            "high".to_string(),
        );
        for (session, content) in [
            (&mut current, "current"),
            (&mut previous, "previous"),
            (&mut other_workspace, "other"),
        ] {
            session.ui_messages.push(ChatMessage {
                role: "user".to_string(),
                content: content.to_string(),
            });
        }
        store.save(&mut current).unwrap();
        store.save(&mut previous).unwrap();
        store.save(&mut other_workspace).unwrap();

        let mut state = AppState::new();
        state.session_store = Some(store);
        state.session = Some(current);
        let (cmd_tx, _) = agent_event::cmd_channel(4);

        assert!(handle_slash_command(
            "/resume".to_string(),
            &mut state,
            &cmd_tx
        ));

        let picker = state.pending_sessions.as_ref().unwrap();
        assert_eq!(picker.sessions.len(), 2);
        assert!(!picker.show_all);
        assert!(picker
            .sessions
            .iter()
            .all(|session| session.workspace_root == "/workspace"));
        assert!(input_status_text(&state).contains("Up/Down select, Enter resume"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn slash_clear_archives_current_session_and_clears_agent() {
        let root = std::env::temp_dir().join(format!("deepcode_clear_{}", uuid::Uuid::new_v4()));
        let store = crate::session::SessionStore::at(root.clone());
        let session = crate::session::SavedSession::new(
            "/workspace".to_string(),
            "deepseek".to_string(),
            "model".to_string(),
            "high".to_string(),
        );
        let old_id = session.id.clone();
        let mut state = AppState::new();
        state.messages.push(ChatMessage {
            role: "user".to_string(),
            content: "old task".to_string(),
        });
        state
            .core_messages
            .push(deepcode_core::types::Message::user("old task"));
        state.current_model = Some("model".to_string());
        state.reasoning_effort = Some("high".to_string());
        state.session_store = Some(store.clone());
        state.session = Some(session.clone());
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_slash_command("/clear".to_string(), &mut state, &cmd_tx);

        assert!(state.messages.is_empty());
        assert!(state.core_messages.is_empty());
        assert_ne!(state.session.as_ref().unwrap().id, old_id);
        assert!(matches!(
            cmd_rx.try_recv().unwrap(),
            agent_event::AgentCommand::ClearSession
        ));
        assert_eq!(store.load(&old_id).unwrap().title, "old task");
        assert_eq!(store.list(None, None).unwrap().len(), 1);
        assert!(store.load(&state.session.as_ref().unwrap().id).is_err());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn generated_title_event_updates_the_existing_session() {
        let root = std::env::temp_dir().join(format!("deepcode_title_{}", uuid::Uuid::new_v4()));
        let store = crate::session::SessionStore::at(root.clone());
        let mut state = AppState::new();
        state.messages.push(ChatMessage {
            role: "user".to_string(),
            content: "Please inspect the parser implementation".to_string(),
        });
        state
            .core_messages
            .push(deepcode_core::types::Message::user(
                "Please inspect the parser implementation",
            ));
        state.session_store = Some(store.clone());
        state.session = Some(crate::session::SavedSession::new(
            "/workspace".to_string(),
            "deepseek".to_string(),
            "model".to_string(),
            "high".to_string(),
        ));
        let id = state.session.as_ref().unwrap().id.clone();
        state.persist_session().unwrap();

        let shared = Arc::new(Mutex::new(state));
        handle_agent_event(
            AgentEvent::SessionTitleGenerated {
                title: "Inspect parser implementation".to_string(),
            },
            &shared,
        );

        let saved = store.load(&id).unwrap();
        assert_eq!(saved.title, "Inspect parser implementation");
        assert!(saved.title_generated);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn agent_permission_title_is_concise() {
        let mut state = AppState::new();
        state.pending_permission = Some(PendingPermission {
            request_id: "perm_1".to_string(),
            tool_name: "agent".to_string(),
            input: serde_json::json!({
                "task": "Analyze the DeepCode project at /Users/gru/Documents/IntellijIDEAProjects/deepcode. Look at the CLI rendering and permission flow."
            }),
            evaluation: None,
            selected: PermissionChoice::AllowOnce,
        });

        let title = input_status_text(&state);
        assert!(title.starts_with(
            "Permission required: subagent (approval) - Analyze the DeepCode project"
        ));
        assert!(!title.contains("Permission required: agent"));
        assert!(title.chars().count() <= 120);
    }

    #[test]
    fn agent_tool_start_waits_for_subagent_event() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.status = "Preparing 1 tool call(s)...".to_string();
        }

        handle_agent_event(
            AgentEvent::ToolCallStarted {
                id: "call_1".to_string(),
                name: "agent".to_string(),
                input: serde_json::json!({"task": "Analyze the project"}),
            },
            &state,
        );

        let state = state.lock().unwrap();
        assert_eq!(state.status, "Preparing 1 tool call(s)...");
        assert!(state.messages.is_empty());
    }

    #[test]
    fn subagent_status_is_truncated() {
        let state = Arc::new(Mutex::new(AppState::new()));
        let long_task = "Analyze ".repeat(50);

        handle_agent_event(
            AgentEvent::SubagentStarted {
                task_id: "task_1".to_string(),
                task: long_task,
            },
            &state,
        );

        let state = state.lock().unwrap();
        assert!(state.status.starts_with("Subagent running: Analyze "));
        assert!(state.status.ends_with("......"));
        assert!(state.status.chars().count() < 130);
    }
}
