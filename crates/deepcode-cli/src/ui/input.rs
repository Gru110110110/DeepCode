use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use deepcode_agent::event::{self as agent_event};
use unicode_width::UnicodeWidthChar;

use super::{
    AppState, ChatMessage, FilePreviewChoice, PastedInputBlock, PermissionChoice, PlanChoice,
    TuiAction,
};
use std::sync::{Arc, Mutex};

const COLLAPSED_PASTE_MIN_WIDTH: usize = 240;

pub(crate) fn handle_key(
    key: KeyEvent,
    cmd_tx: &agent_event::CmdSender,
    state: &Arc<Mutex<AppState>>,
) -> anyhow::Result<bool> {
    let mut s = state.lock().unwrap();

    if matches!(key.code, KeyCode::Char('o') | KeyCode::Char('O'))
        && key.modifiers.contains(KeyModifiers::CONTROL)
    {
        s.reasoning_expanded = !s.reasoning_expanded;
        return Ok(true);
    }

    if let Some(mut picker) = s.pending_sessions.take() {
        match key.code {
            KeyCode::Up => {
                picker.selected = picker.selected.saturating_sub(1);
                s.pending_sessions = Some(picker);
            }
            KeyCode::Down => {
                if !picker.sessions.is_empty() {
                    picker.selected = (picker.selected + 1).min(picker.sessions.len() - 1);
                }
                s.pending_sessions = Some(picker);
            }
            KeyCode::Char('a') | KeyCode::Char('A') => {
                picker.show_all = !picker.show_all;
                let workspace = s
                    .session
                    .as_ref()
                    .map(|session| session.workspace_root.clone());
                if let Some(store) = &s.session_store {
                    let filter = (!picker.show_all).then_some(workspace.as_deref()).flatten();
                    match store.list(filter, None) {
                        Ok(sessions) => {
                            picker.sessions = sessions;
                            picker.selected = 0;
                        }
                        Err(error) => s.status = format!("Could not list sessions: {}", error),
                    }
                }
                s.pending_sessions = Some(picker);
            }
            KeyCode::Enter => {
                if let Some(session) = picker.selected_session() {
                    let id = session.id.clone();
                    if let Err(error) = s.persist_session() {
                        s.status = format!("Could not save current session: {}", error);
                        s.pending_sessions = Some(picker);
                    } else {
                        s.exit_action = TuiAction::Resume(id);
                        s.running = false;
                        return Ok(false);
                    }
                } else {
                    s.status = "No sessions found.".to_string();
                    s.pending_sessions = Some(picker);
                }
            }
            KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => {
                s.status = "Session picker closed.".to_string();
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                s.running = false;
                return Ok(false);
            }
            _ => s.pending_sessions = Some(picker),
        }
        return Ok(true);
    }

    if let Some(mut plan) = s.pending_plan.take() {
        let choice = match key.code {
            KeyCode::Left => {
                plan.selected = plan.selected.previous();
                s.pending_plan = Some(plan);
                return Ok(true);
            }
            KeyCode::Right | KeyCode::Tab => {
                plan.selected = plan.selected.next();
                s.pending_plan = Some(plan);
                return Ok(true);
            }
            KeyCode::Enter => plan.selected,
            KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Char('a') | KeyCode::Char('A') => {
                PlanChoice::Approve
            }
            KeyCode::Char('r') | KeyCode::Char('R') => PlanChoice::Revise,
            KeyCode::Char('n') | KeyCode::Char('N') => PlanChoice::Reject,
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                s.running = false;
                return Ok(false);
            }
            KeyCode::Esc => PlanChoice::Reject,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                s.running = false;
                return Ok(false);
            }
            _ => {
                s.pending_plan = Some(plan);
                return Ok(true);
            }
        };

        match choice {
            PlanChoice::Approve => {
                let send_result = cmd_tx.blocking_send(agent_event::AgentCommand::PlanResponse {
                    request_id: plan.request_id,
                    approved: true,
                });
                s.status = "Executing approved plan...".to_string();
                if send_result.is_err() {
                    s.status = "Agent channel closed.".to_string();
                }
            }
            PlanChoice::Reject => {
                let send_result = cmd_tx.blocking_send(agent_event::AgentCommand::PlanResponse {
                    request_id: plan.request_id,
                    approved: false,
                });
                s.status = "Plan rejected.".to_string();
                if send_result.is_err() {
                    s.status = "Agent channel closed.".to_string();
                }
            }
            PlanChoice::Revise => {
                let send_result = cmd_tx.blocking_send(agent_event::AgentCommand::PlanResponse {
                    request_id: plan.request_id,
                    approved: false,
                });
                s.input = "Revise this plan: ".to_string();
                s.cursor_pos = s.input.len();
                s.clear_input_blocks();
                s.working_since = None;
                s.status = "Plan rejected. Edit the revision request and send it.".to_string();
                if send_result.is_err() {
                    s.status = "Agent channel closed.".to_string();
                }
            }
            PlanChoice::Quit => {
                s.running = false;
                return Ok(false);
            }
        }
        return Ok(true);
    }

    if let Some(mut preview) = s.pending_file_preview.take() {
        let choice = match key.code {
            KeyCode::Left => {
                preview.selected = preview.selected.previous();
                s.pending_file_preview = Some(preview);
                return Ok(true);
            }
            KeyCode::Right | KeyCode::Tab => {
                preview.selected = preview.selected.next();
                s.pending_file_preview = Some(preview);
                return Ok(true);
            }
            KeyCode::Enter => preview.selected,
            KeyCode::Char('y') | KeyCode::Char('Y') => FilePreviewChoice::Apply,
            KeyCode::Char('a') | KeyCode::Char('A') => FilePreviewChoice::Apply,
            KeyCode::Char('n') | KeyCode::Char('N') => FilePreviewChoice::Reject,
            KeyCode::Char('r') | KeyCode::Char('R') => FilePreviewChoice::Reject,
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                s.running = false;
                return Ok(false);
            }
            KeyCode::Esc => {
                let request_id = preview.request_id;
                s.interrupt_requested = true;
                s.status = "Change rejected.".to_string();
                let send_result =
                    cmd_tx.blocking_send(agent_event::AgentCommand::FileChangePreviewResponse {
                        request_id,
                        approved: false,
                    });
                if send_result.is_err() {
                    s.status = "Agent channel closed.".to_string();
                } else {
                    super::events::activate_next_approval(&mut s);
                }
                return Ok(true);
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                s.running = false;
                return Ok(false);
            }
            _ => {
                s.pending_file_preview = Some(preview);
                return Ok(true);
            }
        };

        let Some(approved) = choice.response() else {
            s.running = false;
            return Ok(false);
        };

        let send_result =
            cmd_tx.blocking_send(agent_event::AgentCommand::FileChangePreviewResponse {
                request_id: preview.request_id,
                approved,
            });
        s.status = if approved {
            "Applying changes...".to_string()
        } else {
            "Change rejected.".to_string()
        };
        if send_result.is_err() {
            s.status = "Agent channel closed.".to_string();
        } else {
            super::events::activate_next_approval(&mut s);
        }
        return Ok(true);
    }

    // Handle permission prompt if active
    if let Some(mut perm) = s.pending_permission.take() {
        let choice = match key.code {
            KeyCode::Left => {
                perm.selected = perm.selected.previous();
                s.pending_permission = Some(perm);
                return Ok(true);
            }
            KeyCode::Right | KeyCode::Tab => {
                perm.selected = perm.selected.next();
                s.pending_permission = Some(perm);
                return Ok(true);
            }
            KeyCode::Enter => perm.selected,
            KeyCode::Char('y') | KeyCode::Char('Y') => PermissionChoice::AllowOnce,
            KeyCode::Char('s') | KeyCode::Char('S') => PermissionChoice::AllowSession,
            KeyCode::Char('a') | KeyCode::Char('A') => PermissionChoice::AllowAlways,
            KeyCode::Char('n') | KeyCode::Char('N') => PermissionChoice::Deny,
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                s.running = false;
                return Ok(false);
            }
            KeyCode::Esc => {
                s.pending_permission = None;
                s.interrupt_requested = true;
                s.status = "Interrupting...".to_string();
                let send_result = cmd_tx.blocking_send(agent_event::AgentCommand::Interrupt);
                if send_result.is_err() {
                    s.status = "Agent channel closed.".to_string();
                }
                return Ok(true);
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                s.running = false;
                return Ok(false);
            }
            _ => {
                // Unrecognized key: restore pending permission and ignore
                s.pending_permission = Some(perm);
                return Ok(true);
            }
        };

        let Some((approved, scope)) = choice.response() else {
            s.running = false;
            return Ok(false);
        };

        let send_result = cmd_tx.blocking_send(agent_event::AgentCommand::PermissionResponse {
            request_id: perm.request_id,
            approved,
            scope,
        });
        s.status = if approved {
            "Thinking...".to_string()
        } else {
            "Permission denied.".to_string()
        };
        if send_result.is_err() {
            s.status = "Agent channel closed.".to_string();
        } else {
            super::events::activate_next_approval(&mut s);
        }
        return Ok(true);
    }

    match key.code {
        KeyCode::BackTab => {
            let enabled = !s.plan_mode_enabled;
            super::events::set_plan_mode(&mut s, cmd_tx, enabled);
        }
        KeyCode::Tab if key.modifiers.contains(KeyModifiers::SHIFT) => {
            let enabled = !s.plan_mode_enabled;
            super::events::set_plan_mode(&mut s, cmd_tx, enabled);
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            s.running = false;
            return Ok(false);
        }
        KeyCode::Esc => {
            if s.working_since.is_some() {
                s.interrupt_requested = true;
                s.status = "Interrupting...".to_string();
                let send_result = cmd_tx.blocking_send(agent_event::AgentCommand::Interrupt);
                if send_result.is_err() {
                    s.status = "Agent channel closed.".to_string();
                }
            } else {
                s.status = "Press Ctrl+C to exit.".to_string();
            }
        }
        KeyCode::Up
            if key.modifiers.contains(KeyModifiers::SHIFT)
                || key.modifiers.contains(KeyModifiers::ALT) =>
        {
            s.scroll_up(3);
        }
        KeyCode::Down
            if key.modifiers.contains(KeyModifiers::SHIFT)
                || key.modifiers.contains(KeyModifiers::ALT) =>
        {
            s.scroll_down(3);
        }
        KeyCode::PageUp => {
            let page = s.viewport.last_transcript_visible.max(1);
            s.scroll_up(page);
        }
        KeyCode::PageDown => {
            let page = s.viewport.last_transcript_visible.max(1);
            s.scroll_down(page);
        }
        KeyCode::Enter => {
            if s.working_since.is_some() && !s.input.starts_with('/') {
                s.status = "Already working; press Esc to interrupt.".to_string();
                return Ok(true);
            }
            let input = std::mem::take(&mut s.input);
            s.cursor_pos = 0;
            s.clear_input_blocks();

            if input.starts_with('/') {
                if !super::events::handle_slash_command(input, &mut s, cmd_tx) {
                    return Ok(false);
                }
            } else if !input.is_empty() {
                s.messages.push(ChatMessage {
                    role: "user".to_string(),
                    content: input.clone(),
                });
                s.core_messages
                    .push(deepcode_core::types::Message::user(&input));
                s.scroll_to_bottom();
                s.streaming_text.clear();
                s.last_usage = None;
                s.working_since = Some(std::time::Instant::now());
                s.interrupt_requested = false;
                s.status = if s.plan_mode_enabled {
                    "Planning...".to_string()
                } else {
                    "Working...".to_string()
                };

                if cmd_tx
                    .blocking_send(agent_event::AgentCommand::Process { message: input })
                    .is_err()
                {
                    s.status = "Agent channel closed.".to_string();
                    s.working_since = None;
                } else if let Err(error) = s.persist_session() {
                    s.status = format!("Message sent, but session save failed: {}", error);
                }
            }
        }
        KeyCode::Backspace => {
            if !delete_paste_block_before_cursor(&mut s) {
                let pos = clamp_char_boundary(&s.input, s.cursor_pos);
                if let Some(prev) = prev_char_boundary(&s.input, pos) {
                    delete_input_range(&mut s, prev, pos);
                    s.cursor_pos = prev;
                }
            }
        }
        KeyCode::Delete => {
            if !delete_paste_block_at_cursor(&mut s) {
                let pos = clamp_char_boundary(&s.input, s.cursor_pos);
                if let Some(next) = next_char_boundary(&s.input, pos) {
                    delete_input_range(&mut s, pos, next);
                    s.cursor_pos = pos;
                }
            }
        }
        KeyCode::Left => {
            move_cursor_left(&mut s);
        }
        KeyCode::Right => {
            move_cursor_right(&mut s);
        }
        KeyCode::Home => {
            s.cursor_pos = 0;
        }
        KeyCode::End if s.input.is_empty() => {
            s.scroll_to_bottom();
        }
        KeyCode::End => {
            s.cursor_pos = s.input.len();
        }
        KeyCode::Char(ch) => {
            insert_text(&mut s, &ch.to_string());
        }
        _ => {}
    }

    Ok(true)
}

pub(crate) fn handle_paste(text: String, state: &Arc<Mutex<AppState>>) -> anyhow::Result<bool> {
    let mut s = state.lock().unwrap();
    if s.pending_plan.is_some()
        || s.pending_sessions.is_some()
        || s.pending_file_preview.is_some()
        || s.pending_permission.is_some()
    {
        s.status = "Finish the current prompt before pasting.".to_string();
        return Ok(true);
    }

    insert_paste(&mut s, &text);
    Ok(true)
}

pub(crate) fn insert_paste(state: &mut AppState, text: &str) {
    if should_collapse_paste(text) {
        insert_collapsed_paste(state, text);
    } else {
        insert_text(state, text);
    }
}

fn should_collapse_paste(text: &str) -> bool {
    logical_line_count(text) > 1
        || unicode_width::UnicodeWidthStr::width(text) >= COLLAPSED_PASTE_MIN_WIDTH
}

fn insert_text(state: &mut AppState, text: &str) {
    if text.is_empty() {
        return;
    }

    snap_cursor_out_of_paste_block(state);
    let pos = clamp_char_boundary(&state.input, state.cursor_pos);
    shift_blocks_for_insert(&mut state.pasted_input_blocks, pos, text.len());
    state.input.insert_str(pos, text);
    state.cursor_pos = pos + text.len();
}

fn insert_collapsed_paste(state: &mut AppState, text: &str) {
    if text.is_empty() {
        return;
    }

    snap_cursor_out_of_paste_block(state);
    let pos = clamp_char_boundary(&state.input, state.cursor_pos);
    shift_blocks_for_insert(&mut state.pasted_input_blocks, pos, text.len());
    state.input.insert_str(pos, text);
    let block = PastedInputBlock {
        id: state.next_pasted_input_id,
        start: pos,
        end: pos + text.len(),
        line_count: logical_line_count(text),
        char_count: text.chars().count(),
    };
    state.next_pasted_input_id = state.next_pasted_input_id.saturating_add(1);
    state.pasted_input_blocks.push(block);
    state.pasted_input_blocks.sort_by_key(|block| block.start);
    state.cursor_pos = pos + text.len();
}

fn logical_line_count(text: &str) -> usize {
    if text.is_empty() {
        0
    } else {
        text.split('\n').count()
    }
}

fn move_cursor_left(state: &mut AppState) {
    let pos = clamp_char_boundary(&state.input, state.cursor_pos);
    if let Some(block) = state
        .pasted_input_blocks
        .iter()
        .find(|block| pos > block.start && pos <= block.end)
    {
        state.cursor_pos = block.start;
        return;
    }

    if let Some(prev) = prev_char_boundary(&state.input, pos) {
        state.cursor_pos = prev;
    }
}

fn move_cursor_right(state: &mut AppState) {
    let pos = clamp_char_boundary(&state.input, state.cursor_pos);
    if let Some(block) = state
        .pasted_input_blocks
        .iter()
        .find(|block| pos >= block.start && pos < block.end)
    {
        state.cursor_pos = block.end;
        return;
    }

    if let Some(next) = next_char_boundary(&state.input, pos) {
        state.cursor_pos = next;
    }
}

fn delete_paste_block_before_cursor(state: &mut AppState) -> bool {
    let pos = clamp_char_boundary(&state.input, state.cursor_pos);
    let Some((start, end)) = state
        .pasted_input_blocks
        .iter()
        .find(|block| pos > block.start && pos <= block.end)
        .map(|block| (block.start, block.end))
    else {
        return false;
    };

    delete_input_range(state, start, end);
    state.cursor_pos = start;
    true
}

fn delete_paste_block_at_cursor(state: &mut AppState) -> bool {
    let pos = clamp_char_boundary(&state.input, state.cursor_pos);
    let Some((start, end)) = state
        .pasted_input_blocks
        .iter()
        .find(|block| pos >= block.start && pos < block.end)
        .map(|block| (block.start, block.end))
    else {
        return false;
    };

    delete_input_range(state, start, end);
    state.cursor_pos = start;
    true
}

fn delete_input_range(state: &mut AppState, start: usize, end: usize) {
    if start >= end || end > state.input.len() {
        return;
    }

    state.input.drain(start..end);
    shift_blocks_for_delete(&mut state.pasted_input_blocks, start, end);
    if state.input.is_empty() {
        state.clear_input_blocks();
    }
}

fn snap_cursor_out_of_paste_block(state: &mut AppState) {
    let pos = clamp_char_boundary(&state.input, state.cursor_pos);
    state.cursor_pos = state
        .pasted_input_blocks
        .iter()
        .find(|block| pos > block.start && pos < block.end)
        .map(|block| block.end)
        .unwrap_or(pos);
}

fn shift_blocks_for_insert(blocks: &mut [PastedInputBlock], pos: usize, len: usize) {
    for block in blocks {
        if block.start >= pos {
            block.start = block.start.saturating_add(len);
            block.end = block.end.saturating_add(len);
        } else if block.end > pos {
            block.end = block.end.saturating_add(len);
        }
    }
}

fn shift_blocks_for_delete(blocks: &mut Vec<PastedInputBlock>, start: usize, end: usize) {
    let deleted_len = end.saturating_sub(start);
    blocks.retain(|block| block.end <= start || block.start >= end);
    for block in blocks.iter_mut() {
        if block.start >= end {
            block.start = block.start.saturating_sub(deleted_len);
            block.end = block.end.saturating_sub(deleted_len);
        }
    }
}

fn clamp_char_boundary(input: &str, pos: usize) -> usize {
    let mut pos = pos.min(input.len());
    while pos > 0 && !input.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

fn prev_char_boundary(input: &str, pos: usize) -> Option<usize> {
    let pos = clamp_char_boundary(input, pos);
    if pos == 0 {
        return None;
    }
    input[..pos].char_indices().last().map(|(idx, _)| idx)
}

fn next_char_boundary(input: &str, pos: usize) -> Option<usize> {
    let pos = clamp_char_boundary(input, pos);
    if pos >= input.len() {
        return None;
    }
    input[pos..]
        .char_indices()
        .nth(1)
        .map(|(offset, _)| pos + offset)
        .or(Some(input.len()))
}

pub(crate) fn input_view_with_blocks(
    input: &str,
    blocks: &[PastedInputBlock],
    cursor_pos: usize,
    width: u16,
) -> (String, u16) {
    let (display_input, display_cursor_pos) = input_display_projection(input, blocks, cursor_pos);
    input_view_plain(&display_input, display_cursor_pos, width)
}

fn input_view_plain(input: &str, cursor_pos: usize, width: u16) -> (String, u16) {
    let cursor_pos = clamp_char_boundary(input, cursor_pos);
    let capacity = width as usize;
    if capacity == 0 {
        return (String::new(), 0);
    }

    let before = &input[..cursor_pos];
    let after = &input[cursor_pos..];
    let before_visible = trim_start_to_width(before, capacity);
    let before_width = unicode_width::UnicodeWidthStr::width(before_visible);
    let after_visible = take_width(after, capacity.saturating_sub(before_width));

    (
        format!("{}{}", before_visible, after_visible),
        before_width.min(capacity) as u16,
    )
}

fn input_display_projection(
    input: &str,
    blocks: &[PastedInputBlock],
    cursor_pos: usize,
) -> (String, usize) {
    let cursor_pos = clamp_char_boundary(input, cursor_pos);
    let mut display = String::new();
    let mut display_cursor_pos = None;
    let mut raw_start = 0usize;

    let mut sorted_blocks = blocks
        .iter()
        .filter(|block| block.start <= block.end && block.end <= input.len())
        .collect::<Vec<_>>();
    sorted_blocks.sort_by_key(|block| block.start);

    for block in sorted_blocks {
        if block.start < raw_start {
            continue;
        }

        if display_cursor_pos.is_none() && cursor_pos >= raw_start && cursor_pos <= block.start {
            display_cursor_pos = Some(display.len() + cursor_pos.saturating_sub(raw_start));
        }
        display.push_str(&input[raw_start..block.start]);

        let label = block.label();
        if display_cursor_pos.is_none() && cursor_pos >= block.start && cursor_pos <= block.end {
            display_cursor_pos = Some(if cursor_pos == block.start {
                display.len()
            } else {
                display.len() + label.len()
            });
        }
        display.push_str(&label);
        raw_start = block.end;
    }

    if display_cursor_pos.is_none() {
        display_cursor_pos = Some(display.len() + cursor_pos.saturating_sub(raw_start));
    }
    display.push_str(&input[raw_start..]);

    let display_cursor_pos = display_cursor_pos.unwrap_or(display.len());
    (display, display_cursor_pos)
}

fn trim_start_to_width(value: &str, max_width: usize) -> &str {
    if unicode_width::UnicodeWidthStr::width(value) <= max_width {
        return value;
    }

    for (idx, _) in value.char_indices() {
        let candidate = &value[idx..];
        if unicode_width::UnicodeWidthStr::width(candidate) <= max_width {
            return candidate;
        }
    }

    ""
}

fn take_width(value: &str, max_width: usize) -> &str {
    if max_width == 0 {
        return "";
    }

    let mut end = 0;
    let mut width = 0;
    for (idx, ch) in value.char_indices() {
        let ch_width = ch.width().unwrap_or(0);
        if width + ch_width > max_width {
            break;
        }
        width += ch_width;
        end = idx + ch.len_utf8();
    }
    &value[..end]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::{DeferredApproval, PendingPermission, PendingPlanApproval};
    use deepcode_permissions::policy::ApprovalScope;

    #[test]
    fn shift_tab_toggles_between_plan_and_agent_modes() {
        let state = Arc::new(Mutex::new(AppState::new()));
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            &cmd_tx,
            &state,
        )
        .unwrap();

        {
            let state = state.lock().unwrap();
            assert!(state.plan_mode_enabled);
            assert!(state.status.contains("Plan mode"));
        }
        assert!(matches!(
            cmd_rx.try_recv(),
            Ok(agent_event::AgentCommand::SetPlanMode { enabled: true })
        ));

        handle_key(
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            &cmd_tx,
            &state,
        )
        .unwrap();

        {
            let state = state.lock().unwrap();
            assert!(!state.plan_mode_enabled);
            assert!(state.status.contains("Agent mode"));
        }
        assert!(matches!(
            cmd_rx.try_recv(),
            Ok(agent_event::AgentCommand::SetPlanMode { enabled: false })
        ));
    }

    #[test]
    fn shift_tab_does_not_change_mode_during_plan_review() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.pending_plan = Some(PendingPlanApproval {
                request_id: "plan_1".to_string(),
                plan: "1. Inspect".to_string(),
                selected: PlanChoice::Approve,
            });
        }
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            &cmd_tx,
            &state,
        )
        .unwrap();

        let state = state.lock().unwrap();
        assert!(!state.plan_mode_enabled);
        assert!(state.pending_plan.is_some());
        assert!(cmd_rx.try_recv().is_err());
    }

    #[test]
    fn shift_tab_does_not_change_mode_while_agent_is_working() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.working_since = Some(std::time::Instant::now());
        }
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
            &cmd_tx,
            &state,
        )
        .unwrap();

        let state = state.lock().unwrap();
        assert!(!state.plan_mode_enabled);
        assert!(state.status.contains("interrupt before switching"));
        assert!(cmd_rx.try_recv().is_err());
    }

    #[test]
    fn char_boundary_helpers_handle_multibyte_input() {
        let input = "a中🙂b";
        let after_a = 1;
        let after_zh = after_a + "中".len();
        let after_emoji = after_zh + "🙂".len();

        assert_eq!(next_char_boundary(input, 0), Some(after_a));
        assert_eq!(next_char_boundary(input, after_a), Some(after_zh));
        assert_eq!(next_char_boundary(input, after_zh), Some(after_emoji));
        assert_eq!(prev_char_boundary(input, after_emoji), Some(after_zh));
        assert_eq!(prev_char_boundary(input, after_zh), Some(after_a));
    }

    #[test]
    fn clamp_char_boundary_moves_to_valid_byte_index() {
        let input = "中";
        assert_eq!(clamp_char_boundary(input, 1), 0);
        assert_eq!(clamp_char_boundary(input, 2), 0);
        assert_eq!(clamp_char_boundary(input, 3), 3);
    }

    #[test]
    fn input_view_accounts_for_wide_characters() {
        let input = "ab中文";
        let (visible, cursor_col) = input_view_with_blocks(input, &[], input.len(), 6);

        assert_eq!(visible, input);
        assert_eq!(cursor_col, 6);
    }

    #[test]
    fn input_view_keeps_cursor_inside_capacity() {
        let input = "abcdefghijklmnopqrstuvwxyz";
        let (visible, cursor_col) = input_view_with_blocks(input, &[], input.len(), 8);

        assert_eq!(visible, "stuvwxyz");
        assert_eq!(cursor_col, 8);
    }

    #[test]
    fn multiline_paste_collapses_in_view_and_preserves_input() {
        let mut state = AppState::new();
        let pasted = "one\ntwo\nthree";

        insert_paste(&mut state, pasted);

        assert_eq!(state.input, pasted);
        assert_eq!(state.pasted_input_blocks.len(), 1);
        let (visible, cursor_col) = input_view_with_blocks(
            &state.input,
            &state.pasted_input_blocks,
            state.cursor_pos,
            80,
        );

        assert_eq!(visible, "[Pasted text #1 +3 lines]");
        assert_eq!(usize::from(cursor_col), visible.len());
    }

    #[test]
    fn small_paste_is_inserted_normally() {
        let mut state = AppState::new();

        insert_paste(&mut state, "hello");

        assert_eq!(state.input, "hello");
        assert!(state.pasted_input_blocks.is_empty());
        assert_eq!(state.cursor_pos, "hello".len());
    }

    #[test]
    fn backspace_deletes_collapsed_paste_block() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            insert_paste(&mut state, "one\ntwo\nthree");
        }
        let (cmd_tx, _) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap();

        let state = state.lock().unwrap();
        assert!(state.input.is_empty());
        assert!(state.pasted_input_blocks.is_empty());
        assert_eq!(state.cursor_pos, 0);
    }

    #[test]
    fn delete_deletes_collapsed_paste_block_at_cursor() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.input = "say ".to_string();
            state.cursor_pos = state.input.len();
            insert_paste(&mut state, "one\ntwo\nthree");
            state.cursor_pos = "say ".len();
        }
        let (cmd_tx, _) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::Delete, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap();

        let state = state.lock().unwrap();
        assert_eq!(state.input, "say ");
        assert!(state.pasted_input_blocks.is_empty());
        assert_eq!(state.cursor_pos, "say ".len());
    }

    #[test]
    fn cursor_navigation_skips_collapsed_paste_block() {
        let mut state = AppState::new();
        state.input = "a".to_string();
        state.cursor_pos = state.input.len();
        insert_paste(&mut state, "one\ntwo\nthree");
        state.input.push('z');
        state.cursor_pos = state.input.len();

        move_cursor_left(&mut state);
        assert_eq!(state.cursor_pos, "aone\ntwo\nthree".len());

        move_cursor_left(&mut state);
        assert_eq!(state.cursor_pos, "a".len());

        move_cursor_right(&mut state);
        assert_eq!(state.cursor_pos, "aone\ntwo\nthree".len());
    }

    #[test]
    fn enter_sends_full_pasted_text_and_clears_blocks() {
        let state = Arc::new(Mutex::new(AppState::new()));
        let pasted = "one\ntwo\nthree";
        {
            let mut state = state.lock().unwrap();
            insert_paste(&mut state, pasted);
        }
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap();

        match cmd_rx.try_recv().unwrap() {
            agent_event::AgentCommand::Process { message } => assert_eq!(message, pasted),
            other => panic!("unexpected command: {:?}", other),
        }
        let state = state.lock().unwrap();
        assert!(state.input.is_empty());
        assert!(state.pasted_input_blocks.is_empty());
    }

    #[test]
    fn first_sent_message_creates_a_session_checkpoint() {
        let root = std::env::temp_dir().join(format!("deepcode_input_{}", uuid::Uuid::new_v4()));
        let store = crate::session::SessionStore::at(root.clone());
        let session = crate::session::SavedSession::new(
            "/workspace".to_string(),
            "deepseek".to_string(),
            "model".to_string(),
            "high".to_string(),
        );
        let id = session.id.clone();
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.input = "recover this request".to_string();
            state.cursor_pos = state.input.len();
            state.session_store = Some(store.clone());
            state.session = Some(session);
        }
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap();

        assert!(matches!(
            cmd_rx.try_recv(),
            Ok(agent_event::AgentCommand::Process { message }) if message == "recover this request"
        ));
        let saved = store.load(&id).unwrap();
        assert_eq!(saved.title, "recover this request");
        assert_eq!(saved.ui_messages.len(), 1);
        assert_eq!(saved.core_messages.len(), 1);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn permission_prompt_uses_arrows_and_enter() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.pending_permission = Some(PendingPermission {
                request_id: "perm_1".to_string(),
                tool_name: "web_search".to_string(),
                input: serde_json::json!({"query": "DeepSeek V4"}),
                evaluation: None,
                selected: PermissionChoice::AllowOnce,
            });
        }
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::Right, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap();
        {
            let state = state.lock().unwrap();
            assert_eq!(
                state.pending_permission.as_ref().unwrap().selected,
                PermissionChoice::AllowSession
            );
        }

        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap();

        match cmd_rx.try_recv().unwrap() {
            agent_event::AgentCommand::PermissionResponse {
                request_id,
                approved,
                scope,
            } => {
                assert_eq!(request_id, "perm_1");
                assert!(approved);
                assert_eq!(scope, ApprovalScope::Session);
            }
            other => panic!("unexpected command: {:?}", other),
        }
        assert!(state.lock().unwrap().pending_permission.is_none());
    }

    #[test]
    fn session_picker_selects_and_returns_resume_action() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.pending_sessions = Some(crate::ui::PendingSessionPicker {
                sessions: vec![
                    crate::session::SessionSummary {
                        id: "first".to_string(),
                        title: "First".to_string(),
                        workspace_root: "/one".to_string(),
                        updated_at: "2026-01-01T00:00:00Z".to_string(),
                        provider: "deepseek".to_string(),
                        model: "model".to_string(),
                    },
                    crate::session::SessionSummary {
                        id: "second".to_string(),
                        title: "Second".to_string(),
                        workspace_root: "/one".to_string(),
                        updated_at: "2026-02-01T00:00:00Z".to_string(),
                        provider: "deepseek".to_string(),
                        model: "model".to_string(),
                    },
                ],
                selected: 0,
                show_all: false,
            });
        }
        let (cmd_tx, _) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap();
        assert_eq!(
            state
                .lock()
                .unwrap()
                .pending_sessions
                .as_ref()
                .unwrap()
                .selected,
            1
        );

        assert!(!handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap());
        assert_eq!(
            state.lock().unwrap().exit_action,
            TuiAction::Resume("second".to_string())
        );
    }

    #[test]
    fn file_preview_prompt_applies_with_enter() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.pending_file_preview = Some(crate::ui::PendingFilePreview {
                request_id: "preview_1".to_string(),
                preview: deepcode_tools::tool::FileChangePreview {
                    path: "a.txt".to_string(),
                    before_exists: true,
                    before: "old\n".to_string(),
                    after: "new\n".to_string(),
                    unified_diff: String::new(),
                },
                selected: FilePreviewChoice::Apply,
            });
        }
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap();

        match cmd_rx.try_recv().unwrap() {
            agent_event::AgentCommand::FileChangePreviewResponse {
                request_id,
                approved,
            } => {
                assert_eq!(request_id, "preview_1");
                assert!(approved);
            }
            other => panic!("unexpected command: {:?}", other),
        }
        assert!(state.lock().unwrap().pending_file_preview.is_none());
    }

    #[test]
    fn rejecting_preview_with_escape_activates_deferred_approval() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.pending_file_preview = Some(crate::ui::PendingFilePreview {
                request_id: "preview_1".to_string(),
                preview: deepcode_tools::tool::FileChangePreview {
                    path: "a.txt".to_string(),
                    before_exists: true,
                    before: "old\n".to_string(),
                    after: "new\n".to_string(),
                    unified_diff: String::new(),
                },
                selected: FilePreviewChoice::Apply,
            });
            state
                .deferred_approvals
                .push_back(DeferredApproval::Permission(PendingPermission {
                    request_id: "permission_2".to_string(),
                    tool_name: "shell".to_string(),
                    input: serde_json::json!({"command":"true"}),
                    evaluation: None,
                    selected: PermissionChoice::AllowOnce,
                }));
        }
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap();

        assert!(matches!(
            cmd_rx.try_recv().unwrap(),
            agent_event::AgentCommand::FileChangePreviewResponse {
                request_id,
                approved: false,
            } if request_id == "preview_1"
        ));
        let state = state.lock().unwrap();
        assert_eq!(
            state
                .pending_permission
                .as_ref()
                .map(|permission| permission.request_id.as_str()),
            Some("permission_2")
        );
        assert!(state.deferred_approvals.is_empty());
    }

    #[test]
    fn plan_prompt_approves_with_enter() {
        let state = Arc::new(Mutex::new(AppState::new()));
        {
            let mut state = state.lock().unwrap();
            state.pending_plan = Some(PendingPlanApproval {
                request_id: "plan_1".to_string(),
                plan: "1. Edit files\n2. Run tests".to_string(),
                selected: PlanChoice::Approve,
            });
        }
        let (cmd_tx, mut cmd_rx) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
            &cmd_tx,
            &state,
        )
        .unwrap();

        match cmd_rx.try_recv().unwrap() {
            agent_event::AgentCommand::PlanResponse {
                request_id,
                approved,
            } => {
                assert_eq!(request_id, "plan_1");
                assert!(approved);
            }
            other => panic!("unexpected command: {:?}", other),
        }
        assert!(state.lock().unwrap().pending_plan.is_none());
    }

    #[test]
    fn ctrl_o_toggles_reasoning_even_during_a_preview() {
        let state = Arc::new(Mutex::new(AppState::new()));
        state.lock().unwrap().pending_file_preview = Some(crate::ui::PendingFilePreview {
            request_id: "preview_1".to_string(),
            preview: deepcode_tools::tool::FileChangePreview {
                path: "a.txt".to_string(),
                before_exists: true,
                before: "old\n".to_string(),
                after: "new\n".to_string(),
                unified_diff: String::new(),
            },
            selected: FilePreviewChoice::Apply,
        });
        let (cmd_tx, _) = agent_event::cmd_channel(4);

        handle_key(
            KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL),
            &cmd_tx,
            &state,
        )
        .unwrap();

        let state = state.lock().unwrap();
        assert!(state.reasoning_expanded);
        assert!(state.pending_file_preview.is_some());
    }
}
