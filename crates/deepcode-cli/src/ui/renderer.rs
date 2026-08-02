use std::io::{Stdout, Write};

use crossterm::event::{MouseButton, MouseEvent, MouseEventKind};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Position, Rect};
use ratatui::prelude::*;
use ratatui::widgets::{
    Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
    StatefulWidget,
};
use ratatui::{Frame, Terminal};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::banner::startup_header_lines;
use super::diff::render_diff_message;
use super::events::{
    file_preview_options_line, input_status_text, permission_options_line, plan_options_line,
    should_hide_empty_input_prompt,
};
use super::markdown::render_message_lines;
use super::scrolling::{MouseScrollState, ScrollDirection, TranscriptScroll};
use super::{AppState, ChatMessage, StartupHeader, MESSAGE_GAP_LINES};

const STATUS_STYLE: Style = Style::new().fg(Color::DarkGray);
const INPUT_PROMPT_STYLE: Style = Style::new().fg(Color::Cyan);
const INPUT_IDLE_BG: Color = Color::Rgb(244, 244, 244);
const SELECTION_BG: Color = Color::Rgb(198, 222, 255);
const SCROLL_TRACK: &str = "│";
const SCROLL_THUMB: &str = "┃";

type AppTerminal = Terminal<CrosstermBackend<Stdout>>;

pub(crate) struct FullscreenRenderer {
    terminal: AppTerminal,
}

#[derive(Debug, Default)]
pub(crate) struct ViewportState {
    pub transcript_scroll: TranscriptScroll,
    pub pending_scroll_delta: i32,
    pub mouse_scroll: MouseScrollState,
    pub transcript_scrollbar_dragging: bool,
    pub transcript_selection: TranscriptSelection,
    pub last_transcript_area: Option<Rect>,
    pub last_transcript_top: usize,
    pub last_transcript_visible: usize,
    pub last_transcript_total: usize,
    pub last_transcript_padding_top: usize,
    transcript_cache: TranscriptLineCache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TranscriptSelectionPoint {
    line: usize,
    column: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TranscriptSelection {
    anchor: Option<TranscriptSelectionPoint>,
    head: Option<TranscriptSelectionPoint>,
    dragging: bool,
}

impl TranscriptSelection {
    fn clear(&mut self) {
        self.anchor = None;
        self.head = None;
        self.dragging = false;
    }

    fn ordered_endpoints(self) -> Option<(TranscriptSelectionPoint, TranscriptSelectionPoint)> {
        let anchor = self.anchor?;
        let head = self.head?;
        if (anchor.line, anchor.column) <= (head.line, head.column) {
            Some((anchor, head))
        } else {
            Some((head, anchor))
        }
    }
}

#[derive(Debug, Clone)]
struct TranscriptLineCache {
    width: u16,
    startup_header: Option<StartupHeader>,
    signature: Vec<(String, String)>,
    lines: Vec<Line<'static>>,
}

impl Default for TranscriptLineCache {
    fn default() -> Self {
        Self {
            width: 0,
            startup_header: None,
            signature: Vec::new(),
            lines: vec![Line::from("")],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InputSnapshot {
    safe_width: u16,
    status_text: String,
    mode: InputModeSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InputModeSnapshot {
    HiddenWorking,
    Permission {
        options: String,
    },
    FilePreview {
        options: String,
    },
    PlanApproval {
        options: String,
    },
    Editing {
        visible_input: String,
        cursor_col: u16,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TranscriptScrollbar {
    top: usize,
    visible: usize,
    total: usize,
}

impl FullscreenRenderer {
    pub(crate) fn new(stdout: Stdout) -> anyhow::Result<Self> {
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        terminal.clear()?;
        Ok(Self { terminal })
    }

    pub(crate) fn draw(&mut self, state: &mut AppState) -> anyhow::Result<()> {
        self.terminal.draw(|frame| render_frame(frame, state))?;
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> anyhow::Result<()> {
        self.terminal.show_cursor()?;
        Ok(())
    }

    pub(crate) fn copy_to_clipboard(&mut self, text: &str) -> anyhow::Result<()> {
        let encoded = base64_encode(text.as_bytes());
        let sequence = format!("\x1b]52;c;{}\x07", encoded);
        self.terminal.backend_mut().write_all(sequence.as_bytes())?;
        std::io::Write::flush(self.terminal.backend_mut())?;
        Ok(())
    }
}

pub(crate) fn handle_mouse(mouse: MouseEvent, state: &mut AppState) -> Option<String> {
    match mouse.kind {
        MouseEventKind::ScrollUp => {
            let update = state.viewport.mouse_scroll.on_scroll(ScrollDirection::Up);
            state.viewport.pending_scroll_delta = state
                .viewport
                .pending_scroll_delta
                .saturating_add(update.delta_lines);
        }
        MouseEventKind::ScrollDown => {
            let update = state.viewport.mouse_scroll.on_scroll(ScrollDirection::Down);
            state.viewport.pending_scroll_delta = state
                .viewport
                .pending_scroll_delta
                .saturating_add(update.delta_lines);
        }
        MouseEventKind::Down(MouseButton::Left) => {
            state.viewport.transcript_scrollbar_dragging = false;
            state.viewport.transcript_selection.dragging = false;
            if mouse_hits_transcript_scrollbar(state, mouse) {
                state.viewport.transcript_selection.clear();
                state.viewport.transcript_scrollbar_dragging = true;
            } else if let Some(point) = selection_point_from_mouse(state, mouse) {
                state.viewport.transcript_selection.anchor = Some(point);
                state.viewport.transcript_selection.head = Some(point);
                state.viewport.transcript_selection.dragging = true;
            } else {
                state.viewport.transcript_selection.clear();
            }
        }
        MouseEventKind::Drag(MouseButton::Left) if state.viewport.transcript_scrollbar_dragging => {
            scroll_transcript_to_mouse_row(state, mouse.row);
        }
        MouseEventKind::Drag(MouseButton::Left) if state.viewport.transcript_selection.dragging => {
            update_selection_drag(state, mouse);
        }
        MouseEventKind::Up(MouseButton::Left) if state.viewport.transcript_scrollbar_dragging => {
            state.viewport.transcript_scrollbar_dragging = false;
        }
        MouseEventKind::Up(MouseButton::Left) if state.viewport.transcript_selection.dragging => {
            state.viewport.transcript_selection.dragging = false;
            if let Some(text) = selection_to_text(state).filter(|text| !text.is_empty()) {
                return Some(text);
            }
            state.viewport.transcript_selection.clear();
        }
        _ => {}
    }

    None
}

pub(crate) fn scroll_transcript_to_mouse_row(state: &mut AppState, row: u16) -> bool {
    let Some(area) = state.viewport.last_transcript_area else {
        return false;
    };
    let total = state.viewport.last_transcript_total;
    let visible = state.viewport.last_transcript_visible;
    if area.height == 0 || total <= visible {
        return false;
    }

    let max_start = total.saturating_sub(visible);
    if max_start == 0 {
        state.scroll_to_bottom();
        return true;
    }

    let max_row = usize::from(area.height.saturating_sub(1));
    let relative_row = usize::from(row.saturating_sub(area.y)).min(max_row);
    let top = relative_row
        .saturating_mul(max_start)
        .saturating_add(max_row / 2)
        .checked_div(max_row)
        .unwrap_or(0);

    state.viewport.transcript_scroll = if top >= max_start {
        TranscriptScroll::to_bottom()
    } else {
        TranscriptScroll::at_line(top)
    };
    state.viewport.pending_scroll_delta = 0;
    true
}

fn mouse_hits_transcript_scrollbar(state: &AppState, mouse: MouseEvent) -> bool {
    let Some(area) = state.viewport.last_transcript_area else {
        return false;
    };
    if area.width <= 1
        || state.viewport.last_transcript_total <= state.viewport.last_transcript_visible
    {
        return false;
    }

    let scrollbar_col = area.x.saturating_add(area.width.saturating_sub(1));
    mouse.column == scrollbar_col
        && mouse.row >= area.y
        && mouse.row < area.y.saturating_add(area.height)
}

fn selection_point_from_mouse(
    state: &AppState,
    mouse: MouseEvent,
) -> Option<TranscriptSelectionPoint> {
    let area = state.viewport.last_transcript_area?;
    let text_width = area.width.saturating_sub(1).max(1);
    if area.width == 0
        || mouse.row < area.y
        || mouse.row >= area.y.saturating_add(area.height)
        || mouse.column < area.x
        || mouse.column >= area.x.saturating_add(text_width)
    {
        return None;
    }

    let row_offset = usize::from(mouse.row.saturating_sub(area.y));
    if row_offset < state.viewport.last_transcript_padding_top {
        return None;
    }
    let visual_line = row_offset - state.viewport.last_transcript_padding_top;
    let line = state
        .viewport
        .last_transcript_top
        .saturating_add(visual_line);
    if line >= state.viewport.last_transcript_total {
        return None;
    }

    let raw_column = usize::from(mouse.column.saturating_sub(area.x));
    let max_column = state
        .viewport
        .transcript_cache
        .lines
        .get(line)
        .map(line_width)
        .unwrap_or(0);

    Some(TranscriptSelectionPoint {
        line,
        column: raw_column.min(max_column),
    })
}

fn update_selection_drag(state: &mut AppState, mouse: MouseEvent) {
    if let Some(point) = selection_point_from_mouse(state, mouse) {
        state.viewport.transcript_selection.head = Some(point);
        return;
    }

    let Some(area) = state.viewport.last_transcript_area else {
        return;
    };
    if mouse.row < area.y {
        state.scroll_up(1);
    } else if mouse.row >= area.y.saturating_add(area.height) {
        state.scroll_down(1);
    }
}

fn selection_to_text(state: &AppState) -> Option<String> {
    let (start, end) = state.viewport.transcript_selection.ordered_endpoints()?;
    if start == end {
        return None;
    }

    let mut selected = String::new();
    for line_index in start.line..=end.line {
        let line = state.viewport.transcript_cache.lines.get(line_index)?;
        let text = line_text(line);
        let width = UnicodeWidthStr::width(text.as_str());
        let start_col = if line_index == start.line {
            start.column.min(width)
        } else {
            0
        };
        let end_col = if line_index == end.line {
            end.column.min(width)
        } else {
            width
        };

        if line_index > start.line {
            selected.push('\n');
        }
        selected.push_str(&slice_text_by_display_cols(&text, start_col, end_col));
    }

    Some(selected)
}

fn render_frame(frame: &mut Frame<'_>, state: &mut AppState) {
    let area = frame.area();
    if area.width == 0 || area.height == 0 {
        return;
    }

    let footer_height = area.height.min(2);
    let transcript_area = Rect::new(
        area.x,
        area.y,
        area.width,
        area.height.saturating_sub(footer_height),
    );
    render_transcript(frame, state, transcript_area);

    if footer_height > 0 {
        let footer_area = Rect::new(
            area.x,
            area.y + area.height.saturating_sub(footer_height),
            area.width,
            footer_height,
        );
        render_footer(frame, state, footer_area);
    }
    if state.pending_sessions.is_some() {
        render_session_picker(frame, state);
    }
}

fn render_session_picker(frame: &mut Frame<'_>, state: &AppState) {
    let Some(picker) = &state.pending_sessions else {
        return;
    };
    let outer = frame.area();
    let width = outer.width.saturating_sub(2).clamp(1, 100);
    let height = (picker.sessions.len() as u16 + 4)
        .min(outer.height.saturating_sub(1))
        .max(1);
    let area = Rect::new(
        outer.x + outer.width.saturating_sub(width) / 2,
        outer.y + outer.height.saturating_sub(height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, area);
    let scope = if picker.show_all {
        "All workspaces"
    } else {
        "Current workspace"
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" Sessions - {} ", scope));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let visible = usize::from(inner.height).max(1);
    let start = picker
        .selected
        .saturating_sub(visible / 2)
        .min(picker.sessions.len().saturating_sub(visible));
    let lines = picker
        .sessions
        .iter()
        .enumerate()
        .skip(start)
        .take(visible)
        .map(|(index, session)| {
            let marker = if index == picker.selected { ">" } else { " " };
            let id = session.id.chars().take(8).collect::<String>();
            let date = session.updated_at.chars().take(10).collect::<String>();
            let text = format!(
                "{} {}  {}  {}  {}/{}",
                marker, id, date, session.title, session.provider, session.model
            );
            let style = if index == picker.selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Line::from(Span::styled(
                super::banner::truncate_display_width_end(&text, usize::from(inner.width)),
                style,
            ))
        })
        .collect::<Vec<_>>();
    let lines = if lines.is_empty() {
        vec![Line::from("No saved sessions")]
    } else {
        lines
    };
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_transcript(frame: &mut Frame<'_>, state: &mut AppState, area: Rect) {
    Block::default()
        .style(Style::default().bg(Color::Reset))
        .render(area, frame.buffer_mut());
    let content_area = without_wrap_column(area);
    state.viewport.last_transcript_area = Some(content_area);

    if area.height == 0 {
        state.viewport.last_transcript_visible = 0;
        state.viewport.last_transcript_total = 0;
        state.viewport.last_transcript_top = 0;
        state.viewport.last_transcript_padding_top = 0;
        return;
    }

    let text_width = content_area.width.max(1);
    let lines = state
        .viewport
        .transcript_cache
        .ensure(
            state.startup_header.as_ref(),
            &state.messages,
            &state.streaming_text,
            &state.reasoning_text,
            state.reasoning_elapsed_secs(),
            state.reasoning_expanded,
            text_width,
        )
        .to_vec();
    let total_lines = lines.len();
    let visible_lines = usize::from(area.height);
    let (top, end, scrollbar) =
        resolve_transcript_window(&mut state.viewport, total_lines, visible_lines);
    let mut visible = if total_lines == 0 {
        vec![Line::from("")]
    } else {
        lines[top..end].to_vec()
    };
    apply_transcript_selection(&mut visible, top, state.viewport.transcript_selection);

    state.viewport.last_transcript_padding_top = 0;

    Paragraph::new(visible).render(content_area, frame.buffer_mut());

    if let Some(scrollbar) = scrollbar {
        let scrollable_range = scrollbar.total.saturating_sub(scrollbar.visible);
        let mut scrollbar_state = ScrollbarState::new(scrollable_range)
            .position(scrollbar.top.min(scrollable_range))
            .viewport_content_length(scrollbar.visible);
        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .track_symbol(Some(SCROLL_TRACK))
            .track_style(Style::default().fg(Color::DarkGray))
            .thumb_symbol(SCROLL_THUMB)
            .thumb_style(Style::default().fg(Color::Cyan))
            .render(content_area, frame.buffer_mut(), &mut scrollbar_state);
    }
}

fn without_wrap_column(area: Rect) -> Rect {
    // A write in the terminal's last column can auto-wrap and scroll the whole TUI.
    Rect::new(area.x, area.y, area.width.saturating_sub(1), area.height)
}

fn apply_transcript_selection(
    visible: &mut [Line<'static>],
    top: usize,
    selection: TranscriptSelection,
) {
    let Some((start, end)) = selection.ordered_endpoints() else {
        return;
    };
    if start == end {
        return;
    }

    for (offset, line) in visible.iter_mut().enumerate() {
        let line_index = top + offset;
        if line_index < start.line || line_index > end.line {
            continue;
        }

        let line_width = line_width(line);
        let start_col = if line_index == start.line {
            start.column.min(line_width)
        } else {
            0
        };
        let end_col = if line_index == end.line {
            end.column.min(line_width)
        } else {
            line_width
        };
        if start_col < end_col {
            highlight_line_range(line, start_col, end_col);
        }
    }
}

fn highlight_line_range(line: &mut Line<'static>, start_col: usize, end_col: usize) {
    let spans = std::mem::take(&mut line.spans);
    let mut highlighted = Vec::new();
    let mut col = 0usize;

    for span in spans {
        let style = span.style;
        let content = span.content.into_owned();
        let mut buffer = String::new();
        let mut selected_state: Option<bool> = None;

        for ch in content.chars() {
            let width = char_width(ch);
            let char_start = col;
            let char_end = col.saturating_add(width);
            let selected = char_end > start_col && char_start < end_col;
            if selected_state.is_some_and(|state| state != selected) {
                push_selection_span(&mut highlighted, &mut buffer, style, selected_state);
            }
            selected_state = Some(selected);
            buffer.push(ch);
            col = char_end;
        }

        push_selection_span(&mut highlighted, &mut buffer, style, selected_state);
    }

    line.spans = highlighted;
}

fn push_selection_span(
    out: &mut Vec<Span<'static>>,
    buffer: &mut String,
    style: Style,
    selected: Option<bool>,
) {
    if buffer.is_empty() {
        return;
    }
    let style = if selected.unwrap_or(false) {
        style.bg(SELECTION_BG)
    } else {
        style
    };
    out.push(Span::styled(std::mem::take(buffer), style));
}

fn render_footer(frame: &mut Frame<'_>, state: &AppState, area: Rect) {
    let area = without_wrap_column(area);
    let snapshot = input_snapshot_for_width(state, area.width.saturating_add(1));
    let status_area = Rect::new(area.x, area.y, area.width, 1);
    let input_area = Rect::new(
        area.x,
        area.y.saturating_add(1),
        area.width,
        area.height.saturating_sub(1),
    );

    if area.height >= 1 {
        let status_text = super::banner::truncate_display_width_end(
            &snapshot.status_text,
            usize::from(snapshot.safe_width),
        );
        Paragraph::new(Line::from(Span::styled(status_text, STATUS_STYLE)))
            .render(status_area, frame.buffer_mut());
    }

    if input_area.height == 0 {
        return;
    }

    match snapshot.mode {
        InputModeSnapshot::HiddenWorking => {
            Block::default()
                .style(Style::default().bg(INPUT_IDLE_BG))
                .render(input_area, frame.buffer_mut());
            Paragraph::new(Line::from(Span::styled("> ", STATUS_STYLE)))
                .style(Style::default().bg(INPUT_IDLE_BG))
                .render(input_area, frame.buffer_mut());
            frame.set_cursor_position(Position::new(input_area.x.saturating_add(2), input_area.y));
        }
        InputModeSnapshot::Permission { options }
        | InputModeSnapshot::FilePreview { options }
        | InputModeSnapshot::PlanApproval { options } => {
            let options = super::banner::truncate_display_width_end(
                &options,
                usize::from(snapshot.safe_width),
            );
            let cursor_col =
                UnicodeWidthStr::width(options.as_str()).min(usize::from(snapshot.safe_width));
            Paragraph::new(Line::from(options)).render(input_area, frame.buffer_mut());
            frame.set_cursor_position(Position::new(
                input_area.x.saturating_add(cursor_col as u16),
                input_area.y,
            ));
        }
        InputModeSnapshot::Editing {
            visible_input,
            cursor_col,
        } => {
            let line = Line::from(vec![
                Span::styled("> ", INPUT_PROMPT_STYLE),
                Span::raw(visible_input),
            ]);
            Paragraph::new(line).render(input_area, frame.buffer_mut());
            frame.set_cursor_position(Position::new(
                input_area.x.saturating_add(2).saturating_add(cursor_col),
                input_area.y,
            ));
        }
    }
}

fn resolve_transcript_window(
    viewport: &mut ViewportState,
    total_lines: usize,
    visible_lines: usize,
) -> (usize, usize, Option<TranscriptScrollbar>) {
    if viewport.pending_scroll_delta != 0 {
        viewport.transcript_scroll = viewport.transcript_scroll.scrolled_by(
            viewport.pending_scroll_delta,
            total_lines,
            visible_lines,
        );
        viewport.pending_scroll_delta = 0;
    }

    let max_start = total_lines.saturating_sub(visible_lines);
    let (scroll_state, top) = viewport.transcript_scroll.resolve_top(max_start);
    viewport.transcript_scroll = scroll_state;
    viewport.last_transcript_top = top;
    viewport.last_transcript_visible = visible_lines;
    viewport.last_transcript_total = total_lines;
    let end = top.saturating_add(visible_lines).min(total_lines);
    let scrollbar =
        (total_lines > visible_lines && visible_lines > 0).then_some(TranscriptScrollbar {
            top,
            visible: visible_lines,
            total: total_lines,
        });

    (top, end, scrollbar)
}

impl TranscriptLineCache {
    #[allow(clippy::too_many_arguments)]
    fn ensure(
        &mut self,
        startup_header: Option<&StartupHeader>,
        messages: &[ChatMessage],
        streaming_text: &str,
        reasoning_text: &str,
        reasoning_elapsed_secs: Option<u64>,
        reasoning_expanded: bool,
        width: u16,
    ) -> &[Line<'static>] {
        let signature = transcript_signature(
            messages,
            streaming_text,
            reasoning_text,
            reasoning_elapsed_secs,
            reasoning_expanded,
        );
        if self.width != width
            || self.startup_header.as_ref() != startup_header
            || self.signature != signature
        {
            self.width = width;
            self.startup_header = startup_header.cloned();
            self.signature = signature;
            self.lines = build_transcript_lines(
                startup_header,
                messages,
                streaming_text,
                reasoning_text,
                reasoning_elapsed_secs,
                reasoning_expanded,
                width,
            );
        }
        &self.lines
    }
}

fn transcript_signature(
    messages: &[ChatMessage],
    streaming_text: &str,
    reasoning_text: &str,
    reasoning_elapsed_secs: Option<u64>,
    reasoning_expanded: bool,
) -> Vec<(String, String)> {
    let mut signature = messages
        .iter()
        .map(|msg| (msg.role.clone(), msg.content.clone()))
        .collect::<Vec<_>>();
    signature.push((
        "reasoning:expanded".to_string(),
        reasoning_expanded.to_string(),
    ));
    if !streaming_text.is_empty() {
        signature.push((
            "assistant:streaming".to_string(),
            streaming_text.to_string(),
        ));
    }
    if let Some(elapsed) = reasoning_elapsed_secs {
        signature.push((
            "reasoning:streaming".to_string(),
            format!("{elapsed}:{reasoning_expanded}:{reasoning_text}"),
        ));
    }
    signature
}

fn build_transcript_lines(
    startup_header: Option<&StartupHeader>,
    messages: &[ChatMessage],
    streaming_text: &str,
    reasoning_text: &str,
    reasoning_elapsed_secs: Option<u64>,
    reasoning_expanded: bool,
    width: u16,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if let Some(header) = startup_header {
        lines.extend(startup_header_lines(header, width));
        lines.push(Line::from(""));
    }

    for msg in messages {
        lines.extend(render_chat_message(msg, width, reasoning_expanded));
        lines.extend(std::iter::repeat_n(Line::from(""), MESSAGE_GAP_LINES));
    }

    if let Some(elapsed) = reasoning_elapsed_secs {
        lines.extend(render_reasoning(
            reasoning_text,
            elapsed,
            true,
            reasoning_expanded,
            width,
        ));
        lines.extend(std::iter::repeat_n(Line::from(""), MESSAGE_GAP_LINES));
    }

    if !streaming_text.is_empty() {
        let msg = ChatMessage {
            role: "assistant".to_string(),
            content: streaming_text.to_string(),
        };
        lines.extend(render_chat_message(&msg, width, reasoning_expanded));
        lines.extend(std::iter::repeat_n(Line::from(""), MESSAGE_GAP_LINES));
    }

    if lines.is_empty() {
        lines.push(Line::from(""));
    }
    lines
}

fn render_chat_message(
    msg: &ChatMessage,
    width: u16,
    reasoning_expanded: bool,
) -> Vec<Line<'static>> {
    let prefix = match msg.role.as_str() {
        "user" => Some(("> ", Color::Cyan)),
        "assistant" => Some(("• ", Color::Green)),
        "tool" => Some(("▸ ", Color::DarkGray)),
        "activity" => None,
        "error" => Some(("x ", Color::Red)),
        "diff" => None,
        _ => None,
    };

    if let Some(elapsed) = msg
        .role
        .strip_prefix("reasoning:")
        .and_then(|value| value.parse::<u64>().ok())
    {
        render_reasoning(&msg.content, elapsed, false, reasoning_expanded, width)
    } else if msg.role == "diff" {
        render_diff_message(&msg.content, width)
    } else {
        render_message_lines(&msg.content, prefix, width)
    }
}

fn render_reasoning(
    content: &str,
    elapsed: u64,
    active: bool,
    expanded: bool,
    width: u16,
) -> Vec<Line<'static>> {
    let label = if active {
        "Thinking ... "
    } else {
        "Thought for "
    };
    let mut header = Line::from(vec![
        Span::styled(label, STATUS_STYLE),
        Span::styled(
            format!("{elapsed}s"),
            STATUS_STYLE.add_modifier(Modifier::BOLD),
        ),
    ]);
    if !active {
        header.spans.push(Span::styled(
            if expanded {
                " (ctrl+o to collapse)"
            } else {
                " (ctrl+o to expand)"
            },
            STATUS_STYLE,
        ));
    }

    let mut lines = vec![header];
    if expanded && !content.is_empty() {
        lines.extend(render_message_lines(
            content,
            Some(("  ", Color::DarkGray)),
            width,
        ));
    }
    lines
}

fn input_snapshot_for_width(state: &AppState, width: u16) -> InputSnapshot {
    let width = width.max(1);
    let safe_width = width.saturating_sub(1);
    let status_text = input_status_text(state);
    let mode = if should_hide_empty_input_prompt(state) {
        InputModeSnapshot::HiddenWorking
    } else if let Some(plan) = &state.pending_plan {
        InputModeSnapshot::PlanApproval {
            options: plan_options_line(plan),
        }
    } else if let Some(preview) = &state.pending_file_preview {
        InputModeSnapshot::FilePreview {
            options: file_preview_options_line(preview),
        }
    } else if let Some(permission) = &state.pending_permission {
        InputModeSnapshot::Permission {
            options: permission_options_line(permission),
        }
    } else {
        let input_capacity = safe_width.saturating_sub(2);
        let (visible_input, cursor_col) = super::input::input_view_with_blocks(
            &state.input,
            &state.pasted_input_blocks,
            state.cursor_pos,
            input_capacity,
        );
        InputModeSnapshot::Editing {
            visible_input,
            cursor_col,
        }
    };

    InputSnapshot {
        safe_width,
        status_text,
        mode,
    }
}

fn line_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| UnicodeWidthStr::width(span.content.as_ref()))
        .sum()
}

fn line_text(line: &Line<'_>) -> String {
    line.spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<Vec<_>>()
        .join("")
}

fn slice_text_by_display_cols(text: &str, start_col: usize, end_col: usize) -> String {
    if start_col >= end_col {
        return String::new();
    }

    let mut out = String::new();
    let mut col = 0usize;
    for ch in text.chars() {
        let width = char_width(ch);
        let next = col.saturating_add(width);
        if next > start_col && col < end_col {
            out.push(ch);
        }
        col = next;
        if col >= end_col {
            break;
        }
    }
    out
}

fn char_width(ch: char) -> usize {
    if ch == '\t' {
        4
    } else {
        ch.width().unwrap_or(1)
    }
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);

        out.push(TABLE[(b0 >> 2) as usize] as char);
        out.push(TABLE[(((b0 & 0b0000_0011) << 4) | (b1 >> 4)) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[(((b1 & 0b0000_1111) << 2) | (b2 >> 6)) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(b2 & 0b0011_1111) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::{PendingPermission, PermissionChoice};
    use crossterm::event::KeyModifiers;
    use ratatui::backend::TestBackend;

    #[test]
    fn input_snapshot_is_stable_for_unchanged_state() {
        let mut state = AppState::new();
        state.status = "Thinking...".to_string();

        let first = input_snapshot_for_width(&state, 80);
        let second = input_snapshot_for_width(&state, 80);

        assert_eq!(first, second);
    }

    #[test]
    fn input_snapshot_changes_when_permission_selection_changes() {
        let mut state = AppState::new();
        state.pending_permission = Some(PendingPermission {
            request_id: "perm_1".to_string(),
            tool_name: "agent".to_string(),
            input: serde_json::json!({"task": "Analyze project"}),
            evaluation: None,
            selected: PermissionChoice::AllowOnce,
        });
        let first = input_snapshot_for_width(&state, 80);

        state.pending_permission.as_mut().unwrap().selected = PermissionChoice::AllowAlways;
        let second = input_snapshot_for_width(&state, 80);

        assert_ne!(first, second);
    }

    #[test]
    fn completed_reasoning_is_collapsed_by_default() {
        let lines = render_reasoning("inspect parser\ncheck tests", 10, false, false, 80);
        let text = lines.iter().map(line_text).collect::<Vec<_>>().join("\n");

        assert_eq!(text, "Thought for 10s (ctrl+o to expand)");
        assert!(lines[0].spans[1]
            .style
            .add_modifier
            .contains(Modifier::BOLD));
    }

    #[test]
    fn reasoning_content_renders_when_expanded() {
        let lines = render_reasoning("inspect parser\ncheck tests", 10, false, true, 80);
        let text = lines.iter().map(line_text).collect::<Vec<_>>().join("\n");

        assert!(text.starts_with("Thought for 10s (ctrl+o to collapse)"));
        assert!(text.contains("inspect parser"));
        assert!(text.contains("check tests"));
    }

    #[test]
    fn completed_reasoning_cache_rebuilds_when_expansion_changes() {
        let messages = vec![ChatMessage {
            role: "reasoning:10".to_string(),
            content: "inspect parser".to_string(),
        }];
        let mut cache = TranscriptLineCache::default();
        let collapsed = cache
            .ensure(None, &messages, "", "", None, false, 80)
            .iter()
            .map(line_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!collapsed.contains("inspect parser"));

        let expanded = cache
            .ensure(None, &messages, "", "", None, true, 80)
            .iter()
            .map(line_text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(expanded.contains("inspect parser"));
    }

    #[test]
    fn active_reasoning_shows_elapsed_seconds() {
        let lines = render_reasoning("", 5, true, false, 80);

        assert_eq!(line_text(&lines[0]), "Thinking ... 5s");
    }

    #[test]
    fn transcript_window_slices_visible_range() {
        let mut viewport = ViewportState {
            transcript_scroll: TranscriptScroll::at_line(30),
            ..ViewportState::default()
        };

        let (top, end, _) = resolve_transcript_window(&mut viewport, 100, 20);

        assert_eq!((top, end), (30, 50));
    }

    #[test]
    fn short_transcript_starts_directly_below_scrollable_header() {
        let mut state = AppState::with_messages(vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }]);
        state.startup_header = Some(StartupHeader {
            model: "test-model xhigh".to_string(),
            directory: "/tmp/project".to_string(),
        });
        let header_line_count =
            startup_header_lines(state.startup_header.as_ref().unwrap(), 79).len();
        let mut terminal = Terminal::new(TestBackend::new(80, 30)).unwrap();

        terminal
            .draw(|frame| render_frame(frame, &mut state))
            .unwrap();

        assert_eq!(state.viewport.last_transcript_area.unwrap().y, 0);
        assert_eq!(state.viewport.last_transcript_top, 0);
        assert_eq!(state.viewport.last_transcript_padding_top, 0);
        assert!(line_text(&state.viewport.transcript_cache.lines[0]).contains("____"));
        assert_eq!(
            line_text(&state.viewport.transcript_cache.lines[header_line_count]),
            ""
        );
        assert!(
            line_text(&state.viewport.transcript_cache.lines[header_line_count + 1])
                .starts_with("> hello")
        );
    }

    #[test]
    fn startup_header_scrolls_with_long_transcript() {
        let messages = (0..20)
            .map(|index| ChatMessage {
                role: "user".to_string(),
                content: format!("message {index}"),
            })
            .collect();
        let mut state = AppState::with_messages(messages);
        state.startup_header = Some(StartupHeader {
            model: "test-model xhigh".to_string(),
            directory: "/tmp/project".to_string(),
        });
        let header_line_count =
            startup_header_lines(state.startup_header.as_ref().unwrap(), 79).len();
        let mut terminal = Terminal::new(TestBackend::new(80, 20)).unwrap();

        terminal
            .draw(|frame| render_frame(frame, &mut state))
            .unwrap();

        assert!(state.viewport.last_transcript_top > header_line_count);
        state.scroll_up(usize::MAX);
        terminal
            .draw(|frame| render_frame(frame, &mut state))
            .unwrap();
        assert_eq!(state.viewport.last_transcript_top, 0);
    }

    #[test]
    fn session_picker_fits_small_terminal() {
        let mut state = AppState::new();
        state.pending_sessions = Some(crate::ui::PendingSessionPicker {
            sessions: vec![crate::session::SessionSummary {
                id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
                title: "A long session title".to_string(),
                workspace_root: "/tmp/project".to_string(),
                updated_at: "2026-07-18T00:00:00Z".to_string(),
                provider: "deepseek".to_string(),
                model: "deepseek-v4-pro".to_string(),
            }],
            selected: 0,
            show_all: false,
        });
        let mut terminal = Terminal::new(TestBackend::new(12, 4)).unwrap();

        terminal
            .draw(|frame| render_frame(frame, &mut state))
            .unwrap();
    }

    #[test]
    fn wide_input_keeps_terminal_wrap_column_empty_across_frames() {
        let mut state = AppState::new();
        state.input = "当前项目只是3D地球，我想改成太阳系3D效果".to_string();
        state.cursor_pos = state.input.len();
        let mut terminal = Terminal::new(TestBackend::new(20, 8)).unwrap();

        terminal
            .draw(|frame| render_frame(frame, &mut state))
            .unwrap();
        state.input.push_str("，太阳在正中间");
        state.cursor_pos = state.input.len();
        terminal
            .draw(|frame| render_frame(frame, &mut state))
            .unwrap();

        assert_wrap_column_is_empty(&terminal, 20, 8);
        terminal
            .backend_mut()
            .assert_cursor_position(Position::new(18, 7));
    }

    #[test]
    fn transcript_scrollbar_does_not_use_terminal_wrap_column() {
        let messages = (0..20)
            .map(|index| ChatMessage {
                role: "assistant".to_string(),
                content: format!("第 {index} 行输出包含中文宽字符"),
            })
            .collect();
        let mut state = AppState::with_messages(messages);
        let mut terminal = Terminal::new(TestBackend::new(20, 8)).unwrap();

        terminal
            .draw(|frame| render_frame(frame, &mut state))
            .unwrap();

        assert_wrap_column_is_empty(&terminal, 20, 8);
        assert_eq!(state.viewport.last_transcript_area.unwrap().width, 19);
    }

    #[test]
    fn streaming_does_not_pull_scrolled_view_to_bottom() {
        let mut viewport = ViewportState {
            transcript_scroll: TranscriptScroll::at_line(10),
            ..ViewportState::default()
        };

        let (top, _, _) = resolve_transcript_window(&mut viewport, 100, 20);
        assert_eq!(top, 10);

        let (top, _, _) = resolve_transcript_window(&mut viewport, 120, 20);
        assert_eq!(top, 10);
        assert_ne!(viewport.transcript_scroll, TranscriptScroll::to_bottom());
    }

    #[test]
    fn tail_tracks_new_streaming_content() {
        let mut viewport = ViewportState::default();

        let (top, _, _) = resolve_transcript_window(&mut viewport, 100, 20);
        assert_eq!(top, 80);
        assert_eq!(viewport.transcript_scroll, TranscriptScroll::to_bottom());

        let (top, _, _) = resolve_transcript_window(&mut viewport, 120, 20);
        assert_eq!(top, 100);
        assert_eq!(viewport.transcript_scroll, TranscriptScroll::to_bottom());
    }

    #[test]
    fn scrollbar_drag_maps_rows_to_offsets() {
        let mut state = AppState::new();
        state.viewport.last_transcript_area = Some(Rect::new(0, 10, 40, 11));
        state.viewport.last_transcript_total = 120;
        state.viewport.last_transcript_visible = 20;

        assert!(scroll_transcript_to_mouse_row(&mut state, 10));
        assert_eq!(
            state.viewport.transcript_scroll,
            TranscriptScroll::at_line(0)
        );

        assert!(scroll_transcript_to_mouse_row(&mut state, 15));
        assert_eq!(
            state.viewport.transcript_scroll,
            TranscriptScroll::at_line(50)
        );

        assert!(scroll_transcript_to_mouse_row(&mut state, 20));
        assert_eq!(
            state.viewport.transcript_scroll,
            TranscriptScroll::to_bottom()
        );
    }

    #[test]
    fn selection_point_maps_mouse_to_transcript_line_and_column() {
        let mut state = AppState::new();
        state.viewport.last_transcript_area = Some(Rect::new(2, 3, 20, 5));
        state.viewport.last_transcript_top = 10;
        state.viewport.last_transcript_total = 20;
        state.viewport.last_transcript_padding_top = 1;
        state.viewport.transcript_cache.lines = (0..20)
            .map(|idx| Line::from(format!("line-{idx}")))
            .collect();

        let mouse = mouse_event(MouseEventKind::Down(MouseButton::Left), 7, 5);
        let point = selection_point_from_mouse(&state, mouse).unwrap();

        assert_eq!(
            point,
            TranscriptSelectionPoint {
                line: 11,
                column: 5
            }
        );
    }

    #[test]
    fn selection_to_text_extracts_multiline_text() {
        let mut state = AppState::new();
        state.viewport.transcript_cache.lines = vec![
            Line::from("alpha"),
            Line::from("bravo"),
            Line::from("charlie"),
        ];
        state.viewport.transcript_selection.anchor =
            Some(TranscriptSelectionPoint { line: 0, column: 2 });
        state.viewport.transcript_selection.head =
            Some(TranscriptSelectionPoint { line: 2, column: 4 });

        assert_eq!(selection_to_text(&state).unwrap(), "pha\nbravo\nchar");
    }

    #[test]
    fn selection_highlight_preserves_text_and_adds_background() {
        let mut lines = vec![Line::from(vec![
            Span::styled("alpha", Style::default().fg(Color::Green)),
            Span::raw(" beta"),
        ])];
        let selection = TranscriptSelection {
            anchor: Some(TranscriptSelectionPoint { line: 4, column: 2 }),
            head: Some(TranscriptSelectionPoint { line: 4, column: 8 }),
            dragging: false,
        };

        apply_transcript_selection(&mut lines, 4, selection);

        assert_eq!(line_text(&lines[0]), "alpha beta");
        assert!(lines[0]
            .spans
            .iter()
            .any(|span| span.style.bg == Some(SELECTION_BG)));
    }

    #[test]
    fn handle_mouse_returns_selected_text_on_left_button_up() {
        let mut state = AppState::new();
        state.viewport.last_transcript_area = Some(Rect::new(0, 0, 12, 4));
        state.viewport.last_transcript_top = 0;
        state.viewport.last_transcript_total = 2;
        state.viewport.last_transcript_visible = 4;
        state.viewport.transcript_cache.lines = vec![Line::from("hello"), Line::from("world")];

        assert_eq!(
            handle_mouse(
                mouse_event(MouseEventKind::Down(MouseButton::Left), 1, 0),
                &mut state
            ),
            None
        );
        assert_eq!(
            handle_mouse(
                mouse_event(MouseEventKind::Drag(MouseButton::Left), 3, 1),
                &mut state
            ),
            None
        );

        let copied = handle_mouse(
            mouse_event(MouseEventKind::Up(MouseButton::Left), 3, 1),
            &mut state,
        )
        .unwrap();

        assert_eq!(copied, "ello\nwor");
    }

    #[test]
    fn base64_encode_matches_osc52_payload_encoding() {
        assert_eq!(base64_encode(b"hello"), "aGVsbG8=");
        assert_eq!(base64_encode("复制".as_bytes()), "5aSN5Yi2");
    }

    fn mouse_event(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
        MouseEvent {
            kind,
            column,
            row,
            modifiers: KeyModifiers::NONE,
        }
    }

    fn assert_wrap_column_is_empty(terminal: &Terminal<TestBackend>, width: u16, height: u16) {
        let wrap_column = width.saturating_sub(1);
        for row in 0..height {
            assert_eq!(
                terminal.backend().buffer()[(wrap_column, row)].symbol(),
                " "
            );
        }
    }
}
