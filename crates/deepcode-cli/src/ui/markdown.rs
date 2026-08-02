use pulldown_cmark::{
    Alignment as MarkdownAlignment, CodeBlockKind, Event as MarkdownEvent, HeadingLevel,
    Options as MarkdownOptions, Parser as MarkdownParser, Tag as MarkdownTag, TagEnd,
};
use ratatui::prelude::*;
use std::sync::LazyLock;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const INLINE_CODE_BG: Color = Color::Rgb(249, 249, 249);

pub(crate) fn render_message_lines(
    content: &str,
    prefix: Option<(&str, Color)>,
    full_width: u16,
) -> Vec<Line<'static>> {
    let prefix_width = prefix
        .map(|(text, _)| UnicodeWidthStr::width(text))
        .unwrap_or(0);
    let content_width = full_width.saturating_sub(prefix_width as u16).max(1);
    let rendered = render_markdown_message(content, usize::from(content_width));
    let mut result = Vec::new();
    let mut first = true;

    for line in rendered {
        let mut spans = Vec::new();
        if let Some((prefix_text, prefix_color)) = prefix {
            if first {
                spans.push(Span::styled(
                    prefix_text.to_string(),
                    Style::default().fg(prefix_color),
                ));
            } else {
                spans.push(Span::raw(" ".repeat(prefix_width)));
            }
        }
        spans.extend(line.spans);
        result.push(Line::from(spans));
        first = false;
    }

    if result.is_empty() {
        result.push(Line::from(""));
    }

    result
}

fn push_span_buffer(spans: &mut Vec<Span<'static>>, style: Style, buffer: &mut String) {
    if !buffer.is_empty() {
        spans.push(Span::styled(std::mem::take(buffer), style));
    }
}

/// Render Markdown chat text, preserving syntax highlighting for code blocks.
pub(crate) fn render_markdown_message(content: &str, terminal_width: usize) -> Vec<Line<'static>> {
    let mut options = MarkdownOptions::empty();
    options.insert(MarkdownOptions::ENABLE_TABLES);
    options.insert(MarkdownOptions::ENABLE_STRIKETHROUGH);
    options.insert(MarkdownOptions::ENABLE_TASKLISTS);

    let parser = MarkdownParser::new_ext(content, options);
    MarkdownTerminalRenderer::new(terminal_width).render(parser)
}

struct MarkdownTerminalRenderer {
    terminal_width: usize,
    lines: Vec<Line<'static>>,
    current: Vec<Span<'static>>,
    style: Style,
    style_stack: Vec<Style>,
    quote_depth: usize,
    list_stack: Vec<ListState>,
    code_block: Option<CodeBlockState>,
    table: Option<TableState>,
}

struct ListState {
    next_number: Option<u64>,
}

struct CodeBlockState {
    lang: String,
    code: String,
}

struct TableState {
    alignments: Vec<MarkdownAlignment>,
    rows: Vec<Vec<Vec<Span<'static>>>>,
    current_row: Vec<Vec<Span<'static>>>,
    current_cell: Vec<Span<'static>>,
}

impl MarkdownTerminalRenderer {
    fn new(terminal_width: usize) -> Self {
        Self {
            terminal_width,
            lines: Vec::new(),
            current: Vec::new(),
            style: Style::default(),
            style_stack: Vec::new(),
            quote_depth: 0,
            list_stack: Vec::new(),
            code_block: None,
            table: None,
        }
    }

    fn render<'a, I>(mut self, parser: I) -> Vec<Line<'static>>
    where
        I: IntoIterator<Item = MarkdownEvent<'a>>,
    {
        for event in parser {
            if self.handle_code_block_event(&event) {
                continue;
            }

            match event {
                MarkdownEvent::Start(tag) => self.start_tag(tag),
                MarkdownEvent::End(tag) => self.end_tag(tag),
                MarkdownEvent::Text(text)
                | MarkdownEvent::Html(text)
                | MarkdownEvent::InlineHtml(text) => self.push_text(text.as_ref()),
                MarkdownEvent::Code(code) => self.push_code_span(code.as_ref()),
                MarkdownEvent::SoftBreak | MarkdownEvent::HardBreak => self.finish_line(),
                MarkdownEvent::Rule => self.push_rule(),
                MarkdownEvent::TaskListMarker(checked) => {
                    self.push_text(if checked { "[x] " } else { "[ ] " });
                }
                _ => {}
            }
        }

        self.finish_line();
        self.finish_code_block();

        wrap_rendered_lines_to_width(self.lines, self.terminal_width)
    }

    fn handle_code_block_event(&mut self, event: &MarkdownEvent<'_>) -> bool {
        if self.code_block.is_none() {
            return false;
        }

        match event {
            MarkdownEvent::End(TagEnd::CodeBlock) => self.finish_code_block(),
            MarkdownEvent::Text(text) | MarkdownEvent::Code(text) => {
                if let Some(block) = &mut self.code_block {
                    block.code.push_str(text.as_ref());
                }
            }
            MarkdownEvent::SoftBreak | MarkdownEvent::HardBreak => {
                if let Some(block) = &mut self.code_block {
                    block.code.push('\n');
                }
            }
            _ => {}
        }

        true
    }

    fn start_tag(&mut self, tag: MarkdownTag<'_>) {
        match tag {
            MarkdownTag::Paragraph => {}
            MarkdownTag::Heading { level, .. } => self.push_style(heading_style(level)),
            MarkdownTag::BlockQuote(_) => {
                self.finish_line();
                self.quote_depth += 1;
            }
            MarkdownTag::CodeBlock(kind) => self.start_code_block(kind),
            MarkdownTag::List(next_number) => self.list_stack.push(ListState { next_number }),
            MarkdownTag::Item => self.start_list_item(),
            MarkdownTag::Emphasis => self.push_modifier(Modifier::ITALIC),
            MarkdownTag::Strong => self.push_modifier(Modifier::BOLD),
            MarkdownTag::Strikethrough => self.push_modifier(Modifier::CROSSED_OUT),
            MarkdownTag::Link { .. } => self.push_style(
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::UNDERLINED),
            ),
            MarkdownTag::Table(alignments) => {
                self.finish_line();
                self.table = Some(TableState {
                    alignments,
                    rows: Vec::new(),
                    current_row: Vec::new(),
                    current_cell: Vec::new(),
                });
            }
            MarkdownTag::TableHead | MarkdownTag::TableRow => {
                if let Some(table) = &mut self.table {
                    table.current_row.clear();
                }
            }
            MarkdownTag::TableCell => {
                if let Some(table) = &mut self.table {
                    table.current_cell.clear();
                }
            }
            _ => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph => self.finish_line(),
            TagEnd::Heading(_) => {
                self.pop_style();
                self.finish_line();
            }
            TagEnd::BlockQuote(_) => {
                self.finish_line();
                self.quote_depth = self.quote_depth.saturating_sub(1);
            }
            TagEnd::CodeBlock => self.finish_code_block(),
            TagEnd::List(_) => {
                self.finish_line();
                self.list_stack.pop();
            }
            TagEnd::Item => self.finish_line(),
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough | TagEnd::Link => {
                self.pop_style();
            }
            TagEnd::Table => self.finish_table(),
            TagEnd::TableHead | TagEnd::TableRow => self.finish_table_row(),
            TagEnd::TableCell => self.finish_table_cell(),
            _ => {}
        }
    }

    fn push_style(&mut self, style: Style) {
        self.style_stack.push(self.style);
        self.style = self.style.patch(style);
    }

    fn push_modifier(&mut self, modifier: Modifier) {
        self.style_stack.push(self.style);
        self.style = self.style.add_modifier(modifier);
    }

    fn pop_style(&mut self) {
        if let Some(previous) = self.style_stack.pop() {
            self.style = previous;
        }
    }

    fn push_text(&mut self, text: &str) {
        self.push_span(normalize_terminal_icons(text), self.style);
    }

    fn push_code_span(&mut self, text: &str) {
        self.push_span(
            text.to_string(),
            Style::default().fg(Color::Yellow).bg(INLINE_CODE_BG),
        );
    }

    fn push_span(&mut self, text: String, style: Style) {
        if text.is_empty() {
            return;
        }

        if let Some(table) = &mut self.table {
            table.current_cell.push(Span::styled(text, style));
            return;
        }

        self.ensure_line_prefix();
        self.current.push(Span::styled(text, style));
    }

    fn ensure_line_prefix(&mut self) {
        if !self.current.is_empty() {
            return;
        }

        for _ in 0..self.quote_depth {
            self.current.push(Span::styled(
                "│ ".to_string(),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }

    fn finish_line(&mut self) {
        if self.current.is_empty() {
            return;
        }
        self.lines
            .push(Line::from(std::mem::take(&mut self.current)));
    }

    fn push_rule(&mut self) {
        self.finish_line();
        self.lines.push(Line::from(Span::styled(
            "────────────────────────",
            Style::default().fg(Color::DarkGray),
        )));
    }

    fn start_list_item(&mut self) {
        let depth = self.list_stack.len().saturating_sub(1);
        let marker = match self.list_stack.last_mut() {
            Some(ListState {
                next_number: Some(next),
            }) => {
                let current = *next;
                *next += 1;
                format!("{}. ", current)
            }
            _ => "• ".to_string(),
        };

        self.ensure_line_prefix();
        if depth > 0 {
            self.current.push(Span::raw("  ".repeat(depth)));
        }
        self.current.push(Span::styled(
            marker,
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
    }

    fn start_code_block(&mut self, kind: CodeBlockKind<'_>) {
        self.finish_line();
        let lang = match kind {
            CodeBlockKind::Fenced(lang) => lang.to_string(),
            CodeBlockKind::Indented => String::new(),
        };
        self.code_block = Some(CodeBlockState {
            lang,
            code: String::new(),
        });
    }

    fn finish_code_block(&mut self) {
        let Some(block) = self.code_block.take() else {
            return;
        };
        self.lines.extend(highlight_code(&block.code, &block.lang));
    }

    fn finish_table_cell(&mut self) {
        if let Some(table) = &mut self.table {
            table
                .current_row
                .push(std::mem::take(&mut table.current_cell));
        }
    }

    fn finish_table_row(&mut self) {
        if let Some(table) = &mut self.table {
            if !table.current_row.is_empty() {
                table.rows.push(std::mem::take(&mut table.current_row));
            }
        }
    }

    fn finish_table(&mut self) {
        let Some(table) = self.table.take() else {
            return;
        };
        self.lines.extend(render_table(&table, self.terminal_width));
    }
}

fn heading_style(level: HeadingLevel) -> Style {
    let style = Style::default().add_modifier(Modifier::BOLD);
    match level {
        HeadingLevel::H1 | HeadingLevel::H2 => style.fg(Color::Cyan),
        _ => style,
    }
}

fn render_table(table: &TableState, terminal_width: usize) -> Vec<Line<'static>> {
    if table.rows.is_empty() {
        return Vec::new();
    }

    let column_count = table.rows.iter().map(Vec::len).max().unwrap_or(0);
    if column_count == 0 {
        return Vec::new();
    }

    let mut widths = vec![0usize; column_count];
    for row in &table.rows {
        for (idx, cell) in row.iter().enumerate() {
            widths[idx] = widths[idx].max(spans_width(cell));
        }
    }
    constrain_table_widths(&mut widths, terminal_width);

    let mut lines = Vec::new();
    lines.push(render_table_border('┌', '┬', '┐', &widths));
    for (row_idx, row) in table.rows.iter().enumerate() {
        lines.extend(render_table_row(
            row,
            &widths,
            &table.alignments,
            row_idx == 0,
        ));
        if row_idx + 1 < table.rows.len() {
            lines.push(render_table_border('├', '┼', '┤', &widths));
        }
    }
    lines.push(render_table_border('└', '┴', '┘', &widths));
    lines
}

fn render_table_row(
    row: &[Vec<Span<'static>>],
    widths: &[usize],
    alignments: &[MarkdownAlignment],
    is_header: bool,
) -> Vec<Line<'static>> {
    let wrapped_cells: Vec<Vec<Vec<Span<'static>>>> = widths
        .iter()
        .enumerate()
        .map(|(idx, width)| {
            let cell = row.get(idx).cloned().unwrap_or_default();
            wrap_spans_to_width(cell, *width)
        })
        .collect();
    let row_height = wrapped_cells.iter().map(Vec::len).max().unwrap_or(1).max(1);
    let mut lines = Vec::new();

    for line_idx in 0..row_height {
        let mut spans = vec![Span::styled(
            "│ ".to_string(),
            Style::default().fg(Color::DarkGray),
        )];

        for (idx, width) in widths.iter().enumerate() {
            if idx > 0 {
                spans.push(Span::styled(
                    " │ ".to_string(),
                    Style::default().fg(Color::DarkGray),
                ));
            }

            let cell_line = wrapped_cells
                .get(idx)
                .and_then(|cell| cell.get(line_idx))
                .cloned()
                .unwrap_or_default();
            let cell_width = spans_width(&cell_line);
            let (left_pad, right_pad) =
                table_padding(*width, cell_width, alignments.get(idx).copied());

            if left_pad > 0 {
                spans.push(Span::raw(" ".repeat(left_pad)));
            }
            for span in cell_line {
                let span = if is_header {
                    Span::styled(
                        span.content.into_owned(),
                        span.style.fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    )
                } else {
                    span
                };
                spans.push(span);
            }
            if right_pad > 0 {
                spans.push(Span::raw(" ".repeat(right_pad)));
            }
        }

        spans.push(Span::styled(
            " │".to_string(),
            Style::default().fg(Color::DarkGray),
        ));
        lines.push(Line::from(spans));
    }

    lines
}

fn table_padding(
    column_width: usize,
    cell_width: usize,
    alignment: Option<MarkdownAlignment>,
) -> (usize, usize) {
    let padding = column_width.saturating_sub(cell_width);
    match alignment.unwrap_or(MarkdownAlignment::None) {
        MarkdownAlignment::Right => (padding, 0),
        MarkdownAlignment::Center => (padding / 2, padding - padding / 2),
        _ => (0, padding),
    }
}

fn render_table_border(left: char, middle: char, right: char, widths: &[usize]) -> Line<'static> {
    let mut text = String::new();
    text.push(left);
    for (idx, width) in widths.iter().enumerate() {
        if idx > 0 {
            text.push(middle);
        }
        text.push_str(&"─".repeat(width.saturating_add(2)));
    }
    text.push(right);
    Line::from(Span::styled(text, Style::default().fg(Color::DarkGray)))
}

fn constrain_table_widths(widths: &mut [usize], terminal_width: usize) {
    let column_count = widths.len();
    if column_count == 0 {
        return;
    }

    let border_width = column_count.saturating_add(1);
    let padding_width = column_count.saturating_mul(2);
    let available = terminal_width.saturating_sub(border_width + padding_width);
    if available == 0 {
        widths.fill(1);
        return;
    }

    let current: usize = widths.iter().sum();
    if current <= available {
        return;
    }

    let max_col_width = (available / column_count).max(4);
    for width in widths.iter_mut() {
        *width = (*width).min(max_col_width).max(1);
    }

    while widths.iter().sum::<usize>() > available {
        let Some((idx, _)) = widths.iter().enumerate().max_by_key(|(_, width)| **width) else {
            break;
        };
        if widths[idx] <= 1 {
            break;
        }
        widths[idx] -= 1;
    }
}

fn wrap_spans_to_width(spans: Vec<Span<'static>>, width: usize) -> Vec<Vec<Span<'static>>> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut current = Vec::new();
    let mut current_width = 0usize;

    for span in spans {
        let style = span.style;
        let content = span.content.into_owned();
        let mut buffer = String::new();

        for ch in content.chars() {
            let ch_width = if ch == '\t' {
                4
            } else {
                ch.width().unwrap_or(0)
            };
            if current_width > 0 && current_width + ch_width > width {
                push_span_buffer(&mut current, style, &mut buffer);
                lines.push(std::mem::take(&mut current));
                current_width = 0;
            }
            buffer.push(ch);
            current_width += ch_width;
        }

        push_span_buffer(&mut current, style, &mut buffer);
    }

    if current.is_empty() {
        lines.push(Vec::new());
    } else {
        lines.push(current);
    }

    lines
}

fn wrap_rendered_lines_to_width(lines: Vec<Line<'static>>, width: usize) -> Vec<Line<'static>> {
    lines
        .into_iter()
        .flat_map(|line| wrap_spans_to_width(line.spans, width))
        .map(Line::from)
        .collect()
}

fn spans_width(spans: &[Span<'_>]) -> usize {
    UnicodeWidthStr::width(
        spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>()
            .as_str(),
    )
}

fn normalize_terminal_icons(text: &str) -> String {
    text.replace("✅", "✓")
        .replace("❌", "✗")
        .replace("🔴", "●")
        .replace("🟡", "●")
        .replace("🔵", "●")
        .replace("📊", "▣")
        .replace("⚠️", "!")
        .replace("⚠", "!")
}

fn highlight_code(code: &str, lang: &str) -> Vec<Line<'static>> {
    use syntect::easy::HighlightLines;
    use syntect::highlighting::ThemeSet;
    use syntect::parsing::SyntaxSet;
    use syntect::util::LinesWithEndings;

    static SYNTAX_SET: LazyLock<SyntaxSet> = LazyLock::new(SyntaxSet::load_defaults_newlines);
    static THEME_SET: LazyLock<ThemeSet> = LazyLock::new(ThemeSet::load_defaults);

    let ps = &*SYNTAX_SET;
    let ts = &*THEME_SET;

    let syntax = ps
        .find_syntax_by_token(lang)
        .or_else(|| ps.find_syntax_by_extension(lang))
        .unwrap_or_else(|| ps.find_syntax_plain_text());

    let mut highlighter = HighlightLines::new(syntax, &ts.themes["base16-ocean.dark"]);

    let mut result = Vec::new();
    for line in LinesWithEndings::from(code) {
        let line = line.trim_end_matches(&['\r', '\n'][..]);
        let highlighted = match highlighter.highlight_line(line, ps) {
            Ok(h) => h,
            Err(_) => {
                result.push(Line::from(line.to_string()));
                continue;
            }
        };

        let mut spans = Vec::new();
        for (style, text) in highlighted {
            let fg = style.foreground;
            let ratatui_style = Style::default().fg(Color::Rgb(fg.r, fg.g, fg.b));
            spans.push(Span::styled(text.to_string(), ratatui_style));
        }
        result.push(Line::from(spans));
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<Vec<_>>()
            .join("")
    }

    fn first_markdown_line(markdown: &str) -> Line<'static> {
        render_markdown_message(markdown, 80)
            .into_iter()
            .next()
            .unwrap_or_else(|| Line::from(""))
    }

    #[test]
    fn prefixed_message_wraps_long_text_to_visual_lines() {
        let lines = render_message_lines("abcdef", Some(("• ", Color::Green)), 5);

        assert_eq!(lines.len(), 2);
        assert_eq!(line_text(&lines[0]), "• abc");
        assert_eq!(line_text(&lines[1]), "  def");
        assert!(lines.iter().all(|line| line.width() <= 5));
    }

    #[test]
    fn prefixed_message_indents_explicit_newlines() {
        let lines = render_message_lines("one\ntwo", Some(("> ", Color::Cyan)), 20);

        assert_eq!(line_text(&lines[0]), "> one");
        assert_eq!(line_text(&lines[1]), "  two");
    }

    #[test]
    fn markdown_inline_markers_are_rendered_not_shown() {
        let line = first_markdown_line("**Bold** `code` [site](https://example.com)");
        let text = line_text(&line);

        assert_eq!(text, "Bold code site");
        assert!(!text.contains("**"));
        assert!(!text.contains('`'));
        assert!(!text.contains("https://"));
    }

    #[test]
    fn markdown_long_paragraph_wraps_inside_width() {
        let lines = render_markdown_message("abcdefghijklmnopqrstuvwxyz", 8);

        assert!(lines.len() > 1);
        assert!(lines.iter().all(|line| line.width() <= 8));
        assert_eq!(
            lines.iter().map(line_text).collect::<Vec<_>>().join(""),
            "abcdefghijklmnopqrstuvwxyz"
        );
    }

    #[test]
    fn inline_code_uses_subtle_light_background() {
        let line = first_markdown_line("Run `cargo test` now");
        let span = line
            .spans
            .iter()
            .find(|span| span.content.as_ref() == "cargo test")
            .expect("inline code span should be rendered");

        assert_eq!(span.style.bg, Some(INLINE_CODE_BG));
        assert_ne!(span.style.bg, Some(Color::Rgb(40, 40, 40)));
    }

    #[test]
    fn markdown_nested_styles_preserve_outer_modifiers() {
        let line = first_markdown_line("**[Bold link](https://example.com)**");
        let span = line
            .spans
            .iter()
            .find(|span| span.content.as_ref() == "Bold link")
            .expect("link span should be rendered");

        assert_eq!(span.style.fg, Some(Color::Blue));
        assert!(span.style.add_modifier.contains(Modifier::BOLD));
        assert!(span.style.add_modifier.contains(Modifier::UNDERLINED));
    }

    #[test]
    fn markdown_inline_markers_are_rendered_inside_cjk_text() {
        let line = first_markdown_line("✅ DeepCode **已具备**的功能");
        let text = line_text(&line);

        assert_eq!(text, "✓ DeepCode 已具备的功能");
        assert!(!text.contains("**"));
        assert!(!text.contains("✅"));
    }

    #[test]
    fn markdown_block_markers_are_rendered_not_shown() {
        assert_eq!(line_text(&first_markdown_line("# Title")), "Title");
        assert_eq!(line_text(&first_markdown_line("- item")), "• item");
        assert_eq!(line_text(&first_markdown_line("> quote")), "│ quote");
    }

    #[test]
    fn markdown_tables_are_rendered_without_pipe_syntax() {
        let lines = render_markdown_message(
            "| 功能 | 实现位置 |\n|------|----------|\n| **多 Provider** | `providers/` |",
            80,
        );
        let text = lines.iter().map(line_text).collect::<Vec<_>>().join("\n");

        assert!(text.contains("┌"));
        assert!(text.contains("│"));
        assert!(text.contains("└"));
        assert!(text.contains("功能"));
        assert!(text.contains("实现位置"));
        assert!(text.contains("多 Provider"));
        assert!(text.contains("providers/"));
        assert!(!text.contains("|------|"));
        assert!(!text.contains("**"));
        assert!(!text.contains('`'));
    }

    #[test]
    fn markdown_tables_wrap_long_cells_inside_borders() {
        let lines = render_markdown_message(
            "| 功能 | 说明 |\n|------|------|\n| 工具系统 | 文件读写编辑 Shell Git Glob Grep 网页搜索 子代理 |",
            32,
        );
        let text = lines.iter().map(line_text).collect::<Vec<_>>().join("\n");

        assert!(text.contains("┌"));
        assert!(text.contains("┬"));
        assert!(text.contains("┼"));
        assert!(text.contains("└"));
        assert!(lines.iter().all(|line| line.width() <= 32));
        assert!(text.lines().filter(|line| line.starts_with('│')).count() > 2);
    }

    #[test]
    fn markdown_unicode_rules_are_rendered_as_rules() {
        let line = first_markdown_line("────────────────────────");
        assert_eq!(line_text(&line), "────────────────────────");
    }

    #[test]
    fn markdown_code_fences_are_hidden_but_code_remains() {
        let lines = render_markdown_message("```rust\nlet x = 1;\n```", 80);
        let text = lines.iter().map(line_text).collect::<Vec<_>>().join("\n");

        assert!(text.contains("let x = 1;"));
        assert!(!text.contains("```"));
    }

    #[test]
    fn markdown_code_fences_do_not_embed_newlines_inside_spans() {
        let lines = render_markdown_message("```rust\nlet x = 1;\nlet y = 2;\n```", 80);

        assert_eq!(lines.len(), 2);
        for line in &lines {
            for span in &line.spans {
                assert!(!span.content.contains('\n'));
                assert!(!span.content.contains('\r'));
            }
        }
    }

    #[test]
    fn markdown_code_fences_wrap_long_lines_inside_width() {
        let lines = render_markdown_message("```text\nabcdefghijklmnopqrstuvwxyz\n```", 10);

        assert!(lines.len() > 1);
        assert!(lines.iter().all(|line| line.width() <= 10));
    }
}
