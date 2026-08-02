use ratatui::prelude::*;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const LINE_NUMBER_WIDTH: usize = 4;
const DIFF_INDENT: usize = 2;
const ADDED_BG: Color = Color::Rgb(232, 248, 236);
const DELETED_BG: Color = Color::Rgb(253, 235, 235);
const CODE_FG: Color = Color::Rgb(48, 48, 48);

#[derive(Debug, Clone, PartialEq, Eq)]
struct DiffFileSummary {
    path: String,
    added: usize,
    deleted: usize,
    hunks: usize,
}

pub(crate) fn render_diff_message(content: &str, width: u16) -> Vec<Line<'static>> {
    let width = width.max(1);
    let raw_lines = content.lines().collect::<Vec<_>>();
    let (title, diff_lines) = split_title(&raw_lines);
    let mut lines = Vec::new();

    let summaries = summarize_diff(diff_lines);
    let (operation, target) = diff_title(title, &summaries, diff_lines);
    lines.extend(render_diff_title(&operation, &target, width));
    if !summaries.is_empty() {
        lines.extend(render_diff_summary(&summaries, width));
    }

    lines.extend(render_diff_lines(diff_lines, width));

    if lines.is_empty() {
        lines.push(Line::from(""));
    }

    lines
}

fn diff_title(
    title: Option<&str>,
    summaries: &[DiffFileSummary],
    diff_lines: &[&str],
) -> (String, String) {
    if let Some((operation, path)) = title.and_then(|value| value.split_once(':')) {
        return (operation.trim().to_string(), path.trim().to_string());
    }

    let operation = if diff_lines.contains(&"--- /dev/null") {
        "Create"
    } else {
        "Update"
    };
    let target = if summaries.len() == 1 {
        summaries[0].path.clone()
    } else {
        format!("{} files", summaries.len())
    };
    (operation.to_string(), target)
}

fn render_diff_title(operation: &str, target: &str, width: u16) -> Vec<Line<'static>> {
    let full = format!("● {operation}({target})");
    if full.width() <= usize::from(width) {
        return vec![Line::from(vec![
            Span::styled("● ", Style::default().fg(Color::Green)),
            Span::styled(
                operation.to_string(),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("({target})")),
        ])];
    }

    wrap_with_style(&full, Style::default().add_modifier(Modifier::BOLD), width)
}

fn split_title<'a>(lines: &'a [&'a str]) -> (Option<&'a str>, &'a [&'a str]) {
    match lines.first().copied() {
        Some(first) if first.starts_with("Create:") || first.starts_with("Update:") => {
            (Some(first), &lines[1..])
        }
        _ => (None, lines),
    }
}

fn render_diff_lines(diff_lines: &[&str], width: u16) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut old_line: Option<usize> = None;
    let mut new_line: Option<usize> = None;
    let mut in_hunk = false;

    for raw in diff_lines {
        if raw.starts_with("diff --git ") {
            in_hunk = false;
            continue;
        }

        if !in_hunk && (raw.starts_with("--- ") || raw.starts_with("+++ ")) {
            continue;
        }

        if raw.starts_with("@@") {
            in_hunk = true;
            if let Some((old_start, new_start)) = parse_hunk_header(raw) {
                old_line = Some(old_start);
                new_line = Some(new_start);
            }
            continue;
        }

        if let Some(content) = raw.strip_prefix('+') {
            lines.extend(render_diff_line(
                content,
                width,
                new_line,
                '+',
                Color::Green,
                Style::default().fg(CODE_FG).bg(ADDED_BG),
            ));
            if let Some(line) = new_line.as_mut() {
                *line = line.saturating_add(1);
            }
            continue;
        }

        if let Some(content) = raw.strip_prefix('-') {
            lines.extend(render_diff_line(
                content,
                width,
                old_line,
                '-',
                Color::Red,
                Style::default().fg(CODE_FG).bg(DELETED_BG),
            ));
            if let Some(line) = old_line.as_mut() {
                *line = line.saturating_add(1);
            }
            continue;
        }

        if let Some(content) = raw.strip_prefix(' ') {
            lines.extend(render_diff_line(
                content,
                width,
                new_line.or(old_line),
                ' ',
                Color::DarkGray,
                Style::default().fg(Color::Gray),
            ));
            if let Some(line) = old_line.as_mut() {
                *line = line.saturating_add(1);
            }
            if let Some(line) = new_line.as_mut() {
                *line = line.saturating_add(1);
            }
            continue;
        }

        if raw.starts_with("\\ No newline") {
            lines.extend(render_header_line(
                raw,
                width,
                Style::default()
                    .fg(Color::DarkGray)
                    .add_modifier(Modifier::ITALIC),
            ));
            continue;
        }

        lines.extend(render_header_line(
            raw,
            width,
            Style::default().fg(Color::DarkGray),
        ));
    }

    lines
}

fn summarize_diff(diff_lines: &[&str]) -> Vec<DiffFileSummary> {
    let mut summaries = Vec::new();
    let mut current: Option<DiffFileSummary> = None;

    for raw in diff_lines {
        if raw.starts_with("diff --git ") || raw.starts_with("--- ") {
            if let Some(summary) = current.take().filter(DiffFileSummary::has_changes) {
                summaries.push(summary);
            }
            current = Some(DiffFileSummary {
                path: parse_diff_path(raw).unwrap_or_else(|| "<file>".to_string()),
                added: 0,
                deleted: 0,
                hunks: 0,
            });
            continue;
        }

        if raw.starts_with("+++ ") {
            let path = parse_diff_path(raw).unwrap_or_else(|| "<file>".to_string());
            if path != "/dev/null" {
                if current.is_none() {
                    current = Some(DiffFileSummary {
                        path: path.clone(),
                        added: 0,
                        deleted: 0,
                        hunks: 0,
                    });
                }
                if let Some(summary) = current.as_mut() {
                    summary.path = path;
                }
            }
            continue;
        }

        if raw.starts_with("@@") {
            current
                .get_or_insert_with(|| DiffFileSummary {
                    path: "<file>".to_string(),
                    added: 0,
                    deleted: 0,
                    hunks: 0,
                })
                .hunks += 1;
            continue;
        }

        if raw.starts_with('+') && !raw.starts_with("+++") {
            current
                .get_or_insert_with(|| DiffFileSummary {
                    path: "<file>".to_string(),
                    added: 0,
                    deleted: 0,
                    hunks: 0,
                })
                .added += 1;
        } else if raw.starts_with('-') && !raw.starts_with("---") {
            current
                .get_or_insert_with(|| DiffFileSummary {
                    path: "<file>".to_string(),
                    added: 0,
                    deleted: 0,
                    hunks: 0,
                })
                .deleted += 1;
        }
    }

    if let Some(summary) = current.filter(DiffFileSummary::has_changes) {
        summaries.push(summary);
    }

    summaries
}

impl DiffFileSummary {
    fn has_changes(&self) -> bool {
        self.added > 0 || self.deleted > 0 || self.hunks > 0
    }
}

fn parse_diff_path(line: &str) -> Option<String> {
    if line.starts_with("diff --git ") {
        return line
            .split_whitespace()
            .nth(3)
            .map(|path| path.trim_start_matches("b/").to_string());
    }

    line.strip_prefix("--- ")
        .or_else(|| line.strip_prefix("+++ "))
        .map(|path| {
            path.trim()
                .trim_start_matches("a/")
                .trim_start_matches("b/")
                .to_string()
        })
}

fn render_diff_summary(summaries: &[DiffFileSummary], width: u16) -> Vec<Line<'static>> {
    let files = summaries.len();
    let added: usize = summaries.iter().map(|summary| summary.added).sum();
    let deleted: usize = summaries.iter().map(|summary| summary.deleted).sum();
    let mut parts = Vec::new();
    if added > 0 {
        parts.push(format!(
            "Added {added} line{}",
            if added == 1 { "" } else { "s" }
        ));
    }
    if deleted > 0 {
        parts.push(format!(
            "removed {deleted} line{}",
            if deleted == 1 { "" } else { "s" }
        ));
    }
    if parts.is_empty() {
        parts.push("No changed lines".to_string());
    }
    let file_suffix = if files > 1 {
        format!(" in {files} files")
    } else {
        String::new()
    };
    let summary = format!("  └ {}{file_suffix}", parts.join(", "));
    if summary.width() > usize::from(width) {
        return wrap_with_style(&summary, Style::default().fg(Color::DarkGray), width);
    }

    let base = Style::default().fg(Color::DarkGray);
    let emphasis = Style::default()
        .fg(Color::Gray)
        .add_modifier(Modifier::BOLD);
    let mut spans = vec![Span::styled("  └ ", base)];
    if added > 0 {
        spans.push(Span::styled("Added ", base));
        spans.push(Span::styled(added.to_string(), emphasis));
        spans.push(Span::styled(
            if added == 1 { " line" } else { " lines" },
            base,
        ));
    }
    if added > 0 && deleted > 0 {
        spans.push(Span::styled(", ", base));
    }
    if deleted > 0 {
        spans.push(Span::styled("removed ", base));
        spans.push(Span::styled(deleted.to_string(), emphasis));
        spans.push(Span::styled(
            if deleted == 1 { " line" } else { " lines" },
            base,
        ));
    }
    if added == 0 && deleted == 0 {
        spans.push(Span::styled("No changed lines", base));
    }
    if !file_suffix.is_empty() {
        spans.push(Span::styled(file_suffix, base));
    }
    vec![Line::from(spans)]
}

fn parse_hunk_header(line: &str) -> Option<(usize, usize)> {
    let mut parts = line.split_whitespace();
    let _at = parts.next()?;
    let old = parts.next()?.trim_start_matches('-');
    let new = parts.next()?.trim_start_matches('+');
    let old_start = old.split(',').next()?.parse::<usize>().ok()?;
    let new_start = new.split(',').next()?.parse::<usize>().ok()?;
    Some((old_start, new_start))
}

fn render_header_line(line: &str, width: u16, style: Style) -> Vec<Line<'static>> {
    wrap_with_style(line, style, width)
}

fn render_diff_line(
    content: &str,
    width: u16,
    line_number: Option<usize>,
    marker: char,
    gutter_fg: Color,
    content_style: Style,
) -> Vec<Line<'static>> {
    let prefix = format_line_number(width, line_number, marker);
    let gutter_width = UnicodeWidthStr::width(prefix.as_str());
    let prefix_width = DIFF_INDENT + gutter_width;
    let available = usize::from(width).saturating_sub(prefix_width).max(1);
    let wrapped = wrap_preserving_chars(content, available);
    let mut out = Vec::new();

    for (idx, chunk) in wrapped.into_iter().enumerate() {
        let gutter = if idx == 0 {
            prefix.clone()
        } else {
            " ".repeat(gutter_width)
        };
        let chunk_width = UnicodeWidthStr::width(chunk.as_str());
        let mut gutter_style = Style::default().fg(gutter_fg);
        if let Some(background) = content_style.bg {
            gutter_style = gutter_style.bg(background);
        }
        let mut spans = vec![Span::raw(" ".repeat(DIFF_INDENT))];
        spans.push(Span::styled(gutter, gutter_style));
        spans.push(Span::styled(chunk, content_style));
        if content_style.bg.is_some() && chunk_width < available {
            spans.push(Span::styled(
                " ".repeat(available - chunk_width),
                content_style,
            ));
        }
        out.push(Line::from(spans));
    }

    out
}

fn format_line_number(width: u16, line_number: Option<usize>, marker: char) -> String {
    let gutter_width = LINE_NUMBER_WIDTH + 4;
    if usize::from(width) < gutter_width + 8 {
        return marker.to_string();
    }

    let line_number = line_number
        .map(|line| format!("{line:>LINE_NUMBER_WIDTH$}"))
        .unwrap_or_else(|| " ".repeat(LINE_NUMBER_WIDTH));
    format!("{line_number} {marker}")
}

fn wrap_with_style(text: &str, style: Style, width: u16) -> Vec<Line<'static>> {
    wrap_preserving_chars(text, usize::from(width.max(1)))
        .into_iter()
        .map(|part| Line::from(Span::styled(part, style)))
        .collect()
}

fn wrap_preserving_chars(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    if text.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;

    for ch in text.chars() {
        let ch_width = char_width(ch);
        if current_width > 0 && current_width + ch_width > width {
            lines.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(ch);
        current_width += ch_width;
    }

    if !current.is_empty() {
        lines.push(current);
    }

    lines
}

fn char_width(ch: char) -> usize {
    if ch == '\t' {
        4
    } else {
        ch.width().unwrap_or(1)
    }
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

    #[test]
    fn diff_prepends_compact_file_summary() {
        let lines = render_diff_message(
            "Update: a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-old\n+new",
            80,
        );
        let text = lines.iter().map(line_text).collect::<Vec<_>>().join("\n");

        assert!(text.starts_with("● Update(a.txt)"));
        assert!(text.contains("└ Added 1 line, removed 1 line"));
        assert!(!text.contains("@@"));
    }

    #[test]
    fn diff_lines_include_line_number_gutter_and_backgrounds() {
        let lines = render_diff_message(
            "Update: a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-old\n+new",
            80,
        );

        let deleted = lines
            .iter()
            .find(|line| line_text(line).contains("-old"))
            .expect("deleted line should render");
        let added = lines
            .iter()
            .find(|line| line_text(line).contains("+new"))
            .expect("added line should render");

        assert!(line_text(deleted).contains("   1 -old"));
        assert!(line_text(added).contains("   1 +new"));
        assert_eq!(deleted.spans[0].style.bg, None);
        assert_eq!(added.spans[0].style.bg, None);
        assert_eq!(deleted.spans[1].style.bg, Some(DELETED_BG));
        assert_eq!(added.spans[1].style.bg, Some(ADDED_BG));
        assert_eq!(deleted.spans[2].style.bg, Some(DELETED_BG));
        assert_eq!(added.spans[2].style.bg, Some(ADDED_BG));
    }

    #[test]
    fn diff_wraps_long_lines_inside_width() {
        let long = "a".repeat(60);
        let content =
            format!("Create: a.txt\n--- /dev/null\n+++ b/a.txt\n@@ -0,0 +1,1 @@\n+{long}");
        let lines = render_diff_message(&content, 24);
        let added_lines = lines
            .iter()
            .filter(|line| {
                line.spans
                    .get(1)
                    .is_some_and(|span| span.style.bg == Some(ADDED_BG))
            })
            .collect::<Vec<_>>();

        assert!(added_lines.len() > 1, "long diff line should wrap");
        for line in added_lines {
            assert!(
                line.width() <= 24,
                "wrapped line exceeds width: {:?}",
                line_text(line)
            );
        }
    }

    #[test]
    fn diff_preserves_added_line_indentation() {
        let lines = render_diff_message(
            "Create: a.rs\n--- /dev/null\n+++ b/a.rs\n@@ -0,0 +1,1 @@\n+    let y = 2;",
            80,
        );
        let added = lines
            .iter()
            .find(|line| line_text(line).contains("+    let y = 2;"))
            .expect("added indented line should render");

        assert_eq!(added.spans[2].content.as_ref(), "    let y = 2;");
    }

    #[test]
    fn raw_diff_without_title_still_renders_summary() {
        let lines =
            render_diff_message("--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n-old\n+new", 80);
        let text = lines.iter().map(line_text).collect::<Vec<_>>().join("\n");

        assert!(text.starts_with("● Update(a.txt)"));
        assert!(text.contains("└ Added 1 line, removed 1 line"));
        assert!(!text.contains("--- a/a.txt"));
        assert!(!text.contains("+++ b/a.txt"));
    }

    #[test]
    fn diff_keeps_code_lines_that_resemble_file_headers() {
        let lines = render_diff_message(
            "Update: a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1,1 +1,1 @@\n--- old heading\n+++ new heading",
            80,
        );
        let text = lines.iter().map(line_text).collect::<Vec<_>>().join("\n");

        assert!(text.contains("--- old heading"));
        assert!(text.contains("+++ new heading"));
    }
}
