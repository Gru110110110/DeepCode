use ratatui::prelude::*;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::{StartupHeader, STARTUP_BANNER_WIDTH};

pub(crate) fn startup_header_lines(
    header: &StartupHeader,
    terminal_width: u16,
) -> Vec<Line<'static>> {
    let width = usize::from(terminal_width);
    if width < 16 {
        return vec![Line::from(truncate_display_width("DeepCode", width))];
    }
    let box_inner_width = STARTUP_BANNER_WIDTH.min(width.saturating_sub(2)).max(1);
    let logo_style = Style::default().fg(Color::DarkGray);
    let box_style = Style::default().fg(Color::Gray);
    let accent_style = Style::default()
        .fg(Color::Cyan)
        .add_modifier(Modifier::BOLD);

    let mut lines = Vec::new();
    for line in [
        " ____                  ____          _      ",
        "|  _ \\  ___  ___ _ __ / ___|___   __| | ___ ",
        "| | | |/ _ \\/ _ \\ '_ \\ |   / _ \\ / _` |/ _ \\",
        "| |_| |  __/  __/ |_) | |__| (_) | (_| |  __/",
        "|____/ \\___|\\___| .__/ \\____\\___/ \\__,_|\\___|",
        "                |_|                          ",
    ] {
        lines.push(Line::from(Span::styled(
            truncate_display_width(line, width),
            logo_style,
        )));
    }

    lines.push(Line::from(Span::styled(
        format!("┌{}┐", "─".repeat(box_inner_width)),
        box_style,
    )));
    lines.push(header_box_line(
        ">_ DeepCode (v0.1.0)",
        box_inner_width,
        accent_style,
    ));
    lines.push(header_box_line("", box_inner_width, box_style));
    lines.push(header_box_line(
        &format!(
            "model:     {}   /model to change",
            truncate_display_width(&header.model, 32)
        ),
        box_inner_width,
        box_style,
    ));
    lines.push(header_box_line(
        &format!(
            "directory: {}",
            truncate_display_width(&header.directory, box_inner_width.saturating_sub(11),)
        ),
        box_inner_width,
        box_style,
    ));
    lines.push(Line::from(Span::styled(
        format!("└{}┘", "─".repeat(box_inner_width)),
        box_style,
    )));

    lines
}

fn header_box_line(content: &str, width: usize, style: Style) -> Line<'static> {
    let content_width = width.saturating_sub(1);
    let content = truncate_display_width(content, content_width);
    let padding = content_width.saturating_sub(UnicodeWidthStr::width(content.as_str()));
    Line::from(vec![
        Span::styled("│ ".to_string(), Style::default().fg(Color::Gray)),
        Span::styled(content, style),
        Span::styled(
            format!("{}│", " ".repeat(padding)),
            Style::default().fg(Color::Gray),
        ),
    ])
}

pub(crate) fn truncate_display_width(value: &str, width: usize) -> String {
    if UnicodeWidthStr::width(value) <= width {
        return value.to_string();
    }

    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "...".chars().take(1).collect();
    }

    let mut suffix = Vec::new();
    let mut suffix_width = 0usize;
    for ch in value.chars().rev() {
        let ch_width = ch.width().unwrap_or(0);
        if suffix_width + ch_width > width.saturating_sub(1) {
            break;
        }
        suffix.push(ch);
        suffix_width += ch_width;
    }
    suffix.reverse();
    format!("…{}", suffix.into_iter().collect::<String>())
}

pub(crate) fn truncate_display_width_end(value: &str, width: usize) -> String {
    if UnicodeWidthStr::width(value) <= width {
        return value.to_string();
    }

    if width == 0 {
        return String::new();
    }
    if width == 1 {
        return "…".to_string();
    }

    let prefix = take_width(value, width.saturating_sub(1));
    format!("{}…", prefix)
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

    fn line_text(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<Vec<_>>()
            .join("")
    }

    #[test]
    fn startup_header_contains_banner_model_and_directory() {
        let header = StartupHeader {
            model: "deepseek-test xhigh".to_string(),
            directory: "/tmp/deepcode".to_string(),
        };
        let text = startup_header_lines(&header, 80)
            .iter()
            .map(line_text)
            .collect::<Vec<_>>()
            .join("\n");

        assert!(text.contains(">_ DeepCode (v0.1.0)"));
        assert!(text.contains("model:     deepseek-test xhigh"));
        assert!(text.contains("directory: /tmp/deepcode"));
    }

    #[test]
    fn startup_header_box_respects_terminal_width() {
        let header = StartupHeader {
            model: "model".to_string(),
            directory: "/a/very/long/path/that/needs/truncation".to_string(),
        };
        let lines = startup_header_lines(&header, 40);

        assert!(lines.iter().all(|line| line.width() <= 40));
    }
}
