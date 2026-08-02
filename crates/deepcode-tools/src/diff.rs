use similar::TextDiff;

const CONTEXT_LINES: usize = 3;

pub(crate) fn unified_diff(path: &str, before: &str, after: &str, before_exists: bool) -> String {
    if before == after {
        return String::new();
    }

    let before_header = if before_exists {
        format!("a/{path}")
    } else {
        "/dev/null".to_string()
    };
    let after_header = format!("b/{path}");
    let diff = TextDiff::from_lines(before, after);

    diff.unified_diff()
        .context_radius(CONTEXT_LINES)
        .header(&before_header, &after_header)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unified_diff_for_modified_file() {
        let diff = unified_diff("a.txt", "one\ntwo\nthree\n", "one\nTWO\nthree\n", true);

        assert!(diff.contains("--- a/a.txt"), "{diff}");
        assert!(diff.contains("+++ b/a.txt"), "{diff}");
        assert!(diff.contains("@@"), "{diff}");
        assert!(diff.contains("-two"), "{diff}");
        assert!(diff.contains("+TWO"), "{diff}");
    }

    #[test]
    fn unified_diff_for_new_file() {
        let diff = unified_diff("new.txt", "", "hello\nworld\n", false);

        assert!(diff.starts_with("--- /dev/null\n+++ b/new.txt\n"), "{diff}");
        assert!(diff.contains("@@"), "{diff}");
        assert!(diff.contains("+hello"), "{diff}");
        assert!(diff.contains("+world"), "{diff}");
    }

    #[test]
    fn unified_diff_shows_trailing_newline_changes() {
        let diff = unified_diff("a.txt", "one", "one\n", true);

        assert!(diff.contains("-one"), "{diff}");
        assert!(diff.contains("+one"), "{diff}");
    }

    #[test]
    fn identical_inputs_emit_empty_diff() {
        assert_eq!(unified_diff("same.txt", "same\n", "same\n", true), "");
    }

    #[test]
    fn large_file_local_edit_stays_localized() {
        let before = numbered_lines(1_500);
        let after = before.replace("line 750\n", "line 750 changed\n");
        let diff = unified_diff("large.txt", &before, &after, true);

        assert!(diff.contains("-line 750"), "{diff}");
        assert!(diff.contains("+line 750 changed"), "{diff}");
        assert!(
            !diff.contains("-line 0\n-line 1\n-line 2"),
            "diff should not delete the whole file:\n{diff}"
        );
        assert!(
            !diff.contains("+line 0\n+line 1\n+line 2"),
            "diff should not insert the whole file:\n{diff}"
        );
    }

    #[test]
    fn large_file_multiple_edits_stay_localized() {
        let before = numbered_lines(1_800);
        let after = before
            .replace("line 200\n", "line 200 changed\n")
            .replace("line 1600\n", "line 1600 changed\n");
        let diff = unified_diff("large.txt", &before, &after, true);

        assert!(diff.contains("-line 200"), "{diff}");
        assert!(diff.contains("+line 200 changed"), "{diff}");
        assert!(diff.contains("-line 1600"), "{diff}");
        assert!(diff.contains("+line 1600 changed"), "{diff}");
        assert!(
            !diff.contains("-line 500\n-line 501\n-line 502"),
            "unchanged middle region should not be rewritten:\n{diff}"
        );
    }

    fn numbered_lines(count: usize) -> String {
        (0..count)
            .map(|idx| format!("line {idx}"))
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    }
}
