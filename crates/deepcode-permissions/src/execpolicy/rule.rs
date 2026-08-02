use std::cmp::Ordering;
use std::fmt;
use std::path::Path;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Decision {
    Allow,
    #[default]
    Prompt,
    Forbidden,
}

impl Decision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allow => "allow",
            Self::Prompt => "prompt",
            Self::Forbidden => "forbidden",
        }
    }
}

impl Ord for Decision {
    fn cmp(&self, other: &Self) -> Ordering {
        (*self as u8).cmp(&(*other as u8))
    }
}

impl PartialOrd for Decision {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl FromStr for Decision {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "allow" | "Allow" => Ok(Self::Allow),
            "prompt" | "Prompt" => Ok(Self::Prompt),
            "forbidden" | "deny" | "denied" | "Forbidden" => Ok(Self::Forbidden),
            _ => anyhow::bail!("unknown decision `{}`", value),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternToken {
    Single(String),
    Alternatives(Vec<String>),
}

impl PatternToken {
    fn matches(&self, token: &str) -> bool {
        match self {
            Self::Single(value) => value == token,
            Self::Alternatives(values) => values.iter().any(|value| value == token),
        }
    }

    pub fn display(&self) -> String {
        match self {
            Self::Single(value) => value.clone(),
            Self::Alternatives(values) => format!("[{}]", values.join("|")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixPattern {
    pub tokens: Vec<PatternToken>,
}

impl PrefixPattern {
    pub fn new(tokens: Vec<PatternToken>) -> anyhow::Result<Self> {
        if tokens.is_empty() {
            anyhow::bail!("prefix_rule pattern cannot be empty");
        }
        Ok(Self { tokens })
    }

    pub fn first_program(&self) -> Vec<String> {
        match &self.tokens[0] {
            PatternToken::Single(value) => vec![program_basename(value)],
            PatternToken::Alternatives(values) => {
                values.iter().map(|v| program_basename(v)).collect()
            }
        }
    }

    pub fn matches_prefix(&self, command: &[String]) -> bool {
        if command.len() < self.tokens.len() {
            return false;
        }
        self.tokens
            .iter()
            .zip(command.iter())
            .all(|(pattern, token)| pattern.matches(token))
    }

    pub fn display(&self) -> String {
        self.tokens
            .iter()
            .map(PatternToken::display)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixRule {
    pub pattern: PrefixPattern,
    pub decision: Decision,
    pub match_examples: Vec<Vec<String>>,
    pub not_match_examples: Vec<Vec<String>>,
    pub justification: Option<String>,
    pub source: Option<String>,
}

impl PrefixRule {
    pub fn matches(&self, command: &[String]) -> bool {
        self.pattern.matches_prefix(command)
    }

    pub fn validate_examples(&self) -> anyhow::Result<()> {
        for example in &self.match_examples {
            if !self.matches(example) {
                anyhow::bail!(
                    "match example `{}` does not match prefix `{}`",
                    example.join(" "),
                    self.pattern.display()
                );
            }
        }
        for example in &self.not_match_examples {
            if self.matches(example) {
                anyhow::bail!(
                    "not_match example `{}` matched prefix `{}`",
                    example.join(" "),
                    self.pattern.display()
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleMatch {
    PrefixRuleMatch {
        matched_prefix: String,
        decision: Decision,
        justification: Option<String>,
        source: Option<String>,
    },
    HeuristicsRuleMatch {
        command: String,
        decision: Decision,
        justification: Option<String>,
    },
}

impl RuleMatch {
    pub fn decision(&self) -> Decision {
        match self {
            Self::PrefixRuleMatch { decision, .. } | Self::HeuristicsRuleMatch { decision, .. } => {
                *decision
            }
        }
    }

    pub fn description(&self) -> String {
        match self {
            Self::PrefixRuleMatch {
                matched_prefix,
                decision,
                source,
                ..
            } => format!(
                "prefix_rule {} => {}{}",
                matched_prefix,
                decision.as_str(),
                source
                    .as_ref()
                    .map(|s| format!(" ({})", s))
                    .unwrap_or_default()
            ),
            Self::HeuristicsRuleMatch {
                command, decision, ..
            } => format!("heuristic `{}` => {}", command, decision.as_str()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Evaluation {
    pub decision: Decision,
    pub matches: Vec<RuleMatch>,
}

impl Evaluation {
    pub fn from_matches(matches: Vec<RuleMatch>) -> Self {
        let decision = matches
            .iter()
            .map(RuleMatch::decision)
            .max()
            .unwrap_or(Decision::Prompt);
        Self { decision, matches }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecPolicyCheckCommand {
    pub tokens: Vec<String>,
    pub original: String,
}

impl ExecPolicyCheckCommand {
    pub fn parse(segment: &str) -> anyhow::Result<Self> {
        let tokens = shlex::split(segment)
            .ok_or_else(|| anyhow::anyhow!("could not parse shell segment `{}`", segment))?;
        Ok(Self {
            tokens,
            original: segment.trim().to_string(),
        })
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Policy {
    pub rules: Vec<PrefixRule>,
}

impl Policy {
    pub fn add_prefix_rule(&mut self, rule: PrefixRule) {
        self.rules.push(rule);
    }

    pub fn merge(&mut self, other: Policy) {
        self.rules.extend(other.rules);
    }

    pub fn check(
        &self,
        command: &ExecPolicyCheckCommand,
        heuristics_fallback: Option<Decision>,
    ) -> Evaluation {
        let mut matches = Vec::new();
        let program = command
            .tokens
            .first()
            .map(|token| program_basename(token))
            .unwrap_or_default();
        for rule in &self.rules {
            if !rule.pattern.first_program().iter().any(|p| p == &program) {
                continue;
            }
            if rule.matches(&command.tokens) {
                matches.push(RuleMatch::PrefixRuleMatch {
                    matched_prefix: rule.pattern.display(),
                    decision: rule.decision,
                    justification: rule.justification.clone(),
                    source: rule.source.clone(),
                });
            }
        }
        if matches.is_empty() {
            if let Some(decision) = heuristics_fallback {
                matches.push(RuleMatch::HeuristicsRuleMatch {
                    command: command.original.clone(),
                    decision,
                    justification: None,
                });
            }
        }
        Evaluation::from_matches(matches)
    }

    pub fn check_multiple(
        &self,
        commands: &[ExecPolicyCheckCommand],
        fallback: impl Fn(&ExecPolicyCheckCommand) -> Decision,
    ) -> Evaluation {
        let mut matches = Vec::new();
        for command in commands {
            matches.extend(self.check(command, Some(fallback(command))).matches);
        }
        Evaluation::from_matches(matches)
    }
}

pub fn command_segments(command: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let command = strip_heredoc_bodies(command);
    let mut chars = command.chars().peekable();
    let mut quote: Option<char> = None;
    while let Some(ch) = chars.next() {
        if let Some(q) = quote {
            current.push(ch);
            if ch == q {
                quote = None;
            }
            continue;
        }

        match ch {
            '\'' | '"' => {
                quote = Some(ch);
                current.push(ch);
            }
            '|' | ';' | '\n' => {
                push_segment(&mut segments, &mut current);
                if ch == '|' && matches!(chars.peek(), Some('|')) {
                    let _ = chars.next();
                }
            }
            '&' => {
                push_segment(&mut segments, &mut current);
                if matches!(chars.peek(), Some('&')) {
                    let _ = chars.next();
                }
            }
            _ => current.push(ch),
        }
    }
    push_segment(&mut segments, &mut current);
    segments
}

fn strip_heredoc_bodies(command: &str) -> String {
    let mut stripped = String::new();
    let mut pending = Vec::<HereDocDelimiter>::new();

    for line in command.split_inclusive('\n') {
        let line_without_newline = line.trim_end_matches('\n').trim_end_matches('\r');
        if let Some(delimiter) = pending.first() {
            let candidate = if delimiter.strip_tabs {
                line_without_newline.trim_start_matches('\t')
            } else {
                line_without_newline
            };
            if candidate == delimiter.value {
                pending.remove(0);
            }
            continue;
        }

        stripped.push_str(line);
        pending.extend(heredoc_delimiters(line_without_newline));
    }

    stripped
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HereDocDelimiter {
    value: String,
    strip_tabs: bool,
}

fn heredoc_delimiters(line: &str) -> Vec<HereDocDelimiter> {
    let mut delimiters = Vec::new();
    let chars: Vec<char> = line.chars().collect();
    let mut index = 0usize;
    let mut quote = None;
    let mut escaped = false;

    while index < chars.len() {
        let ch = chars[index];
        if escaped {
            escaped = false;
            index += 1;
            continue;
        }
        if ch == '\\' && quote != Some('\'') {
            escaped = true;
            index += 1;
            continue;
        }
        if let Some(active) = quote {
            if ch == active {
                quote = None;
            }
            index += 1;
            continue;
        }
        if ch == '\'' || ch == '"' {
            quote = Some(ch);
            index += 1;
            continue;
        }
        if ch != '<' || chars.get(index + 1) != Some(&'<') || chars.get(index + 2) == Some(&'<') {
            index += 1;
            continue;
        }

        index += 2;
        let strip_tabs = if chars.get(index) == Some(&'-') {
            index += 1;
            true
        } else {
            false
        };
        while chars.get(index).is_some_and(|value| value.is_whitespace()) {
            index += 1;
        }
        if let Some(delimiter) = read_heredoc_delimiter(&chars, &mut index) {
            delimiters.push(HereDocDelimiter {
                value: delimiter,
                strip_tabs,
            });
        }
    }

    delimiters
}

fn read_heredoc_delimiter(chars: &[char], index: &mut usize) -> Option<String> {
    let first = *chars.get(*index)?;
    if first == '\'' || first == '"' {
        *index += 1;
        let quote = first;
        let mut value = String::new();
        while let Some(ch) = chars.get(*index).copied() {
            *index += 1;
            if ch == quote {
                return Some(value);
            }
            value.push(ch);
        }
        return None;
    }

    let mut value = String::new();
    while let Some(ch) = chars.get(*index).copied() {
        if ch.is_whitespace() || matches!(ch, ';' | '|' | '&' | '(' | ')' | '<' | '>') {
            break;
        }
        value.push(ch);
        *index += 1;
    }

    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

fn push_segment(segments: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        segments.push(trimmed.to_string());
    }
    current.clear();
}

fn program_basename(value: &str) -> String {
    Path::new(value)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(value)
        .to_string()
}

impl fmt::Display for Decision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strictest_decision_wins() {
        let eval = Evaluation::from_matches(vec![
            RuleMatch::HeuristicsRuleMatch {
                command: "ls".to_string(),
                decision: Decision::Allow,
                justification: None,
            },
            RuleMatch::HeuristicsRuleMatch {
                command: "rm".to_string(),
                decision: Decision::Forbidden,
                justification: None,
            },
        ]);
        assert_eq!(eval.decision, Decision::Forbidden);
    }

    #[test]
    fn prefix_rule_supports_alternatives() {
        let rule = PrefixRule {
            pattern: PrefixPattern::new(vec![
                PatternToken::Alternatives(vec!["git".to_string(), "hub".to_string()]),
                PatternToken::Single("status".to_string()),
            ])
            .unwrap(),
            decision: Decision::Allow,
            match_examples: vec![],
            not_match_examples: vec![],
            justification: None,
            source: None,
        };
        assert!(rule.matches(&["git".to_string(), "status".to_string()]));
        assert!(rule.matches(&["hub".to_string(), "status".to_string()]));
        assert!(!rule.matches(&["git".to_string(), "push".to_string()]));
    }

    #[test]
    fn command_segments_ignore_heredoc_body() {
        let segments = command_segments(
            "cat > index.html <<'HTML'\n<!DOCTYPE html>\n<script>console.log('ok')</script>\nHTML\n",
        );
        assert_eq!(segments, vec!["cat > index.html <<'HTML'"]);
    }

    #[test]
    fn command_segments_keep_commands_around_heredoc() {
        let segments = command_segments("cat <<EOF; rm file\nbody\nEOF\n");
        assert_eq!(segments, vec!["cat <<EOF", "rm file"]);
    }
}
