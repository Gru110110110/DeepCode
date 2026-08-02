use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::network_policy::host_from_tool_input;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ApprovalKey(pub String);

impl ApprovalKey {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub fn build_approval_key(tool_name: &str, input: &serde_json::Value) -> ApprovalKey {
    if tool_name == "shell" {
        return ApprovalKey(format!("shell:exact:{}", hash_json(input)));
    }
    if matches!(tool_name, "web_fetch" | "fetch_url" | "web_search") {
        if let Some(host) = host_from_tool_input(tool_name, input) {
            return ApprovalKey(format!("network:{}", host));
        }
    }
    if is_file_tool(tool_name) {
        return ApprovalKey(format!("file:{}:{}", tool_name, hash_json(input)));
    }
    ApprovalKey(format!("tool:{}:{}", tool_name, hash_json(input)))
}

pub fn build_grouping_key(tool_name: &str, input: &serde_json::Value) -> ApprovalKey {
    if tool_name == "shell" {
        return ApprovalKey(format!("shell:group:{}", hash_json(input)));
    }
    if matches!(tool_name, "web_fetch" | "fetch_url" | "web_search") {
        if let Some(host) = host_from_tool_input(tool_name, input) {
            return ApprovalKey(format!("network:{}", host));
        }
    }
    if is_file_tool(tool_name) {
        return ApprovalKey(format!("file:{}:{}", tool_name, file_paths_key(input)));
    }
    build_approval_key(tool_name, input)
}

pub fn hash_json(value: &serde_json::Value) -> String {
    let mut hasher = Sha256::new();
    canonical_json(value, &mut hasher);
    let digest = hasher.finalize();
    digest[..8]
        .iter()
        .map(|byte| format!("{:02x}", byte))
        .collect()
}

fn canonical_json(value: &serde_json::Value, out: &mut Sha256) {
    match value {
        serde_json::Value::Null => out.update(b"null"),
        serde_json::Value::Bool(value) => {
            out.update(if *value { &b"true"[..] } else { &b"false"[..] })
        }
        serde_json::Value::Number(value) => out.update(value.to_string().as_bytes()),
        serde_json::Value::String(value) => {
            out.update(b"\"");
            out.update(escape_json_string(value).as_bytes());
            out.update(b"\"");
        }
        serde_json::Value::Array(values) => {
            out.update(b"[");
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    out.update(b",");
                }
                canonical_json(value, out);
            }
            out.update(b"]");
        }
        serde_json::Value::Object(map) => {
            out.update(b"{");
            let sorted: BTreeMap<_, _> = map.iter().collect();
            for (index, (key, value)) in sorted.iter().enumerate() {
                if index > 0 {
                    out.update(b",");
                }
                out.update(b"\"");
                out.update(escape_json_string(key).as_bytes());
                out.update(b"\":");
                canonical_json(value, out);
            }
            out.update(b"}");
        }
    }
}

fn escape_json_string(value: &str) -> String {
    let mut escaped = String::new();
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            ch if ch <= '\u{1f}' => escaped.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => escaped.push(ch),
        }
    }
    escaped
}

fn is_file_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "write_file" | "edit_file" | "read_file" | "grep" | "glob"
    )
}

fn file_paths_key(input: &serde_json::Value) -> String {
    let mut paths = Vec::new();
    collect_path_values(input, &mut paths);
    paths.sort();
    paths.dedup();
    let joined = paths.join("\n");
    let mut hasher = Sha256::new();
    hasher.update(joined.as_bytes());
    let digest = hasher.finalize();
    digest[..8]
        .iter()
        .map(|byte| format!("{:02x}", byte))
        .collect()
}

fn collect_path_values(value: &serde_json::Value, paths: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                if matches!(key.as_str(), "path" | "directory" | "working_dir") {
                    if let Some(path) = value.as_str() {
                        paths.push(path.to_string());
                    }
                } else {
                    collect_path_values(value, paths);
                }
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_path_values(value, paths);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_hash_ignores_object_key_order() {
        let a = serde_json::json!({"b": 2, "a": [true, null]});
        let b = serde_json::json!({"a": [true, null], "b": 2});
        assert_eq!(hash_json(&a), hash_json(&b));
    }

    #[test]
    fn shell_exact_and_group_keys_differ() {
        let input = serde_json::json!({"command": "cargo test -p deepcode-cli"});
        assert_ne!(
            build_approval_key("shell", &input),
            build_grouping_key("shell", &input)
        );
    }
}
