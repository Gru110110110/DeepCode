use deepcode_tools::tool::Tool;

pub(crate) fn required_tool_input_issues(
    tool: &dyn Tool,
    input: &serde_json::Value,
) -> Vec<String> {
    let schema = tool.input_schema();
    let required = schema
        .get("required")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str);

    let mut issues = Vec::new();
    for name in required {
        let Some(value) = input.get(name) else {
            issues.push(format!("missing required parameter '{}'", name));
            continue;
        };
        if value.is_null() {
            issues.push(format!("missing required parameter '{}'", name));
            continue;
        }

        let expected_type = schema
            .get("properties")
            .and_then(|properties| properties.get(name))
            .and_then(|property| property.get("type"))
            .and_then(serde_json::Value::as_str);

        let type_matches = match expected_type {
            Some("string") => value.is_string(),
            Some("integer") => value.is_i64() || value.is_u64(),
            Some("number") => value.is_number(),
            Some("boolean") => value.is_boolean(),
            Some("object") => value.is_object(),
            Some("array") => value.is_array(),
            _ => true,
        };
        if !type_matches {
            issues.push(format!(
                "parameter '{}' must be a {}",
                name,
                expected_type.unwrap_or("valid value")
            ));
        }
    }

    issues
}

pub(crate) fn invalid_tool_input_message(
    tool_name: &str,
    tool: &dyn Tool,
    input: &serde_json::Value,
) -> Option<String> {
    let issues = required_tool_input_issues(tool, input);
    if issues.is_empty() {
        return None;
    }

    Some(format!(
        "Tool input rejected before execution: {}. Call '{}' again with a JSON object matching its schema.",
        issues.join("; "),
        tool_name
    ))
}
