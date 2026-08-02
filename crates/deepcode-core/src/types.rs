use serde::{Deserialize, Serialize};

/// A conversation message, shared across all providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
            id: Some(uuid::Uuid::new_v4().to_string()),
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content,
            id: Some(uuid::Uuid::new_v4().to_string()),
        }
    }

    pub fn tool_result(tool_use_id: &str, content: &str, is_error: bool) -> Self {
        Self {
            role: Role::Tool,
            content: vec![ContentBlock::tool_result(tool_use_id, content, is_error)],
            id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "reasoning")]
    Reasoning {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<serde_json::Value>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
    #[serde(rename = "image")]
    Image {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<ImageDetail>,
    },
    #[serde(rename = "audio")]
    Audio {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
    #[serde(rename = "file")]
    File {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    /// A provider-owned item which must survive history round-trips unchanged.
    #[serde(rename = "provider_item")]
    ProviderItem {
        provider: String,
        value: serde_json::Value,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MediaSource {
    Url { url: String },
    Base64 { media_type: String, data: String },
    FileId { file_id: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImageDetail {
    Auto,
    Low,
    High,
    Original,
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text { text: text.into() }
    }

    pub fn reasoning(text: impl Into<String>) -> Self {
        ContentBlock::Reasoning {
            text: text.into(),
            metadata: None,
        }
    }

    pub fn reasoning_with_metadata(text: impl Into<String>, metadata: serde_json::Value) -> Self {
        ContentBlock::Reasoning {
            text: text.into(),
            metadata: Some(metadata),
        }
    }

    pub fn tool_use(id: &str, name: &str, input: serde_json::Value) -> Self {
        ContentBlock::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input,
        }
    }

    pub fn tool_result(tool_use_id: &str, content: &str, is_error: bool) -> Self {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.to_string(),
            content: content.to_string(),
            is_error,
        }
    }

    pub fn provider_item(provider: impl Into<String>, value: serde_json::Value) -> Self {
        ContentBlock::ProviderItem {
            provider: provider.into(),
            value,
        }
    }

    /// Get text content if this is a Text block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            _ => None,
        }
    }
}

/// A tool definition sent to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Convert an object schema to the strict function-calling subset while
/// preserving optional fields by making them explicitly nullable.
pub fn strict_json_schema(schema: &serde_json::Value) -> serde_json::Value {
    let mut schema = schema.clone();
    make_schema_strict(&mut schema);
    schema
}

fn make_schema_strict(schema: &mut serde_json::Value) {
    if let Some(items) = schema.get_mut("items") {
        make_schema_strict(items);
    }
    for keyword in ["anyOf", "oneOf", "allOf"] {
        if let Some(variants) = schema
            .get_mut(keyword)
            .and_then(serde_json::Value::as_array_mut)
        {
            for variant in variants {
                make_schema_strict(variant);
            }
        }
    }
    let Some(object) = schema.as_object_mut() else {
        return;
    };
    let is_object = object.get("type").and_then(serde_json::Value::as_str) == Some("object")
        || object.contains_key("properties");
    if !is_object {
        return;
    }
    object.insert("additionalProperties".to_string(), false.into());
    let originally_required = object
        .get("required")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .unwrap_or_default();
    let required_names = originally_required
        .iter()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_string)
        .collect::<std::collections::HashSet<_>>();
    let Some(properties) = object
        .get_mut("properties")
        .and_then(serde_json::Value::as_object_mut)
    else {
        return;
    };
    let names = properties.keys().cloned().collect::<Vec<_>>();
    for (name, property) in properties.iter_mut() {
        make_schema_strict(property);
        if !required_names.contains(name) {
            let original = std::mem::take(property);
            *property = serde_json::json!({
                "anyOf": [original, {"type": "null"}],
            });
        }
    }
    object.insert("required".to_string(), serde_json::json!(names));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_schema_closes_objects_and_preserves_optional_fields_as_nullable() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "required_value": {"type": "string"},
                "optional_value": {"type": "integer"}
            },
            "required": ["required_value"]
        });
        let strict = strict_json_schema(&schema);
        assert_eq!(strict["additionalProperties"], false);
        assert_eq!(
            strict["required"],
            serde_json::json!(["optional_value", "required_value"])
        );
        assert_eq!(
            strict["properties"]["optional_value"]["anyOf"][1]["type"],
            "null"
        );
    }
}
