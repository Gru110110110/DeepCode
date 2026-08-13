use std::collections::HashMap;
use std::sync::Arc;

use deepcode_core::error::Result;
use deepcode_core::types::ToolDefinition;

use crate::tool::{FileChangePreview, Tool, ToolExecutionContext};

/// A collection of registered tools, indexed by name.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool. Replaces any existing tool with the same name.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        self.tools.insert(name, tool);
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    /// Check if a tool is registered.
    pub fn contains(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// All registered tool names.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.tools.keys().cloned().collect();
        names.sort();
        names
    }

    /// All tool definitions (for sending to an LLM).
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions: Vec<ToolDefinition> = self
            .tools
            .values()
            .map(|tool| tool.to_definition())
            .collect();
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        definitions
    }

    /// Filtered definitions, excluding disabled tools.
    pub fn filtered(&self, exclude: &[String]) -> Vec<ToolDefinition> {
        let mut definitions: Vec<ToolDefinition> = self
            .tools
            .iter()
            .filter(|(name, _)| !exclude.contains(name))
            .map(|(_, t)| t.to_definition())
            .collect();
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        definitions
    }

    /// Tool definitions that are safe to expose during planning.
    pub fn planning_definitions(&self) -> Vec<ToolDefinition> {
        let mut definitions: Vec<ToolDefinition> = self
            .tools
            .values()
            .filter(|tool| {
                let safety = tool.safety();
                safety.is_read_only && !safety.requires_approval && !safety.is_destructive
            })
            .map(|tool| tool.to_definition())
            .collect();
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        definitions
    }

    /// A registry containing only local, approval-free read-only tools.
    /// Orchestration tools are excluded so child agents cannot recursively spawn agents.
    pub fn read_only_subset(&self) -> Self {
        let tools = self
            .tools
            .iter()
            .filter(|(name, tool)| {
                let safety = tool.safety();
                safety.is_read_only
                    && !safety.requires_approval
                    && !safety.is_destructive
                    && !matches!(
                        name.as_str(),
                        "spawn_agent" | "wait_agents" | "cancel_agents"
                    )
            })
            .map(|(name, tool)| (name.clone(), Arc::clone(tool)))
            .collect();
        Self { tools }
    }

    /// Execute a tool by name with JSON input.
    pub async fn execute(&self, name: &str, input: serde_json::Value) -> Result<String> {
        self.execute_with_context(name, input, ToolExecutionContext::default())
            .await
    }

    /// Execute a tool by name with JSON input and runtime execution context.
    pub async fn execute_with_context(
        &self,
        name: &str,
        input: serde_json::Value,
        context: ToolExecutionContext,
    ) -> Result<String> {
        let tool = self.tools.get(name).ok_or_else(|| {
            deepcode_core::error::DeepCodeError::ToolExecution {
                tool: name.to_string(),
                message: "Tool not found".to_string(),
            }
        })?;
        tool.execute_with_context(input, context).await
    }

    /// Prepare a file-content preview for a tool, if the tool supports it.
    pub async fn preview_change(
        &self,
        name: &str,
        input: serde_json::Value,
    ) -> Result<Option<FileChangePreview>> {
        let tool = self.tools.get(name).ok_or_else(|| {
            deepcode_core::error::DeepCodeError::ToolExecution {
                tool: name.to_string(),
                message: "Tool not found".to_string(),
            }
        })?;
        tool.preview_change(input).await
    }

    /// Execute a previously approved file-content preview.
    pub async fn execute_previewed(
        &self,
        name: &str,
        input: serde_json::Value,
        preview: FileChangePreview,
    ) -> Result<String> {
        let tool = self.tools.get(name).ok_or_else(|| {
            deepcode_core::error::DeepCodeError::ToolExecution {
                tool: name.to_string(),
                message: "Tool not found".to_string(),
            }
        })?;
        tool.execute_previewed(input, preview).await
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Clone for ToolRegistry {
    fn clone(&self) -> Self {
        Self {
            tools: self.tools.clone(),
        }
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolSafety;
    use async_trait::async_trait;

    struct NamedTool {
        name: &'static str,
        read_only: bool,
    }

    impl NamedTool {
        fn new(name: &'static str) -> Arc<Self> {
            Arc::new(Self {
                name,
                read_only: true,
            })
        }

        fn mutating(name: &'static str) -> Arc<Self> {
            Arc::new(Self {
                name,
                read_only: false,
            })
        }
    }

    #[async_trait]
    impl Tool for NamedTool {
        fn name(&self) -> &str {
            self.name
        }

        fn description(&self) -> &str {
            "test tool"
        }

        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        fn safety(&self) -> ToolSafety {
            ToolSafety {
                is_read_only: self.read_only,
                is_concurrency_safe: true,
                is_destructive: false,
                requires_approval: false,
            }
        }

        async fn execute(&self, _input: serde_json::Value) -> Result<String> {
            Ok(String::new())
        }
    }

    #[test]
    fn definitions_are_sorted_for_stable_provider_requests() {
        let mut registry = ToolRegistry::new();
        registry.register(NamedTool::new("write_file"));
        registry.register(NamedTool::new("grep"));
        registry.register(NamedTool::new("read_file"));

        let names: Vec<String> = registry
            .definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect();

        assert_eq!(names, vec!["grep", "read_file", "write_file"]);
    }

    #[test]
    fn planning_definitions_are_sorted_after_filtering() {
        let mut registry = ToolRegistry::new();
        registry.register(NamedTool::new("read_file"));
        registry.register(NamedTool::mutating("edit_file"));
        registry.register(NamedTool::new("grep"));

        let names: Vec<String> = registry
            .planning_definitions()
            .into_iter()
            .map(|definition| definition.name)
            .collect();

        assert_eq!(names, vec!["grep", "read_file"]);
    }

    #[test]
    fn read_only_subset_excludes_mutations_and_recursive_orchestration() {
        let mut registry = ToolRegistry::new();
        registry.register(NamedTool::new("read_file"));
        registry.register(NamedTool::mutating("edit_file"));
        registry.register(NamedTool::new("spawn_agent"));
        registry.register(NamedTool::new("wait_agents"));

        let subset = registry.read_only_subset();

        assert_eq!(subset.names(), vec!["read_file"]);
    }
}
