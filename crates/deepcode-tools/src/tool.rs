use async_trait::async_trait;
use deepcode_core::error::Result;
use deepcode_core::types::ToolDefinition;
use deepcode_sandbox::SandboxPolicy;

/// A prepared file-content change that can be reviewed before writing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileChangePreview {
    pub path: String,
    pub before_exists: bool,
    pub before: String,
    pub after: String,
    pub unified_diff: String,
}

impl FileChangePreview {
    pub fn is_noop(&self) -> bool {
        self.before_exists && self.before == self.after
    }
}

/// Safety metadata for a tool.
#[derive(Debug, Clone)]
pub struct ToolSafety {
    /// Whether this tool only reads data (no side effects).
    pub is_read_only: bool,
    /// Whether this tool is safe to run concurrently with other instances.
    pub is_concurrency_safe: bool,
    /// Whether this tool can permanently destroy data.
    pub is_destructive: bool,
    /// Whether this tool requires user approval before execution.
    pub requires_approval: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ToolExecutionContext {
    pub sandbox_policy: Option<SandboxPolicy>,
}

/// The unified tool interface.
///
/// Every tool (built-in or plugin) implements this trait.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Unique name (e.g., "read_file", "shell").
    fn name(&self) -> &str;

    /// Human-readable description shown to the LLM.
    fn description(&self) -> &str;

    /// JSON Schema describing the tool's input parameters.
    fn input_schema(&self) -> serde_json::Value;

    /// Safety classification for this tool.
    fn safety(&self) -> ToolSafety;

    /// Reject inputs that violate hard tool-local policy before permission prompts.
    fn preflight(&self, _input: &serde_json::Value) -> Result<()> {
        Ok(())
    }

    /// Produce a `ToolDefinition` for sending to an LLM.
    fn to_definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: self.input_schema(),
        }
    }

    /// Execute the tool with the given JSON input.
    /// Returns the result as a human-readable string.
    async fn execute(&self, input: serde_json::Value) -> Result<String>;

    /// Execute the tool with contextual runtime policy.
    async fn execute_with_context(
        &self,
        input: serde_json::Value,
        _context: ToolExecutionContext,
    ) -> Result<String> {
        self.execute(input).await
    }

    /// Prepare a file-content change for user review without applying it.
    async fn preview_change(&self, _input: serde_json::Value) -> Result<Option<FileChangePreview>> {
        Ok(None)
    }

    /// Execute a previously reviewed file-content change.
    async fn execute_previewed(
        &self,
        input: serde_json::Value,
        _preview: FileChangePreview,
    ) -> Result<String> {
        self.execute(input).await
    }
}
