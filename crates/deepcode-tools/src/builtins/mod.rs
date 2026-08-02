mod edit_file;
mod git_add;
mod git_branch;
mod git_checkout;
mod git_commit;
mod git_common;
mod git_diff;
mod git_log;
mod git_status;
mod glob;
mod grep;
mod read_file;
mod shell;
mod web_fetch;
mod web_search;
mod write_file;

use crate::execution::ToolExecutionConfig;
use crate::registry::ToolRegistry;
use deepcode_core::config::ToolsConfig;
use std::sync::Arc;

/// Register all built-in tools in the given registry.
pub fn register_all(registry: &mut ToolRegistry, tools_config: &ToolsConfig) {
    let execution = ToolExecutionConfig::from_tools_config(tools_config);
    let disabled = &tools_config.disabled;
    let mut register = |tool: Arc<dyn crate::tool::Tool>| {
        if !disabled.iter().any(|name| name == tool.name()) {
            registry.register(tool);
        }
    };

    register(Arc::new(read_file::ReadFileTool::new(execution.clone())));
    register(Arc::new(write_file::WriteFileTool::new(execution.clone())));
    register(Arc::new(edit_file::EditFileTool::new(execution.clone())));
    register(Arc::new(shell::ShellTool::new(execution.clone())));
    register(Arc::new(glob::GlobTool::new(execution.clone())));
    register(Arc::new(grep::GrepTool::new(execution.clone())));
    register(Arc::new(web_fetch::WebFetchTool));
    register(Arc::new(web_search::WebSearchTool));
    register(Arc::new(git_status::GitStatusTool::new(execution.clone())));
    register(Arc::new(git_diff::GitDiffTool::new(execution.clone())));
    register(Arc::new(git_log::GitLogTool::new(execution.clone())));
    register(Arc::new(git_add::GitAddTool::new(execution.clone())));
    register(Arc::new(git_commit::GitCommitTool::new(execution.clone())));
    register(Arc::new(git_checkout::GitCheckoutTool::new(
        execution.clone(),
    )));
    register(Arc::new(git_branch::GitBranchTool::new(execution)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_all_skips_disabled_tools() {
        let mut registry = ToolRegistry::new();
        let config = ToolsConfig {
            disabled: vec!["shell".to_string(), "web_fetch".to_string()],
            max_file_size_bytes: None,
        };

        register_all(&mut registry, &config);

        assert!(!registry.contains("shell"));
        assert!(!registry.contains("web_fetch"));
        assert!(registry.contains("read_file"));
        assert!(registry.contains("git_status"));
    }
}
