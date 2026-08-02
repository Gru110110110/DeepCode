use std::path::{Path, PathBuf};
use std::time::Duration;

use deepcode_core::config::ToolsConfig;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_sandbox::{execute_prepared, CommandSpec, ExecEnv, SandboxManager, SandboxPolicy};

#[derive(Debug, Clone)]
pub(crate) struct ToolExecutionConfig {
    pub workspace_root: Option<PathBuf>,
    pub max_file_size_bytes: Option<usize>,
    pub shell_timeout: Duration,
    pub max_output_bytes: usize,
}

impl ToolExecutionConfig {
    pub(crate) fn from_tools_config(config: &ToolsConfig) -> Self {
        Self {
            workspace_root: std::env::current_dir()
                .ok()
                .and_then(|path| path.canonicalize().ok()),
            max_file_size_bytes: config.max_file_size_bytes,
            shell_timeout: Duration::from_secs(60),
            max_output_bytes: 200_000,
        }
    }

    pub(crate) fn unrestricted() -> Self {
        Self {
            workspace_root: None,
            max_file_size_bytes: None,
            shell_timeout: Duration::from_secs(60),
            max_output_bytes: 200_000,
        }
    }

    #[cfg(test)]
    pub(crate) fn unrestricted_with_max_size(max_file_size_bytes: Option<usize>) -> Self {
        Self {
            max_file_size_bytes,
            ..Self::unrestricted()
        }
    }

    pub(crate) fn sandbox_manager(&self) -> SandboxManager {
        SandboxManager::new(ExecEnv::from_workspace(self.workspace_root.clone()))
    }

    pub(crate) async fn run_sandboxed_command(
        &self,
        tool: &str,
        program: &str,
        args: Vec<String>,
        cwd: PathBuf,
        policy: SandboxPolicy,
    ) -> Result<std::process::Output> {
        let mut spec = CommandSpec::new(program);
        spec.args = args;
        spec.cwd = Some(cwd);
        spec.policy = policy;
        let prepared = self.sandbox_manager().prepare(spec)?;

        execute_prepared(&prepared, self.shell_timeout)
            .await
            .map_err(|error| DeepCodeError::ToolExecution {
                tool: tool.to_string(),
                message: error.to_string(),
            })
    }

    pub(crate) fn resolve_existing_path(&self, tool: &str, path: &str) -> Result<PathBuf> {
        let candidate = self.candidate_path(path)?;
        let canonical = candidate
            .canonicalize()
            .map_err(|e| DeepCodeError::ToolExecution {
                tool: tool.to_string(),
                message: format!("Cannot access {}: {}", candidate.display(), e),
            })?;
        self.ensure_in_workspace(tool, &canonical)?;
        Ok(canonical)
    }

    pub(crate) fn resolve_writable_path(&self, tool: &str, path: &str) -> Result<PathBuf> {
        let candidate = self.candidate_path(path)?;

        let path_to_check = if candidate.exists() {
            candidate
                .canonicalize()
                .map_err(|e| DeepCodeError::ToolExecution {
                    tool: tool.to_string(),
                    message: format!("Cannot access {}: {}", candidate.display(), e),
                })?
        } else {
            let parent = candidate
                .parent()
                .ok_or_else(|| DeepCodeError::ToolExecution {
                    tool: tool.to_string(),
                    message: format!("Path '{}' has no parent directory", path),
                })?;
            parent
                .canonicalize()
                .map_err(|e| DeepCodeError::ToolExecution {
                    tool: tool.to_string(),
                    message: format!("Cannot access parent {}: {}", parent.display(), e),
                })?
        };

        self.ensure_in_workspace(tool, &path_to_check)?;
        Ok(candidate)
    }

    pub(crate) fn resolve_directory(&self, tool: &str, path: &str) -> Result<PathBuf> {
        let resolved = self.resolve_existing_path(tool, path)?;
        if !resolved.is_dir() {
            return Err(DeepCodeError::ToolExecution {
                tool: tool.to_string(),
                message: format!("{} is not a directory", resolved.display()),
            });
        }
        Ok(resolved)
    }

    pub(crate) fn ensure_existing_file_size(&self, tool: &str, path: &Path) -> Result<()> {
        if let Some(max) = self.max_file_size_bytes {
            let size = path
                .metadata()
                .map_err(|e| DeepCodeError::ToolExecution {
                    tool: tool.to_string(),
                    message: format!("Cannot stat {}: {}", path.display(), e),
                })?
                .len() as usize;
            if size > max {
                return Err(DeepCodeError::ToolExecution {
                    tool: tool.to_string(),
                    message: format!(
                        "{} is {} bytes, exceeding configured limit of {} bytes",
                        path.display(),
                        size,
                        max
                    ),
                });
            }
        }
        Ok(())
    }

    pub(crate) fn ensure_content_size(&self, tool: &str, path: &Path, content: &str) -> Result<()> {
        if let Some(max) = self.max_file_size_bytes {
            if content.len() > max {
                return Err(DeepCodeError::ToolExecution {
                    tool: tool.to_string(),
                    message: format!(
                        "Content for {} is {} bytes, exceeding configured limit of {} bytes",
                        path.display(),
                        content.len(),
                        max
                    ),
                });
            }
        }
        Ok(())
    }

    fn candidate_path(&self, path: &str) -> Result<PathBuf> {
        if path.trim().is_empty() {
            return Err(DeepCodeError::ToolExecution {
                tool: "path".to_string(),
                message: "Path cannot be empty".to_string(),
            });
        }

        let raw = Path::new(path);
        if raw.is_absolute() {
            Ok(raw.to_path_buf())
        } else if let Some(root) = &self.workspace_root {
            Ok(root.join(raw))
        } else {
            Ok(raw.to_path_buf())
        }
    }

    fn ensure_in_workspace(&self, tool: &str, path: &Path) -> Result<()> {
        if let Some(root) = &self.workspace_root {
            if !path.starts_with(root) {
                return Err(DeepCodeError::ToolExecution {
                    tool: tool.to_string(),
                    message: format!(
                        "{} is outside workspace root {}",
                        path.display(),
                        root.display()
                    ),
                });
            }
        }
        Ok(())
    }
}

pub(crate) fn truncate_output(input: &str, max_bytes: usize) -> String {
    if input.len() <= max_bytes {
        return input.to_string();
    }

    let mut end = max_bytes;
    while !input.is_char_boundary(end) {
        end -= 1;
    }
    format!(
        "{}\n\n[truncated, {} -> {} bytes]",
        &input[..end],
        input.len(),
        max_bytes
    )
}
