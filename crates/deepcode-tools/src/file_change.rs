use std::path::Path;

use deepcode_core::error::{DeepCodeError, Result};

use crate::execution::ToolExecutionConfig;
use crate::tool::FileChangePreview;

pub(crate) async fn execute_previewed_file_change(
    tool: &str,
    execution: &ToolExecutionConfig,
    preview: FileChangePreview,
    success_message: String,
) -> Result<String> {
    if preview.is_noop() {
        return Ok(format!("No changes needed for {}", preview.path));
    }

    let current = current_content_for_preview(tool, &preview).await?;
    if current != preview.before {
        return Err(DeepCodeError::ToolExecution {
            tool: tool.to_string(),
            message: format!("{} changed after preview; please retry", preview.path),
        });
    }

    let path = Path::new(&preview.path);
    execution.ensure_content_size(tool, path, &preview.after)?;

    tokio::fs::write(path, &preview.after)
        .await
        .map_err(|e| DeepCodeError::ToolExecution {
            tool: tool.to_string(),
            message: format!("Cannot write {}: {}", preview.path, e),
        })?;

    Ok(success_message)
}

async fn current_content_for_preview(tool: &str, preview: &FileChangePreview) -> Result<String> {
    if preview.before_exists {
        tokio::fs::read_to_string(&preview.path)
            .await
            .map_err(|e| DeepCodeError::ToolExecution {
                tool: tool.to_string(),
                message: format!("Cannot read {}: {}", preview.path, e),
            })
    } else if Path::new(&preview.path).exists() {
        Err(DeepCodeError::ToolExecution {
            tool: tool.to_string(),
            message: format!(
                "{} was created after preview; refusing to overwrite without a new preview",
                preview.path
            ),
        })
    } else {
        Ok(String::new())
    }
}
