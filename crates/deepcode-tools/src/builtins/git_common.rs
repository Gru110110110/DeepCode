use deepcode_core::error::{DeepCodeError, Result};

pub(super) fn validate_branch_name(tool: &str, branch: &str) -> Result<()> {
    if branch.is_empty() || branch.starts_with('-') {
        return Err(DeepCodeError::ToolExecution {
            tool: tool.to_string(),
            message: "Branch name must be non-empty and must not start with '-'".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_and_option_like_branch_names() {
        assert!(validate_branch_name("git", "").is_err());
        assert!(validate_branch_name("git", "--help").is_err());
        assert!(validate_branch_name("git", "feature/cleanup").is_ok());
    }
}
