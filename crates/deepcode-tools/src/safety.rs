use crate::tool::ToolSafety;

impl ToolSafety {
    /// Read-only, concurrency-safe. No approval needed.
    pub const READ_ONLY: Self = Self {
        is_read_only: true,
        is_concurrency_safe: true,
        is_destructive: false,
        requires_approval: false,
    };

    /// Safe mutation (e.g., writing to a new file). Needs approval.
    pub const SAFE_MUTATION: Self = Self {
        is_read_only: false,
        is_concurrency_safe: true,
        is_destructive: false,
        requires_approval: true,
    };

    /// Destructive operation (e.g., shell commands). Needs approval.
    pub const DESTRUCTIVE: Self = Self {
        is_read_only: false,
        is_concurrency_safe: false,
        is_destructive: true,
        requires_approval: true,
    };

    /// Network access (e.g., web fetch). Needs approval.
    pub const NETWORK: Self = Self {
        is_read_only: true,
        is_concurrency_safe: true,
        is_destructive: false,
        requires_approval: true,
    };
}
