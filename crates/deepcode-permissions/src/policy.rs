use std::path::PathBuf;

use deepcode_core::config::PermissionsConfig;
use deepcode_sandbox::SandboxPolicy;
use serde::{Deserialize, Serialize};

use crate::approval_key::ApprovalKey;
use crate::execpolicy::{Decision, RuleMatch};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalScope {
    Once,
    Session,
    Persistent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCategory {
    Safe,
    FileRead,
    FileWrite,
    Shell,
    Network,
    Mcp,
    Agent,
    Unknown,
}

impl ToolCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::FileRead => "file-read",
            Self::FileWrite => "file-write",
            Self::Shell => "shell",
            Self::Network => "network",
            Self::Mcp => "mcp",
            Self::Agent => "agent",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    Routine,
    Elevated,
    Critical,
}

impl RiskLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Routine => "routine",
            Self::Elevated => "elevated",
            Self::Critical => "critical",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionAction {
    Read,
    Write,
    Execute,
    Fetch,
    Search,
    SpawnAgent,
    Other,
}

impl PermissionAction {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Execute => "execute",
            Self::Fetch => "fetch",
            Self::Search => "search",
            Self::SpawnAgent => "spawn-agent",
            Self::Other => "other",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PermissionTarget {
    File(PathBuf),
    Files(Vec<PathBuf>),
    NetworkHost(String),
    ShellCommand(String),
    Tool {
        tool: String,
        action: String,
        target: Option<String>,
    },
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionRequest {
    pub tool_name: String,
    pub category: ToolCategory,
    pub action: PermissionAction,
    pub target: PermissionTarget,
    pub input: serde_json::Value,
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyRuleMatch {
    Shell(RuleMatch),
    Filesystem {
        path: String,
        pattern: String,
        decision: Decision,
        source: Option<String>,
        justification: Option<String>,
    },
    Network {
        host: String,
        pattern: String,
        decision: Decision,
        source: Option<String>,
        justification: Option<String>,
    },
    Tool {
        tool: String,
        action: Option<String>,
        target: Option<String>,
        decision: Decision,
        source: Option<String>,
        justification: Option<String>,
    },
    Heuristic {
        subject: String,
        decision: Decision,
        justification: Option<String>,
    },
    Grant {
        key: String,
        decision: Decision,
        scope: ApprovalScope,
    },
}

impl PolicyRuleMatch {
    pub fn decision(&self) -> Decision {
        match self {
            Self::Shell(rule) => rule.decision(),
            Self::Filesystem { decision, .. }
            | Self::Network { decision, .. }
            | Self::Tool { decision, .. }
            | Self::Heuristic { decision, .. }
            | Self::Grant { decision, .. } => *decision,
        }
    }

    pub fn justification(&self) -> Option<String> {
        match self {
            Self::Shell(RuleMatch::PrefixRuleMatch { justification, .. })
            | Self::Shell(RuleMatch::HeuristicsRuleMatch { justification, .. }) => {
                justification.clone()
            }
            Self::Filesystem { justification, .. }
            | Self::Network { justification, .. }
            | Self::Tool { justification, .. }
            | Self::Heuristic { justification, .. } => justification.clone(),
            Self::Grant { scope, .. } => Some(format!("{:?} approval grant", scope)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecision {
    pub decision: Decision,
    pub matched_rules: Vec<PolicyRuleMatch>,
    pub sandbox_policy: SandboxPolicy,
    pub justification: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionEvaluation {
    pub decision: Decision,
    pub category: ToolCategory,
    pub risk: RiskLevel,
    pub matched_rules: Vec<PolicyRuleMatch>,
    pub approval_key: ApprovalKey,
    pub grouping_key: ApprovalKey,
    pub sandbox_policy: SandboxPolicy,
    pub summary: String,
    pub justification: Option<String>,
}

impl PermissionEvaluation {
    pub fn allow(
        category: ToolCategory,
        risk: RiskLevel,
        approval_key: ApprovalKey,
        grouping_key: ApprovalKey,
        sandbox_policy: SandboxPolicy,
        summary: String,
    ) -> Self {
        Self {
            decision: Decision::Allow,
            category,
            risk,
            matched_rules: Vec::new(),
            approval_key,
            grouping_key,
            sandbox_policy,
            summary,
            justification: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionSystemConfig {
    pub default_permissions: Option<String>,
    #[serde(default)]
    pub permissions: PermissionsConfig,
    #[serde(default)]
    pub policy_files: Vec<PathBuf>,
    pub write_policy_file: Option<PathBuf>,
    #[serde(default)]
    pub grants_file: Option<PathBuf>,
}

impl Default for PermissionSystemConfig {
    fn default() -> Self {
        Self {
            default_permissions: Some(":workspace".to_string()),
            permissions: PermissionsConfig::default(),
            policy_files: default_policy_files(),
            write_policy_file: Some(default_write_policy_file()),
            grants_file: Some(default_grants_file()),
        }
    }
}

pub fn default_policy_files() -> Vec<PathBuf> {
    vec![default_policy_dir().join("default.star")]
}

pub fn default_write_policy_file() -> PathBuf {
    default_policy_dir().join("user.star")
}

pub fn default_grants_file() -> PathBuf {
    default_policy_dir().join("permissions.toml")
}

fn default_policy_dir() -> PathBuf {
    deepcode_core::paths::home_dir()
        .join(".config")
        .join("deepcode")
        .join("policies")
}
