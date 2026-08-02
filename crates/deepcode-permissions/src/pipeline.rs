use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use deepcode_core::config::{
    FilesystemAccess, FilesystemOperationRule, FilesystemRuleValue, PermissionDecision,
    PermissionProfileConfig, ProfileNetworkConfig, ShellPatternTokenConfig, ShellRuleConfig,
    ToolRuleConfig,
};
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_sandbox::{contains_dynamic_shell_syntax, FilesystemSandboxRule, SandboxPolicy};
use deepcode_tools::tool::{Tool, ToolSafety};
use glob::Pattern;

use crate::approval_key::{build_approval_key, build_grouping_key, ApprovalKey};
use crate::execpolicy::{
    command_segments, Decision, ExecPolicyCheckCommand, PatternToken, Policy, PolicyParser,
    PrefixPattern, PrefixRule,
};
use crate::network_policy::{
    host_from_tool_input, host_from_url, host_matches, is_private_or_metadata_host,
};
use crate::policy::{
    ApprovalScope, PermissionAction, PermissionEvaluation, PermissionRequest,
    PermissionSystemConfig, PermissionTarget, PolicyDecision, PolicyRuleMatch, RiskLevel,
    ToolCategory,
};

pub struct PermissionSystem {
    engine: PolicyEngine,
    config: PermissionSystemConfig,
    session_allow: HashSet<ApprovalKey>,
    session_deny_exact: HashSet<ApprovalKey>,
}

impl PermissionSystem {
    pub fn new(config: PermissionSystemConfig) -> Self {
        let engine = PolicyEngine::from_config(&config).unwrap_or_else(|err| {
            tracing::warn!(error = %err, "Failed to build permission policy; falling back to :workspace");
            PolicyEngine::workspace()
        });
        let session_allow = load_persistent_grants(config.grants_file.as_ref());
        Self {
            engine,
            config,
            session_allow,
            session_deny_exact: HashSet::new(),
        }
    }

    pub async fn check(
        &mut self,
        tool: &dyn Tool,
        input: &serde_json::Value,
    ) -> Result<PermissionEvaluation> {
        let tool_name = tool.name();
        let approval_key = build_approval_key(tool_name, input);
        let grouping_key = build_grouping_key(tool_name, input);
        let request = build_request(tool, input)?;

        if let Some((reason, risk)) = hard_forbidden(tool_name, input) {
            return Ok(PermissionEvaluation {
                decision: Decision::Forbidden,
                category: request.category,
                risk,
                matched_rules: vec![PolicyRuleMatch::Heuristic {
                    subject: request.summary.clone(),
                    decision: Decision::Forbidden,
                    justification: Some(reason.clone()),
                }],
                approval_key,
                grouping_key,
                sandbox_policy: self.engine.sandbox_policy_for(&request),
                summary: request.summary,
                justification: Some(reason),
            });
        }

        if self.session_deny_exact.contains(&approval_key) {
            return Ok(PermissionEvaluation {
                decision: Decision::Forbidden,
                category: request.category,
                risk: RiskLevel::Elevated,
                matched_rules: vec![PolicyRuleMatch::Grant {
                    key: approval_key.0.clone(),
                    decision: Decision::Forbidden,
                    scope: ApprovalScope::Once,
                }],
                approval_key,
                grouping_key,
                sandbox_policy: self.engine.sandbox_policy_for(&request),
                summary: request.summary,
                justification: Some("Previously denied exact request in this session".to_string()),
            });
        }

        let mut decision = self.engine.evaluate(&request)?;
        let risk = risk_for(tool_name, input, &tool.safety(), request.category);

        if matches!(decision.decision, Decision::Forbidden) {
            return Ok(PermissionEvaluation {
                decision: Decision::Forbidden,
                category: request.category,
                risk,
                matched_rules: decision.matched_rules,
                approval_key,
                grouping_key,
                sandbox_policy: decision.sandbox_policy,
                summary: request.summary,
                justification: decision.justification,
            });
        }

        if self.session_allow.contains(&grouping_key) {
            decision.decision = Decision::Allow;
            decision.matched_rules = vec![PolicyRuleMatch::Grant {
                key: grouping_key.0.clone(),
                decision: Decision::Allow,
                scope: ApprovalScope::Session,
            }];
            decision.justification = Some("Previously allowed grouping".to_string());
        }

        Ok(PermissionEvaluation {
            decision: decision.decision,
            category: request.category,
            risk,
            matched_rules: decision.matched_rules,
            approval_key,
            grouping_key,
            sandbox_policy: decision.sandbox_policy,
            summary: request.summary,
            justification: decision.justification,
        })
    }

    pub fn handle_response(
        &mut self,
        tool_name: &str,
        input: &serde_json::Value,
        approved: bool,
        scope: ApprovalScope,
    ) -> Result<()> {
        let approval_key = build_approval_key(tool_name, input);
        let grouping_key = build_grouping_key(tool_name, input);

        if !approved {
            self.session_deny_exact.insert(approval_key);
            return Ok(());
        }

        match scope {
            ApprovalScope::Once => {}
            ApprovalScope::Session => {
                self.session_allow.insert(grouping_key);
            }
            ApprovalScope::Persistent => {
                self.session_allow.insert(grouping_key.clone());
                append_persistent_grant(
                    self.config.grants_file.as_ref(),
                    tool_name,
                    input,
                    &grouping_key,
                )?;
            }
        }
        Ok(())
    }

    pub fn snapshot_lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!("Permissions profile: {}", self.engine.profile_name),
            format!("Sandbox: {}", self.engine.base_sandbox_label()),
            format!(
                "Workspace roots: {}",
                display_paths(&self.engine.workspace_roots)
            ),
            format!("Filesystem rules: {}", self.engine.filesystem_rules.len()),
            format!("Network: {}", self.engine.network.summary()),
            format!("Shell rules: {}", self.engine.shell_policy.rules.len()),
            format!("Tool rules: {}", self.engine.tool_rules.len()),
            format!("Session grants: {}", self.session_allow.len()),
            format!(
                "Persistent grants file: {}",
                self.config
                    .grants_file
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| "<disabled>".to_string())
            ),
        ];
        if !self.engine.sources.is_empty() {
            lines.push(format!(
                "Policy sources: {}",
                self.engine.sources.join(", ")
            ));
        }
        lines
    }
}

#[derive(Debug, Clone)]
struct PolicyEngine {
    profile_name: String,
    danger_full_access: bool,
    workspace_roots: Vec<PathBuf>,
    filesystem_rules: Vec<FilesystemRule>,
    network: NetworkRules,
    shell_policy: Policy,
    shell_default: Decision,
    tool_rules: Vec<ToolRule>,
    sources: Vec<String>,
}

impl PolicyEngine {
    fn from_config(config: &PermissionSystemConfig) -> anyhow::Result<Self> {
        let active = config
            .default_permissions
            .as_deref()
            .unwrap_or(":workspace")
            .to_string();
        let mut stack = Vec::new();
        let mut engine = Self::resolve_profile(&active, config, &mut stack)?;
        let bundle = PolicyParser::parse_files_with_metadata(&config.policy_files)?;
        engine.merge_policy_bundle(bundle)?;
        Ok(engine)
    }

    fn workspace() -> Self {
        Self::builtin(":workspace").expect("built-in workspace policy must exist")
    }

    fn resolve_profile(
        name: &str,
        config: &PermissionSystemConfig,
        stack: &mut Vec<String>,
    ) -> anyhow::Result<Self> {
        if stack.iter().any(|item| item == name) {
            anyhow::bail!(
                "permission profile inheritance cycle: {}",
                stack.join(" -> ")
            );
        }
        if let Some(builtin) = Self::builtin(name) {
            return Ok(builtin);
        }
        let profile = config
            .permissions
            .profiles
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("unknown permission profile `{}`", name))?;
        stack.push(name.to_string());
        let mut engine = if let Some(parent) = profile.extends.as_deref() {
            Self::resolve_profile(parent, config, stack)?
        } else {
            Self::empty(name)
        };
        let _ = stack.pop();
        engine.profile_name = name.to_string();
        engine.apply_profile(name, profile)?;
        Ok(engine)
    }

    fn empty(name: &str) -> Self {
        Self {
            profile_name: name.to_string(),
            danger_full_access: false,
            workspace_roots: vec![current_dir()],
            filesystem_rules: Vec::new(),
            network: NetworkRules::default(),
            shell_policy: Policy::default(),
            shell_default: Decision::Prompt,
            tool_rules: Vec::new(),
            sources: Vec::new(),
        }
    }

    fn builtin(name: &str) -> Option<Self> {
        let mut policy = Self::empty(name);
        policy.sources.push(format!("builtin {}", name));
        match name {
            ":read-only" => {
                policy.add_builtin_fs_special(":minimal", FilesystemAccess::Read);
                policy.add_builtin_fs_special(":workspace_roots", FilesystemAccess::Read);
                policy.add_sensitive_default_rules();
                Some(policy)
            }
            ":workspace" => {
                policy.add_builtin_fs_special(":minimal", FilesystemAccess::Read);
                policy.add_builtin_fs_special(":tmpdir", FilesystemAccess::Write);
                policy.add_builtin_fs_special(":slash_tmp", FilesystemAccess::Write);
                policy.add_builtin_fs_special(":workspace_roots", FilesystemAccess::Write);
                policy.add_sensitive_default_rules();
                Some(policy)
            }
            ":danger-full-access" => {
                policy.danger_full_access = true;
                policy.shell_default = Decision::Allow;
                policy.network.enabled = true;
                policy.network.default = Decision::Allow;
                policy.filesystem_rules.push(FilesystemRule::special(
                    RuleScope::Root,
                    ".".to_string(),
                    FsRuleDecision::from_access(FilesystemAccess::Write),
                    "builtin :danger-full-access",
                ));
                Some(policy)
            }
            _ => None,
        }
    }

    fn add_builtin_fs_special(&mut self, key: &str, access: FilesystemAccess) {
        let scope = match key {
            ":minimal" => RuleScope::Minimal,
            ":tmpdir" => RuleScope::TmpDir,
            ":slash_tmp" => RuleScope::SlashTmp,
            ":workspace_roots" => RuleScope::WorkspaceRoots,
            _ => RuleScope::Root,
        };
        self.filesystem_rules.push(FilesystemRule::special(
            scope,
            ".".to_string(),
            FsRuleDecision::from_access(access),
            "builtin",
        ));
    }

    fn add_sensitive_default_rules(&mut self) {
        for pattern in [
            ".env*", "**/.env*", "*.pem", "**/*.pem", "*.key", "**/*.key",
        ] {
            self.filesystem_rules.push(FilesystemRule::special(
                RuleScope::WorkspaceRoots,
                pattern.to_string(),
                FsRuleDecision::from_access(FilesystemAccess::Deny),
                "builtin sensitive paths",
            ));
        }
        for pattern in [
            "Cargo.toml",
            "Cargo.lock",
            "package.json",
            "package-lock.json",
            "pnpm-lock.yaml",
            "yarn.lock",
        ] {
            self.filesystem_rules.push(FilesystemRule::special(
                RuleScope::WorkspaceRoots,
                pattern.to_string(),
                FsRuleDecision {
                    read: Decision::Allow,
                    write: Decision::Prompt,
                },
                "builtin manifest prompts",
            ));
        }
        for pattern in ["~/.ssh/**", "~/.aws/**", "/etc/**"] {
            self.filesystem_rules.push(FilesystemRule::absolute(
                pattern.to_string(),
                FsRuleDecision::from_access(FilesystemAccess::Deny),
                "builtin sensitive paths",
            ));
        }
    }

    fn apply_profile(
        &mut self,
        name: &str,
        profile: &PermissionProfileConfig,
    ) -> anyhow::Result<()> {
        self.sources.push(format!("config profile {}", name));
        for (path, enabled) in &profile.workspace_roots {
            if *enabled {
                self.workspace_roots.push(expand_home(path));
            }
        }
        self.workspace_roots.push(current_dir());
        self.workspace_roots = normalize_roots(&self.workspace_roots);
        self.add_filesystem_entries(
            &profile.filesystem.entries,
            RuleScope::Absolute,
            &format!("permissions.{}.filesystem", name),
        )?;
        self.network.apply_config(&profile.network, name);
        self.shell_default = decision_from_config(profile.shell.default);
        for rule in &profile.shell.rules {
            self.shell_policy
                .add_prefix_rule(prefix_rule_from_config(rule, name)?);
        }
        for rule in &profile.tool.rules {
            self.tool_rules.push(ToolRule::from_config(rule, name));
        }
        Ok(())
    }

    fn add_filesystem_entries(
        &mut self,
        entries: &BTreeMap<String, FilesystemRuleValue>,
        inherited_scope: RuleScope,
        source: &str,
    ) -> anyhow::Result<()> {
        for (key, value) in entries {
            self.add_filesystem_entry(key, value, inherited_scope.clone(), source)?;
        }
        Ok(())
    }

    fn add_filesystem_entry(
        &mut self,
        key: &str,
        value: &FilesystemRuleValue,
        inherited_scope: RuleScope,
        source: &str,
    ) -> anyhow::Result<()> {
        match value {
            FilesystemRuleValue::Access(access) => {
                let (scope, pattern) = scope_and_pattern(key, inherited_scope);
                self.filesystem_rules.push(FilesystemRule::new(
                    scope,
                    pattern,
                    FsRuleDecision::from_access(*access),
                    source.to_string(),
                ));
            }
            FilesystemRuleValue::Operations(rule) => {
                let (scope, pattern) = scope_and_pattern(key, inherited_scope);
                self.filesystem_rules.push(FilesystemRule::new(
                    scope,
                    pattern,
                    FsRuleDecision::from_operation(rule),
                    source.to_string(),
                ));
            }
            FilesystemRuleValue::Children(children) => {
                let child_scope = special_scope(key).unwrap_or(inherited_scope);
                self.add_filesystem_entries(children, child_scope, source)?;
            }
        }
        Ok(())
    }

    fn merge_policy_bundle(
        &mut self,
        bundle: crate::execpolicy::PolicyBundle,
    ) -> anyhow::Result<()> {
        self.shell_policy.merge(bundle.policy);
        for (host, decision) in bundle.network_rules {
            self.network.rules.push(NetworkRule {
                pattern: host,
                decision,
                source: Some("policy file".to_string()),
            });
        }
        for rule in bundle.filesystem_rules {
            self.filesystem_rules.push(FilesystemRule::absolute(
                rule.path,
                FsRuleDecision::from_decision(rule.decision),
                rule.source.unwrap_or_else(|| "policy file".to_string()),
            ));
        }
        for rule in bundle.tool_rules {
            self.tool_rules.push(ToolRule {
                tool: rule.tool,
                action: rule.action,
                target: rule.target,
                decision: rule.decision,
                source: rule.source,
                justification: rule.justification,
            });
        }
        Ok(())
    }

    fn evaluate(&self, request: &PermissionRequest) -> Result<PolicyDecision> {
        let mut matches = match request.category {
            ToolCategory::Shell => self.evaluate_shell(request)?,
            ToolCategory::FileRead | ToolCategory::FileWrite => self.evaluate_filesystem(request),
            ToolCategory::Network => self.evaluate_network(request)?,
            ToolCategory::Agent | ToolCategory::Unknown | ToolCategory::Mcp => {
                self.evaluate_tool(request)
            }
            ToolCategory::Safe => vec![PolicyRuleMatch::Heuristic {
                subject: request.summary.clone(),
                decision: Decision::Allow,
                justification: Some("Read-only tool".to_string()),
            }],
        };
        if matches.is_empty() {
            matches.push(PolicyRuleMatch::Heuristic {
                subject: request.summary.clone(),
                decision: Decision::Prompt,
                justification: Some("No matching policy rule".to_string()),
            });
        }
        let decision = matches
            .iter()
            .map(PolicyRuleMatch::decision)
            .max()
            .unwrap_or(Decision::Prompt);
        let justification = matches.iter().find_map(PolicyRuleMatch::justification);
        Ok(PolicyDecision {
            decision,
            matched_rules: matches,
            sandbox_policy: self.sandbox_policy_for(request),
            justification,
        })
    }

    fn evaluate_shell(&self, request: &PermissionRequest) -> Result<Vec<PolicyRuleMatch>> {
        let command = match &request.target {
            PermissionTarget::ShellCommand(command) => command,
            _ => {
                return Ok(vec![PolicyRuleMatch::Heuristic {
                    subject: request.summary.clone(),
                    decision: Decision::Prompt,
                    justification: Some("Missing shell command target".to_string()),
                }])
            }
        };
        if contains_dynamic_shell_syntax(command) {
            return Ok(vec![PolicyRuleMatch::Heuristic {
                subject: command.clone(),
                decision: match self.shell_default {
                    Decision::Forbidden => Decision::Forbidden,
                    _ => Decision::Prompt,
                },
                justification: Some(
                    "Dynamic shell execution requires explicit approval".to_string(),
                ),
            }]);
        }
        let mut parsed = Vec::new();
        for segment in command_segments(command) {
            parsed.push(ExecPolicyCheckCommand::parse(&segment).map_err(|err| {
                DeepCodeError::ToolExecution {
                    tool: "shell".to_string(),
                    message: err.to_string(),
                }
            })?);
        }
        if parsed.is_empty() {
            return Ok(vec![PolicyRuleMatch::Heuristic {
                subject: command.clone(),
                decision: Decision::Prompt,
                justification: Some("Empty shell command".to_string()),
            }]);
        }
        let evaluation = self.shell_policy.check_multiple(&parsed, |command| {
            classify_shell_segment(command, self.shell_default)
        });
        Ok(evaluation
            .matches
            .into_iter()
            .map(PolicyRuleMatch::Shell)
            .collect())
    }

    fn evaluate_filesystem(&self, request: &PermissionRequest) -> Vec<PolicyRuleMatch> {
        let paths = target_paths(request);
        if paths.is_empty() {
            return vec![PolicyRuleMatch::Heuristic {
                subject: request.summary.clone(),
                decision: Decision::Prompt,
                justification: Some("File operation has no path target".to_string()),
            }];
        }
        let mut out = Vec::new();
        for path in paths {
            let op = if request.category == ToolCategory::FileRead {
                PermissionAction::Read
            } else {
                PermissionAction::Write
            };
            let normalized = normalize_path(&path);
            let mut matched = false;
            for rule in &self.filesystem_rules {
                if rule.matches(&normalized, &self.workspace_roots) {
                    matched = true;
                    let decision = rule.decision.for_action(op);
                    out.push(PolicyRuleMatch::Filesystem {
                        path: normalized.display().to_string(),
                        pattern: rule.display_pattern(),
                        decision,
                        source: Some(rule.source.clone()),
                        justification: Some("Filesystem policy".to_string()),
                    });
                }
            }
            if !matched {
                out.push(PolicyRuleMatch::Filesystem {
                    path: normalized.display().to_string(),
                    pattern: "<no match>".to_string(),
                    decision: Decision::Forbidden,
                    source: None,
                    justification: Some("No filesystem rule matched".to_string()),
                });
            }
        }
        out
    }

    fn evaluate_network(&self, request: &PermissionRequest) -> Result<Vec<PolicyRuleMatch>> {
        let host = match &request.target {
            PermissionTarget::NetworkHost(host) => host.clone(),
            _ => host_from_tool_input(&request.tool_name, &request.input)
                .unwrap_or_else(|| "web-search".to_string()),
        };
        Ok(vec![self.network.evaluate(&host)])
    }

    fn evaluate_tool(&self, request: &PermissionRequest) -> Vec<PolicyRuleMatch> {
        let (action, target) = match &request.target {
            PermissionTarget::Tool { action, target, .. } => {
                (Some(action.as_str()), target.as_deref())
            }
            _ => (None, None),
        };
        let mut matches = Vec::new();
        for rule in &self.tool_rules {
            if rule.matches(&request.tool_name, action, target) {
                matches.push(PolicyRuleMatch::Tool {
                    tool: rule.tool.clone(),
                    action: rule.action.clone(),
                    target: rule.target.clone(),
                    decision: rule.decision,
                    source: rule.source.clone(),
                    justification: rule.justification.clone(),
                });
            }
        }
        if matches.is_empty() {
            matches.push(PolicyRuleMatch::Heuristic {
                subject: request.summary.clone(),
                decision: match request.category {
                    ToolCategory::Agent => Decision::Prompt,
                    ToolCategory::Unknown | ToolCategory::Mcp => Decision::Prompt,
                    _ => Decision::Allow,
                },
                justification: Some("Tool default policy".to_string()),
            });
        }
        matches
    }

    fn sandbox_policy_for(&self, request: &PermissionRequest) -> SandboxPolicy {
        if self.danger_full_access {
            return SandboxPolicy::DangerFullAccess;
        }
        let network_access = self.network.enabled
            && matches!(
                request.category,
                ToolCategory::Network | ToolCategory::Shell
            );
        let rules = self.sandbox_filesystem_rules();
        match request.category {
            ToolCategory::Safe | ToolCategory::FileRead | ToolCategory::Network => {
                SandboxPolicy::Profile {
                    filesystem: rules,
                    network_access,
                    label: self.profile_name.clone(),
                }
            }
            _ => SandboxPolicy::Profile {
                filesystem: rules,
                network_access,
                label: self.profile_name.clone(),
            },
        }
    }

    fn sandbox_filesystem_rules(&self) -> Vec<FilesystemSandboxRule> {
        let mut rules = Vec::new();
        for rule in &self.filesystem_rules {
            rules.extend(rule.to_sandbox_rules(&self.workspace_roots));
        }
        rules
    }

    fn base_sandbox_label(&self) -> &'static str {
        if self.danger_full_access {
            "danger-full-access"
        } else {
            "profile"
        }
    }
}

#[derive(Debug, Clone)]
struct FilesystemRule {
    scope: RuleScope,
    pattern: String,
    decision: FsRuleDecision,
    source: String,
}

impl FilesystemRule {
    fn new(scope: RuleScope, pattern: String, decision: FsRuleDecision, source: String) -> Self {
        Self {
            scope,
            pattern,
            decision,
            source,
        }
    }

    fn special(
        scope: RuleScope,
        pattern: String,
        decision: FsRuleDecision,
        source: impl Into<String>,
    ) -> Self {
        Self::new(scope, pattern, decision, source.into())
    }

    fn absolute(pattern: String, decision: FsRuleDecision, source: impl Into<String>) -> Self {
        Self::new(RuleScope::Absolute, pattern, decision, source.into())
    }

    fn matches(&self, path: &Path, workspace_roots: &[PathBuf]) -> bool {
        match &self.scope {
            RuleScope::WorkspaceRoots => workspace_roots.iter().any(|root| {
                path.strip_prefix(root)
                    .ok()
                    .is_some_and(|rel| pattern_matches_path(&self.pattern, rel))
            }),
            RuleScope::Root => pattern_matches_path(&self.pattern, path),
            RuleScope::Absolute => pattern_matches_absolute(&self.pattern, path),
            RuleScope::Minimal => minimal_read_roots()
                .iter()
                .any(|root| path.starts_with(root)),
            RuleScope::TmpDir => path.starts_with(std::env::temp_dir()),
            RuleScope::SlashTmp => path.starts_with(Path::new("/tmp")),
        }
    }

    fn to_sandbox_rules(&self, workspace_roots: &[PathBuf]) -> Vec<FilesystemSandboxRule> {
        let access = if self.decision.write == Decision::Allow {
            deepcode_sandbox::FilesystemAccess::Write
        } else if self.decision.read == Decision::Allow {
            deepcode_sandbox::FilesystemAccess::Read
        } else {
            deepcode_sandbox::FilesystemAccess::Deny
        };
        match &self.scope {
            RuleScope::WorkspaceRoots => workspace_roots
                .iter()
                .flat_map(|root| sandbox_rules_for_pattern(root, &self.pattern, access))
                .collect(),
            RuleScope::Root => sandbox_rules_for_pattern(Path::new("/"), &self.pattern, access),
            RuleScope::Absolute => sandbox_rules_for_absolute(&self.pattern, access),
            RuleScope::Minimal => minimal_read_roots()
                .into_iter()
                .map(|path| FilesystemSandboxRule { path, access })
                .collect(),
            RuleScope::TmpDir => vec![FilesystemSandboxRule {
                path: std::env::temp_dir(),
                access,
            }],
            RuleScope::SlashTmp => slash_tmp_sandbox_rules(access),
        }
    }

    fn display_pattern(&self) -> String {
        format!("{:?}:{}", self.scope, self.pattern)
    }
}

#[derive(Debug, Clone)]
enum RuleScope {
    Root,
    WorkspaceRoots,
    Absolute,
    Minimal,
    TmpDir,
    SlashTmp,
}

#[derive(Debug, Clone, Copy)]
struct FsRuleDecision {
    read: Decision,
    write: Decision,
}

impl FsRuleDecision {
    fn from_access(access: FilesystemAccess) -> Self {
        match access {
            FilesystemAccess::Read => Self {
                read: Decision::Allow,
                write: Decision::Forbidden,
            },
            FilesystemAccess::Write => Self {
                read: Decision::Allow,
                write: Decision::Allow,
            },
            FilesystemAccess::Deny => Self {
                read: Decision::Forbidden,
                write: Decision::Forbidden,
            },
        }
    }

    fn from_operation(rule: &FilesystemOperationRule) -> Self {
        Self {
            read: rule
                .read
                .map(decision_from_config)
                .unwrap_or(Decision::Prompt),
            write: rule
                .write
                .map(decision_from_config)
                .unwrap_or(Decision::Prompt),
        }
    }

    fn from_decision(decision: Decision) -> Self {
        Self {
            read: decision,
            write: decision,
        }
    }

    fn for_action(self, action: PermissionAction) -> Decision {
        match action {
            PermissionAction::Read => self.read,
            PermissionAction::Write => self.write,
            _ => Decision::Prompt,
        }
    }
}

#[derive(Debug, Clone)]
struct NetworkRules {
    enabled: bool,
    default: Decision,
    allow_local_binding: bool,
    rules: Vec<NetworkRule>,
    audit_log: Option<PathBuf>,
}

impl Default for NetworkRules {
    fn default() -> Self {
        Self {
            enabled: false,
            default: Decision::Prompt,
            allow_local_binding: false,
            rules: Vec::new(),
            audit_log: None,
        }
    }
}

impl NetworkRules {
    fn apply_config(&mut self, config: &ProfileNetworkConfig, profile: &str) {
        self.enabled = config.enabled;
        self.default = decision_from_config(config.default);
        self.allow_local_binding = config.allow_local_binding;
        self.audit_log = config.audit_log.clone();
        for (pattern, decision) in &config.domains {
            self.rules.push(NetworkRule {
                pattern: pattern.to_ascii_lowercase(),
                decision: decision_from_config(*decision),
                source: Some(format!("permissions.{}.network.domains", profile)),
            });
        }
    }

    fn evaluate(&self, host: &str) -> PolicyRuleMatch {
        let normalized = host.trim_end_matches('.').to_ascii_lowercase();
        if !self.enabled {
            return PolicyRuleMatch::Network {
                host: normalized,
                pattern: "<network disabled>".to_string(),
                decision: Decision::Forbidden,
                source: None,
                justification: Some("Network is disabled for the active profile".to_string()),
            };
        }
        let mut matched = Vec::new();
        for rule in &self.rules {
            if host_matches(&rule.pattern, &normalized) {
                matched.push(rule.clone());
            }
        }
        let explicit_allow = matched.iter().any(|rule| rule.decision == Decision::Allow);
        if is_private_or_metadata_host(&normalized) && !explicit_allow && !self.allow_local_binding
        {
            return PolicyRuleMatch::Network {
                host: normalized,
                pattern: "<local/private guard>".to_string(),
                decision: Decision::Forbidden,
                source: None,
                justification: Some(
                    "Local/private network targets require explicit allow".to_string(),
                ),
            };
        }
        let decision = matched
            .iter()
            .map(|rule| rule.decision)
            .max()
            .unwrap_or(self.default);
        let pattern = matched
            .last()
            .map(|rule| rule.pattern.clone())
            .unwrap_or_else(|| "<default>".to_string());
        PolicyRuleMatch::Network {
            host: normalized,
            pattern,
            decision,
            source: matched.last().and_then(|rule| rule.source.clone()),
            justification: Some("Network host policy".to_string()),
        }
    }

    fn summary(&self) -> String {
        if !self.enabled {
            "disabled".to_string()
        } else {
            format!(
                "enabled, default {}, {} domain rules",
                self.default.as_str(),
                self.rules.len()
            )
        }
    }
}

#[derive(Debug, Clone)]
struct NetworkRule {
    pattern: String,
    decision: Decision,
    source: Option<String>,
}

#[derive(Debug, Clone)]
struct ToolRule {
    tool: String,
    action: Option<String>,
    target: Option<String>,
    decision: Decision,
    source: Option<String>,
    justification: Option<String>,
}

impl ToolRule {
    fn from_config(rule: &ToolRuleConfig, profile: &str) -> Self {
        Self {
            tool: rule.tool.clone(),
            action: rule.action.clone(),
            target: rule.target.clone(),
            decision: decision_from_config(rule.decision),
            source: Some(format!("permissions.{}.tool.rules", profile)),
            justification: rule.justification.clone(),
        }
    }

    fn matches(&self, tool: &str, action: Option<&str>, target: Option<&str>) -> bool {
        if self.tool != tool {
            return false;
        }
        if let Some(rule_action) = self.action.as_deref() {
            if Some(rule_action) != action {
                return false;
            }
        }
        if let Some(rule_target) = self.target.as_deref() {
            let Some(target) = target else {
                return false;
            };
            return Pattern::new(rule_target)
                .map(|pattern| pattern.matches(target))
                .unwrap_or_else(|_| rule_target == target);
        }
        true
    }
}

fn build_request(tool: &dyn Tool, input: &serde_json::Value) -> Result<PermissionRequest> {
    let tool_name = tool.name().to_string();
    let safety = tool.safety();
    let category = classify_category(&tool_name, &safety);
    let summary = summarize_tool(&tool_name, input);
    let (action, target) = match tool_name.as_str() {
        "shell" => (
            PermissionAction::Execute,
            PermissionTarget::ShellCommand(required_str(input, "command", "shell")?.to_string()),
        ),
        "web_fetch" | "fetch_url" => (
            PermissionAction::Fetch,
            PermissionTarget::NetworkHost(
                host_from_url(required_str(input, "url", &tool_name)?).map_err(|err| {
                    DeepCodeError::ToolExecution {
                        tool: tool_name.clone(),
                        message: format!("Malformed URL host: {}", err),
                    }
                })?,
            ),
        ),
        "web_search" => (
            PermissionAction::Search,
            PermissionTarget::NetworkHost("web-search".to_string()),
        ),
        "read_file" => (
            PermissionAction::Read,
            PermissionTarget::File(PathBuf::from(required_str(input, "path", &tool_name)?)),
        ),
        "grep" | "glob" => (
            PermissionAction::Read,
            input
                .get("directory")
                .and_then(serde_json::Value::as_str)
                .or_else(|| input.get("path").and_then(serde_json::Value::as_str))
                .map(|path| PermissionTarget::File(PathBuf::from(path)))
                .unwrap_or_else(|| PermissionTarget::File(PathBuf::from("."))),
        ),
        "write_file" | "edit_file" => (
            PermissionAction::Write,
            PermissionTarget::File(PathBuf::from(required_str(input, "path", &tool_name)?)),
        ),
        "git_add" => (
            PermissionAction::Write,
            PermissionTarget::Tool {
                tool: tool_name.clone(),
                action: "stage".to_string(),
                target: json_array_strings(input, "files").map(|values| values.join(",")),
            },
        ),
        "git_commit" => (
            PermissionAction::Write,
            PermissionTarget::Tool {
                tool: tool_name.clone(),
                action: "commit".to_string(),
                target: input
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
            },
        ),
        "git_checkout" => {
            let (action, target) = if let Some(files) = json_array_strings(input, "files") {
                ("restore_files".to_string(), Some(files.join(",")))
            } else if input.get("create").and_then(serde_json::Value::as_bool) == Some(true) {
                (
                    "create_branch".to_string(),
                    input
                        .get("branch")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string),
                )
            } else {
                (
                    "switch_branch".to_string(),
                    input
                        .get("branch")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_string),
                )
            };
            (
                PermissionAction::Write,
                PermissionTarget::Tool {
                    tool: tool_name.clone(),
                    action,
                    target,
                },
            )
        }
        "git_branch" => (
            PermissionAction::Write,
            PermissionTarget::Tool {
                tool: tool_name.clone(),
                action: input
                    .get("action")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("list")
                    .to_string(),
                target: input
                    .get("branch")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
            },
        ),
        "agent" => (
            PermissionAction::SpawnAgent,
            PermissionTarget::Tool {
                tool: tool_name.clone(),
                action: "spawn".to_string(),
                target: input
                    .get("task")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
            },
        ),
        _ => (PermissionAction::Other, PermissionTarget::None),
    };
    Ok(PermissionRequest {
        tool_name,
        category,
        action,
        target,
        input: input.clone(),
        summary,
    })
}

fn classify_category(tool_name: &str, safety: &ToolSafety) -> ToolCategory {
    match tool_name {
        "shell" => ToolCategory::Shell,
        "web_fetch" | "web_search" | "fetch_url" => ToolCategory::Network,
        "read_file" | "grep" | "glob" => ToolCategory::FileRead,
        "write_file" | "edit_file" => ToolCategory::FileWrite,
        "git_add" | "git_commit" | "git_checkout" | "git_branch" => ToolCategory::Unknown,
        "agent" => ToolCategory::Agent,
        _ if safety.is_read_only && !safety.requires_approval => ToolCategory::Safe,
        _ => ToolCategory::Unknown,
    }
}

fn hard_forbidden(tool_name: &str, input: &serde_json::Value) -> Option<(String, RiskLevel)> {
    if tool_name == "shell" {
        let command = input
            .get("command")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        if command.contains("rm -rf /") || command.contains("rm -rf --no-preserve-root /") {
            return Some((
                "Destructive system-wide deletion blocked".to_string(),
                RiskLevel::Critical,
            ));
        }
        if command.contains("dd if=") || command.contains("mkfs.") || command.contains("> /dev/sda")
        {
            return Some((
                "Raw disk or formatting operation blocked".to_string(),
                RiskLevel::Critical,
            ));
        }
    }
    None
}

fn risk_for(
    tool_name: &str,
    input: &serde_json::Value,
    safety: &ToolSafety,
    category: ToolCategory,
) -> RiskLevel {
    if tool_name == "shell" {
        let command = input
            .get("command")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_ascii_lowercase();
        if command.contains("sudo")
            || command.contains("rm ")
            || command.contains("chmod")
            || command.contains("chown")
            || command.contains("curl")
            || command.contains("wget")
        {
            return RiskLevel::Critical;
        }
    }
    match category {
        ToolCategory::Safe | ToolCategory::FileRead | ToolCategory::Network => RiskLevel::Routine,
        ToolCategory::FileWrite | ToolCategory::Shell | ToolCategory::Agent => RiskLevel::Elevated,
        _ if safety.is_destructive => RiskLevel::Critical,
        _ if safety.requires_approval => RiskLevel::Elevated,
        _ => RiskLevel::Routine,
    }
}

fn prefix_rule_from_config(rule: &ShellRuleConfig, profile: &str) -> anyhow::Result<PrefixRule> {
    let tokens = rule
        .pattern
        .iter()
        .map(|token| match token {
            ShellPatternTokenConfig::Token(value) => PatternToken::Single(value.clone()),
            ShellPatternTokenConfig::AnyOf(values) => PatternToken::Alternatives(values.clone()),
        })
        .collect();
    Ok(PrefixRule {
        pattern: PrefixPattern::new(tokens)?,
        decision: decision_from_config(rule.decision),
        match_examples: Vec::new(),
        not_match_examples: Vec::new(),
        justification: rule.justification.clone(),
        source: Some(format!("permissions.{}.shell.rules", profile)),
    })
}

fn classify_shell_segment(command: &ExecPolicyCheckCommand, default: Decision) -> Decision {
    let text = command.original.to_ascii_lowercase();
    if text.contains("rm -rf /")
        || text.contains("mkfs.")
        || text.contains("dd if=")
        || text.contains("> /dev/sda")
    {
        return Decision::Forbidden;
    }
    default
}

fn scope_and_pattern(key: &str, inherited_scope: RuleScope) -> (RuleScope, String) {
    if let Some(scope) = special_scope(key) {
        return (scope, ".".to_string());
    }
    (inherited_scope, key.to_string())
}

fn special_scope(key: &str) -> Option<RuleScope> {
    match key {
        ":root" => Some(RuleScope::Root),
        ":workspace_roots" => Some(RuleScope::WorkspaceRoots),
        ":minimal" => Some(RuleScope::Minimal),
        ":tmpdir" => Some(RuleScope::TmpDir),
        ":slash_tmp" => Some(RuleScope::SlashTmp),
        _ => None,
    }
}

fn decision_from_config(decision: PermissionDecision) -> Decision {
    match decision {
        PermissionDecision::Allow => Decision::Allow,
        PermissionDecision::Prompt => Decision::Prompt,
        PermissionDecision::Deny => Decision::Forbidden,
    }
}

fn target_paths(request: &PermissionRequest) -> Vec<PathBuf> {
    match &request.target {
        PermissionTarget::File(path) => vec![path.clone()],
        PermissionTarget::Files(paths) => paths.clone(),
        _ => Vec::new(),
    }
}

fn required_str<'a>(input: &'a serde_json::Value, key: &str, tool: &str) -> Result<&'a str> {
    input
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| DeepCodeError::ToolExecution {
            tool: tool.to_string(),
            message: format!("Missing '{}' parameter", key),
        })
}

fn json_array_strings(input: &serde_json::Value, key: &str) -> Option<Vec<String>> {
    input
        .get(key)
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(str::to_string)
                .collect()
        })
}

fn summarize_tool(tool_name: &str, input: &serde_json::Value) -> String {
    match tool_name {
        "shell" => input
            .get("command")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("shell command")
            .to_string(),
        "web_fetch" => input
            .get("url")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("web fetch")
            .to_string(),
        "web_search" => input
            .get("query")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("web search")
            .to_string(),
        _ => serde_json::to_string(input).unwrap_or_else(|_| tool_name.to_string()),
    }
}

fn current_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn normalize_roots(paths: &[PathBuf]) -> Vec<PathBuf> {
    let mut roots = paths
        .iter()
        .map(|path| normalize_path(path))
        .collect::<Vec<_>>();
    roots.sort();
    roots.dedup();
    roots
}

fn normalize_path(path: &Path) -> PathBuf {
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        current_dir().join(path)
    };
    if let Ok(canonical) = candidate.canonicalize() {
        return canonical;
    }
    if let Some(parent) = candidate.parent() {
        if let Ok(parent) = parent.canonicalize() {
            if let Some(name) = candidate.file_name() {
                return parent.join(name);
            }
        }
    }
    candidate
}

fn expand_home(path: &str) -> PathBuf {
    if let Some(rest) = path.strip_prefix("~/") {
        home_dir().join(rest)
    } else if path == "~" {
        home_dir()
    } else {
        PathBuf::from(path)
    }
}

fn home_dir() -> PathBuf {
    deepcode_core::paths::home_dir()
}

fn pattern_matches_absolute(pattern: &str, path: &Path) -> bool {
    let expanded = expand_home(pattern);
    if pattern.contains('*') {
        return Pattern::new(&expanded.display().to_string())
            .map(|glob| glob.matches_path(path))
            .unwrap_or(false);
    }
    let normalized = normalize_path(&expanded);
    path == normalized || path.starts_with(&normalized)
}

fn pattern_matches_path(pattern: &str, path: &Path) -> bool {
    if pattern == "." {
        return true;
    }
    if pattern.contains('*') {
        return Pattern::new(pattern)
            .map(|glob| glob.matches_path(path))
            .unwrap_or(false);
    }
    path == Path::new(pattern) || path.starts_with(Path::new(pattern))
}

fn sandbox_rules_for_pattern(
    root: &Path,
    pattern: &str,
    access: deepcode_sandbox::FilesystemAccess,
) -> Vec<FilesystemSandboxRule> {
    if pattern == "." {
        return vec![FilesystemSandboxRule {
            path: root.to_path_buf(),
            access,
        }];
    }
    if pattern.contains('*') {
        if matches!(access, deepcode_sandbox::FilesystemAccess::Deny) {
            let glob_pattern = root.join(pattern).display().to_string();
            return glob::glob(&glob_pattern)
                .ok()
                .into_iter()
                .flatten()
                .filter_map(std::result::Result::ok)
                .map(|path| FilesystemSandboxRule { path, access })
                .collect();
        }
        return vec![FilesystemSandboxRule {
            path: root.to_path_buf(),
            access,
        }];
    }
    vec![FilesystemSandboxRule {
        path: root.join(pattern),
        access,
    }]
}

fn sandbox_rules_for_absolute(
    pattern: &str,
    access: deepcode_sandbox::FilesystemAccess,
) -> Vec<FilesystemSandboxRule> {
    if pattern.contains('*') {
        if matches!(access, deepcode_sandbox::FilesystemAccess::Deny) {
            return glob::glob(&expand_home(pattern).display().to_string())
                .ok()
                .into_iter()
                .flatten()
                .filter_map(std::result::Result::ok)
                .map(|path| FilesystemSandboxRule { path, access })
                .collect();
        }
        let prefix = pattern
            .split('*')
            .next()
            .map(|value| value.trim_end_matches('/'))
            .unwrap_or(pattern)
            .trim_end_matches('/');
        if prefix.is_empty() {
            return Vec::new();
        }
        return vec![FilesystemSandboxRule {
            path: expand_home(prefix),
            access,
        }];
    }
    vec![FilesystemSandboxRule {
        path: expand_home(pattern),
        access,
    }]
}

#[cfg(not(target_os = "windows"))]
fn minimal_read_roots() -> Vec<PathBuf> {
    ["/bin", "/usr", "/System", "/Library", "/opt", "/dev/null"]
        .into_iter()
        .map(PathBuf::from)
        .collect()
}

#[cfg(target_os = "windows")]
fn minimal_read_roots() -> Vec<PathBuf> {
    let mut roots = [
        "SystemRoot",
        "ProgramFiles",
        "ProgramFiles(x86)",
        "ProgramData",
    ]
    .into_iter()
    .filter_map(std::env::var_os)
    .map(PathBuf::from)
    .collect::<Vec<_>>();
    if let Ok(executable) = std::env::current_exe() {
        if let Some(parent) = executable.parent() {
            roots.push(parent.to_path_buf());
        }
    }
    roots.sort();
    roots.dedup();
    roots
}

#[cfg(not(target_os = "windows"))]
fn slash_tmp_sandbox_rules(
    access: deepcode_sandbox::FilesystemAccess,
) -> Vec<FilesystemSandboxRule> {
    vec![FilesystemSandboxRule {
        path: PathBuf::from("/tmp"),
        access,
    }]
}

#[cfg(target_os = "windows")]
fn slash_tmp_sandbox_rules(
    _access: deepcode_sandbox::FilesystemAccess,
) -> Vec<FilesystemSandboxRule> {
    Vec::new()
}

fn load_persistent_grants(path: Option<&PathBuf>) -> HashSet<ApprovalKey> {
    let mut keys = HashSet::new();
    let Some(path) = path else {
        return keys;
    };
    let Ok(content) = std::fs::read_to_string(path) else {
        return keys;
    };
    for line in content.lines() {
        let line = line.trim();
        let Some(value) = line.strip_prefix("grouping = ") else {
            continue;
        };
        let value = value.trim().trim_matches('"');
        if !value.is_empty() {
            keys.insert(ApprovalKey(value.to_string()));
        }
    }
    keys
}

fn append_persistent_grant(
    path: Option<&PathBuf>,
    tool_name: &str,
    input: &serde_json::Value,
    grouping_key: &ApprovalKey,
) -> Result<()> {
    let Some(path) = path else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| DeepCodeError::Config(e.to_string()))?;
    }
    let line = format!(
        "\n[[grants]]\ntool = {}\ngrouping = {}\nsummary = {}\ndecision = \"allow\"\ncreated_at = {}\n",
        serde_json::to_string(tool_name).unwrap_or_default(),
        serde_json::to_string(grouping_key.as_str()).unwrap_or_default(),
        serde_json::to_string(&summarize_tool(tool_name, input)).unwrap_or_default(),
        serde_json::to_string(&chrono::Utc::now().to_rfc3339()).unwrap_or_default(),
    );
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut file| std::io::Write::write_all(&mut file, line.as_bytes()))
        .map_err(|e| DeepCodeError::Config(e.to_string()))
}

fn display_paths(paths: &[PathBuf]) -> String {
    if paths.is_empty() {
        return "<none>".to_string();
    }
    paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use deepcode_core::config::{
        FilesystemPermissionsConfig, PermissionProfileConfig, PermissionsConfig, ProfileShellConfig,
    };
    use deepcode_tools::tool::{Tool, ToolSafety};

    #[derive(Debug)]
    struct MockTool {
        name: &'static str,
        safety: ToolSafety,
    }

    #[async_trait::async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            "mock"
        }
        fn input_schema(&self) -> serde_json::Value {
            serde_json::json!({})
        }
        fn safety(&self) -> ToolSafety {
            self.safety.clone()
        }
        async fn execute(&self, _input: serde_json::Value) -> Result<String> {
            Ok("ok".to_string())
        }
    }

    fn tool(name: &'static str, safety: ToolSafety) -> MockTool {
        MockTool { name, safety }
    }

    #[tokio::test]
    async fn workspace_read_is_allowed() {
        let mut perm = PermissionSystem::new(PermissionSystemConfig::default());
        let eval = perm
            .check(
                &tool("read_file", ToolSafety::READ_ONLY),
                &serde_json::json!({"path": "src/lib.rs"}),
            )
            .await
            .unwrap();
        assert_eq!(eval.decision, Decision::Allow);
    }

    #[tokio::test]
    async fn workspace_env_read_is_denied() {
        let mut perm = PermissionSystem::new(PermissionSystemConfig::default());
        let eval = perm
            .check(
                &tool("read_file", ToolSafety::READ_ONLY),
                &serde_json::json!({"path": ".env"}),
            )
            .await
            .unwrap();
        assert_eq!(eval.decision, Decision::Forbidden);
    }

    #[tokio::test]
    async fn manifest_write_prompts_despite_workspace_write() {
        let mut perm = PermissionSystem::new(PermissionSystemConfig::default());
        let eval = perm
            .check(
                &tool("write_file", ToolSafety::SAFE_MUTATION),
                &serde_json::json!({"path": "Cargo.toml", "content": ""}),
            )
            .await
            .unwrap();
        assert_eq!(eval.decision, Decision::Prompt);
    }

    #[tokio::test]
    async fn dangerous_shell_is_forbidden() {
        let mut perm = PermissionSystem::new(PermissionSystemConfig::default());
        let eval = perm
            .check(
                &tool("shell", ToolSafety::DESTRUCTIVE),
                &serde_json::json!({"command": "rm -rf /"}),
            )
            .await
            .unwrap();
        assert_eq!(eval.decision, Decision::Forbidden);
        assert_eq!(eval.risk, RiskLevel::Critical);
    }

    #[tokio::test]
    async fn shell_rule_allows_matching_command() {
        let mut profile = PermissionProfileConfig {
            extends: Some(":workspace".to_string()),
            ..Default::default()
        };
        profile.shell = ProfileShellConfig {
            rules: vec![ShellRuleConfig {
                pattern: vec![
                    ShellPatternTokenConfig::Token("git".to_string()),
                    ShellPatternTokenConfig::Token("status".to_string()),
                ],
                decision: PermissionDecision::Allow,
                justification: None,
            }],
            ..Default::default()
        };
        let mut profiles = BTreeMap::new();
        profiles.insert("test".to_string(), profile);
        let mut perm = PermissionSystem::new(PermissionSystemConfig {
            default_permissions: Some("test".to_string()),
            permissions: PermissionsConfig {
                profiles,
                ..Default::default()
            },
            ..Default::default()
        });
        let shell = tool("shell", ToolSafety::DESTRUCTIVE);
        let allowed = perm
            .check(
                &shell,
                &serde_json::json!({"command": "git status --short"}),
            )
            .await
            .unwrap();
        assert_eq!(allowed.decision, Decision::Allow);
    }

    #[tokio::test]
    async fn session_deny_does_not_block_variant() {
        let mut perm = PermissionSystem::new(PermissionSystemConfig::default());
        let shell = tool("shell", ToolSafety::DESTRUCTIVE);
        let input = serde_json::json!({"command": "cargo test -p a"});
        perm.handle_response("shell", &input, false, ApprovalScope::Once)
            .unwrap();
        let denied = perm.check(&shell, &input).await.unwrap();
        assert_eq!(denied.decision, Decision::Forbidden);
        let variant = perm
            .check(&shell, &serde_json::json!({"command": "cargo test -p b"}))
            .await
            .unwrap();
        assert_eq!(variant.decision, Decision::Prompt);
    }

    #[tokio::test]
    async fn explicit_filesystem_rule_can_deny_workspace_path() {
        let mut profile = PermissionProfileConfig {
            extends: Some(":workspace".to_string()),
            filesystem: FilesystemPermissionsConfig {
                entries: BTreeMap::from([(
                    ":workspace_roots".to_string(),
                    FilesystemRuleValue::Children(BTreeMap::from([(
                        "README.md".to_string(),
                        FilesystemRuleValue::Access(FilesystemAccess::Deny),
                    )])),
                )]),
                ..Default::default()
            },
            ..Default::default()
        };
        profile
            .workspace_roots
            .insert(current_dir().display().to_string(), true);
        let mut profiles = BTreeMap::new();
        profiles.insert("test".to_string(), profile);
        let mut perm = PermissionSystem::new(PermissionSystemConfig {
            default_permissions: Some("test".to_string()),
            permissions: PermissionsConfig {
                profiles,
                ..Default::default()
            },
            ..Default::default()
        });
        let eval = perm
            .check(
                &tool("read_file", ToolSafety::READ_ONLY),
                &serde_json::json!({"path": "README.md"}),
            )
            .await
            .unwrap();
        assert_eq!(eval.decision, Decision::Forbidden);
    }
}
