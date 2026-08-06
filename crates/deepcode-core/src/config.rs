use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeepCodeConfig {
    pub active_provider: Option<String>,
    pub default_permissions: Option<String>,
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub permissions: PermissionsConfig,
}

impl DeepCodeConfig {
    pub fn parse(content: &str) -> crate::error::Result<Self> {
        let config: Self = toml::from_str(content)
            .map_err(|e| crate::error::DeepCodeError::Config(format!("Invalid config: {}", e)))?;
        config.validate()?;
        Ok(config)
    }

    pub fn load(path: &PathBuf) -> crate::error::Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            crate::error::DeepCodeError::Config(format!(
                "Cannot read config file {:?}: {}",
                path, e
            ))
        })?;
        Self::parse(&content)
    }

    pub fn active_provider_name(&self) -> crate::error::Result<&str> {
        if let Some(name) = self.active_provider.as_deref() {
            return Ok(name);
        }
        if self.providers.len() == 1 {
            return self
                .providers
                .keys()
                .next()
                .map(String::as_str)
                .ok_or_else(|| config_error("At least one provider must be configured"));
        }
        Err(config_error(
            "active_provider is required when more than one provider is configured",
        ))
    }

    pub fn resolve_provider(
        &self,
        name: Option<&str>,
    ) -> crate::error::Result<(String, &ProviderConfig)> {
        let key = match name {
            Some(name) => name,
            None => self.active_provider_name()?,
        };
        self.providers
            .get(key)
            .map(|provider| (key.to_string(), provider))
            .ok_or_else(|| {
                crate::error::DeepCodeError::Config(format!(
                    "Provider '{}' not found in config",
                    key
                ))
            })
    }

    fn validate(&self) -> crate::error::Result<()> {
        if self.providers.is_empty() {
            return Err(config_error("At least one provider must be configured"));
        }
        for (name, provider) in &self.providers {
            provider.validate(name)?;
        }
        let (active_name, active_provider) = self.resolve_provider(None)?;
        if active_provider.kind != "ollama" && active_provider.resolve_api_key().is_none() {
            return Err(config_error(format!(
                "Active provider '{}' requires api_key",
                active_name
            )));
        }
        self.tools.validate()?;
        self.permissions
            .validate(self.default_permissions.as_deref())?;

        Ok(())
    }
}

fn config_error(message: impl Into<String>) -> crate::error::DeepCodeError {
    crate::error::DeepCodeError::Config(message.into())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    #[serde(rename = "type")]
    pub kind: String,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub max_concurrent_requests: Option<usize>,
    pub request_timeout_secs: Option<u64>,
    pub model: Option<String>,
    pub reasoning_effort: Option<ReasoningEffort>,
    pub wire_api: Option<String>,
    #[serde(default)]
    pub models: HashMap<String, ModelOverride>,
}

impl ProviderConfig {
    pub fn resolve_api_key(&self) -> Option<String> {
        if let Some(ref key) = self.api_key {
            if !key.is_empty() {
                return Some(key.clone());
            }
        }
        None
    }

    fn validate(&self, name: &str) -> crate::error::Result<()> {
        if !matches!(
            self.kind.as_str(),
            "openai" | "anthropic" | "deepseek" | "ollama" | "kimi"
        ) {
            return Err(config_error(format!(
                "Provider '{}' has unsupported type '{}'; supported: openai, anthropic, deepseek, ollama, kimi",
                name, self.kind
            )));
        }
        if self.api_key.as_deref() == Some("") {
            return Err(config_error(format!(
                "Provider '{}' contains an empty API key setting",
                name
            )));
        }
        if self.max_concurrent_requests == Some(0) {
            return Err(config_error(format!(
                "Provider '{}' max_concurrent_requests must be greater than zero",
                name
            )));
        }
        if self.request_timeout_secs == Some(0) {
            return Err(config_error(format!(
                "Provider '{}' request_timeout_secs must be greater than zero",
                name
            )));
        }
        if let Some(wire_api) = self.wire_api.as_deref() {
            if !matches!(self.kind.as_str(), "openai" | "deepseek")
                || !matches!(wire_api, "responses" | "chat_completions")
            {
                return Err(config_error(format!(
                    "Provider '{}' has invalid wire_api '{}'; OpenAI and DeepSeek support responses or chat_completions",
                    name, wire_api
                )));
            }
        }
        for (model, override_config) in &self.models {
            override_config.validate(model)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub id: String,
    pub provider: String,
    pub display_name: Option<String>,
    pub context_window: usize,
    pub max_output_tokens: usize,
    pub reasoning_efforts: Vec<ReasoningEffort>,
}

impl ModelProfile {
    pub fn supports_effort(&self, effort: ReasoningEffort) -> bool {
        self.reasoning_efforts
            .iter()
            .any(|candidate| candidate == &effort)
    }

    pub fn supports_effort_str(&self, effort: &str) -> bool {
        effort
            .parse()
            .is_ok_and(|effort| self.supports_effort(effort))
    }

    pub fn effort_names(&self) -> Vec<&'static str> {
        self.reasoning_efforts
            .iter()
            .map(|effort| effort.as_str())
            .collect()
    }

    pub fn validate(&self) -> crate::error::Result<()> {
        if self.id.trim().is_empty() || self.provider.trim().is_empty() {
            return Err(config_error("Model id and provider must not be empty"));
        }
        if self.context_window == 0 || self.max_output_tokens == 0 {
            return Err(config_error(format!(
                "Model '{}' token limits must be greater than zero",
                self.id
            )));
        }
        if self.max_output_tokens >= self.context_window {
            return Err(config_error(format!(
                "Model '{}' max_output_tokens must be smaller than context_window",
                self.id
            )));
        }
        if self.reasoning_efforts.is_empty() {
            return Err(config_error(format!(
                "Model '{}' must declare at least one reasoning_effort (use ['off'] for non-reasoning models)",
                self.id
            )));
        }
        let mut efforts = HashSet::new();
        for effort in &self.reasoning_efforts {
            if !efforts.insert(*effort) {
                return Err(config_error(format!(
                    "Model '{}' contains duplicate reasoning effort '{}'",
                    self.id, effort
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOverride {
    pub display_name: Option<String>,
    pub context_window: Option<usize>,
    pub max_output_tokens: Option<usize>,
    pub reasoning_efforts: Option<Vec<ReasoningEffort>>,
    pub default_reasoning_effort: Option<ReasoningEffort>,
}

impl ModelOverride {
    fn validate(&self, model: &str) -> crate::error::Result<()> {
        if self.context_window == Some(0) || self.max_output_tokens == Some(0) {
            return Err(config_error(format!(
                "Model override '{}' token limits must be greater than zero",
                model
            )));
        }
        if let (Some(output), Some(context)) = (self.max_output_tokens, self.context_window) {
            if output >= context {
                return Err(config_error(format!(
                    "Model override '{}' max_output_tokens must be smaller than context_window",
                    model
                )));
            }
        }
        if self.reasoning_efforts.as_ref().is_some_and(Vec::is_empty) {
            return Err(config_error(format!(
                "Model override '{}' reasoning_efforts must not be empty",
                model
            )));
        }
        if let (Some(default), Some(efforts)) = (
            self.default_reasoning_effort,
            self.reasoning_efforts.as_ref(),
        ) {
            if !efforts.contains(&default) {
                return Err(config_error(format!(
                    "Model override '{}' default reasoning effort '{}' is not supported",
                    model, default
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    Off,
    Minimal,
    Low,
    Medium,
    High,
    Xhigh,
    Max,
}

impl ReasoningEffort {
    pub const ALL: [Self; 7] = [
        Self::Off,
        Self::Minimal,
        Self::Low,
        Self::Medium,
        Self::High,
        Self::Xhigh,
        Self::Max,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Xhigh => "xhigh",
            Self::Max => "max",
        }
    }
}

impl fmt::Display for ReasoningEffort {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for ReasoningEffort {
    type Err = crate::error::DeepCodeError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|candidate| candidate.as_str().eq_ignore_ascii_case(value))
            .ok_or_else(|| config_error(format!("Unsupported reasoning effort '{}'", value)))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ToolsConfig {
    #[serde(default)]
    pub disabled: Vec<String>,
    pub max_file_size_bytes: Option<usize>,
}

impl ToolsConfig {
    fn validate(&self) -> crate::error::Result<()> {
        const KNOWN_TOOLS: &[&str] = &[
            "read_file",
            "write_file",
            "edit_file",
            "shell",
            "glob",
            "grep",
            "web_fetch",
            "web_search",
            "git_status",
            "git_diff",
            "git_log",
            "git_add",
            "git_commit",
            "git_checkout",
            "git_branch",
            "agent",
        ];
        if let Some(unknown) = self
            .disabled
            .iter()
            .find(|name| !KNOWN_TOOLS.contains(&name.as_str()))
        {
            return Err(config_error(format!(
                "Unknown tool '{}' in tools.disabled",
                unknown
            )));
        }
        if self.max_file_size_bytes == Some(0) {
            return Err(config_error(
                "tools.max_file_size_bytes must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PermissionsConfig {
    #[serde(default = "default_policy_files")]
    pub policy_files: Vec<PathBuf>,
    pub write_policy_file: Option<PathBuf>,
    #[serde(default)]
    #[serde(flatten)]
    pub profiles: BTreeMap<String, PermissionProfileConfig>,
}

impl PermissionsConfig {
    fn validate(&self, default_permissions: Option<&str>) -> crate::error::Result<()> {
        for (name, profile) in &self.profiles {
            validate_profile_name(name)?;
            profile.validate(name)?;
        }
        if let Some(default) = default_permissions {
            validate_profile_reference(default)?;
            if !is_builtin_permission_profile(default) && !self.profiles.contains_key(default) {
                return Err(config_error(format!(
                    "default_permissions references unknown profile '{}'",
                    default
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PermissionProfileConfig {
    pub description: Option<String>,
    pub extends: Option<String>,
    #[serde(default)]
    pub workspace_roots: BTreeMap<String, bool>,
    #[serde(default)]
    pub filesystem: FilesystemPermissionsConfig,
    #[serde(default)]
    pub network: ProfileNetworkConfig,
    #[serde(default)]
    pub shell: ProfileShellConfig,
    #[serde(default)]
    pub tool: ProfileToolConfig,
}

impl PermissionProfileConfig {
    fn validate(&self, name: &str) -> crate::error::Result<()> {
        if let Some(parent) = self.extends.as_deref() {
            validate_profile_reference(parent)?;
            if parent == ":danger-full-access" {
                return Err(config_error(format!(
                    "permissions.{} cannot extend :danger-full-access",
                    name
                )));
            }
        }
        for (root, enabled) in &self.workspace_roots {
            if root.trim().is_empty() {
                return Err(config_error(format!(
                    "permissions.{}.workspace_roots cannot contain an empty path",
                    name
                )));
            }
            let _ = enabled;
        }
        self.filesystem.validate(name)?;
        self.network.validate(name)?;
        self.shell.validate(name)?;
        self.tool.validate(name)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FilesystemPermissionsConfig {
    pub glob_scan_max_depth: Option<usize>,
    #[serde(default)]
    #[serde(flatten)]
    pub entries: BTreeMap<String, FilesystemRuleValue>,
}

impl FilesystemPermissionsConfig {
    fn validate(&self, profile: &str) -> crate::error::Result<()> {
        if self.glob_scan_max_depth == Some(0) {
            return Err(config_error(format!(
                "permissions.{}.filesystem.glob_scan_max_depth must be greater than zero",
                profile
            )));
        }
        for (path, value) in &self.entries {
            if path.trim().is_empty() {
                return Err(config_error(format!(
                    "permissions.{}.filesystem cannot contain an empty path",
                    profile
                )));
            }
            value.validate(profile)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FilesystemRuleValue {
    Access(FilesystemAccess),
    Operations(FilesystemOperationRule),
    Children(BTreeMap<String, FilesystemRuleValue>),
}

impl FilesystemRuleValue {
    fn validate(&self, profile: &str) -> crate::error::Result<()> {
        match self {
            Self::Access(_) => Ok(()),
            Self::Operations(rule) => rule.validate(profile),
            Self::Children(children) => {
                for (path, value) in children {
                    if path.trim().is_empty() {
                        return Err(config_error(format!(
                            "permissions.{}.filesystem cannot contain an empty child path",
                            profile
                        )));
                    }
                    value.validate(profile)?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FilesystemAccess {
    Read,
    Write,
    Deny,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct FilesystemOperationRule {
    pub read: Option<PermissionDecision>,
    pub write: Option<PermissionDecision>,
}

impl FilesystemOperationRule {
    fn validate(&self, profile: &str) -> crate::error::Result<()> {
        if self.read.is_none() && self.write.is_none() {
            return Err(config_error(format!(
                "permissions.{}.filesystem operation rules must set read or write",
                profile
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileNetworkConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub default: PermissionDecision,
    #[serde(default)]
    pub domains: BTreeMap<String, PermissionDecision>,
    #[serde(default)]
    pub allow_local_binding: bool,
    pub audit_log: Option<PathBuf>,
}

impl Default for ProfileNetworkConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            default: PermissionDecision::Prompt,
            domains: BTreeMap::new(),
            allow_local_binding: false,
            audit_log: None,
        }
    }
}

impl ProfileNetworkConfig {
    fn validate(&self, profile: &str) -> crate::error::Result<()> {
        for host in self.domains.keys() {
            if host.trim().is_empty() {
                return Err(config_error(format!(
                    "permissions.{}.network.domains cannot contain an empty host",
                    profile
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ProfileShellConfig {
    #[serde(default)]
    pub default: PermissionDecision,
    #[serde(default)]
    pub rules: Vec<ShellRuleConfig>,
}

impl ProfileShellConfig {
    fn validate(&self, profile: &str) -> crate::error::Result<()> {
        for rule in &self.rules {
            rule.validate(profile)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShellRuleConfig {
    pub pattern: Vec<ShellPatternTokenConfig>,
    pub decision: PermissionDecision,
    pub justification: Option<String>,
}

impl ShellRuleConfig {
    fn validate(&self, profile: &str) -> crate::error::Result<()> {
        if self.pattern.is_empty() {
            return Err(config_error(format!(
                "permissions.{}.shell.rules pattern cannot be empty",
                profile
            )));
        }
        for token in &self.pattern {
            token.validate(profile)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ShellPatternTokenConfig {
    Token(String),
    AnyOf(Vec<String>),
}

impl ShellPatternTokenConfig {
    fn validate(&self, profile: &str) -> crate::error::Result<()> {
        match self {
            Self::Token(value) if value.trim().is_empty() => Err(config_error(format!(
                "permissions.{}.shell.rules pattern cannot contain an empty token",
                profile
            ))),
            Self::AnyOf(values)
                if values.is_empty() || values.iter().any(|value| value.trim().is_empty()) =>
            {
                Err(config_error(format!(
                    "permissions.{}.shell.rules alternatives cannot be empty",
                    profile
                )))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ProfileToolConfig {
    #[serde(default)]
    pub rules: Vec<ToolRuleConfig>,
}

impl ProfileToolConfig {
    fn validate(&self, profile: &str) -> crate::error::Result<()> {
        for rule in &self.rules {
            rule.validate(profile)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolRuleConfig {
    pub tool: String,
    pub decision: PermissionDecision,
    pub action: Option<String>,
    pub target: Option<String>,
    pub justification: Option<String>,
}

impl ToolRuleConfig {
    fn validate(&self, profile: &str) -> crate::error::Result<()> {
        if self.tool.trim().is_empty() {
            return Err(config_error(format!(
                "permissions.{}.tool.rules tool cannot be empty",
                profile
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionDecision {
    Allow,
    #[default]
    Prompt,
    Deny,
}

fn default_policy_files() -> Vec<PathBuf> {
    vec![default_policy_dir().join("default.star")]
}

fn default_policy_dir() -> PathBuf {
    crate::paths::home_dir()
        .join(".config")
        .join("deepcode")
        .join("policies")
}

fn is_builtin_permission_profile(name: &str) -> bool {
    matches!(name, ":read-only" | ":workspace" | ":danger-full-access")
}

fn validate_profile_reference(name: &str) -> crate::error::Result<()> {
    if name.trim().is_empty() {
        return Err(config_error("permission profile names cannot be empty"));
    }
    if name.starts_with(':') && !is_builtin_permission_profile(name) {
        return Err(config_error(format!(
            "Unknown built-in permission profile '{}'",
            name
        )));
    }
    Ok(())
}

fn validate_profile_name(name: &str) -> crate::error::Result<()> {
    if name.trim().is_empty() || name.starts_with(':') {
        return Err(config_error(format!(
            "Invalid permission profile name '{}'",
            name
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"
        active_provider = "deepseek"

        [providers.deepseek]
        type = "deepseek"
        api_key = "test-key"
        model = "deepseek-v4-pro"
        reasoning_effort = "high"

        [providers.deepseek.models."private-model"]
        context_window = 1000000
        max_output_tokens = 32768
        reasoning_efforts = ["off", "high", "max"]
    "#;

    #[test]
    fn provider_config_validates() {
        let config: DeepCodeConfig = toml::from_str(VALID).unwrap();
        config.validate().unwrap();
        let (name, provider) = config.resolve_provider(None).unwrap();
        assert_eq!(name, "deepseek");
        assert_eq!(provider.model.as_deref(), Some("deepseek-v4-pro"));
        assert!(provider.models.contains_key("private-model"));
    }

    #[test]
    fn unknown_top_level_fields_are_rejected() {
        let invalid = format!("unexpected = true\n{VALID}");
        assert!(toml::from_str::<DeepCodeConfig>(&invalid).is_err());
    }

    #[test]
    fn unknown_provider_fields_are_rejected() {
        let invalid = VALID.replace(
            "reasoning_effort = \"high\"",
            "reasoning_effort = \"high\"\n        temperature = 0.2",
        );
        assert!(toml::from_str::<DeepCodeConfig>(&invalid).is_err());
    }

    #[test]
    fn unknown_tools_fields_are_rejected() {
        let invalid = format!(
            "{}\n[tools]\nshell_allowed_commands = [\"ls\", \"cat\"]\n",
            VALID
        );
        assert!(toml::from_str::<DeepCodeConfig>(&invalid).is_err());
    }

    #[test]
    fn profile_permissions_config_validates() {
        let source = format!(
            "default_permissions = \"project\"\n{}\n[permissions.project]\nextends = \":workspace\"\n\n[permissions.project.filesystem.\":workspace_roots\"]\n\"Cargo.toml\" = {{ write = \"prompt\" }}\n\n[permissions.project.network]\nenabled = true\ndefault = \"prompt\"\n\n[permissions.project.network.domains]\n\"github.com\" = \"allow\"\n\n[[permissions.project.shell.rules]]\npattern = [\"git\", \"status\"]\ndecision = \"allow\"\njustification = \"read git state\"\n",
            VALID
        );
        let config = DeepCodeConfig::parse(&source).unwrap();
        assert_eq!(config.default_permissions.as_deref(), Some("project"));
        let profile = config.permissions.profiles.get("project").unwrap();
        assert_eq!(profile.extends.as_deref(), Some(":workspace"));
        assert_eq!(profile.shell.rules.len(), 1);
        assert_eq!(
            profile.network.domains.get("github.com"),
            Some(&PermissionDecision::Allow)
        );
    }

    #[test]
    fn example_config_uses_current_permissions_schema() {
        let source = include_str!("../../../config.example.toml");
        let config = DeepCodeConfig::parse(source).unwrap();
        assert_eq!(config.default_permissions.as_deref(), Some("project-edit"));
        assert!(config.permissions.profiles.contains_key("project-edit"));
    }

    #[test]
    fn unknown_permission_profile_fields_are_rejected() {
        let source = format!(
            "{}\n[permissions.project]\nallow_prefixes = [\"\"]\n",
            VALID
        );
        let err = DeepCodeConfig::parse(&source).unwrap_err();
        assert!(
            err.to_string().contains("allow_prefixes") || err.to_string().contains("unknown field")
        );
    }

    #[test]
    fn unknown_default_permission_profile_is_rejected() {
        let source = format!("default_permissions = \"missing\"\n{}", VALID);
        let err = DeepCodeConfig::parse(&source).unwrap_err();
        assert!(err.to_string().contains("unknown profile"));
    }

    #[test]
    fn single_provider_is_activated_automatically() {
        let config: DeepCodeConfig =
            toml::from_str(&VALID.replace("active_provider = \"deepseek\"\n\n", "")).unwrap();
        config.validate().unwrap();
        assert_eq!(config.active_provider_name().unwrap(), "deepseek");
    }

    #[test]
    fn deepseek_accepts_wire_api_selection() {
        let source = VALID.replace(
            "reasoning_effort = \"high\"",
            "reasoning_effort = \"high\"\n        wire_api = \"responses\"",
        );
        let config: DeepCodeConfig = toml::from_str(&source).unwrap();
        config.validate().unwrap();
        assert_eq!(
            config.providers["deepseek"].wire_api.as_deref(),
            Some("responses")
        );
    }

    #[test]
    fn inactive_provider_may_be_configured_before_its_key() {
        let source = format!(
            "{}\n[providers.later]\ntype = \"openai\"\nbase_url = \"https://gateway.test/v1\"\n",
            VALID
        );
        let config = DeepCodeConfig::parse(&source).unwrap();
        assert!(config.providers["later"].resolve_api_key().is_none());
    }

    #[test]
    fn kimi_provider_accepts_code_models() {
        let source = r#"
            [providers.membership]
            type = "kimi"
            api_key = "test-key"
            model = "k3-256k"
            reasoning_effort = "high"
        "#;
        let config = DeepCodeConfig::parse(source).unwrap();

        assert_eq!(config.providers["membership"].kind, "kimi");
        assert_eq!(
            config.providers["membership"].model.as_deref(),
            Some("k3-256k")
        );
    }

    #[test]
    fn kimi_code_alias_is_rejected() {
        let error = DeepCodeConfig::parse(
            r#"
                [providers.membership]
                type = "kimi-code"
                api_key = "test-key"
            "#,
        )
        .unwrap_err();

        assert!(error.to_string().contains("unsupported type 'kimi-code'"));
    }
}
