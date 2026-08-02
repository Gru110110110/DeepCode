use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use deepcode_core::error::{DeepCodeError, Result};
use serde::{Deserialize, Serialize};

#[cfg(target_os = "windows")]
mod windows;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilesystemAccess {
    Read,
    Write,
    Deny,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FilesystemSandboxRule {
    pub path: PathBuf,
    pub access: FilesystemAccess,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxPolicy {
    DangerFullAccess,
    ReadOnly {
        network_access: bool,
    },
    WorkspaceWrite {
        writable_roots: Vec<PathBuf>,
        network_access: bool,
    },
    Profile {
        filesystem: Vec<FilesystemSandboxRule>,
        network_access: bool,
        label: String,
    },
    ExternalSandbox {
        name: String,
        network_access: bool,
    },
}

impl SandboxPolicy {
    pub fn workspace_write(root: impl Into<PathBuf>) -> Self {
        Self::WorkspaceWrite {
            writable_roots: vec![root.into()],
            network_access: false,
        }
    }

    pub fn network_access(&self) -> bool {
        match self {
            Self::DangerFullAccess => true,
            Self::ReadOnly { network_access }
            | Self::WorkspaceWrite { network_access, .. }
            | Self::Profile { network_access, .. }
            | Self::ExternalSandbox { network_access, .. } => *network_access,
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Self::DangerFullAccess => "danger-full-access",
            Self::ReadOnly { .. } => "read-only",
            Self::WorkspaceWrite { .. } => "workspace-write",
            Self::Profile { label, .. } => label.as_str(),
            Self::ExternalSandbox { .. } => "external",
        }
    }
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self::ReadOnly {
            network_access: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxType {
    MacosSeatbelt,
    LinuxBwrap,
    LinuxLandlock,
    WindowsRestrictedToken,
    External,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandSpec {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
    pub policy: SandboxPolicy,
}

impl CommandSpec {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            cwd: None,
            env: BTreeMap::new(),
            policy: SandboxPolicy::default(),
        }
    }

    pub fn shell(
        command: impl Into<String>,
        cwd: impl Into<PathBuf>,
        policy: SandboxPolicy,
    ) -> Self {
        let (program, args) = shell_invocation(command.into());
        Self {
            program,
            args,
            cwd: Some(cwd.into()),
            env: BTreeMap::new(),
            policy,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecEnv {
    pub workspace_root: Option<PathBuf>,
    pub writable_roots: Vec<PathBuf>,
    pub tmp_roots: Vec<PathBuf>,
}

impl ExecEnv {
    pub fn from_workspace(root: Option<PathBuf>) -> Self {
        let mut tmp_roots = vec![std::env::temp_dir()];
        if let Ok(tmpdir) = std::env::var("TMPDIR") {
            tmp_roots.push(PathBuf::from(tmpdir));
        }
        tmp_roots.sort();
        tmp_roots.dedup();

        Self {
            workspace_root: root,
            writable_roots: Vec::new(),
            tmp_roots,
        }
    }

    fn effective_writable_roots(&self, policy: &SandboxPolicy, cwd: Option<&Path>) -> Vec<PathBuf> {
        if !matches!(policy, SandboxPolicy::WorkspaceWrite { .. }) {
            return Vec::new();
        }

        let mut roots = Vec::new();
        if let Some(root) = &self.workspace_root {
            roots.push(root.clone());
        }
        if let Some(cwd) = cwd {
            roots.push(cwd.to_path_buf());
        }
        roots.extend(self.writable_roots.iter().cloned());
        if let SandboxPolicy::WorkspaceWrite { writable_roots, .. } = policy {
            roots.extend(writable_roots.iter().cloned());
        }
        roots.extend(self.tmp_roots.iter().cloned());
        roots.sort();
        roots.dedup();
        roots
    }

    fn profile_rules(
        &self,
        policy: &SandboxPolicy,
        cwd: Option<&Path>,
    ) -> Vec<FilesystemSandboxRule> {
        match policy {
            SandboxPolicy::ReadOnly { .. } => {
                let mut rules = Vec::new();
                if let Some(root) = &self.workspace_root {
                    rules.push(FilesystemSandboxRule {
                        path: root.clone(),
                        access: FilesystemAccess::Read,
                    });
                }
                if let Some(cwd) = cwd {
                    rules.push(FilesystemSandboxRule {
                        path: cwd.to_path_buf(),
                        access: FilesystemAccess::Read,
                    });
                }
                rules
            }
            SandboxPolicy::WorkspaceWrite { .. } => self
                .effective_writable_roots(policy, cwd)
                .into_iter()
                .map(|path| FilesystemSandboxRule {
                    path,
                    access: FilesystemAccess::Write,
                })
                .collect(),
            SandboxPolicy::Profile { filesystem, .. } => filesystem.clone(),
            SandboxPolicy::DangerFullAccess | SandboxPolicy::ExternalSandbox { .. } => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparedCommand {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub env: BTreeMap<String, String>,
    pub sandboxed: bool,
    pub sandbox_type: Option<SandboxType>,
    pub sandbox_policy: SandboxPolicy,
    #[serde(default)]
    pub filesystem_rules: Vec<FilesystemSandboxRule>,
}

#[derive(Debug, Clone)]
pub struct SandboxManager {
    env: ExecEnv,
}

impl SandboxManager {
    pub fn new(env: ExecEnv) -> Self {
        Self { env }
    }

    pub fn prepare(&self, spec: CommandSpec) -> Result<PreparedCommand> {
        if matches!(spec.policy, SandboxPolicy::DangerFullAccess) {
            return Ok(PreparedCommand::unsandboxed(spec));
        }

        #[cfg(target_os = "macos")]
        {
            if seatbelt_available() {
                return self.prepare_macos_seatbelt(spec);
            }
        }

        #[cfg(target_os = "linux")]
        {
            if bwrap_available() {
                return self.prepare_linux_bwrap(spec);
            }
            if landlock_available() {
                return self.prepare_linux_landlock(spec);
            }
        }

        #[cfg(target_os = "windows")]
        {
            if windows_restricted_token_available() {
                return self.prepare_windows_restricted_token(spec);
            }
        }

        Err(sandbox_unavailable_error(&spec.policy))
    }

    #[cfg(target_os = "macos")]
    fn prepare_macos_seatbelt(&self, spec: CommandSpec) -> Result<PreparedCommand> {
        let profile = macos_sbpl(
            &spec.policy,
            &self.env.profile_rules(&spec.policy, spec.cwd.as_deref()),
        );
        let original_program = spec.program.clone();
        let mut args = vec!["-p".to_string(), profile, original_program];
        args.extend(spec.args.clone());
        Ok(PreparedCommand {
            program: "/usr/bin/sandbox-exec".to_string(),
            args,
            cwd: spec.cwd,
            env: spec.env,
            sandboxed: true,
            sandbox_type: Some(SandboxType::MacosSeatbelt),
            sandbox_policy: spec.policy,
            filesystem_rules: Vec::new(),
        })
    }

    #[cfg(target_os = "linux")]
    fn prepare_linux_bwrap(&self, spec: CommandSpec) -> Result<PreparedCommand> {
        let mut args = vec!["--die-with-parent".to_string()];
        if !matches!(spec.policy, SandboxPolicy::Profile { .. }) {
            args.extend(["--ro-bind".to_string(), "/".to_string(), "/".to_string()]);
        }
        args.extend([
            "--proc".to_string(),
            "/proc".to_string(),
            "--dev".to_string(),
            "/dev".to_string(),
        ]);

        if !spec.policy.network_access() {
            args.push("--unshare-net".to_string());
        }

        let profile_rules = self.env.profile_rules(&spec.policy, spec.cwd.as_deref());
        for rule in profile_rules
            .iter()
            .filter(|rule| !matches!(rule.access, FilesystemAccess::Deny))
        {
            if !rule.path.exists() {
                continue;
            }
            let p = rule.path.display().to_string();
            match rule.access {
                FilesystemAccess::Read => args.extend(["--ro-bind".to_string(), p.clone(), p]),
                FilesystemAccess::Write => args.extend(["--bind".to_string(), p.clone(), p]),
                FilesystemAccess::Deny => {}
            }
        }
        for rule in profile_rules
            .iter()
            .filter(|rule| matches!(rule.access, FilesystemAccess::Deny))
        {
            if !rule.path.exists() {
                continue;
            }
            let p = rule.path.display().to_string();
            if rule.path.is_dir() {
                args.extend(["--tmpfs".to_string(), p]);
            } else {
                args.extend(["--ro-bind".to_string(), "/dev/null".to_string(), p]);
            }
        }

        if let Some(cwd) = &spec.cwd {
            args.extend(["--chdir".to_string(), cwd.display().to_string()]);
        }

        args.push("--".to_string());
        args.push(spec.program.clone());
        args.extend(spec.args.clone());
        Ok(PreparedCommand {
            program: "bwrap".to_string(),
            args,
            cwd: spec.cwd,
            env: spec.env,
            sandboxed: true,
            sandbox_type: Some(SandboxType::LinuxBwrap),
            sandbox_policy: spec.policy,
            filesystem_rules: Vec::new(),
        })
    }

    #[cfg(target_os = "linux")]
    fn prepare_linux_landlock(&self, spec: CommandSpec) -> Result<PreparedCommand> {
        Ok(PreparedCommand {
            sandboxed: false,
            sandbox_type: Some(SandboxType::LinuxLandlock),
            ..PreparedCommand::unsandboxed(spec)
        })
    }

    #[cfg(target_os = "windows")]
    fn prepare_windows_restricted_token(&self, mut spec: CommandSpec) -> Result<PreparedCommand> {
        if !spec.policy.network_access() {
            windows::apply_no_network_environment(&mut spec.env);
        }
        let filesystem_rules = self.env.profile_rules(&spec.policy, spec.cwd.as_deref());
        Ok(PreparedCommand {
            program: spec.program,
            args: spec.args,
            cwd: spec.cwd,
            env: spec.env,
            sandboxed: true,
            sandbox_type: Some(SandboxType::WindowsRestrictedToken),
            sandbox_policy: spec.policy,
            filesystem_rules,
        })
    }
}

impl PreparedCommand {
    fn unsandboxed(spec: CommandSpec) -> Self {
        Self {
            program: spec.program,
            args: spec.args,
            cwd: spec.cwd,
            env: spec.env,
            sandboxed: false,
            sandbox_type: None,
            sandbox_policy: spec.policy,
            filesystem_rules: Vec::new(),
        }
    }
}

fn shell_invocation(command: String) -> (String, Vec<String>) {
    #[cfg(target_os = "windows")]
    {
        let shell = std::env::var("DEEPCODE_SHELL").unwrap_or_else(|_| "powershell.exe".into());
        let name = Path::new(&shell)
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let args = if matches!(name.as_str(), "powershell" | "pwsh") {
            vec![
                "-NoLogo".into(),
                "-NoProfile".into(),
                "-NonInteractive".into(),
                "-Command".into(),
                command,
            ]
        } else if name == "cmd" {
            vec!["/D".into(), "/S".into(), "/C".into(), command]
        } else {
            vec!["-c".into(), command]
        };
        (shell, args)
    }
    #[cfg(not(target_os = "windows"))]
    {
        (
            std::env::var("SHELL").unwrap_or_else(|_| "sh".to_string()),
            vec!["-c".to_string(), command],
        )
    }
}

#[cfg(target_os = "macos")]
fn seatbelt_available() -> bool {
    Path::new("/usr/bin/sandbox-exec").exists()
}

#[cfg(target_os = "macos")]
fn macos_sbpl(policy: &SandboxPolicy, filesystem_rules: &[FilesystemSandboxRule]) -> String {
    let mut profile = String::from(
        "(version 1)\n\
         (deny default)\n\
         (allow process*)\n\
         (allow signal (target self))\n\
         (allow file-map-executable)\n\
         (allow file-write-data (literal \"/dev/null\"))\n\
         (allow sysctl-read)\n\
         (allow mach-lookup)\n\
         (allow ipc*)\n",
    );

    if policy.network_access() {
        profile.push_str("(allow network*)\n");
    } else {
        profile.push_str("(deny network*)\n");
    }

    if matches!(
        policy,
        SandboxPolicy::ReadOnly { .. } | SandboxPolicy::WorkspaceWrite { .. }
    ) {
        profile.push_str("(allow file-read*)\n");
    }

    if matches!(policy, SandboxPolicy::Profile { .. }) {
        for rule in filesystem_rules {
            match rule.access {
                FilesystemAccess::Read => profile.push_str(&format!(
                    "(allow file-read* (subpath \"{}\"))\n",
                    sbpl_escape(&rule.path)
                )),
                FilesystemAccess::Write => profile.push_str(&format!(
                    "(allow file-read* file-write* (subpath \"{}\"))\n",
                    sbpl_escape(&rule.path)
                )),
                FilesystemAccess::Deny => profile.push_str(&format!(
                    "(deny file-read* file-write* (subpath \"{}\"))\n",
                    sbpl_escape(&rule.path)
                )),
            }
        }
    } else if matches!(policy, SandboxPolicy::WorkspaceWrite { .. }) {
        for rule in filesystem_rules {
            if matches!(rule.access, FilesystemAccess::Write) {
                profile.push_str(&format!(
                    "(allow file-write* (subpath \"{}\"))\n",
                    sbpl_escape(&rule.path)
                ));
            }
        }
    }

    profile
}

#[cfg(target_os = "macos")]
fn sbpl_escape(path: &Path) -> String {
    path.display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
}

#[cfg(target_os = "linux")]
fn bwrap_available() -> bool {
    Path::new("/usr/bin/bwrap").exists() || Path::new("/bin/bwrap").exists()
}

#[cfg(target_os = "linux")]
fn landlock_available() -> bool {
    false
}

#[cfg(target_os = "windows")]
fn windows_restricted_token_available() -> bool {
    true
}

/// Executes a prepared command and enforces the timeout across its process tree.
pub async fn execute_prepared(
    prepared: &PreparedCommand,
    timeout: Duration,
) -> Result<std::process::Output> {
    #[cfg(target_os = "windows")]
    if matches!(
        prepared.sandbox_type,
        Some(SandboxType::WindowsRestrictedToken)
    ) {
        let prepared = prepared.clone();
        return tokio::task::spawn_blocking(move || windows::execute(&prepared, timeout))
            .await
            .map_err(|error| DeepCodeError::ToolExecution {
                tool: "sandbox".to_string(),
                message: format!("Windows sandbox worker failed: {error}"),
            })?;
    }

    let mut command = tokio::process::Command::new(&prepared.program);
    command.kill_on_drop(true);
    command.args(&prepared.args);
    if let Some(cwd) = &prepared.cwd {
        command.current_dir(cwd);
    }
    command.envs(&prepared.env);
    tokio::time::timeout(timeout, command.output())
        .await
        .map_err(|_| DeepCodeError::ToolExecution {
            tool: "sandbox".to_string(),
            message: format!("Command timed out after {} seconds", timeout.as_secs_f64()),
        })?
        .map_err(|error| DeepCodeError::ToolExecution {
            tool: "sandbox".to_string(),
            message: format!("Failed to execute command: {error}"),
        })
}

pub fn sandbox_unavailable_error(policy: &SandboxPolicy) -> DeepCodeError {
    DeepCodeError::ToolExecution {
        tool: "sandbox".to_string(),
        message: format!(
            "No real sandbox backend is available for policy {} on this platform",
            policy.label()
        ),
    }
}

/// Returns whether a shell command contains substitution or process substitution syntax.
pub fn contains_dynamic_shell_syntax(command: &str) -> bool {
    let mut chars = command.chars().peekable();
    let mut single_quoted = false;
    let mut escaped = false;

    while let Some(ch) = chars.next() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' && !single_quoted {
            escaped = true;
            continue;
        }
        if ch == '\'' {
            single_quoted = !single_quoted;
            continue;
        }
        if single_quoted {
            continue;
        }
        if ch == '`' || (matches!(ch, '$' | '<' | '>') && matches!(chars.peek(), Some('('))) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn danger_full_access_is_never_marked_sandboxed() {
        let manager = SandboxManager::new(ExecEnv::default());
        let mut spec = CommandSpec::new("echo");
        spec.policy = SandboxPolicy::DangerFullAccess;
        let prepared = manager.prepare(spec).unwrap();
        assert!(!prepared.sandboxed);
        assert!(prepared.sandbox_type.is_none());
    }

    #[test]
    fn read_only_policy_has_no_writable_roots() {
        let workspace = PathBuf::from("/workspace");
        let cwd = workspace.join("project");
        let env = ExecEnv {
            workspace_root: Some(workspace),
            writable_roots: vec![PathBuf::from("/configured-write")],
            tmp_roots: vec![PathBuf::from("/tmp")],
        };

        assert!(env
            .effective_writable_roots(
                &SandboxPolicy::ReadOnly {
                    network_access: false,
                },
                Some(&cwd),
            )
            .is_empty());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_read_only_profile_allows_dev_null() {
        let profile = macos_sbpl(
            &SandboxPolicy::ReadOnly {
                network_access: false,
            },
            &[],
        );

        assert!(profile.contains("(allow file-write-data (literal \"/dev/null\"))"));
        assert!(!profile.contains("(allow file-write* (subpath \"/workspace\"))"));
    }

    #[test]
    fn workspace_write_policy_includes_effective_roots() {
        let workspace = PathBuf::from("/workspace");
        let cwd = workspace.join("project");
        let env = ExecEnv {
            workspace_root: Some(workspace.clone()),
            writable_roots: vec![PathBuf::from("/configured-write")],
            tmp_roots: vec![PathBuf::from("/tmp")],
        };
        let roots = env.effective_writable_roots(
            &SandboxPolicy::WorkspaceWrite {
                writable_roots: vec![PathBuf::from("/policy-write")],
                network_access: false,
            },
            Some(&cwd),
        );

        assert!(roots.contains(&workspace));
        assert!(roots.contains(&cwd));
        assert!(roots.contains(&PathBuf::from("/configured-write")));
        assert!(roots.contains(&PathBuf::from("/policy-write")));
        assert!(roots.contains(&PathBuf::from("/tmp")));
    }

    #[test]
    fn dynamic_shell_syntax_respects_quotes_and_escapes() {
        assert!(contains_dynamic_shell_syntax("echo $(whoami)"));
        assert!(contains_dynamic_shell_syntax("echo `whoami`"));
        assert!(!contains_dynamic_shell_syntax("echo '$(whoami)'"));
        assert!(!contains_dynamic_shell_syntax(r"echo \$(whoami)"));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_shell_uses_powershell_command_arguments() {
        let spec = CommandSpec::shell(
            "Write-Output hello",
            std::env::current_dir().unwrap(),
            SandboxPolicy::DangerFullAccess,
        );
        assert_eq!(spec.program, "powershell.exe");
        assert_eq!(
            spec.args,
            [
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Write-Output hello"
            ]
        );
    }

    #[cfg(target_os = "windows")]
    #[tokio::test]
    async fn windows_workspace_token_allows_only_configured_write_root() {
        let nonce = format!(
            "{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let base = std::env::current_dir()
            .unwrap()
            .join(format!(".deepcode_windows_sandbox_{nonce}"));
        let workspace = base.join("workspace");
        let outside = base.join("outside");
        std::fs::create_dir_all(&workspace).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let manager = SandboxManager::new(ExecEnv::from_workspace(Some(workspace.clone())));
        let allowed = workspace.join("allowed.txt");
        let blocked = outside.join("blocked.txt");
        let command = format!(
            "Set-Content -LiteralPath '{}' -Value allowed -ErrorAction Stop; \
             Set-Content -LiteralPath '{}' -Value blocked -ErrorAction Stop",
            allowed.display().to_string().replace('\'', "''"),
            blocked.display().to_string().replace('\'', "''")
        );
        let spec = CommandSpec::shell(
            command,
            &workspace,
            SandboxPolicy::WorkspaceWrite {
                writable_roots: vec![workspace.clone()],
                network_access: false,
            },
        );
        let prepared = manager.prepare(spec).unwrap();
        let output = execute_prepared(&prepared, Duration::from_secs(10))
            .await
            .unwrap();

        assert!(!output.status.success());
        assert!(allowed.exists());
        assert!(!blocked.exists());
        std::fs::remove_dir_all(&base).unwrap();
    }
}
