use std::fs;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use chrono::Utc;
use crossterm::cursor::{Hide, RestorePosition, SavePosition, Show};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType};
use crossterm::{execute, queue};
use serde::{Deserialize, Serialize};

const TRUST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrustDecision {
    Trusted,
    Exit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WorkspaceIdentity {
    path: PathBuf,
    device: Option<u64>,
    inode: Option<u64>,
}

impl WorkspaceIdentity {
    fn resolve(path: &Path) -> anyhow::Result<Self> {
        let path = fs::canonicalize(path)
            .with_context(|| format!("Cannot access workspace {}", path.display()))?;
        let metadata = fs::metadata(&path)
            .with_context(|| format!("Cannot inspect workspace {}", path.display()))?;
        if !metadata.is_dir() {
            anyhow::bail!("Workspace is not a directory: {}", path.display());
        }

        #[cfg(unix)]
        let (device, inode) = {
            use std::os::unix::fs::MetadataExt;
            (Some(metadata.dev()), Some(metadata.ino()))
        };
        #[cfg(not(unix))]
        let (device, inode) = (None, None);

        Ok(Self {
            path,
            device,
            inode,
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TrustDatabase {
    schema_version: u32,
    workspaces: Vec<TrustedWorkspace>,
}

impl Default for TrustDatabase {
    fn default() -> Self {
        Self {
            schema_version: TRUST_SCHEMA_VERSION,
            workspaces: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct TrustedWorkspace {
    path: PathBuf,
    device: Option<u64>,
    inode: Option<u64>,
    trusted_at: String,
}

impl TrustedWorkspace {
    fn matches(&self, workspace: &WorkspaceIdentity) -> bool {
        if self.path != workspace.path {
            return false;
        }

        #[cfg(unix)]
        return self.device == workspace.device
            && self.inode == workspace.inode
            && self.device.is_some()
            && self.inode.is_some();

        #[cfg(not(unix))]
        return true;
    }
}

#[derive(Debug, Clone)]
struct WorkspaceTrustStore {
    path: PathBuf,
}

impl WorkspaceTrustStore {
    fn default() -> Self {
        let base = std::env::var_os("DEEPCODE_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| home_dir().join(".local/share/deepcode"));
        Self {
            path: base.join("trusted_workspaces.json"),
        }
    }

    #[cfg(test)]
    fn at(path: PathBuf) -> Self {
        Self { path }
    }

    fn is_trusted(&self, workspace: &WorkspaceIdentity) -> anyhow::Result<bool> {
        Ok(self
            .load()?
            .workspaces
            .iter()
            .any(|trusted| trusted.matches(workspace)))
    }

    fn trust(&self, workspace: &WorkspaceIdentity) -> anyhow::Result<()> {
        let mut database = self.load()?;
        database
            .workspaces
            .retain(|trusted| trusted.path != workspace.path);
        database.workspaces.push(TrustedWorkspace {
            path: workspace.path.clone(),
            device: workspace.device,
            inode: workspace.inode,
            trusted_at: Utc::now().to_rfc3339(),
        });
        self.save(&database)
    }

    fn load(&self) -> anyhow::Result<TrustDatabase> {
        let data = match fs::read(&self.path) {
            Ok(data) => data,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(TrustDatabase::default());
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("Cannot read workspace trust store {}", self.path.display())
                });
            }
        };
        let database: TrustDatabase = serde_json::from_slice(&data).with_context(|| {
            format!("Workspace trust store is invalid: {}", self.path.display())
        })?;
        if database.schema_version != TRUST_SCHEMA_VERSION {
            anyhow::bail!(
                "Unsupported workspace trust schema version {} in {}",
                database.schema_version,
                self.path.display()
            );
        }
        Ok(database)
    }

    fn save(&self, database: &TrustDatabase) -> anyhow::Result<()> {
        let parent = self
            .path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("Workspace trust store has no parent directory"))?;
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "Cannot create workspace trust directory {}",
                parent.display()
            )
        })?;

        let temporary = parent.join(format!(".trusted_workspaces.{}.tmp", std::process::id()));
        fs::write(&temporary, serde_json::to_vec_pretty(database)?)?;
        restrict_file_permissions(&temporary)?;
        replace_file(&temporary, &self.path)?;
        Ok(())
    }
}

pub(crate) fn ensure_current_workspace_trusted() -> anyhow::Result<TrustDecision> {
    ensure_workspace_trusted(&std::env::current_dir()?)
}

pub(crate) fn ensure_workspace_trusted(path: &Path) -> anyhow::Result<TrustDecision> {
    let workspace = WorkspaceIdentity::resolve(path)?;
    let store = WorkspaceTrustStore::default();
    if store.is_trusted(&workspace)? {
        return Ok(TrustDecision::Trusted);
    }

    if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
        anyhow::bail!(
            "Workspace is not trusted: {}. Run DeepCode from an interactive terminal once to review and trust this folder.",
            workspace.path.display()
        );
    }

    match prompt_for_trust(&workspace.path)? {
        TrustDecision::Trusted => {
            store.trust(&workspace)?;
            Ok(TrustDecision::Trusted)
        }
        TrustDecision::Exit => Ok(TrustDecision::Exit),
    }
}

fn prompt_for_trust(workspace: &Path) -> anyhow::Result<TrustDecision> {
    let mut stderr = io::stderr();
    writeln!(stderr)?;
    execute!(
        stderr,
        SetForegroundColor(Color::DarkYellow),
        SetAttribute(Attribute::Bold),
        Print("Accessing workspace:\n\n"),
        ResetColor,
        SetAttribute(Attribute::Reset),
        SetAttribute(Attribute::Bold),
        Print(format!("{}\n\n", workspace.display())),
        SetAttribute(Attribute::Reset),
    )?;
    writeln!(
        stderr,
        "Quick safety check: Is this a project you created or one you trust?"
    )?;
    writeln!(
        stderr,
        "If not, take a moment to review what is in this folder first.\n"
    )?;
    writeln!(
        stderr,
        "DeepCode will be able to read, edit, and execute files here.\n"
    )?;
    execute!(
        stderr,
        SetForegroundColor(Color::DarkGrey),
        Print("Security guide\n\n"),
        ResetColor
    )?;
    terminal::enable_raw_mode()?;
    let _guard = PromptCleanup;
    execute!(stderr, SavePosition, Hide)?;
    let mut selected = 0usize;
    render_choices(&mut stderr, selected)?;

    loop {
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        match trust_key(key, selected) {
            PromptAction::Select(next) => {
                selected = next;
                execute!(stderr, RestorePosition, Clear(ClearType::FromCursorDown))?;
                render_choices(&mut stderr, selected)?;
            }
            PromptAction::Confirm(decision) => {
                writeln!(stderr)?;
                return Ok(decision);
            }
            PromptAction::Ignore => {}
        }
    }
}

fn render_choices(stderr: &mut io::Stderr, selected: usize) -> anyhow::Result<()> {
    for (index, label) in ["1. Yes, I trust this folder", "2. No, exit"]
        .iter()
        .enumerate()
    {
        if index == selected {
            queue!(
                stderr,
                SetForegroundColor(Color::Blue),
                SetAttribute(Attribute::Bold),
                Print(format!("> {}\r\n", label)),
                ResetColor,
                SetAttribute(Attribute::Reset)
            )?;
        } else {
            queue!(stderr, Print(format!("  {}\r\n", label)))?;
        }
    }
    queue!(
        stderr,
        Print("\r\n"),
        SetForegroundColor(Color::DarkGrey),
        Print("Enter to confirm · Esc to cancel\r\n"),
        ResetColor
    )?;
    stderr.flush()?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PromptAction {
    Select(usize),
    Confirm(TrustDecision),
    Ignore,
}

fn trust_key(key: KeyEvent, selected: usize) -> PromptAction {
    match key.code {
        KeyCode::Up | KeyCode::Char('k') => PromptAction::Select(0),
        KeyCode::Down | KeyCode::Char('j') | KeyCode::Tab => PromptAction::Select(1),
        KeyCode::Char('1') | KeyCode::Char('y') | KeyCode::Char('Y') => {
            PromptAction::Confirm(TrustDecision::Trusted)
        }
        KeyCode::Char('2') | KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Char('q') => {
            PromptAction::Confirm(TrustDecision::Exit)
        }
        KeyCode::Enter => PromptAction::Confirm(if selected == 0 {
            TrustDecision::Trusted
        } else {
            TrustDecision::Exit
        }),
        KeyCode::Esc => PromptAction::Confirm(TrustDecision::Exit),
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            PromptAction::Confirm(TrustDecision::Exit)
        }
        _ => PromptAction::Ignore,
    }
}

struct PromptCleanup;

impl Drop for PromptCleanup {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(
            io::stderr(),
            Show,
            ResetColor,
            SetAttribute(Attribute::Reset)
        );
    }
}

fn home_dir() -> PathBuf {
    deepcode_core::paths::home_dir()
}

#[cfg(unix)]
fn restrict_file_permissions(path: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn restrict_file_permissions(_path: &Path) -> anyhow::Result<()> {
    Ok(())
}

#[cfg(windows)]
fn replace_file(source: &Path, destination: &Path) -> anyhow::Result<()> {
    if destination.exists() {
        fs::remove_file(destination)?;
    }
    fs::rename(source, destination)?;
    Ok(())
}

#[cfg(not(windows))]
fn replace_file(source: &Path, destination: &Path) -> anyhow::Result<()> {
    fs::rename(source, destination)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "deepcode_workspace_trust_{}_{}_{}",
            name,
            std::process::id(),
            uuid::Uuid::new_v4()
        ))
    }

    #[test]
    fn trust_is_persisted_for_the_same_workspace_identity() {
        let root = temp_path("persist");
        let workspace = root.join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        let identity = WorkspaceIdentity::resolve(&workspace).unwrap();
        let store = WorkspaceTrustStore::at(root.join("trust.json"));

        assert!(!store.is_trusted(&identity).unwrap());
        store.trust(&identity).unwrap();
        assert!(store.is_trusted(&identity).unwrap());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_different_directory_identity_does_not_inherit_trust() {
        let root = temp_path("identity");
        let workspace = root.join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        let identity = WorkspaceIdentity::resolve(&workspace).unwrap();
        let store = WorkspaceTrustStore::at(root.join("trust.json"));
        store.trust(&identity).unwrap();

        #[cfg(unix)]
        let replacement = WorkspaceIdentity {
            inode: identity.inode.map(|inode| inode.saturating_add(1)),
            ..identity.clone()
        };
        #[cfg(not(unix))]
        let replacement = WorkspaceIdentity {
            path: root.join("other-workspace"),
            ..identity.clone()
        };

        assert!(!store.is_trusted(&replacement).unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn corrupt_store_is_not_silently_accepted() {
        let root = temp_path("corrupt");
        let workspace = root.join("workspace");
        fs::create_dir_all(&workspace).unwrap();
        let store = WorkspaceTrustStore::at(root.join("trust.json"));
        fs::write(&store.path, b"not json").unwrap();

        let identity = WorkspaceIdentity::resolve(&workspace).unwrap();
        assert!(store.is_trusted(&identity).is_err());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn keyboard_choices_are_fail_closed() {
        let none = KeyModifiers::NONE;
        assert_eq!(
            trust_key(KeyEvent::new(KeyCode::Enter, none), 0),
            PromptAction::Confirm(TrustDecision::Trusted)
        );
        assert_eq!(
            trust_key(KeyEvent::new(KeyCode::Enter, none), 1),
            PromptAction::Confirm(TrustDecision::Exit)
        );
        assert_eq!(
            trust_key(KeyEvent::new(KeyCode::Esc, none), 0),
            PromptAction::Confirm(TrustDecision::Exit)
        );
        assert_eq!(
            trust_key(KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL), 0),
            PromptAction::Confirm(TrustDecision::Exit)
        );
    }
}
