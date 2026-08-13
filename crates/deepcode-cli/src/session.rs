use std::fs;
use std::path::{Path, PathBuf};

use chrono::Utc;
use deepcode_core::types::{Message, Role};
use serde::{Deserialize, Serialize};

use crate::ui::ChatMessage;

pub(crate) const SESSION_SCHEMA_VERSION: u32 = 1;
const UNTITLED: &str = "Untitled session";
const TITLE_MAX_CHARS: usize = 60;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct SavedPendingPlan {
    pub plan_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SavedSession {
    pub schema_version: u32,
    pub id: String,
    pub title: String,
    #[serde(default)]
    pub title_generated: bool,
    pub workspace_root: String,
    pub created_at: String,
    pub updated_at: String,
    pub provider: String,
    pub model: String,
    pub reasoning_effort: String,
    #[serde(default)]
    pub plan_mode_enabled: bool,
    #[serde(default)]
    pub pending_plan: Option<SavedPendingPlan>,
    pub ui_messages: Vec<ChatMessage>,
    pub core_messages: Vec<Message>,
}

impl SavedSession {
    pub(crate) fn new(
        workspace_root: String,
        provider: String,
        model: String,
        reasoning_effort: String,
    ) -> Self {
        let now = Utc::now().to_rfc3339();
        Self {
            schema_version: SESSION_SCHEMA_VERSION,
            id: uuid::Uuid::new_v4().to_string(),
            title: UNTITLED.to_string(),
            title_generated: false,
            workspace_root,
            created_at: now.clone(),
            updated_at: now,
            provider,
            model,
            reasoning_effort,
            plan_mode_enabled: false,
            pending_plan: None,
            ui_messages: Vec::new(),
            core_messages: Vec::new(),
        }
    }

    pub(crate) fn refresh_metadata(&mut self) {
        if !self.title_generated {
            self.title = session_title(&self.ui_messages);
        }
        self.updated_at = Utc::now().to_rfc3339();
    }

    pub(crate) fn set_generated_title(&mut self, title: &str) -> bool {
        if self.title_generated {
            return false;
        }
        let Some(title) = clean_title(title) else {
            return false;
        };
        self.title = title;
        self.title_generated = true;
        true
    }

    pub(crate) fn has_user_message(&self) -> bool {
        self.ui_messages
            .iter()
            .any(|message| message.role == "user" && !message.content.trim().is_empty())
            || self.core_messages.iter().any(|message| {
                message.role == Role::User
                    && message
                        .content
                        .iter()
                        .filter_map(|block| block.as_text())
                        .any(|text| !text.trim().is_empty())
            })
    }

    pub(crate) fn summary(&self) -> SessionSummary {
        SessionSummary {
            id: self.id.clone(),
            title: self.title.clone(),
            workspace_root: self.workspace_root.clone(),
            updated_at: self.updated_at.clone(),
            provider: self.provider.clone(),
            model: self.model.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionSummary {
    pub id: String,
    pub title: String,
    pub workspace_root: String,
    pub updated_at: String,
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone)]
pub(crate) struct SessionStore {
    root: PathBuf,
}

impl SessionStore {
    pub(crate) fn default() -> Self {
        let root = std::env::var("DEEPCODE_DATA_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dirs_home().join(".local/share/deepcode"))
            .join("sessions");
        Self { root }
    }

    #[cfg(test)]
    pub(crate) fn at(root: PathBuf) -> Self {
        Self { root }
    }

    pub(crate) fn workspace_root() -> anyhow::Result<String> {
        let cwd = std::env::current_dir()?;
        Ok(fs::canonicalize(&cwd).unwrap_or(cwd).display().to_string())
    }

    pub(crate) fn save(&self, session: &mut SavedSession) -> anyhow::Result<()> {
        if !session.has_user_message() {
            return Ok(());
        }
        self.ensure_root()?;
        session.refresh_metadata();
        let destination = self.root.join(format!("{}.json", session.id));
        let temporary = self.root.join(format!(".{}.tmp", session.id));
        let data = serde_json::to_vec_pretty(session)?;
        fs::write(&temporary, data)?;
        restrict_file_permissions(&temporary)?;
        fs::rename(&temporary, &destination)?;
        Ok(())
    }

    pub(crate) fn load(&self, id: &str) -> anyhow::Result<SavedSession> {
        if uuid::Uuid::parse_str(id).is_err() {
            anyhow::bail!("Invalid session id: {}", id);
        }
        let path = self.root.join(format!("{}.json", id));
        let session: SavedSession = serde_json::from_slice(
            &fs::read(&path).map_err(|e| anyhow::anyhow!("Cannot read session '{}': {}", id, e))?,
        )?;
        validate_session(&session)?;
        Ok(session)
    }

    pub(crate) fn list(
        &self,
        workspace: Option<&str>,
        limit: Option<usize>,
    ) -> anyhow::Result<Vec<SessionSummary>> {
        if !self.root.exists() {
            return Ok(Vec::new());
        }
        let mut sessions = Vec::new();
        for entry in fs::read_dir(&self.root)? {
            let entry = match entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let Ok(data) = fs::read(&path) else {
                continue;
            };
            let Ok(session) = serde_json::from_slice::<SavedSession>(&data) else {
                tracing::warn!(path = %path.display(), "Ignoring corrupt session file");
                continue;
            };
            if validate_session(&session).is_err()
                || !session.has_user_message()
                || workspace.is_some_and(|root| session.workspace_root != root)
            {
                continue;
            }
            sessions.push(session.summary());
        }
        sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
        if let Some(limit) = limit {
            sessions.truncate(limit);
        }
        Ok(sessions)
    }

    pub(crate) fn latest(&self, workspace: &str) -> anyhow::Result<SavedSession> {
        let summary = self
            .list(Some(workspace), Some(1))?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("No sessions found for workspace {}", workspace))?;
        self.load(&summary.id)
    }

    fn ensure_root(&self) -> anyhow::Result<()> {
        fs::create_dir_all(&self.root)?;
        Ok(())
    }
}

fn validate_session(session: &SavedSession) -> anyhow::Result<()> {
    if session.schema_version != SESSION_SCHEMA_VERSION {
        anyhow::bail!(
            "Unsupported session schema version {}",
            session.schema_version
        );
    }
    if uuid::Uuid::parse_str(&session.id).is_err() {
        anyhow::bail!("Session contains invalid UUID");
    }
    if session.pending_plan.as_ref().is_some_and(|pending| {
        pending.plan_path.trim().is_empty() || !Path::new(&pending.plan_path).is_absolute()
    }) {
        anyhow::bail!("Session contains invalid pending plan path");
    }
    Ok(())
}

fn session_title(messages: &[ChatMessage]) -> String {
    let Some(text) = messages
        .iter()
        .find(|message| message.role == "user" && !message.content.trim().is_empty())
        .map(|message| {
            message
                .content
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
    else {
        return UNTITLED.to_string();
    };
    let count = text.chars().count();
    if count <= TITLE_MAX_CHARS {
        text
    } else {
        format!(
            "{}...",
            text.chars().take(TITLE_MAX_CHARS - 3).collect::<String>()
        )
    }
}

fn clean_title(title: &str) -> Option<String> {
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    let title = title
        .trim_matches(|character: char| matches!(character, '"' | '\'' | '`' | '#' | ':' | '：'))
        .trim();
    if title.is_empty() {
        return None;
    }
    let count = title.chars().count();
    Some(if count <= TITLE_MAX_CHARS {
        title.to_string()
    } else {
        format!(
            "{}...",
            title.chars().take(TITLE_MAX_CHARS - 3).collect::<String>()
        )
    })
}

fn dirs_home() -> PathBuf {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> (SessionStore, PathBuf) {
        let root = std::env::temp_dir().join(format!("deepcode_sessions_{}", uuid::Uuid::new_v4()));
        (SessionStore::at(root.clone()), root)
    }

    fn session(workspace: &str, title: &str) -> SavedSession {
        let mut session = SavedSession::new(
            workspace.to_string(),
            "deepseek".to_string(),
            "deepseek-v4-pro".to_string(),
            "high".to_string(),
        );
        session.ui_messages.push(ChatMessage {
            role: "user".to_string(),
            content: title.to_string(),
        });
        session
    }

    #[test]
    fn save_load_and_filter_by_workspace() {
        let (store, root) = store();
        let mut first = session("/one", "First task");
        let mut second = session("/two", "Second task");
        store.save(&mut first).unwrap();
        store.save(&mut second).unwrap();

        assert_eq!(store.load(&first.id).unwrap().title, "First task");
        assert_eq!(store.list(Some("/one"), None).unwrap().len(), 1);
        assert_eq!(store.list(None, None).unwrap().len(), 2);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn pending_plan_and_plan_mode_round_trip() {
        let (store, root) = store();
        let mut session = session("/one", "Pending plan");
        session.plan_mode_enabled = true;
        session.pending_plan = Some(SavedPendingPlan {
            plan_path: "/tmp/plan-00000000-0000-0000-0000-000000000001.md".to_string(),
        });

        store.save(&mut session).unwrap();
        let saved = store.load(&session.id).unwrap();

        assert!(saved.plan_mode_enabled);
        assert_eq!(saved.pending_plan, session.pending_plan);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn legacy_session_defaults_to_no_pending_plan() {
        let session = session("/one", "Legacy session");
        let mut value = serde_json::to_value(&session).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("plan_mode_enabled");
        object.remove("pending_plan");

        let restored: SavedSession = serde_json::from_value(value).unwrap();

        assert!(!restored.plan_mode_enabled);
        assert!(restored.pending_plan.is_none());
    }

    #[test]
    fn empty_session_is_not_saved_or_listed() {
        let (store, root) = store();
        let mut empty = SavedSession::new(
            "/one".to_string(),
            "deepseek".to_string(),
            "model".to_string(),
            "high".to_string(),
        );

        store.save(&mut empty).unwrap();

        assert!(store.load(&empty.id).is_err());
        assert!(store.list(None, None).unwrap().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn generated_title_is_cleaned_and_not_overwritten() {
        let (store, root) = store();
        let mut session = session(
            "/one",
            "A very long initial request that is only a fallback",
        );
        assert!(session.set_generated_title("  `Parser   recovery summary`  "));
        assert!(!session.set_generated_title("Second title"));

        store.save(&mut session).unwrap();

        let saved = store.load(&session.id).unwrap();
        assert_eq!(saved.title, "Parser recovery summary");
        assert!(saved.title_generated);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn corrupt_files_are_ignored() {
        let (store, root) = store();
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("broken.json"), b"not json").unwrap();
        assert!(store.list(None, None).unwrap().is_empty());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn latest_is_sorted_by_updated_at() {
        let (store, root) = store();
        let mut first = session("/one", "First");
        let mut second = session("/one", "Second");
        first.updated_at = "2026-01-01T00:00:00Z".to_string();
        second.updated_at = "2026-02-01T00:00:00Z".to_string();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join(format!("{}.json", first.id)),
            serde_json::to_vec(&first).unwrap(),
        )
        .unwrap();
        fs::write(
            root.join(format!("{}.json", second.id)),
            serde_json::to_vec(&second).unwrap(),
        )
        .unwrap();
        assert_eq!(store.latest("/one").unwrap().id, second.id);
        let _ = fs::remove_dir_all(root);
    }
}
