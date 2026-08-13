use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use crate::session::{SavedSession, SessionStore};
use crate::workspace_trust::{
    ensure_current_workspace_trusted, ensure_workspace_trusted, TrustDecision,
};
use deepcode_agent::event;
use deepcode_core::config::{DeepCodeConfig, ModelProfile, ReasoningEffort};
use deepcode_core::error::DeepCodeError;
use deepcode_core::provider::traits::LlmProvider;
use deepcode_core::types::Message;
use deepcode_permissions::pipeline::PermissionSystem;
use deepcode_permissions::policy::{ApprovalScope, PermissionSystemConfig};
use deepcode_tools::builtins;
use deepcode_tools::registry::ToolRegistry;

fn dirs_home() -> PathBuf {
    deepcode_core::paths::home_dir()
}

pub(crate) fn default_config_path() -> PathBuf {
    dirs_home()
        .join(".config")
        .join("deepcode")
        .join("config.toml")
}

fn expand_home(path: &Path) -> PathBuf {
    if path.starts_with("~") {
        dirs_home().join(path.strip_prefix("~").unwrap_or(path))
    } else {
        path.to_path_buf()
    }
}

fn permissions_config(config: &DeepCodeConfig) -> PermissionSystemConfig {
    let mut policy_files = config
        .permissions
        .policy_files
        .iter()
        .map(|path| expand_home(path.as_path()))
        .collect::<Vec<_>>();
    if policy_files.is_empty() {
        policy_files = deepcode_permissions::policy::default_policy_files()
            .into_iter()
            .map(|p| expand_home(&p))
            .collect();
    }

    let write_policy_file = config
        .permissions
        .write_policy_file
        .as_deref()
        .map(expand_home)
        .or_else(|| Some(deepcode_permissions::policy::default_write_policy_file()));
    let grants_file = Some(deepcode_permissions::policy::default_grants_file());

    PermissionSystemConfig {
        default_permissions: config
            .default_permissions
            .clone()
            .or_else(|| Some(":workspace".to_string())),
        permissions: config.permissions.clone(),
        policy_files,
        write_policy_file,
        grants_file,
    }
}

fn register_subagent_tools(
    registry: &mut ToolRegistry,
    manager: Arc<deepcode_agent::subagent::AgentTaskManager>,
    tools_config: &deepcode_core::config::ToolsConfig,
) {
    if tool_enabled(tools_config, "spawn_agent") {
        registry.register(Arc::new(deepcode_agent::subagent::SpawnAgentTool::new(
            Arc::clone(&manager),
        )));
    }
    if tool_enabled(tools_config, "wait_agents") {
        registry.register(Arc::new(deepcode_agent::subagent::WaitAgentsTool::new(
            Arc::clone(&manager),
        )));
    }
    if tool_enabled(tools_config, "cancel_agents") {
        registry.register(Arc::new(deepcode_agent::subagent::CancelAgentsTool::new(
            manager,
        )));
    }
}

fn tool_enabled(tools_config: &deepcode_core::config::ToolsConfig, name: &str) -> bool {
    !tools_config
        .disabled
        .iter()
        .any(|disabled| disabled == name)
}

fn build_system_prompt(model: &str, subagents_enabled: bool) -> String {
    let directory = std::env::current_dir()
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| ".".to_string());

    let delegation = if subagents_enabled {
        "For independent investigations, start multiple explorer tasks with \
         spawn_agent. When wait_agents is available, call it once with all task IDs; \
         otherwise the runtime will collect outstanding results before the next model \
         request. Always review those results before giving the final answer. Use \
         worker tasks only for clearly isolated implementation work.\n\n"
    } else {
        ""
    };

    format!(
        "You are DeepCode, an AI coding agent running in a terminal.\n\
         You are backed by the configured model `{}`. If asked what you are, say \
         you are DeepCode using this configured model; do not claim to be a generic \
         provider assistant.\n\
         Workspace: {}\n\n\
         Use the available tools to inspect and modify the workspace. For questions \
         about this project, source code, files, git state, build status, tests, or \
         implementation details, inspect the workspace with tools before answering. \
         Prefer read-only tools first, such as glob, grep, read_file, git_status, \
         and safe shell commands when needed. {}\
         Keep responses concise, factual, and in the user's language. Explain the \
         actions you took and the findings from tools instead of guessing.",
        model, directory, delegation
    )
}

type ResolvedProvider = (
    Arc<dyn LlmProvider>,
    String,
    String,
    String,
    (usize, usize),
    Vec<ModelProfile>,
    Option<tokio::task::JoinHandle<()>>,
);

fn data_root() -> PathBuf {
    std::env::var_os("DEEPCODE_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs_home().join(".local/share/deepcode"))
}

/// Resolve provider + model from config.
async fn resolve(
    config: &DeepCodeConfig,
    provider_name: Option<String>,
    model_name: Option<String>,
    reasoning_effort: Option<String>,
) -> anyhow::Result<ResolvedProvider> {
    let (provider_key, provider_config) = config.resolve_provider(provider_name.as_deref())?;
    let llm: Arc<dyn LlmProvider> = deepcode_providers::create_provider(provider_config)?;
    let catalog = deepcode_providers::catalog::resolve_model_catalog(
        &provider_key,
        provider_config,
        &data_root(),
        false,
    )
    .await?;
    let catalog_refresh = if catalog.status.background_refresh {
        let provider_key = provider_key.clone();
        let provider_config = provider_config.clone();
        let root = data_root();
        Some(tokio::spawn(async move {
            if let Err(error) = deepcode_providers::catalog::refresh_model_catalog(
                &provider_key,
                &provider_config,
                &root,
            )
            .await
            {
                tracing::warn!(provider = %provider_key, error = %error, "Background model catalog refresh failed");
            }
        }))
    } else {
        None
    };
    if let Some(message) = catalog.status.message.as_deref() {
        tracing::warn!(provider = %provider_key, message, "Using model catalog fallback");
    }
    let available_models = catalog.models;
    if available_models.is_empty() {
        anyhow::bail!("Provider '{}' has no discovered models", provider_key);
    }
    let model = model_name
        .or_else(|| provider_config.model.clone())
        .or_else(|| available_models.first().map(|model| model.id.clone()))
        .ok_or_else(|| anyhow::anyhow!("Provider '{}' has no available model", provider_key))?;
    let model_cfg = available_models
        .iter()
        .find(|candidate| candidate.id == model)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "Model '{}' is not present in the live or cached catalog for provider '{}'",
                model,
                provider_key
            )
        })?;
    let reasoning_effort = match reasoning_effort {
        Some(value) => value.parse::<ReasoningEffort>()?,
        None => provider_config
            .models
            .get(&model)
            .and_then(|profile| profile.default_reasoning_effort)
            .or(provider_config.reasoning_effort)
            .unwrap_or_else(|| {
                deepcode_providers::catalog::recommended_effort(&provider_config.kind, model_cfg)
            }),
    };
    if !model_cfg.supports_effort(reasoning_effort) {
        anyhow::bail!(
            "Model '{}' does not support configured reasoning effort '{}'",
            model,
            reasoning_effort
        );
    }
    let max_tokens = model_cfg.max_output_tokens;
    let context_window = model_cfg.context_window;
    Ok((
        llm,
        provider_key,
        model,
        reasoning_effort.to_string(),
        (max_tokens, context_window),
        available_models,
        catalog_refresh,
    ))
}

/// Shared context built by both `run_command` and `chat_command`.
pub(crate) struct AgentContext {
    pub llm: Arc<dyn LlmProvider>,
    pub tools: Arc<ToolRegistry>,
    pub permissions: Arc<tokio::sync::Mutex<PermissionSystem>>,
    pub provider: String,
    pub model: String,
    pub model_config: (usize, usize),
    pub available_models: Vec<ModelProfile>,
    pub reasoning_effort: Option<String>,
    pub system_prompt: Option<String>,
    pub initial_messages: Option<Vec<Message>>,
    pub cmd_tx: event::CmdSender,
    pub event_tx: event::EventSender,
    cmd_rx: event::CmdReceiver,
    catalog_refresh: Option<tokio::task::JoinHandle<()>>,
    task_manager: Option<Arc<deepcode_agent::subagent::AgentTaskManager>>,
}

impl AgentContext {
    pub(crate) fn spawn_agent(
        self,
    ) -> (
        tokio::task::JoinHandle<Result<(), DeepCodeError>>,
        Option<tokio::task::JoinHandle<()>>,
    ) {
        let permissions_clone = self.permissions.clone();
        let catalog_refresh = self.catalog_refresh;
        let agent = tokio::spawn(async move {
            if let Some(manager) = self.task_manager {
                deepcode_agent::r#loop::run_managed(
                    self.llm,
                    self.tools,
                    permissions_clone,
                    self.model,
                    self.model_config,
                    self.reasoning_effort,
                    self.system_prompt,
                    self.initial_messages,
                    true,
                    self.cmd_rx,
                    self.event_tx,
                    manager,
                )
                .await
            } else {
                deepcode_agent::r#loop::run(
                    self.llm,
                    self.tools,
                    permissions_clone,
                    self.model,
                    self.model_config,
                    self.reasoning_effort,
                    self.system_prompt,
                    self.initial_messages,
                    true,
                    self.cmd_rx,
                    self.event_tx,
                )
                .await
            }
        });
        (agent, catalog_refresh)
    }
}

pub(crate) async fn build_agent_context(
    config_path: &PathBuf,
    provider_name: Option<String>,
    model_name: Option<String>,
    reasoning_effort: Option<String>,
    command_channel_buffer: usize,
) -> anyhow::Result<(AgentContext, event::EventReceiver)> {
    let config = DeepCodeConfig::load(config_path)?;
    let (llm, provider, model, reasoning_effort, model_config, available_models, catalog_refresh) =
        resolve(&config, provider_name, model_name, reasoning_effort).await?;

    let mut registry = ToolRegistry::new();
    builtins::register_all(&mut registry, &config.tools);

    let permissions = Arc::new(tokio::sync::Mutex::new(PermissionSystem::new(
        permissions_config(&config),
    )));

    let (cmd_tx, cmd_rx) = event::cmd_channel(command_channel_buffer);
    let (event_tx, event_rx) = event::event_channel();

    let task_manager =
        if tool_enabled(&config.tools, "agent") && tool_enabled(&config.tools, "spawn_agent") {
            let settings = Arc::new(tokio::sync::RwLock::new(
                deepcode_agent::subagent::AgentRuntimeSettings {
                    model: model.clone(),
                    reasoning_effort: Some(reasoning_effort.clone()),
                    default_subagent_model: config.agents.default_model.clone(),
                    default_subagent_reasoning_effort: config
                        .agents
                        .default_reasoning_effort
                        .map(|effort| effort.to_string()),
                },
            ));
            let manager = deepcode_agent::subagent::AgentTaskManager::new(
                Arc::clone(&llm),
                Arc::new(registry.clone()),
                Arc::clone(&permissions),
                settings,
                available_models.clone(),
                event_tx.clone(),
                config.agents.max_concurrent,
            );
            register_subagent_tools(&mut registry, Arc::clone(&manager), &config.tools);
            Some(manager)
        } else {
            None
        };
    let tools = Arc::new(registry);

    let system_prompt = Some(build_system_prompt(&model, task_manager.is_some()));

    let context = AgentContext {
        llm,
        tools,
        permissions,
        provider,
        model,
        model_config,
        available_models,
        reasoning_effort: Some(reasoning_effort),
        system_prompt,
        initial_messages: None,
        cmd_tx,
        event_tx,
        cmd_rx,
        catalog_refresh,
        task_manager,
    };

    Ok((context, event_rx))
}

pub(crate) async fn cleanup_agent(
    command_label: &str,
    cmd_tx: event::CmdSender,
    agent_handle: tokio::task::JoinHandle<Result<(), DeepCodeError>>,
    timeout: Duration,
    catalog_refresh: Option<tokio::task::JoinHandle<()>>,
) {
    drop(cmd_tx);
    match tokio::time::timeout(timeout, agent_handle).await {
        Ok(Ok(Ok(()))) => tracing::info!(command = command_label, "Agent cleanup completed"),
        Ok(Ok(Err(e))) => tracing::warn!(
            command = command_label,
            error = %e,
            "Agent returned error during cleanup"
        ),
        Ok(Err(e)) => tracing::warn!(
            command = command_label,
            error = %e,
            "Agent task join failed during cleanup"
        ),
        Err(_) => tracing::warn!(command = command_label, "Agent cleanup timed out"),
    }
    if let Some(refresh) = catalog_refresh {
        if tokio::time::timeout(Duration::from_secs(2), refresh)
            .await
            .is_err()
        {
            tracing::debug!(
                command = command_label,
                "Model catalog refresh cleanup timed out"
            );
        }
    }
}

fn print_colored_diff_to_stderr(preview: &deepcode_tools::tool::FileChangePreview) {
    let operation = if preview.before_exists {
        "Update"
    } else {
        "Create"
    };
    eprintln!("\n{operation}: {}", preview.path);
    let mut in_hunk = false;
    for line in preview.unified_diff.lines() {
        if line.starts_with("diff --git ") {
            in_hunk = false;
            continue;
        }
        if !in_hunk && (line.starts_with("--- ") || line.starts_with("+++ ")) {
            continue;
        }
        if line.starts_with("@@") {
            in_hunk = true;
        }

        let color = if line.starts_with("@@") {
            "\x1b[1;36m"
        } else if line.starts_with("+++") || line.starts_with('+') {
            "\x1b[32m"
        } else if line.starts_with("---") || line.starts_with('-') {
            "\x1b[31m"
        } else {
            "\x1b[90m"
        };
        eprintln!("{}{}\x1b[0m", color, line);
    }
}

fn read_file_preview_decision() -> bool {
    eprint!("Apply changes? [y/N] ");
    let _ = std::io::Write::flush(&mut std::io::stderr());

    let mut line = String::new();
    match std::io::stdin().read_line(&mut line) {
        Ok(_) => matches!(
            line.trim().to_lowercase().as_str(),
            "y" | "yes" | "a" | "apply"
        ),
        Err(_) => false,
    }
}

pub(crate) async fn run_command(
    prompt: Vec<String>,
    provider_name: Option<String>,
    model_name: Option<String>,
    config_path: PathBuf,
) -> anyhow::Result<()> {
    if ensure_current_workspace_trusted()? == TrustDecision::Exit {
        return Ok(());
    }
    let (context, mut event_rx) =
        build_agent_context(&config_path, provider_name, model_name, None, 32).await?;
    let cmd_tx = context.cmd_tx.clone();
    let workspace_root = SessionStore::workspace_root()?;
    let mut saved_session = SavedSession::new(
        workspace_root,
        context.provider.clone(),
        context.model.clone(),
        context
            .reasoning_effort
            .clone()
            .unwrap_or_else(|| "off".to_string()),
    );

    let prompt_text = prompt.join(" ");
    tracing::info!(
        command = "run",
        model = %context.model,
        prompt_chars = prompt_text.chars().count(),
        "Starting one-shot command"
    );

    let (agent_handle, catalog_refresh) = context.spawn_agent();

    cmd_tx
        .send(event::AgentCommand::Process {
            message: prompt_text.clone(),
        })
        .await?;
    saved_session.ui_messages = vec![crate::ui::ChatMessage {
        role: "user".to_string(),
        content: prompt_text.clone(),
    }];
    saved_session.core_messages = vec![Message::user(&prompt_text)];
    if let Err(error) = SessionStore::default().save(&mut saved_session) {
        tracing::warn!(command = "run", error = %error, "Failed to create session checkpoint");
    }

    let mut final_output = String::new();
    let mut latest_core_messages = Vec::new();
    let mut last_usage: Option<crate::ui::TurnUsage> = None;
    let mut agent_error = None;
    let mut quit_requested = false;
    while let Some(ev) = event_rx.recv().await {
        match ev {
            event::AgentEvent::TextDelta(text) => {
                print!("{}", text);
                final_output.push_str(&text);
            }
            event::AgentEvent::ToolCallStarted { name, input, .. } => {
                if matches!(
                    name.as_str(),
                    "agent" | "spawn_agent" | "wait_agents" | "cancel_agents"
                ) {
                    continue;
                }
                let (group, detail) = crate::ui::tool_activity(&name, &input);
                eprintln!("\n• {}: {}", group, detail);
            }
            event::AgentEvent::ToolCallCompleted { .. } => {}
            event::AgentEvent::ToolCallFailed { name, error, .. } => {
                eprintln!("  {}", crate::ui::tool_issue_status(&name, &error));
            }
            event::AgentEvent::PermissionNeeded {
                request_id,
                tool_name,
                input,
                evaluation,
            } => {
                eprintln!("\n╔══════════════════════════════════════════╗");
                eprintln!("║  Permission Required                     ║");
                eprintln!("╠══════════════════════════════════════════╣");
                eprintln!("║  Tool: {:<34}║", tool_name);
                eprintln!("║  Risk: {:<34}║", evaluation.risk.as_str());
                eprintln!("║  Sandbox: {:<31}║", evaluation.sandbox_policy.label());
                let args = serde_json::to_string(&input).unwrap_or_default();
                let args = truncate_with_ellipsis(&args, 34);
                eprintln!("║  Args: {:<34}║", args);
                eprintln!("╠══════════════════════════════════════════╣");
                eprintln!("║  [y] once  [s] session  [a] always       ║");
                eprintln!("║  [n] deny  [q] quit                      ║");
                eprintln!("╚══════════════════════════════════════════╝");
                eprint!("> ");
                let _ = std::io::Write::flush(&mut std::io::stderr());

                let mut line = String::new();
                let (approved, scope, quit) = match std::io::stdin().read_line(&mut line) {
                    Ok(_) => match line.trim().to_lowercase().as_str() {
                        "y" | "yes" => (true, ApprovalScope::Once, false),
                        "s" | "session" => (true, ApprovalScope::Session, false),
                        "a" | "always" => (true, ApprovalScope::Persistent, false),
                        "q" | "quit" => (false, ApprovalScope::Once, true),
                        "n" | "no" => (false, ApprovalScope::Once, false),
                        _ => (false, ApprovalScope::Once, false),
                    },
                    Err(_) => (false, ApprovalScope::Once, false),
                };

                let _ = cmd_tx
                    .send(event::AgentCommand::PermissionResponse {
                        request_id,
                        approved,
                        scope,
                    })
                    .await;

                if !approved {
                    eprintln!("Permission denied.");
                }
                if quit {
                    quit_requested = true;
                    break;
                }
            }
            event::AgentEvent::FileChangePreviewNeeded {
                request_id,
                preview,
                ..
            } => {
                print_colored_diff_to_stderr(&preview);
                let approved = read_file_preview_decision();
                let _ = cmd_tx
                    .send(event::AgentCommand::FileChangePreviewResponse {
                        request_id,
                        approved,
                    })
                    .await;
                if !approved {
                    eprintln!("Change rejected.");
                }
            }
            event::AgentEvent::AgentFinished { .. } => {
                if !final_output.is_empty() && !final_output.ends_with('\n') {
                    println!();
                }
                if let Some(usage) = last_usage.as_ref() {
                    eprintln!("({})", usage_summary(usage));
                }
                eprintln!("(agent finished)");
                break;
            }
            event::AgentEvent::TurnComplete {
                input_tokens,
                output_tokens,
                cached_input_tokens,
                cache_miss_input_tokens,
                reasoning_output_tokens,
            } => {
                let usage = crate::ui::TurnUsage {
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    cache_miss_input_tokens,
                    reasoning_output_tokens,
                };
                if usage.has_reported_tokens() {
                    if let Some(total) = last_usage.as_mut() {
                        total.add_assign(&usage);
                    } else {
                        last_usage = Some(usage);
                    }
                }
            }
            event::AgentEvent::SessionUpdated { messages } => {
                latest_core_messages = messages.clone();
                saved_session.core_messages = messages;
                if let Err(error) = SessionStore::default().save(&mut saved_session) {
                    tracing::warn!(command = "run", error = %error, "Failed to update session checkpoint");
                }
            }
            event::AgentEvent::SessionTitleGenerated { title } => {
                if saved_session.set_generated_title(&title) {
                    if let Err(error) = SessionStore::default().save(&mut saved_session) {
                        tracing::warn!(command = "run", error = %error, "Failed to save session title");
                    }
                }
            }
            event::AgentEvent::AgentError { message } => {
                eprintln!("\nError: {}", message);
                agent_error = Some(message);
                break;
            }
            event::AgentEvent::SubagentStarted { task, .. } => {
                let input = serde_json::json!({ "task": task });
                let (_, detail) = crate::ui::tool_activity("agent", &input);
                eprintln!("\n• Delegated: {}", detail);
            }
            event::AgentEvent::SubagentCompleted { .. } => {}
            event::AgentEvent::SubagentEvent { event, .. } => match event.as_ref() {
                event::AgentEvent::ToolCallStarted { name, input, .. } => {
                    let (group, detail) = crate::ui::tool_activity(name, input);
                    eprintln!("  • Subagent {}: {}", group, detail);
                }
                event::AgentEvent::ToolCallCompleted { .. } => {}
                event::AgentEvent::ToolCallFailed { name, error, .. } => {
                    eprintln!("  Subagent {}", crate::ui::tool_issue_status(name, error));
                }
                event::AgentEvent::AgentError { message } => {
                    eprintln!("  Subagent issue: {}", message);
                }
                event::AgentEvent::TurnComplete {
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    cache_miss_input_tokens,
                    reasoning_output_tokens,
                } => {
                    let usage = crate::ui::TurnUsage {
                        input_tokens: *input_tokens,
                        output_tokens: *output_tokens,
                        cached_input_tokens: *cached_input_tokens,
                        cache_miss_input_tokens: *cache_miss_input_tokens,
                        reasoning_output_tokens: *reasoning_output_tokens,
                    };
                    if usage.has_reported_tokens() {
                        if let Some(total) = last_usage.as_mut() {
                            total.add_assign(&usage);
                        } else {
                            last_usage = Some(usage);
                        }
                    }
                }
                event::AgentEvent::PermissionNeeded {
                    request_id,
                    tool_name,
                    input,
                    evaluation,
                } => {
                    eprintln!(
                        "\nSubagent permission required: {} ({}, {})",
                        tool_name,
                        evaluation.risk.as_str(),
                        serde_json::to_string(input).unwrap_or_default()
                    );
                    eprint!("Allow once? [y/N] ");
                    let _ = std::io::Write::flush(&mut std::io::stderr());
                    let mut line = String::new();
                    let approved = std::io::stdin().read_line(&mut line).is_ok_and(|_| {
                        matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes")
                    });
                    let _ = cmd_tx
                        .send(event::AgentCommand::PermissionResponse {
                            request_id: request_id.clone(),
                            approved,
                            scope: ApprovalScope::Once,
                        })
                        .await;
                }
                event::AgentEvent::FileChangePreviewNeeded {
                    request_id,
                    preview,
                    ..
                } => {
                    print_colored_diff_to_stderr(preview);
                    let approved = read_file_preview_decision();
                    let _ = cmd_tx
                        .send(event::AgentCommand::FileChangePreviewResponse {
                            request_id: request_id.clone(),
                            approved,
                        })
                        .await;
                }
                _ => {}
            },
            _ => {}
        }
    }

    cleanup_agent(
        "run",
        cmd_tx,
        agent_handle,
        Duration::from_secs(5),
        catalog_refresh,
    )
    .await;

    if quit_requested {
        return Ok(());
    }

    if !latest_core_messages.is_empty() {
        saved_session.ui_messages = vec![
            crate::ui::ChatMessage {
                role: "user".to_string(),
                content: prompt_text,
            },
            crate::ui::ChatMessage {
                role: "assistant".to_string(),
                content: final_output,
            },
        ];
        saved_session.core_messages = latest_core_messages;
        if let Err(e) = SessionStore::default().save(&mut saved_session) {
            tracing::warn!(command = "run", error = %e, "Failed to save session");
            eprintln!("Failed to save session: {}", e);
        }
    }

    if let Some(message) = agent_error {
        anyhow::bail!(message);
    }
    Ok(())
}

fn usage_summary(usage: &crate::ui::TurnUsage) -> String {
    let mut summary = format!(
        "usage: in/out {}/{}",
        usage.input_tokens, usage.output_tokens
    );
    let cache_total = usage.cached_input_tokens + usage.cache_miss_input_tokens;
    if let Some(hit_rate) = usage
        .cached_input_tokens
        .saturating_mul(100)
        .checked_div(cache_total)
    {
        summary.push_str(&format!(
            ", cache hit {}% ({}/{})",
            hit_rate, usage.cached_input_tokens, cache_total
        ));
    }
    if usage.reasoning_output_tokens > 0 {
        summary.push_str(&format!(", reasoning {}", usage.reasoning_output_tokens));
    }
    summary
}

fn truncate_with_ellipsis(value: &str, max_chars: usize) -> String {
    let count = value.chars().count();
    if count <= max_chars {
        return value.to_string();
    }
    let visible = max_chars.saturating_sub(3);
    format!("{}...", value.chars().take(visible).collect::<String>())
}

pub(crate) async fn config_command(config_path: PathBuf) -> anyhow::Result<()> {
    let config = DeepCodeConfig::load(&config_path)?;
    println!("Config file: {:?}", config_path);
    let mut safe_config = config.clone();
    for provider in safe_config.providers.values_mut() {
        if provider.api_key.is_some() {
            provider.api_key = Some("***redacted***".to_string());
        }
    }
    println!("{:#?}", safe_config);

    for (name, pc) in &config.providers {
        println!(
            "  Provider '{}': api_key={}",
            name,
            if pc.resolve_api_key().is_some() {
                "configured"
            } else {
                "not set"
            }
        );
        match deepcode_providers::catalog::resolve_model_catalog(name, pc, &data_root(), false)
            .await
        {
            Ok(catalog) => print_catalog_status(&catalog.status),
            Err(error) => println!("    catalog: unavailable ({})", error),
        }
    }

    Ok(())
}

pub(crate) async fn models_command(
    config_path: PathBuf,
    provider_name: Option<String>,
    refresh: bool,
) -> anyhow::Result<()> {
    let config = DeepCodeConfig::load(&config_path)?;
    let (name, provider) = config.resolve_provider(provider_name.as_deref())?;
    let catalog =
        deepcode_providers::catalog::resolve_model_catalog(&name, provider, &data_root(), refresh)
            .await?;

    println!("Provider: {} ({})", name, provider.kind);
    print_catalog_status(&catalog.status);
    println!("MODEL                            CONTEXT     OUTPUT   EFFORTS");
    for model in catalog.models {
        println!(
            "{:<32} {:>10} {:>10}   {}",
            truncate_with_ellipsis(&model.id, 32),
            model.context_window,
            model.max_output_tokens,
            model.effort_names().join(",")
        );
    }
    Ok(())
}

fn print_catalog_status(status: &deepcode_providers::catalog::CatalogStatus) {
    let refreshed = status
        .refreshed_at
        .map(format_unix_time)
        .unwrap_or_else(|| "never".to_string());
    let next = status
        .next_refresh_at
        .map(format_unix_time)
        .unwrap_or_else(|| "manual only".to_string());
    println!(
        "    catalog: source={}, refreshed={}, stale={}, next_refresh={}",
        status.source, refreshed, status.stale, next
    );
    if let Some(message) = status.message.as_deref() {
        println!("    status: {}", message);
    }
}

fn format_unix_time(value: u64) -> String {
    chrono::DateTime::from_timestamp(value as i64, 0)
        .map(|time| time.to_rfc3339())
        .unwrap_or_else(|| value.to_string())
}

pub(crate) fn sessions_command(
    all: bool,
    limit: Option<usize>,
    _config_path: &PathBuf,
) -> anyhow::Result<()> {
    let store = SessionStore::default();
    let workspace = SessionStore::workspace_root()?;
    let sessions = store.list((!all).then_some(workspace.as_str()), limit)?;
    if sessions.is_empty() {
        println!("No saved sessions.");
        return Ok(());
    }
    println!(
        "ID                                    UPDATED     PROVIDER/MODEL                 TITLE"
    );
    for session in sessions {
        let date = session.updated_at.chars().take(10).collect::<String>();
        println!(
            "{:<36}  {:<10}  {:<30}  {}",
            session.id,
            date,
            format!("{}/{}", session.provider, session.model),
            session.title
        );
        if all {
            println!(
                "                                      workspace: {}",
                session.workspace_root
            );
        }
    }
    Ok(())
}

pub(crate) async fn resume_command(
    session_id: Option<String>,
    last: bool,
    config_path: PathBuf,
) -> anyhow::Result<()> {
    if session_id.is_some() == last {
        anyhow::bail!("Provide exactly one of <SESSION_ID> or --last");
    }
    let store = SessionStore::default();
    let id = if last {
        if ensure_current_workspace_trusted()? == TrustDecision::Exit {
            return Ok(());
        }
        store.latest(&SessionStore::workspace_root()?)?.id
    } else {
        session_id.expect("validated session id")
    };
    chat_command(None, None, config_path, Some(id)).await
}

pub(crate) async fn chat_command(
    provider_name: Option<String>,
    model_name: Option<String>,
    config_path: PathBuf,
    initial_session_id: Option<String>,
) -> anyhow::Result<()> {
    let config_path = std::fs::canonicalize(&config_path).unwrap_or(config_path);
    let store = SessionStore::default();
    let mut requested_session = initial_session_id;

    loop {
        let loaded_session = if let Some(id) = requested_session.take() {
            let session = store.load(&id)?;
            let workspace = PathBuf::from(&session.workspace_root);
            if !workspace.is_dir() {
                anyhow::bail!(
                    "Session workspace no longer exists: {}",
                    session.workspace_root
                );
            }
            if ensure_workspace_trusted(&workspace)? == TrustDecision::Exit {
                return Ok(());
            }
            std::env::set_current_dir(&workspace)?;
            Some(session)
        } else {
            if ensure_current_workspace_trusted()? == TrustDecision::Exit {
                return Ok(());
            }
            None
        };

        let (session_provider, session_model, session_effort) = loaded_session
            .as_ref()
            .map(|session| {
                (
                    Some(session.provider.clone()),
                    Some(session.model.clone()),
                    Some(session.reasoning_effort.clone()),
                )
            })
            .unwrap_or_else(|| (provider_name.clone(), model_name.clone(), None));
        let (mut context, event_rx) = build_agent_context(
            &config_path,
            session_provider,
            session_model,
            session_effort,
            256,
        )
        .await?;
        if let Some(session) = &loaded_session {
            context.initial_messages = Some(session.core_messages.clone());
        }

        let cmd_tx = context.cmd_tx.clone();
        let startup_header =
            crate::ui::StartupHeader::current(&context.model, context.reasoning_effort.as_deref());
        let session = loaded_session.unwrap_or_else(|| {
            SavedSession::new(
                SessionStore::workspace_root().unwrap_or_else(|_| ".".to_string()),
                context.provider.clone(),
                context.model.clone(),
                context
                    .reasoning_effort
                    .clone()
                    .unwrap_or_else(|| "off".to_string()),
            )
        });
        let mut app_state = if session.ui_messages.is_empty() && session.core_messages.is_empty() {
            crate::ui::AppState::new()
        } else {
            crate::ui::AppState::with_session(
                session.ui_messages.clone(),
                session.core_messages.clone(),
            )
        };
        app_state.startup_header = Some(startup_header);
        app_state.available_models = context.available_models.clone();
        let config = DeepCodeConfig::load(&config_path)?;
        let (_, provider_config) = config.resolve_provider(Some(&context.provider))?;
        app_state.model_catalog = Some(crate::ui::ModelCatalogContext {
            provider: context.provider.clone(),
            config: provider_config.clone(),
            data_root: data_root(),
        });
        app_state.current_model = Some(context.model.clone());
        app_state.reasoning_effort = context.reasoning_effort.clone();
        app_state.session_store = Some(store.clone());
        app_state.session = Some(session);
        let state = Arc::new(std::sync::Mutex::new(app_state));

        tracing::info!(
            command = "chat",
            provider = %context.provider,
            model = %context.model,
            "Starting chat session"
        );
        let (agent_handle, catalog_refresh) = context.spawn_agent();
        let tui_state = state.clone();
        let tui_cmd_tx = cmd_tx.clone();
        let tui_handle =
            std::thread::spawn(move || crate::ui::run_tui(tui_cmd_tx, event_rx, tui_state));
        let action = match tui_handle.join() {
            Ok(result) => result?,
            Err(payload) => {
                let message = payload
                    .downcast_ref::<&str>()
                    .map(|value| (*value).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic payload".to_string());
                anyhow::bail!("TUI thread panicked: {}", message);
            }
        };
        {
            let mut state = state.lock().unwrap();
            state.save_committed_session()?;
        }
        cleanup_agent(
            "chat",
            cmd_tx,
            agent_handle,
            Duration::from_secs(2),
            catalog_refresh,
        )
        .await;

        match action {
            crate::ui::TuiAction::Exit => return Ok(()),
            crate::ui::TuiAction::Resume(id) => requested_session = Some(id),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{build_system_prompt, tool_enabled};
    use deepcode_core::config::ToolsConfig;

    #[test]
    fn tool_enabled_honors_disabled_agent() {
        let tools_config = ToolsConfig {
            disabled: vec!["agent".to_string()],
            max_file_size_bytes: None,
        };

        assert!(!tool_enabled(&tools_config, "agent"));
        assert!(tool_enabled(&tools_config, "shell"));
    }

    #[test]
    fn system_prompt_only_mentions_subagents_when_enabled() {
        assert!(!build_system_prompt("test-model", false).contains("spawn_agent"));
        assert!(build_system_prompt("test-model", true).contains("spawn_agent"));
    }
}
