use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};

use async_trait::async_trait;
use deepcode_core::config::ModelProfile;
use deepcode_core::error::{DeepCodeError, Result};
use deepcode_core::provider::traits::{LlmProvider, Usage};
use deepcode_permissions::pipeline::PermissionSystem;
use deepcode_tools::registry::ToolRegistry;
use deepcode_tools::tool::{Tool, ToolSafety};
use futures::FutureExt;
use serde_json::json;
use tokio::sync::{mpsc, Mutex, Notify, RwLock, Semaphore};

use crate::event::{AgentCommand, AgentEvent, EventSender};
use crate::r#loop;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    Explorer,
    Worker,
}

impl AgentRole {
    fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("explorer") {
            "explorer" => Ok(Self::Explorer),
            "worker" => Ok(Self::Worker),
            other => Err(DeepCodeError::ToolExecution {
                tool: "spawn_agent".to_string(),
                message: format!(
                    "Unknown agent role '{}'; expected explorer or worker",
                    other
                ),
            }),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Queued,
    Running,
    Completed,
    Failed,
    Interrupted,
    Cancelled,
}

impl TaskStatus {
    fn terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Interrupted | Self::Cancelled
        )
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentTaskResult {
    pub task_id: String,
    pub role: AgentRole,
    pub status: TaskStatus,
    pub model: String,
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub usage: Usage,
}

#[derive(Debug, Clone)]
pub struct AgentRuntimeSettings {
    pub model: String,
    pub reasoning_effort: Option<String>,
    pub default_subagent_model: Option<String>,
    pub default_subagent_reasoning_effort: Option<String>,
}

struct TaskRecord {
    role: AgentRole,
    status: TaskStatus,
    model: String,
    reasoning_effort: Option<String>,
    result: Option<String>,
    error: Option<String>,
    usage: Usage,
    command_tx: Option<mpsc::Sender<AgentCommand>>,
    scheduler_abort: Option<tokio::task::AbortHandle>,
    agent_abort: Option<tokio::task::AbortHandle>,
    collected: bool,
}

impl TaskRecord {
    fn snapshot(&self, task_id: &str) -> AgentTaskResult {
        AgentTaskResult {
            task_id: task_id.to_string(),
            role: self.role,
            status: self.status,
            model: self.model.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            result: self.result.clone(),
            error: self.error.clone(),
            usage: self.usage.clone(),
        }
    }
}

#[derive(Clone)]
pub struct AgentTaskManager {
    llm: Arc<dyn LlmProvider>,
    base_tools: Arc<ToolRegistry>,
    permissions: Arc<Mutex<PermissionSystem>>,
    settings: Arc<RwLock<AgentRuntimeSettings>>,
    models: Arc<RwLock<Vec<ModelProfile>>>,
    event_tx: EventSender,
    tasks: Arc<Mutex<HashMap<String, TaskRecord>>>,
    notify: Arc<Notify>,
    slots: Arc<Semaphore>,
    worker_slot: Arc<Semaphore>,
}

impl AgentTaskManager {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        llm: Arc<dyn LlmProvider>,
        base_tools: Arc<ToolRegistry>,
        permissions: Arc<Mutex<PermissionSystem>>,
        settings: Arc<RwLock<AgentRuntimeSettings>>,
        models: Vec<ModelProfile>,
        event_tx: EventSender,
        max_concurrent: usize,
    ) -> Arc<Self> {
        Arc::new(Self {
            llm,
            base_tools,
            permissions,
            settings,
            models: Arc::new(RwLock::new(models)),
            event_tx,
            tasks: Arc::new(Mutex::new(HashMap::new())),
            notify: Arc::new(Notify::new()),
            slots: Arc::new(Semaphore::new(max_concurrent.max(1))),
            worker_slot: Arc::new(Semaphore::new(1)),
        })
    }

    pub async fn update_main_model(&self, model: String) {
        self.settings.write().await.model = model;
    }

    pub async fn update_main_reasoning_effort(&self, effort: Option<String>) {
        self.settings.write().await.reasoning_effort = effort;
    }

    pub async fn update_models(&self, models: Vec<ModelProfile>) {
        *self.models.write().await = models;
    }

    async fn effective_settings(
        &self,
        model_override: Option<&str>,
        effort_override: Option<&str>,
    ) -> Result<(String, (usize, usize), Option<String>)> {
        let settings = self.settings.read().await;
        let model = model_override
            .map(str::to_string)
            .or_else(|| settings.default_subagent_model.clone())
            .unwrap_or_else(|| settings.model.clone());
        let models = self.models.read().await;
        let profile = models
            .iter()
            .find(|candidate| candidate.id == model)
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: "spawn_agent".to_string(),
                message: format!("Unknown subagent model '{}'", model),
            })?;
        let effort = effort_override
            .map(str::to_string)
            .or_else(|| settings.default_subagent_reasoning_effort.clone())
            .or_else(|| settings.reasoning_effort.clone());
        if let Some(effort) = effort.as_deref() {
            if !profile.supports_effort_str(effort) {
                return Err(DeepCodeError::ToolExecution {
                    tool: "spawn_agent".to_string(),
                    message: format!(
                        "Model '{}' does not support reasoning effort '{}'",
                        model, effort
                    ),
                });
            }
        }
        Ok((
            model,
            (profile.max_output_tokens, profile.context_window),
            effort,
        ))
    }

    pub async fn spawn(
        self: &Arc<Self>,
        task: String,
        role: AgentRole,
        model_override: Option<&str>,
        effort_override: Option<&str>,
    ) -> Result<AgentTaskResult> {
        if task.trim().is_empty() {
            return Err(DeepCodeError::ToolExecution {
                tool: "spawn_agent".to_string(),
                message: "Task must not be empty".to_string(),
            });
        }
        let (model, model_config, reasoning_effort) = self
            .effective_settings(model_override, effort_override)
            .await?;
        let task_id = uuid::Uuid::new_v4().to_string();
        let (command_tx, command_rx) = mpsc::channel(32);
        let record = TaskRecord {
            role,
            status: TaskStatus::Queued,
            model: model.clone(),
            reasoning_effort: reasoning_effort.clone(),
            result: None,
            error: None,
            usage: Usage::default(),
            command_tx: Some(command_tx.clone()),
            scheduler_abort: None,
            agent_abort: None,
            collected: false,
        };
        let initial = record.snapshot(&task_id);
        self.tasks.lock().await.insert(task_id.clone(), record);

        let manager = Arc::clone(self);
        let spawned_task_id = task_id.clone();
        let scheduler = tokio::spawn(async move {
            let outcome = std::panic::AssertUnwindSafe(Arc::clone(&manager).run_task(
                spawned_task_id.clone(),
                task,
                role,
                model,
                model_config,
                reasoning_effort,
                command_tx,
                command_rx,
            ))
            .catch_unwind()
            .await;
            if let Err(payload) = outcome {
                let message = panic_message(payload);
                manager
                    .finish_failed(
                        &spawned_task_id,
                        &format!("Subagent scheduler panicked: {message}"),
                    )
                    .await;
            }
        });
        if let Some(record) = self.tasks.lock().await.get_mut(&task_id) {
            if !record.status.terminal() {
                record.scheduler_abort = Some(scheduler.abort_handle());
            }
        }
        Ok(initial)
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_task(
        self: Arc<Self>,
        task_id: String,
        task: String,
        role: AgentRole,
        model: String,
        model_config: (usize, usize),
        reasoning_effort: Option<String>,
        command_tx: mpsc::Sender<AgentCommand>,
        command_rx: mpsc::Receiver<AgentCommand>,
    ) {
        // A queued worker must not consume general capacity while waiting for the
        // single-writer gate; otherwise queued workers can starve explorers.
        let worker_permit = if role == AgentRole::Worker {
            match self.worker_slot.clone().acquire_owned().await {
                Ok(permit) => Some(permit),
                Err(_) => {
                    self.finish_failed(&task_id, "Worker scheduler closed")
                        .await;
                    return;
                }
            }
        } else {
            None
        };
        let overall_permit = match self.slots.clone().acquire_owned().await {
            Ok(permit) => permit,
            Err(_) => {
                self.finish_failed(&task_id, "Subagent scheduler closed")
                    .await;
                return;
            }
        };
        {
            let mut tasks = self.tasks.lock().await;
            let Some(record) = tasks.get_mut(&task_id) else {
                return;
            };
            if matches!(
                record.status,
                TaskStatus::Cancelled | TaskStatus::Interrupted
            ) {
                return;
            }
            record.status = TaskStatus::Running;
        }
        self.notify.notify_waiters();
        let _ = self.event_tx.send(AgentEvent::SubagentStarted {
            task_id: task_id.clone(),
            task: task.clone(),
        });

        let tools = match role {
            AgentRole::Explorer => Arc::new(self.base_tools.read_only_subset()),
            AgentRole::Worker => Arc::clone(&self.base_tools),
        };
        let (sub_event_tx, mut sub_event_rx) = mpsc::unbounded_channel();
        let llm = Arc::clone(&self.llm);
        let permissions = Arc::clone(&self.permissions);
        let system_prompt = match role {
            AgentRole::Explorer => "You are a read-only code explorer. Inspect the workspace using available tools, do not modify anything, and return concise evidence-backed findings.",
            AgentRole::Worker => "You are a coding worker. Complete the assigned task using available tools, respect all permission and review prompts, verify the result, and return a concise summary.",
        }
        .to_string();
        let effort = reasoning_effort.clone();
        let mut agent_handle = tokio::spawn(async move {
            r#loop::run(
                llm,
                tools,
                permissions,
                model,
                model_config,
                effort,
                Some(system_prompt),
                None,
                deepcode_core::paths::home_dir()
                    .join(".deepcode")
                    .join("plans"),
                false,
                command_rx,
                sub_event_tx,
            )
            .await
        });
        {
            let mut tasks = self.tasks.lock().await;
            let Some(record) = tasks.get_mut(&task_id) else {
                agent_handle.abort();
                return;
            };
            if record.status == TaskStatus::Cancelled || record.status == TaskStatus::Interrupted {
                agent_handle.abort();
                record.scheduler_abort = None;
                return;
            }
            record.agent_abort = Some(agent_handle.abort_handle());
        }

        if command_tx
            .send(AgentCommand::Process { message: task })
            .await
            .is_err()
        {
            agent_handle.abort();
            self.finish_failed(&task_id, "Subagent command channel closed")
                .await;
            return;
        }

        let mut terminal_status = TaskStatus::Failed;
        let mut final_result = None;
        let mut final_error = None;
        while let Some(event) = sub_event_rx.recv().await {
            match &event {
                AgentEvent::TextDelta(_) | AgentEvent::ReasoningDelta(_) => {}
                AgentEvent::TurnComplete {
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    cache_miss_input_tokens,
                    reasoning_output_tokens,
                } => {
                    if let Some(record) = self.tasks.lock().await.get_mut(&task_id) {
                        record.usage.add_assign(&Usage {
                            input_tokens: *input_tokens,
                            output_tokens: *output_tokens,
                            cached_input_tokens: *cached_input_tokens,
                            cache_miss_input_tokens: *cache_miss_input_tokens,
                            reasoning_output_tokens: *reasoning_output_tokens,
                        });
                    }
                    let _ = self.event_tx.send(AgentEvent::SubagentEvent {
                        task_id: task_id.clone(),
                        event: Arc::new(event),
                    });
                }
                AgentEvent::PermissionNeeded { request_id, .. }
                | AgentEvent::FileChangePreviewNeeded { request_id, .. } => {
                    if let Ok(mut routes) = approval_routes().lock() {
                        routes.insert(
                            request_id.clone(),
                            ApprovalRoute {
                                task_id: task_id.clone(),
                                sender: command_tx.clone(),
                            },
                        );
                    }
                    let _ = self.event_tx.send(AgentEvent::SubagentEvent {
                        task_id: task_id.clone(),
                        event: Arc::new(event),
                    });
                }
                AgentEvent::AgentFinished { final_message } => {
                    terminal_status = TaskStatus::Completed;
                    final_result = Some(final_message.clone());
                    break;
                }
                AgentEvent::AgentError { message } => {
                    terminal_status = TaskStatus::Failed;
                    final_error = Some(message.clone());
                    break;
                }
                AgentEvent::Interrupted => {
                    terminal_status = TaskStatus::Interrupted;
                    final_error = Some("Subagent interrupted".to_string());
                    break;
                }
                AgentEvent::ToolCallStarted { .. }
                | AgentEvent::ToolCallCompleted { .. }
                | AgentEvent::ToolCallFailed { .. } => {
                    let _ = self.event_tx.send(AgentEvent::SubagentEvent {
                        task_id: task_id.clone(),
                        event: Arc::new(event),
                    });
                }
                _ => {}
            }
        }
        let _ = command_tx.send(AgentCommand::Shutdown).await;
        match tokio::time::timeout(std::time::Duration::from_secs(2), &mut agent_handle).await {
            Ok(Ok(Ok(()))) => {}
            Ok(Ok(Err(error))) => {
                terminal_status = TaskStatus::Failed;
                final_error.get_or_insert_with(|| error.to_string());
            }
            Ok(Err(error)) => {
                terminal_status = TaskStatus::Failed;
                final_error.get_or_insert_with(|| format!("Subagent runtime task failed: {error}"));
            }
            Err(_) => {
                agent_handle.abort();
                let _ = agent_handle.await;
                final_error.get_or_insert_with(|| "Subagent cleanup timed out".to_string());
                terminal_status = TaskStatus::Failed;
            }
        }
        if final_result.is_none() && final_error.is_none() {
            terminal_status = TaskStatus::Failed;
            final_error = Some("Subagent event stream closed before a terminal event".to_string());
        }
        drop(worker_permit);
        drop(overall_permit);
        let display = final_result
            .clone()
            .or_else(|| final_error.clone())
            .unwrap_or_default();
        {
            let mut tasks = self.tasks.lock().await;
            if let Some(record) = tasks.get_mut(&task_id) {
                if record.status != TaskStatus::Cancelled
                    && record.status != TaskStatus::Interrupted
                {
                    record.status = terminal_status;
                    record.result = final_result;
                    record.error = final_error;
                }
                record.command_tx = None;
                record.scheduler_abort = None;
                record.agent_abort = None;
            }
        }
        cleanup_approval_routes(&task_id);
        self.notify.notify_waiters();
        let _ = self.event_tx.send(AgentEvent::SubagentCompleted {
            task_id,
            result: display,
        });
    }

    async fn finish_failed(&self, task_id: &str, error: &str) {
        let agent_abort = {
            let mut tasks = self.tasks.lock().await;
            let Some(record) = tasks.get_mut(task_id) else {
                return;
            };
            if record.status.terminal() {
                return;
            }
            record.status = TaskStatus::Failed;
            record.error = Some(error.to_string());
            record.command_tx = None;
            record.scheduler_abort = None;
            record.agent_abort.take()
        };
        if let Some(abort) = agent_abort {
            abort.abort();
        }
        cleanup_approval_routes(task_id);
        self.notify.notify_waiters();
    }

    pub async fn wait(&self, task_ids: &[String]) -> Result<Vec<AgentTaskResult>> {
        loop {
            let notified = self.notify.notified();
            let snapshots = {
                let tasks = self.tasks.lock().await;
                let mut snapshots = Vec::with_capacity(task_ids.len());
                for task_id in task_ids {
                    let record =
                        tasks
                            .get(task_id)
                            .ok_or_else(|| DeepCodeError::ToolExecution {
                                tool: "wait_agents".to_string(),
                                message: format!("Unknown subagent task '{}'", task_id),
                            })?;
                    snapshots.push(record.snapshot(task_id));
                }
                snapshots
            };
            if snapshots.iter().all(|task| task.status.terminal()) {
                let mut tasks = self.tasks.lock().await;
                for task_id in task_ids {
                    if let Some(record) = tasks.get_mut(task_id) {
                        record.collected = true;
                    }
                }
                return Ok(snapshots);
            }
            notified.await;
        }
    }

    pub async fn cancel(&self, task_ids: &[String]) -> Result<Vec<AgentTaskResult>> {
        self.stop(task_ids, TaskStatus::Cancelled, "Cancelled by parent agent")
            .await
    }

    async fn stop(
        &self,
        task_ids: &[String],
        status: TaskStatus,
        reason: &str,
    ) -> Result<Vec<AgentTaskResult>> {
        debug_assert!(matches!(
            status,
            TaskStatus::Cancelled | TaskStatus::Interrupted
        ));
        let mut senders = Vec::new();
        let mut aborts = Vec::new();
        let mut cleanup_ids = Vec::new();
        {
            let mut tasks = self.tasks.lock().await;
            for task_id in task_ids {
                if !tasks.contains_key(task_id) {
                    return Err(DeepCodeError::ToolExecution {
                        tool: "cancel_agents".to_string(),
                        message: format!("Unknown subagent task '{}'", task_id),
                    });
                }
            }
            for task_id in task_ids {
                let Some(record) = tasks.get_mut(task_id) else {
                    continue;
                };
                if !record.status.terminal() {
                    record.status = status;
                    record.error = Some(reason.to_string());
                    if let Some(sender) = record.command_tx.clone() {
                        senders.push(sender);
                    }
                    if let Some(abort) = record.agent_abort.take() {
                        aborts.push(abort);
                    }
                    if let Some(abort) = record.scheduler_abort.take() {
                        aborts.push(abort);
                    }
                    record.command_tx = None;
                    cleanup_ids.push(task_id.clone());
                }
            }
        }
        let has_active_handles = !senders.is_empty() || !aborts.is_empty();
        for sender in senders {
            let _ = sender.try_send(AgentCommand::Interrupt);
        }
        if has_active_handles {
            // Give cooperative cancellation a short chance, then guarantee termination.
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            for abort in aborts {
                abort.abort();
            }
        }
        for task_id in cleanup_ids {
            cleanup_approval_routes(&task_id);
        }
        self.notify.notify_waiters();
        let snapshots = self.snapshots(task_ids).await?;
        let mut tasks = self.tasks.lock().await;
        for task_id in task_ids {
            if let Some(record) = tasks.get_mut(task_id) {
                record.collected = true;
            }
        }
        Ok(snapshots)
    }

    async fn snapshots(&self, task_ids: &[String]) -> Result<Vec<AgentTaskResult>> {
        let tasks = self.tasks.lock().await;
        task_ids
            .iter()
            .map(|task_id| {
                tasks
                    .get(task_id)
                    .map(|record| record.snapshot(task_id))
                    .ok_or_else(|| DeepCodeError::ToolExecution {
                        tool: "wait_agents".to_string(),
                        message: format!("Unknown subagent task '{}'", task_id),
                    })
            })
            .collect()
    }

    pub async fn cancel_all(&self) {
        let ids: Vec<String> = self.tasks.lock().await.keys().cloned().collect();
        let _ = self.cancel(&ids).await;
    }

    pub async fn interrupt_all(&self) {
        let ids: Vec<String> = self.tasks.lock().await.keys().cloned().collect();
        let _ = self
            .stop(
                &ids,
                TaskStatus::Interrupted,
                "Interrupted with parent agent",
            )
            .await;
    }

    pub async fn reset_for_parent_turn(&self) {
        self.cancel_all().await;
        let task_ids = {
            let mut tasks = self.tasks.lock().await;
            let ids = tasks.keys().cloned().collect::<Vec<_>>();
            tasks.clear();
            ids
        };
        for task_id in task_ids {
            cleanup_approval_routes(&task_id);
        }
        self.notify.notify_waiters();
    }

    pub async fn wait_for_workers_idle(&self) {
        loop {
            let notified = self.notify.notified();
            let workers_active = self
                .tasks
                .lock()
                .await
                .values()
                .any(|record| record.role == AgentRole::Worker && !record.status.terminal());
            if !workers_active {
                return;
            }
            notified.await;
        }
    }

    pub(crate) async fn uncollected_task_ids(&self) -> Vec<String> {
        self.tasks
            .lock()
            .await
            .iter()
            .filter(|(_, record)| !record.collected)
            .map(|(task_id, _)| task_id.clone())
            .collect()
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".to_string())
}

struct ApprovalRoute {
    task_id: String,
    sender: mpsc::Sender<AgentCommand>,
}

fn approval_routes() -> &'static StdMutex<HashMap<String, ApprovalRoute>> {
    static ROUTES: OnceLock<StdMutex<HashMap<String, ApprovalRoute>>> = OnceLock::new();
    ROUTES.get_or_init(|| StdMutex::new(HashMap::new()))
}

fn cleanup_approval_routes(task_id: &str) {
    if let Ok(mut routes) = approval_routes().lock() {
        routes.retain(|_, route| route.task_id != task_id);
    }
}

pub(crate) fn route_nested_command(command: &AgentCommand) -> bool {
    let request_id = match command {
        AgentCommand::PermissionResponse { request_id, .. }
        | AgentCommand::FileChangePreviewResponse { request_id, .. }
        | AgentCommand::PlanResponse { request_id, .. }
        | AgentCommand::PlanFeedback { request_id, .. } => request_id,
        _ => return false,
    };
    let sender = approval_routes()
        .lock()
        .ok()
        .and_then(|mut routes| routes.remove(request_id));
    sender.is_some_and(|route| route.sender.try_send(command.clone()).is_ok())
}

fn task_ids(input: &serde_json::Value, tool: &str) -> Result<Vec<String>> {
    input
        .get("task_ids")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| DeepCodeError::ToolExecution {
            tool: tool.to_string(),
            message: "Missing 'task_ids' array".to_string(),
        })?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_string)
                .ok_or_else(|| DeepCodeError::ToolExecution {
                    tool: tool.to_string(),
                    message: "Every task id must be a string".to_string(),
                })
        })
        .collect()
}

pub struct SpawnAgentTool {
    manager: Arc<AgentTaskManager>,
}

impl SpawnAgentTool {
    pub fn new(manager: Arc<AgentTaskManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for SpawnAgentTool {
    fn name(&self) -> &str {
        "spawn_agent"
    }

    fn description(&self) -> &str {
        "Start a background subagent. Spawn independent explorer tasks in parallel and review their collected results before reaching a conclusion. Explorer is read-only; worker may modify the workspace and is serialized with other workers and parent mutations."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "role": {"type": "string", "enum": ["explorer", "worker"]},
                "model": {"type": "string"},
                "reasoning_effort": {"type": "string"}
            },
            "required": ["task", "role"]
        })
    }

    fn safety(&self) -> ToolSafety {
        // Every child performs its own permission checks. Keeping orchestration itself
        // approval-free is required to launch multiple explorers in one model response.
        ToolSafety::READ_ONLY
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let task = input
            .get("task")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| DeepCodeError::ToolExecution {
                tool: self.name().to_string(),
                message: "Missing 'task' parameter".to_string(),
            })?;
        let role = AgentRole::parse(input.get("role").and_then(serde_json::Value::as_str))?;
        let result = self
            .manager
            .spawn(
                task.to_string(),
                role,
                input.get("model").and_then(serde_json::Value::as_str),
                input
                    .get("reasoning_effort")
                    .and_then(serde_json::Value::as_str),
            )
            .await?;
        serde_json::to_string(&result).map_err(|error| DeepCodeError::ToolExecution {
            tool: self.name().to_string(),
            message: error.to_string(),
        })
    }
}

pub struct WaitAgentsTool {
    manager: Arc<AgentTaskManager>,
}

impl WaitAgentsTool {
    pub fn new(manager: Arc<AgentTaskManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for WaitAgentsTool {
    fn name(&self) -> &str {
        "wait_agents"
    }

    fn description(&self) -> &str {
        "Wait for specific background subagents and return structured results, errors, statuses, models, and token usage. Use this after spawn_agent before forming a conclusion."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({"type":"object","properties":{"task_ids":{"type":"array","items":{"type":"string"}}},"required":["task_ids"]})
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::READ_ONLY
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let results = self.manager.wait(&task_ids(&input, self.name())?).await?;
        serde_json::to_string(&results).map_err(|error| DeepCodeError::ToolExecution {
            tool: self.name().to_string(),
            message: error.to_string(),
        })
    }
}

pub struct CancelAgentsTool {
    manager: Arc<AgentTaskManager>,
}

impl CancelAgentsTool {
    pub fn new(manager: Arc<AgentTaskManager>) -> Self {
        Self { manager }
    }
}

#[async_trait]
impl Tool for CancelAgentsTool {
    fn name(&self) -> &str {
        "cancel_agents"
    }

    fn description(&self) -> &str {
        "Cancel specific queued or running background subagents and return their final statuses."
    }

    fn input_schema(&self) -> serde_json::Value {
        json!({"type":"object","properties":{"task_ids":{"type":"array","items":{"type":"string"}}},"required":["task_ids"]})
    }

    fn safety(&self) -> ToolSafety {
        ToolSafety::READ_ONLY
    }

    async fn execute(&self, input: serde_json::Value) -> Result<String> {
        let results = self.manager.cancel(&task_ids(&input, self.name())?).await?;
        serde_json::to_string(&results).map_err(|error| DeepCodeError::ToolExecution {
            tool: self.name().to_string(),
            message: error.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use deepcode_core::config::ReasoningEffort;
    use deepcode_core::error::Result as CoreResult;
    use deepcode_core::provider::traits::{
        ContextCompressor, FinishReason, GenerateParams, GenerateResponse, RequestBuilder,
        ResponseParser, StreamDelta,
    };
    use deepcode_core::types::{Message, ToolDefinition};
    use deepcode_permissions::policy::PermissionSystemConfig;
    use futures::stream;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct TestRequestBuilder;

    impl RequestBuilder for TestRequestBuilder {
        fn build_request(
            &self,
            _model: &str,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _system_prompt: Option<&str>,
            _params: &GenerateParams,
        ) -> CoreResult<serde_json::Value> {
            Ok(json!({}))
        }
    }

    struct TestParser;

    impl ResponseParser for TestParser {
        fn parse_response(&self, _raw_body: &serde_json::Value) -> CoreResult<GenerateResponse> {
            Err(DeepCodeError::Provider("unused test parser".to_string()))
        }

        fn parse_stream_chunk(&self, _raw_line: &str) -> CoreResult<Option<StreamDelta>> {
            Ok(None)
        }
    }

    struct TestCompressor;

    #[async_trait]
    impl ContextCompressor for TestCompressor {
        fn needs_compression(&self, _token_count: usize, _context_window: usize) -> bool {
            false
        }

        fn estimate_tokens(&self, messages: &[Message]) -> usize {
            messages.len()
        }

        async fn compress(
            &self,
            messages: &[Message],
            _current_tokens: usize,
            _target_tokens: usize,
        ) -> CoreResult<(Vec<Message>, usize)> {
            Ok((messages.to_vec(), messages.len()))
        }
    }

    struct DelayProvider {
        delay: Duration,
        panic_on_generate: bool,
        active: Arc<AtomicUsize>,
        max_active: Arc<AtomicUsize>,
        request_builder: TestRequestBuilder,
        parser: TestParser,
        compressor: TestCompressor,
    }

    impl DelayProvider {
        fn new(delay: Duration) -> (Arc<Self>, Arc<AtomicUsize>) {
            let max_active = Arc::new(AtomicUsize::new(0));
            (
                Arc::new(Self {
                    delay,
                    panic_on_generate: false,
                    active: Arc::new(AtomicUsize::new(0)),
                    max_active: Arc::clone(&max_active),
                    request_builder: TestRequestBuilder,
                    parser: TestParser,
                    compressor: TestCompressor,
                }),
                max_active,
            )
        }

        fn panicking() -> Arc<Self> {
            let (provider, _) = Self::new(Duration::from_millis(1));
            Arc::new(Self {
                panic_on_generate: true,
                delay: provider.delay,
                active: Arc::clone(&provider.active),
                max_active: Arc::clone(&provider.max_active),
                request_builder: TestRequestBuilder,
                parser: TestParser,
                compressor: TestCompressor,
            })
        }
    }

    #[async_trait]
    impl LlmProvider for DelayProvider {
        fn name(&self) -> &str {
            "test"
        }

        fn request_builder(&self) -> &dyn RequestBuilder {
            &self.request_builder
        }

        fn response_parser(&self) -> &dyn ResponseParser {
            &self.parser
        }

        fn context_compressor(&self) -> &dyn ContextCompressor {
            &self.compressor
        }

        async fn generate_stream(
            &self,
            _model: &str,
            _messages: &[Message],
            _tools: &[ToolDefinition],
            _system_prompt: Option<&str>,
            _params: &GenerateParams,
        ) -> CoreResult<Pin<Box<dyn futures::Stream<Item = CoreResult<StreamDelta>> + Send>>>
        {
            assert!(!self.panic_on_generate, "intentional provider panic");
            let active = Arc::clone(&self.active);
            let max_active = Arc::clone(&self.max_active);
            let delay = self.delay;
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_active.fetch_max(current, Ordering::SeqCst);
            Ok(Box::pin(stream::once(async move {
                tokio::time::sleep(delay).await;
                active.fetch_sub(1, Ordering::SeqCst);
                Ok(StreamDelta::Batch(vec![
                    StreamDelta::TextDelta("done".to_string()),
                    StreamDelta::Usage {
                        input_tokens: 10,
                        output_tokens: 2,
                        cached_input_tokens: 3,
                        cache_miss_input_tokens: 7,
                        reasoning_output_tokens: 1,
                    },
                    StreamDelta::Finished(FinishReason::Stop),
                ]))
            })))
        }

        async fn send_request(&self, _body: &serde_json::Value) -> CoreResult<serde_json::Value> {
            Ok(json!({}))
        }
    }

    fn model_profile() -> ModelProfile {
        ModelProfile {
            id: "test-model".to_string(),
            provider: "test".to_string(),
            display_name: None,
            context_window: 16_384,
            max_output_tokens: 1_024,
            reasoning_efforts: vec![ReasoningEffort::Off, ReasoningEffort::High],
        }
    }

    fn manager(
        provider: Arc<DelayProvider>,
        event_tx: EventSender,
        max_concurrent: usize,
    ) -> Arc<AgentTaskManager> {
        let permission_config = PermissionSystemConfig {
            policy_files: Vec::new(),
            write_policy_file: None,
            grants_file: None,
            ..PermissionSystemConfig::default()
        };
        AgentTaskManager::new(
            provider,
            Arc::new(ToolRegistry::new()),
            Arc::new(Mutex::new(PermissionSystem::new(permission_config))),
            Arc::new(RwLock::new(AgentRuntimeSettings {
                model: "test-model".to_string(),
                reasoning_effort: Some("off".to_string()),
                default_subagent_model: None,
                default_subagent_reasoning_effort: None,
            })),
            vec![model_profile()],
            event_tx,
            max_concurrent,
        )
    }

    #[tokio::test]
    async fn explorers_run_concurrently_and_report_complete_usage() {
        let (provider, max_active) = DelayProvider::new(Duration::from_millis(30));
        let (event_tx, _event_rx) = crate::event::event_channel();
        let manager = manager(provider, event_tx, 3);
        let mut ids = Vec::new();
        for index in 0..3 {
            ids.push(
                manager
                    .spawn(format!("inspect {index}"), AgentRole::Explorer, None, None)
                    .await
                    .unwrap()
                    .task_id,
            );
        }

        let results = manager.wait(&ids).await.unwrap();

        assert_eq!(max_active.load(Ordering::SeqCst), 3);
        assert!(results
            .iter()
            .all(|result| result.status == TaskStatus::Completed));
        assert!(results.iter().all(|result| result.usage.input_tokens == 10
            && result.usage.output_tokens == 2
            && result.usage.reasoning_output_tokens == 1));
    }

    #[tokio::test]
    async fn workers_are_serialized_even_when_agent_capacity_is_higher() {
        let (provider, max_active) = DelayProvider::new(Duration::from_millis(25));
        let (event_tx, _event_rx) = crate::event::event_channel();
        let manager = manager(provider, event_tx, 3);
        let first = manager
            .spawn("worker one".to_string(), AgentRole::Worker, None, None)
            .await
            .unwrap();
        let second = manager
            .spawn("worker two".to_string(), AgentRole::Worker, None, None)
            .await
            .unwrap();

        manager
            .wait(&[first.task_id, second.task_id])
            .await
            .unwrap();

        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cancellation_marks_running_children_and_returns_promptly() {
        let (provider, _max_active) = DelayProvider::new(Duration::from_secs(5));
        let (event_tx, mut event_rx) = crate::event::event_channel();
        let manager = manager(provider, event_tx, 1);
        let task = manager
            .spawn("long task".to_string(), AgentRole::Explorer, None, None)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while !matches!(
                event_rx.recv().await,
                Some(AgentEvent::SubagentStarted { .. })
            ) {}
        })
        .await
        .unwrap();

        let results = manager.cancel(&[task.task_id]).await.unwrap();

        assert_eq!(results[0].status, TaskStatus::Cancelled);
    }

    #[tokio::test]
    async fn task_model_override_is_validated_before_spawn() {
        let (provider, _max_active) = DelayProvider::new(Duration::from_millis(1));
        let (event_tx, _event_rx) = crate::event::event_channel();
        let manager = manager(provider, event_tx, 1);

        let error = manager
            .spawn(
                "inspect".to_string(),
                AgentRole::Explorer,
                Some("missing-model"),
                None,
            )
            .await
            .unwrap_err();

        assert!(error.to_string().contains("Unknown subagent model"));
    }

    #[tokio::test]
    async fn queued_workers_do_not_starve_explorers() {
        let (provider, max_active) = DelayProvider::new(Duration::from_millis(40));
        let (event_tx, _event_rx) = crate::event::event_channel();
        let manager = manager(provider, event_tx, 2);
        let first = manager
            .spawn("worker one".to_string(), AgentRole::Worker, None, None)
            .await
            .unwrap();
        let second = manager
            .spawn("worker two".to_string(), AgentRole::Worker, None, None)
            .await
            .unwrap();
        let explorer = manager
            .spawn("explore".to_string(), AgentRole::Explorer, None, None)
            .await
            .unwrap();

        manager
            .wait(&[first.task_id, second.task_id, explorer.task_id])
            .await
            .unwrap();

        assert_eq!(max_active.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn parent_write_barrier_waits_for_worker_completion() {
        let (provider, _max_active) = DelayProvider::new(Duration::from_millis(50));
        let (event_tx, mut event_rx) = crate::event::event_channel();
        let manager = manager(provider, event_tx, 2);
        let worker = manager
            .spawn("worker".to_string(), AgentRole::Worker, None, None)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while !matches!(
                event_rx.recv().await,
                Some(AgentEvent::SubagentStarted { .. })
            ) {}
        })
        .await
        .unwrap();

        assert!(
            tokio::time::timeout(Duration::from_millis(5), manager.wait_for_workers_idle())
                .await
                .is_err()
        );
        manager.wait_for_workers_idle().await;
        let results = manager.wait(&[worker.task_id]).await.unwrap();

        assert_eq!(results[0].status, TaskStatus::Completed);
    }

    #[tokio::test]
    async fn provider_panic_is_reported_instead_of_leaving_task_running() {
        let (event_tx, _event_rx) = crate::event::event_channel();
        let manager = manager(DelayProvider::panicking(), event_tx, 1);
        let task = manager
            .spawn("panic".to_string(), AgentRole::Explorer, None, None)
            .await
            .unwrap();

        let results = tokio::time::timeout(Duration::from_secs(1), manager.wait(&[task.task_id]))
            .await
            .expect("wait_agents must not hang after a child panic")
            .unwrap();

        assert_eq!(results[0].status, TaskStatus::Failed);
        assert!(results[0]
            .error
            .as_deref()
            .is_some_and(|error| error.contains("runtime task failed")));
    }

    #[tokio::test]
    async fn interrupt_all_reports_interrupted_status() {
        let (provider, _max_active) = DelayProvider::new(Duration::from_secs(5));
        let (event_tx, mut event_rx) = crate::event::event_channel();
        let manager = manager(provider, event_tx, 1);
        let task = manager
            .spawn("long task".to_string(), AgentRole::Explorer, None, None)
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(1), async {
            while !matches!(
                event_rx.recv().await,
                Some(AgentEvent::SubagentStarted { .. })
            ) {}
        })
        .await
        .unwrap();

        manager.interrupt_all().await;
        let results = manager.snapshots(&[task.task_id]).await.unwrap();

        assert_eq!(results[0].status, TaskStatus::Interrupted);
        assert_eq!(
            results[0].error.as_deref(),
            Some("Interrupted with parent agent")
        );
    }

    #[tokio::test]
    async fn reset_removes_previous_parent_turn_tasks() {
        let (provider, _max_active) = DelayProvider::new(Duration::from_millis(1));
        let (event_tx, _event_rx) = crate::event::event_channel();
        let manager = manager(provider, event_tx, 1);
        let task = manager
            .spawn("inspect".to_string(), AgentRole::Explorer, None, None)
            .await
            .unwrap();
        manager.wait(&[task.task_id]).await.unwrap();
        assert_eq!(manager.tasks.lock().await.len(), 1);

        manager.reset_for_parent_turn().await;

        assert!(manager.tasks.lock().await.is_empty());
    }

    #[tokio::test]
    async fn refreshed_models_are_available_to_subagents() {
        let (provider, _max_active) = DelayProvider::new(Duration::from_millis(1));
        let (event_tx, _event_rx) = crate::event::event_channel();
        let manager = manager(provider, event_tx, 1);
        let mut refreshed = model_profile();
        refreshed.id = "refreshed-model".to_string();
        manager
            .update_models(vec![model_profile(), refreshed])
            .await;
        manager
            .update_main_model("refreshed-model".to_string())
            .await;

        let task = manager
            .spawn("inspect".to_string(), AgentRole::Explorer, None, None)
            .await
            .unwrap();
        let results = manager.wait(&[task.task_id]).await.unwrap();

        assert_eq!(results[0].model, "refreshed-model");
        assert_eq!(results[0].status, TaskStatus::Completed);
    }

    #[test]
    fn task_cleanup_removes_stale_approval_routes() {
        let task_id = uuid::Uuid::new_v4().to_string();
        let request_id = uuid::Uuid::new_v4().to_string();
        let (sender, _receiver) = mpsc::channel(1);
        approval_routes().lock().unwrap().insert(
            request_id.clone(),
            ApprovalRoute {
                task_id: task_id.clone(),
                sender,
            },
        );

        cleanup_approval_routes(&task_id);

        assert!(!approval_routes().lock().unwrap().contains_key(&request_id));
    }

    #[test]
    fn panic_messages_preserve_string_payloads() {
        assert_eq!(
            panic_message(Box::new("specific failure")),
            "specific failure"
        );
        assert_eq!(
            panic_message(Box::new("owned failure".to_string())),
            "owned failure"
        );
    }
}
