use deepcode_core::config::{ModelOverride, ModelProfile, ProviderConfig, ReasoningEffort};
use deepcode_core::error::{DeepCodeError, Result};
use futures::{stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const HOSTED_TTL_SECS: u64 = 24 * 60 * 60;
const OLLAMA_TTL_SECS: u64 = 5 * 60;
const HARD_STALE_SECS: u64 = 7 * 24 * 60 * 60;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CatalogSource {
    Live,
    Cache,
    Builtin,
    Config,
}

impl std::fmt::Display for CatalogSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Live => "live",
            Self::Cache => "cache",
            Self::Builtin => "builtin",
            Self::Config => "config",
        })
    }
}

#[derive(Debug, Clone)]
pub struct CatalogStatus {
    pub source: CatalogSource,
    pub refreshed_at: Option<u64>,
    pub stale: bool,
    pub background_refresh: bool,
    pub next_refresh_at: Option<u64>,
    pub message: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CatalogResolution {
    pub models: Vec<ModelProfile>,
    pub status: CatalogStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CacheEntry {
    models: Vec<ModelProfile>,
    last_success: u64,
    last_attempt: u64,
    #[serde(default)]
    consecutive_failures: u32,
    #[serde(default)]
    unsupported: bool,
    etag: Option<String>,
    last_modified: Option<String>,
}

#[derive(Debug)]
struct DiscoveryResult {
    models: Vec<ModelProfile>,
    etag: Option<String>,
    last_modified: Option<String>,
    not_modified: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureKind {
    Authentication,
    Unsupported,
    Transient,
}

#[derive(Debug)]
struct DiscoveryFailure {
    kind: FailureKind,
    message: String,
}

pub fn cache_root(data_root: &Path) -> PathBuf {
    data_root.join("model-catalogs")
}

pub fn cache_key(config: &ProviderConfig) -> String {
    let mut hasher = Sha256::new();
    hasher.update(config.kind.as_bytes());
    hasher.update(b"\0");
    hasher.update(normalized_base_url(config).as_bytes());
    hasher.update(b"\0");
    if let Some(key) = config.resolve_api_key() {
        hasher.update(key.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

pub async fn resolve_model_catalog(
    provider_name: &str,
    config: &ProviderConfig,
    data_root: &Path,
    force_refresh: bool,
) -> Result<CatalogResolution> {
    let root = cache_root(data_root);
    let path = root.join(format!("{}.json", cache_key(config)));
    let mut cached = load_cache(&path);
    let now = unix_now();
    let ttl = if config.kind == "ollama" {
        OLLAMA_TTL_SECS
    } else {
        HOSTED_TTL_SECS
    };

    if !force_refresh {
        if let Some(entry) = cached.as_ref() {
            let age = now.saturating_sub(entry.last_success);
            if age <= ttl {
                return Ok(from_cache(provider_name, config, entry, false, false, None));
            }
            if !retry_due(entry, now) {
                return Ok(from_cache(
                    provider_name,
                    config,
                    entry,
                    true,
                    false,
                    Some("Model catalog refresh is in failure backoff".to_string()),
                ));
            }
            if age <= HARD_STALE_SECS {
                return Ok(from_cache(provider_name, config, entry, true, true, None));
            }
        }
    }

    match refresh_locked(provider_name, config, &root, &path, cached.as_ref()).await {
        Ok(entry) => Ok(CatalogResolution {
            models: apply_overrides(provider_name, config, entry.models.clone()),
            status: CatalogStatus {
                source: CatalogSource::Live,
                refreshed_at: Some(entry.last_success),
                stale: false,
                background_refresh: false,
                next_refresh_at: Some(entry.last_success.saturating_add(ttl_for(config))),
                message: None,
            },
        }),
        Err(failure) => {
            if let Some(entry) = cached.as_mut() {
                if failure.kind == FailureKind::Authentication {
                    return Ok(from_cache(
                        provider_name,
                        config,
                        entry,
                        true,
                        false,
                        Some(format!("Credential error: {}", failure.message)),
                    ));
                }
                entry.last_attempt = now;
                entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
                entry.unsupported = failure.kind == FailureKind::Unsupported;
                let _ = write_cache(&path, entry);
                return Ok(from_cache(
                    provider_name,
                    config,
                    entry,
                    true,
                    false,
                    Some(failure.message),
                ));
            }
            if failure.kind == FailureKind::Authentication {
                return Err(DeepCodeError::Provider(failure.message));
            }
            let fallback = fallback_catalog(provider_name, config, Some(failure.message.clone()))?;
            let entry = CacheEntry {
                models: fallback.models.clone(),
                last_success: 0,
                last_attempt: now,
                consecutive_failures: 1,
                unsupported: failure.kind == FailureKind::Unsupported,
                etag: None,
                last_modified: None,
            };
            let _ = write_cache(&path, &entry);
            Ok(fallback)
        }
    }
}

pub async fn refresh_model_catalog(
    provider_name: &str,
    config: &ProviderConfig,
    data_root: &Path,
) -> Result<()> {
    let root = cache_root(data_root);
    let path = root.join(format!("{}.json", cache_key(config)));
    let cached = load_cache(&path);
    refresh_locked(provider_name, config, &root, &path, cached.as_ref())
        .await
        .map(|_| ())
        .map_err(|failure| DeepCodeError::Provider(failure.message))
}

fn from_cache(
    provider_name: &str,
    config: &ProviderConfig,
    entry: &CacheEntry,
    stale: bool,
    background_refresh: bool,
    message: Option<String>,
) -> CatalogResolution {
    CatalogResolution {
        models: apply_overrides(provider_name, config, entry.models.clone()),
        status: CatalogStatus {
            source: CatalogSource::Cache,
            refreshed_at: Some(entry.last_success),
            stale,
            background_refresh,
            next_refresh_at: next_refresh(entry, config),
            message,
        },
    }
}

fn fallback_catalog(
    provider_name: &str,
    config: &ProviderConfig,
    message: Option<String>,
) -> Result<CatalogResolution> {
    let mut builtins = builtin_profiles(provider_name, &config.kind);
    let source = if builtins.is_empty() {
        CatalogSource::Config
    } else {
        CatalogSource::Builtin
    };
    if let Some(selected) = config.model.as_deref() {
        if !builtins.iter().any(|model| model.id == selected) {
            builtins.push(unknown_profile(provider_name, selected));
        }
    }
    let models = apply_overrides(provider_name, config, builtins);
    if models.is_empty() {
        return Err(DeepCodeError::Config(format!(
            "Provider '{}' could not list models and has no configured model override{}",
            provider_name,
            message
                .as_deref()
                .map(|value| format!(": {}", value))
                .unwrap_or_default()
        )));
    }
    Ok(CatalogResolution {
        models,
        status: CatalogStatus {
            source,
            refreshed_at: None,
            stale: true,
            background_refresh: false,
            next_refresh_at: None,
            message,
        },
    })
}

async fn refresh_locked(
    provider_name: &str,
    config: &ProviderConfig,
    root: &Path,
    path: &Path,
    cached: Option<&CacheEntry>,
) -> std::result::Result<CacheEntry, DiscoveryFailure> {
    fs::create_dir_all(root).map_err(io_failure)?;
    let lock_path = path.with_extension("lock");
    let lock = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path);
    let _lock = match lock {
        Ok(lock) => lock,
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            return load_cache(path).ok_or_else(|| DiscoveryFailure {
                kind: FailureKind::Transient,
                message: "Another process is refreshing the model catalog".to_string(),
            });
        }
        Err(error) => return Err(io_failure(error)),
    };

    let timeout = if cached.is_some() {
        Duration::from_secs(8)
    } else {
        Duration::from_secs(10)
    };
    let result = tokio::time::timeout(timeout, discover(provider_name, config, cached)).await;
    let _ = fs::remove_file(&lock_path);
    let discovered = result.map_err(|_| DiscoveryFailure {
        kind: FailureKind::Transient,
        message: "Model discovery timed out".to_string(),
    })??;
    let now = unix_now();
    let entry = if discovered.not_modified {
        let mut entry = cached.cloned().ok_or_else(|| DiscoveryFailure {
            kind: FailureKind::Transient,
            message: "Provider returned not-modified without a cached catalog".to_string(),
        })?;
        entry.last_success = now;
        entry.last_attempt = now;
        entry.consecutive_failures = 0;
        entry.unsupported = false;
        entry
    } else {
        CacheEntry {
            models: discovered.models,
            last_success: now,
            last_attempt: now,
            consecutive_failures: 0,
            unsupported: false,
            etag: discovered.etag,
            last_modified: discovered.last_modified,
        }
    };
    write_cache(path, &entry).map_err(io_failure)?;
    Ok(entry)
}

async fn discover(
    provider_name: &str,
    config: &ProviderConfig,
    cached: Option<&CacheEntry>,
) -> std::result::Result<DiscoveryResult, DiscoveryFailure> {
    match config.kind.as_str() {
        "anthropic" => discover_anthropic(provider_name, config).await,
        "ollama" => discover_ollama(provider_name, config).await,
        "openai" | "deepseek" | "kimi" => {
            discover_openai_style(provider_name, config, cached).await
        }
        _ => Err(DiscoveryFailure {
            kind: FailureKind::Unsupported,
            message: format!("Unsupported provider type '{}'", config.kind),
        }),
    }
}

async fn discover_openai_style(
    provider_name: &str,
    config: &ProviderConfig,
    cached: Option<&CacheEntry>,
) -> std::result::Result<DiscoveryResult, DiscoveryFailure> {
    let client = discovery_client(config)?;
    let mut request = client.get(format!("{}/models", normalized_base_url(config)));
    if let Some(key) = config.resolve_api_key() {
        request = request.bearer_auth(key);
    }
    if config.kind == "kimi" {
        request = request.header("User-Agent", crate::kimi::USER_AGENT);
    }
    if let Some(etag) = cached.and_then(|entry| entry.etag.as_deref()) {
        request = request.header("If-None-Match", etag);
    }
    if let Some(modified) = cached.and_then(|entry| entry.last_modified.as_deref()) {
        request = request.header("If-Modified-Since", modified);
    }
    let response = request.send().await.map_err(http_failure)?;
    if response.status() == reqwest::StatusCode::NOT_MODIFIED {
        return Ok(DiscoveryResult {
            models: Vec::new(),
            etag: None,
            last_modified: None,
            not_modified: true,
        });
    }
    let status = response.status();
    let etag = header_string(&response, "etag");
    let last_modified = header_string(&response, "last-modified");
    if !status.is_success() {
        return Err(status_failure(status));
    }
    let json: serde_json::Value = response.json().await.map_err(http_failure)?;
    let rows = json["data"].as_array().ok_or_else(|| DiscoveryFailure {
        kind: FailureKind::Transient,
        message: "Model list response does not contain a data array".to_string(),
    })?;
    let builtins = builtin_profiles(provider_name, &config.kind);
    let mut models = Vec::new();
    for row in rows {
        let Some(id) = row["id"].as_str() else {
            continue;
        };
        let builtin = builtins.iter().find(|model| model.id == id);
        if config.kind == "openai"
            && builtin.is_none()
            && config.model.as_deref() != Some(id)
            && !config.models.contains_key(id)
        {
            continue;
        }
        let context = row["context_length"]
            .as_u64()
            .map(|value| value as usize)
            .or_else(|| builtin.map(|model| model.context_window))
            .unwrap_or(32_768);
        let output = row["max_output_tokens"]
            .as_u64()
            .map(|value| value as usize)
            .or_else(|| builtin.map(|model| model.max_output_tokens))
            .unwrap_or(4_096)
            .min(context.saturating_sub(1).max(1));
        let efforts = builtin
            .map(|model| model.reasoning_efforts.clone())
            .unwrap_or_else(|| {
                if config.kind == "kimi" && row["supports_reasoning"].as_bool() == Some(true) {
                    kimi_reasoning_efforts(id, true)
                } else {
                    vec![ReasoningEffort::Off]
                }
            });
        models.push(ModelProfile {
            id: id.to_string(),
            provider: provider_name.to_string(),
            display_name: row["display_name"]
                .as_str()
                .map(str::to_string)
                .or_else(|| builtin.and_then(|model| model.display_name.clone())),
            context_window: context,
            max_output_tokens: output,
            reasoning_efforts: efforts,
        });
    }
    sort_profiles(&mut models, &builtins);
    Ok(DiscoveryResult {
        models,
        etag,
        last_modified,
        not_modified: false,
    })
}

async fn discover_anthropic(
    provider_name: &str,
    config: &ProviderConfig,
) -> std::result::Result<DiscoveryResult, DiscoveryFailure> {
    let client = discovery_client(config)?;
    let mut after_id: Option<String> = None;
    let mut models = Vec::new();
    loop {
        let mut request = client
            .get(format!("{}/models", normalized_base_url(config)))
            .header("anthropic-version", "2023-06-01")
            .header("x-api-key", config.resolve_api_key().unwrap_or_default())
            .query(&[("limit", "1000")]);
        if let Some(after) = after_id.as_deref() {
            request = request.query(&[("after_id", after)]);
        }
        let response = request.send().await.map_err(http_failure)?;
        if !response.status().is_success() {
            return Err(status_failure(response.status()));
        }
        let json: serde_json::Value = response.json().await.map_err(http_failure)?;
        let rows = json["data"].as_array().ok_or_else(|| DiscoveryFailure {
            kind: FailureKind::Transient,
            message: "Anthropic model response does not contain a data array".to_string(),
        })?;
        for row in rows {
            let Some(id) = row["id"].as_str() else {
                continue;
            };
            let mut efforts = vec![ReasoningEffort::Off];
            for effort in ReasoningEffort::ALL.into_iter().filter(|value| {
                *value != ReasoningEffort::Off && *value != ReasoningEffort::Minimal
            }) {
                if row["capabilities"]["effort"][effort.as_str()]["supported"].as_bool()
                    == Some(true)
                {
                    efforts.push(effort);
                }
            }
            let builtin = builtin_profiles(provider_name, "anthropic")
                .into_iter()
                .find(|model| model.id == id);
            let context = nonzero(row["max_input_tokens"].as_u64())
                .map(|value| value as usize)
                .or_else(|| builtin.as_ref().map(|model| model.context_window))
                .unwrap_or(200_000);
            let output = nonzero(row["max_tokens"].as_u64())
                .map(|value| value as usize)
                .or_else(|| builtin.as_ref().map(|model| model.max_output_tokens))
                .unwrap_or(32_768)
                .min(context.saturating_sub(1));
            models.push(ModelProfile {
                id: id.to_string(),
                provider: provider_name.to_string(),
                display_name: row["display_name"].as_str().map(str::to_string),
                context_window: context,
                max_output_tokens: output,
                reasoning_efforts: efforts,
            });
        }
        if json["has_more"].as_bool() != Some(true) {
            break;
        }
        after_id = json["last_id"].as_str().map(str::to_string);
        if after_id.is_none() {
            break;
        }
    }
    let builtins = builtin_profiles(provider_name, "anthropic");
    sort_profiles(&mut models, &builtins);
    Ok(DiscoveryResult {
        models,
        etag: None,
        last_modified: None,
        not_modified: false,
    })
}

async fn discover_ollama(
    provider_name: &str,
    config: &ProviderConfig,
) -> std::result::Result<DiscoveryResult, DiscoveryFailure> {
    let client = discovery_client(config)?;
    let root = normalized_base_url(config)
        .strip_suffix("/v1")
        .unwrap_or(&normalized_base_url(config))
        .to_string();
    let mut request = client.get(format!("{}/api/tags", root));
    if let Some(key) = config.resolve_api_key() {
        request = request.bearer_auth(key);
    }
    let response = request.send().await.map_err(http_failure)?;
    if !response.status().is_success() {
        return Err(status_failure(response.status()));
    }
    let json: serde_json::Value = response.json().await.map_err(http_failure)?;
    let ids = json["models"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| row["model"].as_str().or_else(|| row["name"].as_str()))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let key = config.resolve_api_key();
    let models = stream::iter(ids.into_iter().map(|id| {
        let client = client.clone();
        let root = root.clone();
        let provider_name = provider_name.to_string();
        let key = key.clone();
        async move {
            let mut request = client
                .post(format!("{}/api/show", root))
                .json(&serde_json::json!({"model": id}));
            if let Some(key) = key {
                request = request.bearer_auth(key);
            }
            let response = request.send().await.map_err(http_failure)?;
            if !response.status().is_success() {
                return Err(status_failure(response.status()));
            }
            let row: serde_json::Value = response.json().await.map_err(http_failure)?;
            let capabilities = row["capabilities"].as_array().cloned().unwrap_or_default();
            if !capabilities
                .iter()
                .any(|value| value.as_str() == Some("completion"))
            {
                return Ok(None);
            }
            let context = row["model_info"]
                .as_object()
                .and_then(|values| {
                    values.iter().find_map(|(name, value)| {
                        name.ends_with(".context_length")
                            .then(|| value.as_u64())
                            .flatten()
                    })
                })
                .unwrap_or(32_768) as usize;
            let thinking = capabilities
                .iter()
                .any(|value| value.as_str() == Some("thinking"));
            let efforts = if thinking {
                vec![
                    ReasoningEffort::Off,
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Max,
                ]
            } else {
                vec![ReasoningEffort::Off]
            };
            Ok(Some(ModelProfile {
                id: id.clone(),
                provider: provider_name,
                display_name: Some(id),
                context_window: context,
                max_output_tokens: context.saturating_sub(1).max(1),
                reasoning_efforts: efforts,
            }))
        }
    }))
    .buffer_unordered(4)
    .try_collect::<Vec<_>>()
    .await?
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    Ok(DiscoveryResult {
        models,
        etag: None,
        last_modified: None,
        not_modified: false,
    })
}

fn kimi_reasoning_efforts(model_id: &str, supports_reasoning: bool) -> Vec<ReasoningEffort> {
    match model_id {
        "k3" | "k3-256k" => vec![
            ReasoningEffort::Low,
            ReasoningEffort::High,
            ReasoningEffort::Max,
        ],
        "kimi-for-coding" | "kimi-for-coding-highspeed" => vec![ReasoningEffort::High],
        // Unknown Kimi Code models cannot safely inherit the built-in request
        // contract. Keep them usable with conservative non-reasoning defaults
        // until their parameter mapping is explicitly supported.
        _ if supports_reasoning => vec![ReasoningEffort::Off],
        _ => vec![ReasoningEffort::Off],
    }
}

pub fn builtin_profiles(provider_name: &str, kind: &str) -> Vec<ModelProfile> {
    let specs: &[(&str, &str, usize, usize, &[ReasoningEffort])] = match kind {
        "deepseek" => &[
            (
                "deepseek-v4-pro",
                "DeepSeek V4 Pro",
                1_000_000,
                393_216,
                &[
                    ReasoningEffort::Off,
                    ReasoningEffort::Low,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "deepseek-v4-flash",
                "DeepSeek V4 Flash",
                1_000_000,
                393_216,
                &[
                    ReasoningEffort::Off,
                    ReasoningEffort::Low,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ],
            ),
        ],
        "openai" => &[
            (
                "gpt-5.6",
                "GPT-5.6",
                1_050_000,
                131_072,
                &[
                    ReasoningEffort::Off,
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "gpt-5.6-sol",
                "GPT-5.6 Sol",
                1_050_000,
                131_072,
                &[
                    ReasoningEffort::Off,
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "gpt-5.6-terra",
                "GPT-5.6 Terra",
                1_050_000,
                131_072,
                &[
                    ReasoningEffort::Off,
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "gpt-5.6-luna",
                "GPT-5.6 Luna",
                1_050_000,
                131_072,
                &[
                    ReasoningEffort::Off,
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "gpt-4.1",
                "GPT-4.1",
                1_000_000,
                32_768,
                &[ReasoningEffort::Off],
            ),
        ],
        "anthropic" => &[
            (
                "claude-fable-5",
                "Claude Fable 5",
                1_000_000,
                131_072,
                &[
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "claude-opus-5",
                "Claude Opus 5",
                1_000_000,
                131_072,
                &[
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "claude-sonnet-5",
                "Claude Sonnet 5",
                1_000_000,
                131_072,
                &[
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "claude-sonnet-4-6",
                "Claude Sonnet 4.6",
                1_000_000,
                131_072,
                &[
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "claude-opus-4-8",
                "Claude Opus 4.8",
                1_000_000,
                131_072,
                &[
                    ReasoningEffort::Low,
                    ReasoningEffort::Medium,
                    ReasoningEffort::High,
                    ReasoningEffort::Xhigh,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "claude-haiku-4-5",
                "Claude Haiku 4.5",
                200_000,
                65_536,
                &[ReasoningEffort::Off],
            ),
        ],
        "kimi" => &[
            (
                "kimi-for-coding",
                "Kimi K2.7 Code",
                262_144,
                32_768,
                &[ReasoningEffort::High],
            ),
            (
                "k3",
                "Kimi K3",
                1_048_576,
                131_072,
                &[
                    ReasoningEffort::Low,
                    ReasoningEffort::High,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "k3-256k",
                "Kimi K3 256K",
                262_144,
                131_072,
                &[
                    ReasoningEffort::Low,
                    ReasoningEffort::High,
                    ReasoningEffort::Max,
                ],
            ),
            (
                "kimi-for-coding-highspeed",
                "Kimi K2.7 Code Highspeed",
                262_144,
                32_768,
                &[ReasoningEffort::High],
            ),
        ],
        _ => &[],
    };
    specs
        .iter()
        .map(|(id, display, context, output, efforts)| ModelProfile {
            id: (*id).to_string(),
            provider: provider_name.to_string(),
            display_name: Some((*display).to_string()),
            context_window: *context,
            max_output_tokens: *output,
            reasoning_efforts: efforts.to_vec(),
        })
        .collect()
}

pub fn recommended_effort(kind: &str, model: &ModelProfile) -> ReasoningEffort {
    let preferred = match kind {
        "openai" => ReasoningEffort::Medium,
        "anthropic" | "deepseek" => ReasoningEffort::High,
        "kimi" => ReasoningEffort::High,
        "ollama" => ReasoningEffort::Medium,
        _ => ReasoningEffort::Off,
    };
    if model.supports_effort(preferred) {
        preferred
    } else {
        model
            .reasoning_efforts
            .first()
            .copied()
            .unwrap_or(ReasoningEffort::Off)
    }
}

fn apply_overrides(
    provider_name: &str,
    config: &ProviderConfig,
    mut models: Vec<ModelProfile>,
) -> Vec<ModelProfile> {
    if let Some(selected) = config.model.as_deref() {
        if !models.iter().any(|model| model.id == selected) {
            models.push(unknown_profile(provider_name, selected));
        }
    }
    for (id, override_config) in &config.models {
        let index = models.iter().position(|model| model.id == *id);
        if index.is_none() {
            models.push(unknown_profile(provider_name, id));
        }
        if let Some(model) = models.iter_mut().find(|model| model.id == *id) {
            merge_override(model, override_config);
        }
    }
    models.retain(|model| model.validate().is_ok());
    models
}

fn merge_override(model: &mut ModelProfile, override_config: &ModelOverride) {
    if let Some(display) = override_config.display_name.as_ref() {
        model.display_name = Some(display.clone());
    }
    if let Some(context) = override_config.context_window {
        model.context_window = context;
    }
    if let Some(output) = override_config.max_output_tokens {
        model.max_output_tokens = output;
    }
    if let Some(efforts) = override_config.reasoning_efforts.as_ref() {
        model.reasoning_efforts = efforts.clone();
    }
}

fn unknown_profile(provider_name: &str, id: &str) -> ModelProfile {
    ModelProfile {
        id: id.to_string(),
        provider: provider_name.to_string(),
        display_name: None,
        context_window: 32_768,
        max_output_tokens: 4_096,
        reasoning_efforts: vec![ReasoningEffort::Off],
    }
}

fn sort_profiles(models: &mut [ModelProfile], builtins: &[ModelProfile]) {
    models.sort_by_key(|model| {
        (
            builtins
                .iter()
                .position(|candidate| candidate.id == model.id)
                .unwrap_or(usize::MAX),
            model.id.clone(),
        )
    });
}

fn normalized_base_url(config: &ProviderConfig) -> String {
    config
        .base_url
        .clone()
        .unwrap_or_else(|| match config.kind.as_str() {
            "openai" => "https://api.openai.com/v1".to_string(),
            "anthropic" => "https://api.anthropic.com/v1".to_string(),
            "deepseek" => "https://api.deepseek.com".to_string(),
            "kimi" => crate::kimi::DEFAULT_BASE_URL.to_string(),
            "ollama" => "http://localhost:11434".to_string(),
            _ => String::new(),
        })
        .trim_end_matches('/')
        .to_string()
}

fn discovery_client(
    config: &ProviderConfig,
) -> std::result::Result<reqwest::Client, DiscoveryFailure> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(
            config.request_timeout_secs.unwrap_or(10).min(10),
        ))
        .build()
        .map_err(http_failure)
}

fn load_cache(path: &Path) -> Option<CacheEntry> {
    let bytes = fs::read(path).ok()?;
    match serde_json::from_slice(&bytes) {
        Ok(entry) => Some(entry),
        Err(_) => {
            let corrupt = path.with_extension(format!("corrupt.{}", unix_now()));
            let _ = fs::rename(path, corrupt);
            None
        }
    }
}

fn write_cache(path: &Path, entry: &CacheEntry) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("tmp.{}", std::process::id()));
    fs::write(&temp, serde_json::to_vec_pretty(entry)?)?;
    fs::rename(temp, path)
}

fn retry_due(entry: &CacheEntry, now: u64) -> bool {
    if entry.unsupported {
        return false;
    }
    let delay = match entry.consecutive_failures {
        0 => 0,
        1 => 15 * 60,
        2 => 60 * 60,
        3 => 6 * 60 * 60,
        _ => 24 * 60 * 60,
    };
    now.saturating_sub(entry.last_attempt) >= delay
}

fn ttl_for(config: &ProviderConfig) -> u64 {
    if config.kind == "ollama" {
        OLLAMA_TTL_SECS
    } else {
        HOSTED_TTL_SECS
    }
}

fn next_refresh(entry: &CacheEntry, config: &ProviderConfig) -> Option<u64> {
    if entry.unsupported {
        return None;
    }
    let fresh_until = entry.last_success.saturating_add(ttl_for(config));
    let retry_at = entry
        .last_attempt
        .saturating_add(match entry.consecutive_failures {
            0 => 0,
            1 => 15 * 60,
            2 => 60 * 60,
            3 => 6 * 60 * 60,
            _ => 24 * 60 * 60,
        });
    Some(fresh_until.max(retry_at))
}

fn header_string(response: &reqwest::Response, name: &str) -> Option<String> {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string)
}

fn status_failure(status: reqwest::StatusCode) -> DiscoveryFailure {
    let kind = match status.as_u16() {
        401 | 403 => FailureKind::Authentication,
        404 | 405 => FailureKind::Unsupported,
        _ => FailureKind::Transient,
    };
    DiscoveryFailure {
        kind,
        message: format!("Model discovery returned HTTP {}", status),
    }
}

fn http_failure(error: impl std::fmt::Display) -> DiscoveryFailure {
    DiscoveryFailure {
        kind: FailureKind::Transient,
        message: error.to_string(),
    }
}

fn io_failure(error: impl std::fmt::Display) -> DiscoveryFailure {
    DiscoveryFailure {
        kind: FailureKind::Transient,
        message: format!("Model cache error: {}", error),
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn nonzero(value: Option<u64>) -> Option<u64> {
    value.filter(|value| *value > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn provider(kind: &str) -> ProviderConfig {
        ProviderConfig {
            kind: kind.to_string(),
            api_key: Some("secret".to_string()),
            base_url: Some("https://example.test/v1/".to_string()),
            max_concurrent_requests: None,
            request_timeout_secs: None,
            model: None,
            reasoning_effort: None,
            wire_api: None,
            models: Default::default(),
        }
    }

    #[test]
    fn cache_key_never_contains_secret_and_changes_with_credentials() {
        let first = provider("openai");
        let mut second = first.clone();
        second.api_key = Some("other-secret".to_string());
        let key = cache_key(&first);
        assert!(!key.contains("secret"));
        assert_ne!(key, cache_key(&second));
    }

    #[test]
    fn custom_model_gets_conservative_defaults() {
        let mut config = provider("openai");
        config.model = Some("custom".to_string());
        let catalog = fallback_catalog("gateway", &config, None).unwrap();
        let model = catalog
            .models
            .iter()
            .find(|model| model.id == "custom")
            .unwrap();
        assert_eq!(model.context_window, 32_768);
        assert_eq!(model.reasoning_efforts, vec![ReasoningEffort::Off]);
    }

    #[test]
    fn configured_model_is_kept_when_live_catalog_omits_it() {
        let mut config = provider("kimi");
        config.model = Some("private-kimi".to_string());
        let catalog = apply_overrides("work", &config, builtin_profiles("work", "kimi"));
        let model = catalog
            .iter()
            .find(|model| model.id == "private-kimi")
            .unwrap();

        assert_eq!(model.context_window, 32_768);
        assert_eq!(model.max_output_tokens, 4_096);
        assert_eq!(model.reasoning_efforts, vec![ReasoningEffort::Off]);
    }

    #[test]
    fn builtin_catalog_tracks_current_official_model_ids() {
        let anthropic = builtin_profiles("anthropic", "anthropic");
        assert!(anthropic.iter().any(|model| model.id == "claude-fable-5"));
        assert!(anthropic.iter().any(|model| model.id == "claude-opus-5"));

        let kimi = builtin_profiles("kimi", "kimi");
        assert_eq!(kimi.len(), 4);
        assert!(kimi.iter().any(|model| model.id == "k3"));
        assert!(kimi.iter().any(|model| model.id == "k3-256k"));
        assert!(kimi.iter().any(|model| model.id == "kimi-for-coding"));
        assert!(kimi
            .iter()
            .any(|model| model.id == "kimi-for-coding-highspeed"));
        assert_eq!(kimi[0].id, "kimi-for-coding");
        assert_eq!(kimi[0].context_window, 262_144);
        let k3 = kimi.iter().find(|model| model.id == "k3").unwrap();
        assert_eq!(k3.context_window, 1_048_576);
        assert_eq!(recommended_effort("kimi", k3), ReasoningEffort::High);
    }

    #[test]
    fn retry_backoff_grows_and_unsupported_never_retries() {
        let now = unix_now();
        let mut entry = CacheEntry {
            models: Vec::new(),
            last_success: now,
            last_attempt: now,
            consecutive_failures: 1,
            unsupported: false,
            etag: None,
            last_modified: None,
        };
        assert!(!retry_due(&entry, now + 60));
        assert!(retry_due(&entry, now + 901));
        entry.unsupported = true;
        assert!(!retry_due(&entry, now + 100_000));
    }

    async fn test_server<F>(requests: usize, handler: F) -> (String, tokio::task::JoinHandle<()>)
    where
        F: Fn(&str) -> (u16, Vec<(&'static str, &'static str)>, String) + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handler = Arc::new(handler);
        let task = tokio::spawn(async move {
            for _ in 0..requests {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut request = Vec::new();
                let mut buffer = [0_u8; 4096];
                loop {
                    let read = socket.read(&mut buffer).await.unwrap();
                    if read == 0 {
                        break;
                    }
                    request.extend_from_slice(&buffer[..read]);
                    let header_end = request
                        .windows(4)
                        .position(|window| window == b"\r\n\r\n")
                        .map(|position| position + 4);
                    let Some(header_end) = header_end else {
                        continue;
                    };
                    let headers = String::from_utf8_lossy(&request[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|value| value.trim().parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if request.len() >= header_end + content_length {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&request);
                let (status, headers, body) = handler(&request);
                let reason = if status == 304 { "Not Modified" } else { "OK" };
                let mut response = format!(
                    "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
                    status,
                    reason,
                    body.len()
                );
                for (name, value) in headers {
                    response.push_str(&format!("{}: {}\r\n", name, value));
                }
                response.push_str("\r\n");
                response.push_str(&body);
                socket.write_all(response.as_bytes()).await.unwrap();
            }
        });
        (format!("http://{}", address), task)
    }

    #[tokio::test]
    async fn discovers_openai_deepseek_and_kimi_model_metadata() {
        for kind in ["openai", "deepseek", "kimi"] {
            let kind_owned = kind.to_string();
            let (base, server) = test_server(1, move |request| {
                assert!(request.starts_with("GET /v1/models "));
                if kind_owned == "kimi" {
                    assert!(request
                        .to_ascii_lowercase()
                        .contains(&format!("user-agent: {}", crate::kimi::USER_AGENT)));
                }
                let row = if kind_owned == "kimi" {
                    serde_json::json!({
                        "id": "private-kimi",
                        "context_length": 131072,
                        "max_output_tokens": 8192,
                        "supports_reasoning": true
                    })
                } else {
                    serde_json::json!({"id": "private-model"})
                };
                (
                    200,
                    vec![("ETag", "catalog-v1")],
                    serde_json::json!({"data": [row]}).to_string(),
                )
            })
            .await;
            let mut config = provider(kind);
            if kind == "openai" {
                config.model = Some("private-model".to_string());
            }
            config.base_url = Some(format!("{}/v1", base));
            let discovered = discover_openai_style("work", &config, None).await.unwrap();
            assert_eq!(discovered.models.len(), 1);
            assert_eq!(discovered.etag.as_deref(), Some("catalog-v1"));
            if kind == "kimi" {
                assert_eq!(discovered.models[0].context_window, 131072);
                assert_eq!(
                    discovered.models[0].reasoning_efforts,
                    vec![ReasoningEffort::Off]
                );
            } else {
                assert_eq!(discovered.models[0].context_window, 32_768);
                assert_eq!(
                    discovered.models[0].reasoning_efforts,
                    vec![ReasoningEffort::Off]
                );
            }
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn openai_discovery_excludes_unconfigured_non_generation_models() {
        let (base, server) = test_server(1, |request| {
            assert!(request.starts_with("GET /v1/models "));
            (
                200,
                vec![],
                serde_json::json!({
                    "data": [
                        {"id": "gpt-5.6"},
                        {"id": "text-embedding-3-large"},
                        {"id": "gpt-image-1"}
                    ]
                })
                .to_string(),
            )
        })
        .await;
        let mut config = provider("openai");
        config.base_url = Some(format!("{}/v1", base));
        let discovered = discover_openai_style("work", &config, None).await.unwrap();
        assert_eq!(
            discovered
                .models
                .iter()
                .map(|model| model.id.as_str())
                .collect::<Vec<_>>(),
            vec!["gpt-5.6"]
        );
        server.await.unwrap();
    }

    #[tokio::test]
    async fn discovers_anthropic_pages_and_effort_capabilities() {
        let (base, server) = test_server(2, |request| {
            assert!(request.starts_with("GET /v1/models?"));
            if request.contains("after_id=page-1") {
                (
                    200,
                    vec![],
                    serde_json::json!({
                        "data": [{"id": "claude-b", "max_input_tokens": 100000, "max_tokens": 8000}],
                        "has_more": false
                    })
                    .to_string(),
                )
            } else {
                (
                    200,
                    vec![],
                    serde_json::json!({
                        "data": [{
                            "id": "claude-a",
                            "max_input_tokens": 200000,
                            "max_tokens": 16000,
                            "capabilities": {"effort": {"low": {"supported": true}, "high": {"supported": true}}}
                        }],
                        "has_more": true,
                        "last_id": "page-1"
                    })
                    .to_string(),
                )
            }
        })
        .await;
        let mut config = provider("anthropic");
        config.base_url = Some(format!("{}/v1", base));
        let discovered = discover_anthropic("work", &config).await.unwrap();
        assert_eq!(discovered.models.len(), 2);
        let first = discovered
            .models
            .iter()
            .find(|model| model.id == "claude-a")
            .unwrap();
        assert!(first.supports_effort(ReasoningEffort::Off));
        assert!(first.supports_effort(ReasoningEffort::Low));
        assert!(first.supports_effort(ReasoningEffort::High));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn ollama_show_filters_non_completion_models() {
        let (base, server) = test_server(3, |request| {
            if request.starts_with("GET /api/tags ") {
                return (
                    200,
                    vec![],
                    serde_json::json!({"models": [{"model": "coder"}, {"model": "embed"}]}).to_string(),
                );
            }
            let completion = request.contains("\"model\":\"coder\"");
            (
                200,
                vec![],
                serde_json::json!({
                    "capabilities": if completion { vec!["completion", "thinking"] } else { vec!["embedding"] },
                    "model_info": {"family.context_length": 65536}
                })
                .to_string(),
            )
        })
        .await;
        let mut config = provider("ollama");
        config.api_key = None;
        config.base_url = Some(base);
        let discovered = discover_ollama("local", &config).await.unwrap();
        assert_eq!(discovered.models.len(), 1);
        assert_eq!(discovered.models[0].id, "coder");
        assert_eq!(discovered.models[0].context_window, 65536);
        assert_eq!(discovered.models[0].max_output_tokens, 65535);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn sends_conditional_headers_and_accepts_not_modified() {
        let (base, server) = test_server(1, |request| {
            assert!(request
                .to_ascii_lowercase()
                .contains("if-none-match: old-tag"));
            (304, vec![], String::new())
        })
        .await;
        let now = unix_now();
        let cached = CacheEntry {
            models: vec![unknown_profile("work", "cached")],
            last_success: now,
            last_attempt: now,
            consecutive_failures: 0,
            unsupported: false,
            etag: Some("old-tag".to_string()),
            last_modified: None,
        };
        let mut config = provider("openai");
        config.base_url = Some(format!("{}/v1", base));
        let discovered = discover_openai_style("work", &config, Some(&cached))
            .await
            .unwrap();
        assert!(discovered.not_modified);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn fresh_and_soft_stale_cache_do_not_block_on_network() {
        let root = std::env::temp_dir().join(format!("deepcode-catalog-{}", uuid::Uuid::new_v4()));
        let config = provider("openai");
        let path = cache_root(&root).join(format!("{}.json", cache_key(&config)));
        let now = unix_now();
        let mut entry = CacheEntry {
            models: vec![unknown_profile("work", "cached")],
            last_success: now,
            last_attempt: now,
            consecutive_failures: 0,
            unsupported: false,
            etag: None,
            last_modified: None,
        };
        write_cache(&path, &entry).unwrap();
        let fresh = resolve_model_catalog("work", &config, &root, false)
            .await
            .unwrap();
        assert!(!fresh.status.stale);
        assert!(!fresh.status.background_refresh);

        entry.last_success = now - HOSTED_TTL_SECS - 1;
        write_cache(&path, &entry).unwrap();
        let stale = resolve_model_catalog("work", &config, &root, false)
            .await
            .unwrap();
        assert!(stale.status.stale);
        assert!(stale.status.background_refresh);
    }

    #[test]
    fn corrupt_cache_is_quarantined() {
        let root = std::env::temp_dir().join(format!("deepcode-catalog-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("catalog.json");
        fs::write(&path, b"not-json").unwrap();
        assert!(load_cache(&path).is_none());
        assert!(!path.exists());
        assert!(fs::read_dir(&root)
            .unwrap()
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().contains("corrupt")));
    }

    #[tokio::test]
    async fn existing_lock_uses_cache_without_network() {
        let root = std::env::temp_dir().join(format!("deepcode-catalog-{}", uuid::Uuid::new_v4()));
        let config = provider("openai");
        let catalog_root = cache_root(&root);
        let path = catalog_root.join(format!("{}.json", cache_key(&config)));
        let entry = CacheEntry {
            models: vec![unknown_profile("work", "cached")],
            last_success: unix_now(),
            last_attempt: unix_now(),
            consecutive_failures: 0,
            unsupported: false,
            etag: None,
            last_modified: None,
        };
        write_cache(&path, &entry).unwrap();
        fs::write(path.with_extension("lock"), b"locked").unwrap();
        let refreshed = refresh_locked("work", &config, &catalog_root, &path, Some(&entry))
            .await
            .unwrap();
        assert_eq!(refreshed.models[0].id, "cached");
    }

    #[tokio::test]
    async fn auth_failure_keeps_old_cache_unchanged() {
        let (base, server) = test_server(1, |_| (401, vec![], "{}".to_string())).await;
        let root = std::env::temp_dir().join(format!("deepcode-catalog-{}", uuid::Uuid::new_v4()));
        let mut config = provider("openai");
        config.base_url = Some(format!("{}/v1", base));
        let path = cache_root(&root).join(format!("{}.json", cache_key(&config)));
        let entry = CacheEntry {
            models: vec![unknown_profile("work", "cached")],
            last_success: unix_now() - HARD_STALE_SECS - 1,
            last_attempt: 0,
            consecutive_failures: 0,
            unsupported: false,
            etag: None,
            last_modified: None,
        };
        write_cache(&path, &entry).unwrap();
        let before = fs::read(&path).unwrap();
        let catalog = resolve_model_catalog("work", &config, &root, true)
            .await
            .unwrap();
        assert!(catalog.status.stale);
        assert!(catalog.status.message.unwrap().contains("Credential error"));
        assert_eq!(fs::read(&path).unwrap(), before);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn unsupported_catalog_is_persisted_without_automatic_retry() {
        let (base, server) = test_server(1, |_| (404, vec![], "{}".to_string())).await;
        let root = std::env::temp_dir().join(format!("deepcode-catalog-{}", uuid::Uuid::new_v4()));
        let mut config = provider("deepseek");
        config.base_url = Some(base);
        let first = resolve_model_catalog("work", &config, &root, false)
            .await
            .unwrap();
        assert_eq!(first.status.source, CatalogSource::Builtin);
        server.await.unwrap();

        let second = resolve_model_catalog("work", &config, &root, false)
            .await
            .unwrap();
        assert!(second.status.stale);
        assert!(!second.status.background_refresh);
        assert!(second.status.next_refresh_at.is_none());
    }
}
