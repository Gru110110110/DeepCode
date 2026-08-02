use std::collections::HashSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::PathBuf;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::execpolicy::Decision;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkDecision {
    Allow,
    Deny,
    Prompt,
}

impl NetworkDecision {
    pub fn into_core(self) -> Decision {
        match self {
            Self::Allow => Decision::Allow,
            Self::Prompt => Decision::Prompt,
            Self::Deny => Decision::Forbidden,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPolicy {
    #[serde(default = "default_decision")]
    pub default: NetworkDecision,
    #[serde(default)]
    pub allow: Vec<String>,
    #[serde(default)]
    pub deny: Vec<String>,
    #[serde(default)]
    pub audit_log: Option<PathBuf>,
}

fn default_decision() -> NetworkDecision {
    NetworkDecision::Prompt
}

impl Default for NetworkPolicy {
    fn default() -> Self {
        Self {
            default: NetworkDecision::Prompt,
            allow: Vec::new(),
            deny: Vec::new(),
            audit_log: None,
        }
    }
}

impl NetworkPolicy {
    pub fn evaluate_host(&self, host: &str) -> NetworkDecision {
        let normalized = host.trim_end_matches('.').to_ascii_lowercase();
        if self
            .deny
            .iter()
            .any(|pattern| host_matches(pattern, &normalized))
        {
            return NetworkDecision::Deny;
        }
        if is_private_or_metadata_host(&normalized) {
            return NetworkDecision::Prompt;
        }
        if self
            .allow
            .iter()
            .any(|pattern| host_matches(pattern, &normalized))
        {
            return NetworkDecision::Allow;
        }
        self.default
    }

    pub fn persist_allow(&mut self, host: &str) {
        if !self.allow.iter().any(|h| h == host) {
            self.allow.push(host.to_ascii_lowercase());
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct NetworkSessionCache {
    approved_hosts: HashSet<String>,
    denied_hosts: HashSet<String>,
}

impl NetworkSessionCache {
    pub fn allow(&mut self, host: &str) {
        self.approved_hosts.insert(host.to_ascii_lowercase());
        self.denied_hosts.remove(&host.to_ascii_lowercase());
    }

    pub fn deny(&mut self, host: &str) {
        self.denied_hosts.insert(host.to_ascii_lowercase());
        self.approved_hosts.remove(&host.to_ascii_lowercase());
    }

    pub fn cached(&self, host: &str) -> Option<NetworkDecision> {
        let host = host.to_ascii_lowercase();
        if self.denied_hosts.contains(&host) {
            return Some(NetworkDecision::Deny);
        }
        if self.approved_hosts.contains(&host) {
            return Some(NetworkDecision::Allow);
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct NetworkPolicyDecider {
    pub policy: NetworkPolicy,
    pub session: NetworkSessionCache,
}

impl NetworkPolicyDecider {
    pub fn new(policy: NetworkPolicy) -> Self {
        Self {
            policy,
            session: NetworkSessionCache::default(),
        }
    }

    pub fn evaluate(&self, host: &str) -> NetworkDecision {
        self.session
            .cached(host)
            .unwrap_or_else(|| self.policy.evaluate_host(host))
    }

    pub fn audit(
        &self,
        host: &str,
        tool: &str,
        decision: NetworkDecision,
    ) -> deepcode_core::error::Result<()> {
        let Some(path) = &self.policy.audit_log else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| deepcode_core::error::DeepCodeError::Config(e.to_string()))?;
        }
        let line = format!(
            "{} network {} {} {:?}\n",
            Utc::now().to_rfc3339(),
            host,
            tool,
            decision
        );
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .and_then(|mut file| std::io::Write::write_all(&mut file, line.as_bytes()))
            .map_err(|e| deepcode_core::error::DeepCodeError::Config(e.to_string()))
    }
}

pub fn host_from_url(value: &str) -> anyhow::Result<String> {
    let url = Url::parse(value)?;
    let Some(host) = url.host_str() else {
        anyhow::bail!("URL has no host");
    };
    Ok(host.trim_end_matches('.').to_ascii_lowercase())
}

pub(crate) fn host_from_tool_input(tool_name: &str, input: &serde_json::Value) -> Option<String> {
    if tool_name == "web_search" {
        return Some("web-search".to_string());
    }
    input
        .get("url")
        .and_then(serde_json::Value::as_str)
        .or_else(|| input.get("input").and_then(serde_json::Value::as_str))
        .and_then(|url| host_from_url(url).ok())
}

pub fn host_matches(pattern: &str, host: &str) -> bool {
    let pattern = pattern.trim_end_matches('.').to_ascii_lowercase();
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if pattern == "*" {
        return true;
    }
    if pattern == host {
        return true;
    }
    if let Some(suffix) = pattern.strip_prefix("**.") {
        return host == suffix || host.ends_with(&format!(".{}", suffix));
    }
    if let Some(suffix) = pattern
        .strip_prefix("*.")
        .or_else(|| pattern.strip_prefix('.'))
    {
        return host.ends_with(&format!(".{}", suffix)) && host != suffix;
    }
    false
}

pub fn is_private_or_metadata_host(host: &str) -> bool {
    if matches!(
        host,
        "localhost" | "ip6-localhost" | "metadata.google.internal"
    ) {
        return true;
    }
    if host == "169.254.169.254" {
        return true;
    }
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => is_private_ipv4(ip),
        Ok(IpAddr::V6(ip)) => is_private_ipv6(ip),
        Err(_) => false,
    }
}

fn is_private_ipv4(ip: Ipv4Addr) -> bool {
    ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_documentation()
        || ip.octets()[0] == 0
}

fn is_private_ipv6(ip: Ipv6Addr) -> bool {
    ip.is_loopback() || ip.is_unique_local() || ip.is_unspecified()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subdomain_matching_requires_real_subdomain() {
        assert!(host_matches("*.example.com", "api.example.com"));
        assert!(host_matches(".example.com", "a.b.example.com"));
        assert!(!host_matches("*.example.com", "example.com"));
    }

    #[test]
    fn deny_wins_over_allow() {
        let policy = NetworkPolicy {
            allow: vec!["*.example.com".to_string()],
            deny: vec!["bad.example.com".to_string()],
            ..Default::default()
        };
        assert_eq!(
            policy.evaluate_host("ok.example.com"),
            NetworkDecision::Allow
        );
        assert_eq!(
            policy.evaluate_host("bad.example.com"),
            NetworkDecision::Deny
        );
    }

    #[test]
    fn metadata_is_prompted() {
        assert!(is_private_or_metadata_host("169.254.169.254"));
        assert!(is_private_or_metadata_host("localhost"));
    }

    #[test]
    fn explicit_private_host_deny_wins_over_prompt() {
        let policy = NetworkPolicy {
            deny: vec!["localhost".to_string()],
            ..Default::default()
        };
        assert_eq!(policy.evaluate_host("localhost"), NetworkDecision::Deny);
    }
}
