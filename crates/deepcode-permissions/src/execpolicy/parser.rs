use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use starlark::any::ProvidesStaticType;
use starlark::environment::{GlobalsBuilder, Module};
use starlark::eval::Evaluator;
use starlark::syntax::{AstModule, Dialect};
use starlark::values::list::ListRef;
use starlark::values::none::NoneType;
use starlark::values::Value;

use super::rule::{Decision, PatternToken, Policy, PrefixPattern, PrefixRule};

#[derive(Debug, Default, ProvidesStaticType)]
struct RuleStore {
    source: RefCell<Option<String>>,
    rules: RefCell<Vec<PrefixRule>>,
    network_rules: RefCell<Vec<(String, Decision)>>,
    filesystem_rules: RefCell<Vec<FilesystemPolicyRule>>,
    tool_rules: RefCell<Vec<ToolPolicyRule>>,
}

impl RuleStore {
    fn push(&self, mut rule: PrefixRule) {
        rule.source = self.source.borrow().clone();
        self.rules.borrow_mut().push(rule);
    }

    fn take_rules(&self) -> Vec<PrefixRule> {
        std::mem::take(&mut *self.rules.borrow_mut())
    }

    fn push_network(&self, host: String, decision: Decision) {
        self.network_rules.borrow_mut().push((host, decision));
    }

    fn take_network_rules(&self) -> Vec<(String, Decision)> {
        std::mem::take(&mut *self.network_rules.borrow_mut())
    }

    fn push_filesystem(&self, path: String, decision: Decision) {
        self.filesystem_rules
            .borrow_mut()
            .push(FilesystemPolicyRule {
                path,
                decision,
                source: self.source.borrow().clone(),
            });
    }

    fn take_filesystem_rules(&self) -> Vec<FilesystemPolicyRule> {
        std::mem::take(&mut *self.filesystem_rules.borrow_mut())
    }

    fn push_tool(
        &self,
        tool: String,
        action: Option<String>,
        target: Option<String>,
        decision: Decision,
        justification: Option<String>,
    ) {
        self.tool_rules.borrow_mut().push(ToolPolicyRule {
            tool,
            action,
            target,
            decision,
            justification,
            source: self.source.borrow().clone(),
        });
    }

    fn take_tool_rules(&self) -> Vec<ToolPolicyRule> {
        std::mem::take(&mut *self.tool_rules.borrow_mut())
    }
}

#[derive(Debug, Clone, Default)]
pub struct PolicyParser;

#[derive(Debug, Clone, Default)]
pub struct PolicyBundle {
    pub policy: Policy,
    pub network_rules: Vec<(String, Decision)>,
    pub filesystem_rules: Vec<FilesystemPolicyRule>,
    pub tool_rules: Vec<ToolPolicyRule>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilesystemPolicyRule {
    pub path: String,
    pub decision: Decision,
    pub source: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolPolicyRule {
    pub tool: String,
    pub action: Option<String>,
    pub target: Option<String>,
    pub decision: Decision,
    pub justification: Option<String>,
    pub source: Option<String>,
}

impl PolicyParser {
    pub fn parse_files(paths: &[PathBuf]) -> anyhow::Result<Policy> {
        Ok(Self::parse_files_with_metadata(paths)?.policy)
    }

    pub fn parse_files_with_metadata(paths: &[PathBuf]) -> anyhow::Result<PolicyBundle> {
        let mut bundle = PolicyBundle::default();
        for path in paths {
            if path.exists() {
                bundle.merge(Self::parse_file_with_metadata(path)?);
            }
        }
        Ok(bundle)
    }

    pub fn parse_file(path: &Path) -> anyhow::Result<Policy> {
        Ok(Self::parse_file_with_metadata(path)?.policy)
    }

    pub fn parse_file_with_metadata(path: &Path) -> anyhow::Result<PolicyBundle> {
        let content = std::fs::read_to_string(path)?;
        Self::parse_source(path.display().to_string(), &content)
    }

    pub fn parse_source(
        filename: impl Into<String>,
        content: &str,
    ) -> anyhow::Result<PolicyBundle> {
        let filename = filename.into();
        let ast = AstModule::parse(&filename, content.to_string(), &Dialect::Standard)
            .map_err(|err| anyhow::anyhow!(err.to_string()))?;
        let globals = GlobalsBuilder::new().with(execpolicy_globals).build();
        let store = RuleStore::default();
        store.source.replace(Some(filename));
        Module::with_temp_heap(|module| {
            let mut eval = Evaluator::new(&module);
            eval.extra = Some(&store);
            eval.eval_module(ast, &globals)?;
            starlark::Result::Ok(())
        })
        .map_err(|err| anyhow::anyhow!(err.to_string()))?;

        let mut bundle = PolicyBundle::default();
        for rule in store.take_rules() {
            rule.validate_examples()?;
            bundle.policy.add_prefix_rule(rule);
        }
        for (host, decision) in store.take_network_rules() {
            bundle.network_rules.push((host, decision));
        }
        bundle
            .network_rules
            .sort_by(|left, right| left.0.cmp(&right.0));
        bundle.network_rules.dedup();
        bundle.filesystem_rules = store.take_filesystem_rules();
        bundle.tool_rules = store.take_tool_rules();
        Ok(bundle)
    }
}

impl PolicyBundle {
    fn merge(&mut self, other: PolicyBundle) {
        self.policy.merge(other.policy);
        self.network_rules.extend(other.network_rules);
        self.network_rules
            .sort_by(|left, right| left.0.cmp(&right.0));
        self.network_rules.dedup();
        self.filesystem_rules.extend(other.filesystem_rules);
        self.tool_rules.extend(other.tool_rules);
    }
}

#[starlark::starlark_module]
fn execpolicy_globals(builder: &mut GlobalsBuilder) {
    fn prefix_rule<'v>(
        pattern: Value<'v>,
        #[starlark(require = named, default = NoneType)] decision: Value<'v>,
        #[starlark(require = named, default = NoneType)] r#match: Value<'v>,
        #[starlark(require = named, default = NoneType)] not_match: Value<'v>,
        #[starlark(require = named, default = NoneType)] justification: Value<'v>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<NoneType> {
        let decision = optional_str(decision)
            .map(Decision::from_str)
            .transpose()?
            .unwrap_or(Decision::Allow);
        let justification = optional_str(justification).map(ToString::to_string);
        let rule = PrefixRule {
            pattern: value_to_pattern(pattern)?,
            decision,
            match_examples: value_to_examples(r#match)?,
            not_match_examples: value_to_examples(not_match)?,
            justification,
            source: None,
        };
        eval.extra
            .ok_or_else(|| anyhow::anyhow!("missing execpolicy rule store"))?
            .downcast_ref::<RuleStore>()
            .ok_or_else(|| anyhow::anyhow!("invalid execpolicy rule store"))?
            .push(rule);
        Ok(NoneType)
    }

    fn network_rule<'v>(
        host: &'v str,
        #[starlark(require = named, default = NoneType)] decision: Value<'v>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<NoneType> {
        let decision = optional_str(decision)
            .map(Decision::from_str)
            .transpose()?
            .unwrap_or(Decision::Allow);
        eval.extra
            .ok_or_else(|| anyhow::anyhow!("missing execpolicy rule store"))?
            .downcast_ref::<RuleStore>()
            .ok_or_else(|| anyhow::anyhow!("invalid execpolicy rule store"))?
            .push_network(host.to_ascii_lowercase(), decision);
        Ok(NoneType)
    }

    fn filesystem_rule<'v>(
        path: &'v str,
        #[starlark(require = named, default = NoneType)] decision: Value<'v>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<NoneType> {
        let decision = optional_str(decision)
            .map(Decision::from_str)
            .transpose()?
            .unwrap_or(Decision::Allow);
        eval.extra
            .ok_or_else(|| anyhow::anyhow!("missing execpolicy rule store"))?
            .downcast_ref::<RuleStore>()
            .ok_or_else(|| anyhow::anyhow!("invalid execpolicy rule store"))?
            .push_filesystem(path.to_string(), decision);
        Ok(NoneType)
    }

    fn tool_rule<'v>(
        tool: &'v str,
        #[starlark(require = named, default = NoneType)] action: Value<'v>,
        #[starlark(require = named, default = NoneType)] target: Value<'v>,
        #[starlark(require = named, default = NoneType)] decision: Value<'v>,
        #[starlark(require = named, default = NoneType)] justification: Value<'v>,
        eval: &mut Evaluator<'v, '_, '_>,
    ) -> anyhow::Result<NoneType> {
        let decision = optional_str(decision)
            .map(Decision::from_str)
            .transpose()?
            .unwrap_or(Decision::Allow);
        eval.extra
            .ok_or_else(|| anyhow::anyhow!("missing execpolicy rule store"))?
            .downcast_ref::<RuleStore>()
            .ok_or_else(|| anyhow::anyhow!("invalid execpolicy rule store"))?
            .push_tool(
                tool.to_string(),
                optional_str(action).map(str::to_string),
                optional_str(target).map(str::to_string),
                decision,
                optional_str(justification).map(str::to_string),
            );
        Ok(NoneType)
    }
}

fn optional_str(value: Value<'_>) -> Option<&str> {
    if value.is_none() {
        None
    } else {
        value.unpack_str()
    }
}

fn value_to_pattern(value: Value<'_>) -> anyhow::Result<PrefixPattern> {
    if let Some(text) = value.unpack_str() {
        let tokens = shlex::split(text)
            .ok_or_else(|| anyhow::anyhow!("could not parse prefix_rule pattern `{}`", text))?;
        return PrefixPattern::new(tokens.into_iter().map(PatternToken::Single).collect());
    }

    let Some(list) = ListRef::from_value(value) else {
        anyhow::bail!("prefix_rule pattern must be a string or list");
    };
    let mut tokens = Vec::new();
    for item in list.iter() {
        if let Some(token) = item.unpack_str() {
            tokens.push(PatternToken::Single(token.to_string()));
            continue;
        }
        let Some(alts) = ListRef::from_value(item) else {
            anyhow::bail!("prefix_rule alternatives must be a list of strings");
        };
        let mut alternatives = Vec::new();
        for alt in alts.iter() {
            let Some(text) = alt.unpack_str() else {
                anyhow::bail!("prefix_rule alternative must be a string");
            };
            alternatives.push(text.to_string());
        }
        if alternatives.is_empty() {
            anyhow::bail!("prefix_rule alternative list cannot be empty");
        }
        tokens.push(PatternToken::Alternatives(alternatives));
    }
    PrefixPattern::new(tokens)
}

fn value_to_examples(value: Value<'_>) -> anyhow::Result<Vec<Vec<String>>> {
    if value.is_none() {
        return Ok(Vec::new());
    }
    if let Some(text) = value.unpack_str() {
        return Ok(vec![parse_example(text)?]);
    }
    let Some(list) = ListRef::from_value(value) else {
        anyhow::bail!("match/not_match must be a string or list of strings");
    };
    let mut examples = Vec::new();
    for item in list.iter() {
        let Some(text) = item.unpack_str() else {
            anyhow::bail!("match/not_match examples must be strings");
        };
        examples.push(parse_example(text)?);
    }
    Ok(examples)
}

fn parse_example(value: &str) -> anyhow::Result<Vec<String>> {
    shlex::split(value).ok_or_else(|| anyhow::anyhow!("could not parse example `{}`", value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execpolicy::ExecPolicyCheckCommand;

    #[test]
    fn parses_prefix_rule_with_alternatives() {
        let bundle = PolicyParser::parse_source(
            "test.star",
            r#"
prefix_rule(["git", ["status", "diff"]], decision = "allow", match = ["git status", "git diff"], not_match = "git push")
"#,
        )
        .unwrap();
        let eval = bundle.policy.check(
            &ExecPolicyCheckCommand::parse("git diff").unwrap(),
            Some(Decision::Prompt),
        );
        assert_eq!(eval.decision, Decision::Allow);
    }

    #[test]
    fn invalid_example_fails_load() {
        let err = PolicyParser::parse_source(
            "test.star",
            r#"prefix_rule("git status", match = "git push")"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("does not match"));
    }

    #[test]
    fn parses_network_rules() {
        let bundle = PolicyParser::parse_source(
            "test.star",
            r#"
network_rule("api.example.com", decision = "allow")
network_rule("bad.example.com", decision = "forbidden")
"#,
        )
        .unwrap();
        assert!(bundle
            .network_rules
            .contains(&("api.example.com".to_string(), Decision::Allow)));
        assert!(bundle
            .network_rules
            .contains(&("bad.example.com".to_string(), Decision::Forbidden)));
    }

    #[test]
    fn parses_filesystem_and_tool_rules() {
        let bundle = PolicyParser::parse_source(
            "test.star",
            r#"
filesystem_rule("~/.ssh/**", decision = "forbidden")
tool_rule("git_checkout", action = "restore_files", decision = "prompt", justification = "restore can overwrite")
"#,
        )
        .unwrap();
        assert_eq!(bundle.filesystem_rules.len(), 1);
        assert_eq!(bundle.filesystem_rules[0].path, "~/.ssh/**");
        assert_eq!(bundle.tool_rules.len(), 1);
        assert_eq!(bundle.tool_rules[0].tool, "git_checkout");
    }
}
