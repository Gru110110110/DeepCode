# Third-Party Notices

## OpenAI Codex

DeepCode's Windows restricted-token sandbox design is adapted from the Windows
sandbox architecture in [OpenAI Codex](https://github.com/openai/codex), which
is licensed under the Apache License 2.0.

The adapted design includes restricted process tokens, path-scoped capability
SIDs, filesystem ACL grants, explicit process handle lists, and Job Object
process-tree containment. DeepCode's implementation is maintained independently
in `crates/deepcode-sandbox/src/windows.rs`.
