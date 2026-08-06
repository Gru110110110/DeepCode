# DeepCode

[English](README.md)

**DeepCode** —— 一款用 Rust 打造的 CLI 编程智能体。支持 OpenAI、Anthropic、DeepSeek、Kimi 和 Ollama多家供应商，通过终端 TUI 流式对话，帮你读取和编辑文件，执行经过审批且受沙盒策略约束的 shell / git 命令，在允许时还能联网搜索或抓取网页，并自动保存会话以便随时恢复。

这个项目的初心，是因为 DeepSeek 缺少官方的编程智能体。项目从0到1都是Vibe Coding。现在虽然核心流程已经跑通，但还有大量功能等待补齐。欢迎你 fork 项目，把它改造成属于你自己的编程助手，我们一起让它变得更强！

## 项目截图

![DeepCode 启动界面](ScreenShot_1.png)

![DeepCode 交互界面](ScreenShot_2.png)

## 功能特性

- **提供商支持**：OpenAI、Anthropic、DeepSeek、Kimi Code 和 Ollama。
- **以提供商为中心的配置**：只配置一个提供商时会自动选中；只有配置多个提供商时才需要 `active_provider`。
- **模型目录**：DeepCode 会从提供商 API 发现模型并缓存目录；发现不可用时回退到内置或配置的模型 profile；支持 `deepcode models` 和 `/model refresh`。
- **交互式 TUI**：流式响应、Markdown 渲染、语法高亮、对话滚动、推理面板折叠/展开、文件变更预览、权限提示和会话选择器。
- **工具系统**：文件读取/写入/编辑、shell 执行、glob/grep 搜索、网页获取/搜索、git status/diff/log/add/commit/checkout/branch，以及子智能体。
- **权限与沙盒**：基于 profile 的 filesystem、network、shell 和 tool 规则；支持单次、会话级和持久批准；使用 macOS Seatbelt、Linux bubblewrap 和 Windows 受限 token 命令沙盒。
- **Plan-Act 模式**：先让 DeepCode 生成计划，批准或要求修改后，再允许执行会产生变更的工作。
- **会话持久化**：以 UUID 命名的会话文件保存对话、工作空间、提供商、模型、推理 effort 和生成标题。
- **上下文压缩**：当估算 prompt 接近当前模型上下文预算时，会执行提供商专属的历史压缩。

## 安装

### 前置要求

- 已安装 [Rust](https://rustup.rs/) 和 Cargo。

### 构建

```bash
git clone <repository-url>
cd deepcode
cargo build --release
```

二进制文件将位于 `target/release/deepcode`。

### 安装到 PATH

在项目根目录下：

```bash
cargo install --path crates/deepcode-cli
```

Cargo 会将二进制文件安装为 `deepcode`，通常位于 `~/.cargo/bin`。请确保该目录在您的 `PATH` 中：

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## 配置

将示例配置复制到配置目录：

```bash
mkdir -p ~/.config/deepcode
cp config.example.toml ~/.config/deepcode/config.toml
```

只配置一个提供商时，只需要提供类型、端点和 API key：

```toml
[providers.deepseek]
type = "deepseek"
api_key = "your-api-key-here"
base_url = "https://api.deepseek.com"
```

配置多个提供商时，需要设置 `active_provider`：

```toml
active_provider = "deepseek"

[providers.deepseek]
type = "deepseek"
api_key = "your-api-key-here"
base_url = "https://api.deepseek.com"

[providers.ollama]
type = "ollama"
base_url = "http://localhost:11434"
```

当前支持的提供商类型是 `openai`、`anthropic`、`deepseek`、`kimi` 和 `ollama`。`kimi` 提供商使用 Kimi Code 会员密钥，默认连接 `https://api.kimi.com/coding/v1`，提供 `k3`、`k3-256k`、`kimi-for-coding` 和 `kimi-for-coding-highspeed`。未显式配置模型时，会选择所有会员均可用的 `kimi-for-coding`。官方 OpenAI 端点默认使用 Responses API。OpenAI 风格网关和 DeepSeek 默认使用 Chat Completions；可以在 OpenAI 或 DeepSeek 提供商表中设置 `wire_api = "responses"` 或 `wire_api = "chat_completions"` 来覆盖。当前 DeepSeek 实现只接受 `deepseek-v4-flash` 使用 Responses API。

可选提供商字段示例：

```toml
[providers.deepseek]
type = "deepseek"
api_key = "your-api-key-here"
base_url = "https://api.deepseek.com"
model = "deepseek-v4-flash"
reasoning_effort = "high"
request_timeout_secs = 300
max_concurrent_requests = 4
```

未知或私有模型可以覆盖保守的默认 profile：

```toml
[providers.deepseek.models."private-model"]
display_name = "Private Model"
context_window = 131072
max_output_tokens = 16384
reasoning_efforts = ["off", "low", "high", "xhigh", "max"]
default_reasoning_effort = "high"
```

如果没有配置 `model`，DeepCode 会使用解析后的模型目录中的第一个模型。目录排序会优先放置已知的内置模型 profile。如果没有配置 `reasoning_effort`，DeepCode 会选择 provider/model 默认值，并在需要时回退到模型声明的第一个 effort。`/effort` 只接受当前模型支持的值。

托管模型目录的有效期是 24 小时；Ollama 模型目录的有效期是 5 分钟。软过期目录会立即使用，并在后台刷新。超过 7 天的目录会尝试同步刷新；如果已有缓存且只是瞬时失败，缓存仍可继续使用。模型目录缓存位于 `DEEPCODE_DATA_DIR/model-catalogs` 或 `~/.local/share/deepcode/model-catalogs`；API key 不会写入这些缓存文件。

查看或强制刷新当前提供商的模型目录：

```bash
deepcode models
deepcode models --refresh
```

### 工具和权限设置

如果省略 `default_permissions`，实际默认使用内置 `:workspace` profile。示例配置定义了一个名为 `project-edit` 的 profile：

```toml
default_permissions = "project-edit"

[tools]
disabled = []
max_file_size_bytes = 1048576

[permissions]
# policy_files = ["~/.config/deepcode/policies/default.star"]
# write_policy_file = "~/.config/deepcode/policies/user.star"

[permissions.project-edit]
description = "Workspace editing with explicit network and sensitive-file boundaries."
extends = ":workspace"

[permissions.project-edit.filesystem.":workspace_roots"]
"." = "write"
"**/.env*" = "deny"
"Cargo.toml" = { write = "prompt" }
"package.json" = { write = "prompt" }

[permissions.project-edit.filesystem]
"~/.ssh/**" = "deny"
"~/.aws/**" = "deny"
"/etc/**" = "deny"

[permissions.project-edit.network]
enabled = true
default = "prompt"

[permissions.project-edit.network.domains]
"github.com" = "allow"
"*.githubusercontent.com" = "allow"
"169.254.169.254" = "deny"

[[permissions.project-edit.shell.rules]]
pattern = ["git", ["status", "diff", "log"]]
decision = "allow"

[[permissions.project-edit.shell.rules]]
pattern = ["git", "push"]
decision = "deny"

[[permissions.project-edit.tool.rules]]
tool = "git_checkout"
action = "restore_files"
decision = "prompt"
justification = "Restoring files can overwrite local work."
```

- `disabled` 会在创建工具 registry 时跳过匹配的内置工具。
- `max_file_size_bytes` 适用于 `read_file`、`write_file` 和 `edit_file`。
- 权限 profile 会把 filesystem、network、shell 和 tool 规则合并为 `allow`、`prompt`、`deny` 决策。
- 交互式批准可以是单次、当前会话或持久批准。持久批准默认保存到 `~/.config/deepcode/policies/permissions.toml`。
- `write_file` 和 `edit_file` 会先准备 unified diff 预览；如果策略已经允许该操作，则直接应用。
- Shell 和 git 命令会使用权限系统选择的沙盒策略执行。需要沙盒的执行在找不到支持的后端时会失败关闭。

## 使用方法

### 在当前目录启动智能体

```bash
cd /path/to/your/project
deepcode
```

不带子命令运行 `deepcode` 会启动交互式聊天 TUI。工作空间根目录是启动命令时所在的目录。文件和搜索工具会限制在该工作空间根目录内。

DeepCode 首次进入工作空间时，会显示规范化后的文件夹路径，并要求确认是否信任该工作空间，然后才加载提供商、注册工具或启动智能体。信任按文件夹身份记忆；在同一路径删除并重新创建文件夹后，需要再次确认。

等效显式命令：

```bash
deepcode chat
```

列出会话，并通过 UUID 或当前工作空间的最新会话恢复：

```bash
deepcode sessions
deepcode sessions --all --limit 50
deepcode resume 550e8400-e29b-41d4-a716-446655440000
deepcode resume --last
```

### 单次执行

```bash
deepcode run "向我解释这个代码库"
```

单次模式和聊天模式一样使用工作空间信任、提供商解析、工具、权限、文件变更预览和会话检查点。

### TUI 斜杠命令

| 命令 | 说明 |
|------|------|
| `/model` | 显示当前模型和可用选择 |
| `/model <name>` | 切换到当前模型目录中的另一个模型 |
| `/model refresh` | 强制刷新当前提供商的模型目录 |
| `/effort` | 显示当前推理 effort 和当前模型可用选项 |
| `/effort <tier>` | 设置当前模型支持的 effort |
| `/permissions` | 显示当前权限策略的只读快照 |
| `/sessions` | 浏览当前工作空间的已保存会话 |
| `/resume` | 浏览当前工作空间的已保存会话 |
| `/resume <id>` | 直接恢复指定会话 |
| `/plan` | 为后续任务启用持久 Plan-Act 模式 |
| `/plan off` | 禁用持久 Plan-Act 模式 |
| `/plan <task>` | 为单个任务生成计划并等待批准 |
| `/act` | 禁用 Plan-Act 模式并回到直接执行 |
| `/clear` | 保存当前会话并开始一个干净的新会话 |
| `/help` | 显示可用命令 |
| `/exit` | 退出会话 |
| `/quit` | 退出会话 |

在 Plan-Act 模式下，DeepCode 会先生成逐步计划，等待批准后才执行。在规划期间，只暴露不需要批准的只读工具。

### TUI 键盘与鼠标

| 快捷键 | 操作 |
|--------|------|
| `Ctrl+C` | 退出 DeepCode |
| `Esc` | 中断当前工作；拒绝当前提示 |
| `Enter` | 发送输入或接受当前选中的提示操作 |
| `Left` / `Right` / `Tab` | 在权限、计划或文件预览提示中切换选项 |
| `Shift+Up` / `Shift+Down` 或 `Alt+Up` / `Alt+Down` | 滚动对话记录 |
| `PageUp` / `PageDown` | 按页滚动对话记录 |
| `Home` / `End` | 在输入框中移动；输入为空时 `End` 跳到对话底部 |
| `Ctrl+O` | 展开或折叠推理面板 |
| 鼠标滚轮 | 滚动对话记录 |
| 鼠标点击 | 将选区复制到剪贴板 |

会话选择器按键：`Up`/`Down` 选择，`Enter` 恢复，`a` 在当前工作空间和所有工作空间会话之间切换，`Esc` 或 `q` 关闭。

权限提示按键：`y` 单次允许，`s` 会话允许，`a` 始终允许，`n` 拒绝，`q` 退出。计划提示按键：`y`/`a` 批准，`r` 修改，`n` 拒绝，`q` 退出。文件预览按键：`y`/`a` 应用，`n`/`r` 拒绝，`q` 退出。

## CLI 参考

```bash
deepcode --provider openai --model gpt-4.1 chat
deepcode --provider deepseek --model deepseek-v4-flash chat
deepcode --config /path/to/config.toml run "Hello"
deepcode --log-level debug chat
```

全局标志：

| 标志 | 说明 |
|------|------|
| `--config <path>` | 配置文件路径（默认：`~/.config/deepcode/config.toml`） |
| `--provider <name>` | 覆盖配置中的提供商 |
| `--model <name>` | 覆盖配置中的模型 |
| `--log-level <level>` | 日志级别：`trace`、`debug`、`info`、`warn`、`error`、`off`（默认：`info`） |

子命令：

| 命令 | 说明 |
|------|------|
| `deepcode` / `deepcode chat` | 启动交互式 TUI |
| `deepcode run <prompt>` | 执行单次 prompt |
| `deepcode config` | 打印已脱敏配置和模型目录状态 |
| `deepcode models [--refresh]` | 列出当前提供商模型目录 |
| `deepcode sessions [--all] [--limit N]` | 列出已保存会话 |
| `deepcode resume <id>` | 恢复指定会话 |
| `deepcode resume --last` | 恢复当前工作空间的最新会话 |

## 数据与日志

默认情况下，DeepCode 会保存：

- 会话：`~/.local/share/deepcode/sessions`
- 模型目录：`~/.local/share/deepcode/model-catalogs`
- 工作空间信任记录：`~/.local/share/deepcode/trusted_workspaces.json`
- 持久权限批准：`~/.config/deepcode/policies/permissions.toml`
- 日志：`~/.local/state/deepcode/logs/deepcode.log`

覆盖示例：

```bash
DEEPCODE_DATA_DIR=/tmp/deepcode-data deepcode run "hello"
DEEPCODE_LOG_FILE=/tmp/deepcode.log deepcode chat
DEEPCODE_STATE_DIR=/tmp/deepcode-state deepcode chat
```

## 内置工具

| 工具 | 说明 | 安全性 |
|------|------|--------|
| `read_file` | 读取文件内容，支持可选 offset/limit | 只读 |
| `write_file` | 写入或覆盖文件，支持预览 | 安全变更 |
| `edit_file` | 替换或插入现有文件的部分内容，支持预览 | 安全变更 |
| `shell` | 按 shell 权限策略执行命令 | 破坏性 |
| `glob` | 文件模式匹配 | 只读 |
| `grep` | 在文件中搜索文本 | 只读 |
| `web_fetch` | 获取 HTTP(S) 内容并返回文本 | 网络 |
| `web_search` | 搜索 DuckDuckGo HTML 结果 | 网络 |
| `git_status` | 显示 git 工作树状态 | 只读 |
| `git_diff` | 显示 git 差异 | 只读 |
| `git_log` | 显示 git 提交历史 | 只读 |
| `git_add` | 暂存文件以便提交 | 安全变更 |
| `git_commit` | 创建 git 提交 | 安全变更 |
| `git_checkout` | 检出分支或恢复文件 | 破坏性 |
| `git_branch` | 列出、创建或删除分支 | 安全变更 |
| `agent` | 使用全新对话上下文启动子智能体 | 安全变更 |

文件和搜索工具限制在启动 `deepcode` 的工作空间目录内。Shell 命令从经过工作空间验证的工作目录运行，遵守配置的 shell 权限策略，60 秒后超时，并截断非常大的输出。

## 项目结构

```text
crates/
├── deepcode-core/        # 核心类型、配置、错误、提供商 trait
├── deepcode-providers/   # LLM 提供商实现和模型目录
├── deepcode-tools/       # 内置工具 registry 和实现
├── deepcode-agent/       # 智能体循环、状态管理、规划、流式处理
├── deepcode-permissions/ # 权限策略管道和批准
├── deepcode-sandbox/     # OS 沙盒命令准备
└── deepcode-cli/         # CLI 入口、命令、会话、TUI
```

## 开发

```bash
# 编译
cargo check

# 运行测试
cargo test --workspace --all-targets

# 带文件日志运行
cargo run -- --log-level debug --config config.example.toml chat

# 检查格式化
cargo fmt --all -- --check

# 运行 clippy
cargo clippy --workspace --all-targets -- -D warnings
```

Windows 后端使用受限 token、按路径隔离的 capability SID、继承 ACL 和 Job Object 强制执行文件写入边界并清理整个进程树。读取权限仍继承当前 Windows 用户，因此 filesystem read-deny 规则继续由工具权限层执行。非管理员模式下的网络阻断属于尽力而为（代理和包管理器环境变量），因此联网命令仍需经过正常权限批准。可用 `DEEPCODE_SHELL` 覆盖默认的 PowerShell 可执行文件。

CI workflow 会在 Ubuntu、macOS 和 Windows 上运行格式化、clippy 和测试。Linux CI 会先安装 `bubblewrap`，再测试沙盒命令执行。

## 许可证

DeepCode 使用 [MIT License](LICENSE) 授权。参考的上游实现见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
