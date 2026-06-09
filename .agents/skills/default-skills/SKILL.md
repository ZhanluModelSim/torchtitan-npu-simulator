---
name: default-skills
description: "用于安装或更新 torchtitan-npu 项目依赖的远程 skills；默认集为 GitCode 的 gitcode-pr、gitcode-pipeline，并支持按需安装 cann-operator-env-config（CANN 环境配置，来自 GitHub Ascend 仓，仅搭环境时需要）。仅当用户提到缺失/安装/更新这些 skills，或创建/推送 PR、按本仓 PR 模板生成描述、查看 PR 改动/评论、触发/查看/等待 GitCode CI pipeline、需要 CANN 环境配置 skill 时发现本地缺少对应 skill 时使用。已安装时直接使用对应 skill。"
---

# 远程 Skills 安装

本 skill 管理需要从外部仓库拉取的远程工具；torchtitan-npu 专属 skills 仍维护在 `.agents/skills/`。

默认集：

- `gitcode-pr`
- `gitcode-pipeline`

按需安装：

- `cann-operator-env-config`：CANN 环境配置 skill，由 `torchtitan-npu-env-setup` 在搭环境时触发安装。

## 安装流程

脚本支持按名安装：不带参数装默认集，带 skill 名则按需安装对应 skill（源仓库由脚本内 `skill_repo_url` 映射决定）。

1. 读取 `.agents/skills/default-skills/scripts/install-default-skills.sh`，确认 `DEFAULT_SKILLS` 列表与 `skill_repo_url` 映射。
2. 执行安装脚本：

```bash
# 默认集（gitcode-pr、gitcode-pipeline）
bash .agents/skills/default-skills/scripts/install-default-skills.sh

# 按需安装（仅搭环境时）
bash .agents/skills/default-skills/scripts/install-default-skills.sh cann-operator-env-config
```

3. 检查以下结果（按实际安装的 skill 对应）：
   - `.agents/skills/_remote/<skill>/`
   - `.agents/skills/<skill> -> _remote/<skill>`
4. 若 agent 侧入口未刷新，运行：

```bash
bash .agents/setup_agent.sh --agent claude --quiet
bash .agents/setup_agent.sh --agent codex --quiet
bash .agents/setup_agent.sh --agent opencode --quiet
```

`_remote` 目录和远程 skill 软链是本地安装产物，不应提交。

## 手动兜底

如果安装脚本失败，但网络可以访问 GitCode：

```bash
tmp_dir="$(mktemp -d)"
git clone --depth 1 https://gitcode.com/cann-agent/skills.git "$tmp_dir/skills"
mkdir -p .agents/skills/_remote
cp -R "$tmp_dir/skills/skills/gitcode-pr" .agents/skills/_remote/
cp -R "$tmp_dir/skills/skills/gitcode-pipeline" .agents/skills/_remote/
ln -sfn _remote/gitcode-pr .agents/skills/gitcode-pr
ln -sfn _remote/gitcode-pipeline .agents/skills/gitcode-pipeline
rm -rf "$tmp_dir"
```

## 使用场景

使用 `gitcode-pr`：

- 创建/提交 PR、pull request、merge request；
- 推送代码到 GitCode；
- 获取 PR 文件列表、diff、评论、discussion；
- 删除或回复 PR 评论；
- 按 PR 模板生成描述。

### torchtitan-npu PR 模板约束

本仓 PR 模板路径是：

```bash
.gitcode/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md
```

使用 `gitcode-pr` 创建或更新 PR 描述时，必须优先遵循本节；如果远程 `gitcode-pr` skill 中出现其他固定模板路径（例如 `.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`），以本仓路径为准。

创建 PR 前必须：

1. 读取 `.gitcode/PULL_REQUEST_TEMPLATE/PULL_REQUEST_TEMPLATE.md`。
2. 保留模板中的全部二级标题：`描述`、`类型`、`Checklist:`、`如何测试`、`其他信息`。
3. `类型` 默认只勾选一个最主要类型；只有模板或用户明确允许多选时才勾多个。
4. `Checklist` 只勾选已经真实完成的项目，不得默认全勾。
5. PR 标题必须使用带中括号的类型标签，格式为 `[type] 描述`；优先使用 `[feat]`、`[fix]`、`[refactor]`、`[docs]`、`[test]`，如使用 `[chore]`、`[ci]`、`[style]`、`[perf]` 等其他仓库已有类型，需要确保类型说明与勾选项一致。禁止使用 `feat: 描述`、`test: 描述` 或无标签标题。
6. `如何测试` 必须写清实际执行过的命令；未执行的测试不能写成已通过。

使用 `gitcode-pipeline`：

- 触发流水线、启动 CI、跑 pipeline；
- 查看流水线状态、CI 状态、pipeline 结果；
- 等待流水线结果；
- 分析 CI/pipeline 失败。
