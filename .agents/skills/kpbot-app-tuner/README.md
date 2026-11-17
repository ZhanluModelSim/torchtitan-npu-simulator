# KPBot App Tuner

本目录为 torchtitan-npu 提供 KPBot App Tuner 插件的项目级入口，实际工作流和配套资源复用固定版本的上游 submodule。项目级 Agent 运行时仅注册一个`kpbot-app-tuner` 入口；上游 `subskills` 不单独注册，由主流程按需调用。该设计避免同名入口冲突，不影响用户 prompt 触发主 Skill。

## 初始化

场景一：克隆 torchtitan-npu 时同时初始化 KPBot 子模块

```shell
git clone --recurse-submodules https://gitcode.com/cann/torchtitan-npu.git
```

场景二：已通过普通方式克隆 torchtitan-npu 但尚未初始化 KPBot 子模块

```shell
git submodule update --init --recursive third_party/agent-skills/KPBot
```

如需仅在本地跟随 KPBot 上游分支更新，可在 torchtitan-npu 仓库根目录执行以下操作，该操作仅实现本地更新：

```shell
git submodule update --remote -- third_party/agent-skills/KPBot
```

## OpenCode 使用

在仓库根目录启动 OpenCode，确认 Skill 已注册：

```shell
opencode debug skill | grep kpbot-app-tuner
opencode
```

在对话中调用：

```text
/kpbot-app-tuner 帮我优化当前 torchtitan-npu 训练任务中host bound场景的性能。请先确认运行环境、启动脚本、训练参数配置、模型参数配置以及明确关注的核心指标，再尝试 CPU 亲和性优化、OS 优化、BIOS优化、编译优化等各方向，涉及运行环境变更操作时先向我确认。
```

入口会读取 `third_party/agent-skills/KPBot/Plugins/app-tuner/opencode/` 覆盖层，并按上游流程执行场景确认、基线建立、证据采集、优化验证、回退和报告输出。

## Claude Code 使用

在仓库根目录启动 Claude Code；项目中的 `.claude/skills` 会自动发现 `.agents/skills/kpbot-app-tuner/SKILL.md`：

```shell
claude
```

在对话中调用：

```text
/kpbot-app-tuner 帮我优化当前 torchtitan-npu 训练任务中host bound场景的性能。请先确认运行环境、启动脚本、训练参数配置、模型参数配置以及明确关注的核心指标，再尝试 CPU 亲和性优化、OS 优化、BIOS优化、编译优化等各方向，涉及运行环境变更操作时先向我确认。
```

Claude Code 入口会读取 `third_party/agent-skills/KPBot/Plugins/app-tuner/skills/kpbot-app-tuner/SKILL.md`，复用上游主 Skill 及其 `references`、`subskills` 和 `scripts`。
