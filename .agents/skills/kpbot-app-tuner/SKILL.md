---
name: kpbot-app-tuner
description: 使用该 skill 构建和执行服务器应用优化 Agent 的自顶向下优化流程，覆盖用户场景输入、Agent 可选操作确认、基线数据确认、环境诊断与备份、磁盘/网卡/内存/CPU/GPU/NPU/硬件规格瓶颈识别、基线后按进程火焰图/热点函数/热点 so/topdown L1 icache miss/L3 cache miss/线程切换等性能信息采集、根据采集信息生成候选优化 skill 列表、按候选列表依次执行 CPU 亲和性、网络参数、性能库、应用配置、BIOS、OS、编译和 Other 等优化 skill、候选完成后覆盖执行未命中 skill、分轮收益统计、review 与环境还原、案例归档和最终报告输出。适用于 Claude Code、Codex、Cursor、OpenCode 以及其他支持目录式 SKILL.md 的编程 Agent 环境。
---

# KPBot App Tuner

本入口复用固定 commit 的 KPBot 上游流程，不复制其工作流实现。首次 clone 或更新仓库后执行：

```bash
git submodule update --init --recursive third_party/agent-skills/KPBot
```

执行前确认 `third_party/agent-skills/KPBot/` 已初始化；未初始化时不得自行重写或降级 KPBot 流程。

## 平台入口

- Claude Code：读取 `third_party/agent-skills/KPBot/Plugins/app-tuner/skills/kpbot-app-tuner/SKILL.md`。
- OpenCode：读取 `third_party/agent-skills/KPBot/Plugins/app-tuner/opencode/kpbot-app-tuner/SKILL.md`。

除平台工具调用方式和路径映射外，严格遵循上游 Skill 的用户确认、基线、证据采集、单变量迭代、回退和报告规则。主 Skill 所需的 `scripts`、`references`、`subskills` 和 overlay 均从 `third_party/agent-skills/KPBot` 的同一 submodule commit 读取。

## torchtitan-npu 约束

1. 先用训练证据确认 host bound，不以单一 NPU 利用率指标下结论。
2. 固定模型、数据、checkpoint、随机性、并行策略、batch size、sequence length 和精度配置后建立基线。
3. 默认只分析或调整 host 侧；涉及模型、算子、并行或数值语义变化时单独说明并验证。
4. BIOS、sysctl、IRQ、重启、重编译、远程执行和持久化配置须经用户明确授权，并保留回退步骤。
