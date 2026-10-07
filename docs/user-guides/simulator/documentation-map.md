# Simulator 文档导航与历史方案索引

> 现状核对：2026-10-07，`feat/npu-simulator`，实现基线 `ecde2f0`。依据源码与 Git 历史静态核对，未执行测试或仿真；后续行为以当前源码为准。

返回[使用入口](../simulator.md)。本页区分当前使用/消费契约与原始方案。原始需求、spec、实施计划和分析保持原文件、原路径、原内容；历史文档中出现的“当前”“待实现”只对应当时版本。

## 当前现状文档

| 文档 | 范围 |
|---|---|
| [使用入口](../simulator.md) | 环境前提与当前配置启动方式 |
| [配置参考](configuration.md) | Simulator 核心字段、SAC、并行公式和假设优化 |
| [模型矩阵](model-matrix.md) | 正式 registry 与模型特例，区分 raw_model/训练接入 |
| [输出与内存](outputs.md) | 导出结构、字段语义、峰值与 inventory 区别、PP replay 边界 |
| [当前架构](../../design/simulator-architecture.md) | 从 launcher 到 L0–L3、通信 ownership 与插件 |
| [历史修复与当前边界](../../design/simulator-fixes-and-limits.md) | 已合入根因处理与尚未落地范围 |

## 当前细粒度契约

以下文档保留，承担与简要现状说明不同的细粒度职责。描述存在冲突时，先核对实现，再按专门契约读取：

| 文档 | 当前职责 |
|---|---|
| [通信归属契约](../../design/communication-ownership-contract.md) | L1_STAGE、L2_PIPELINE、L2_PREFETCH、L2_STANDALONE，避免重复通信 |
| [PP L2 架构](../../design/pp-l2-capture-architecture.md) | 语义 action、rank 身份、SchedulePlan 与兼容投影 |
| [依赖重建契约](../../design/schedule-plan-dependency-reconstruction-contract.md) | schema v2 输入、DataSlot、transfer_id、FSDP readiness 与禁止兜底项 |
| [DES 消费指南](../../design/schedule-plan-des-consumer-guide.md) | 状态机与消费细节；冲突时以上一项契约为准 |
| [DualPipeV 消费指南](../../design/dualpipev-schedule-plan-consumer-guide.md) | 虚拟 stage、overlap parent/sub-action、同 rank V transfer |

## 保留的原始方案与分析

这些文件有起因、推导或阶段证据，不是无用副本；本次不编辑、不移动、不删除。通过本页限定其阅读范围，而不改写历史结论。

| 原始文档 | 阅读范围 / 当前替代入口 |
|---|---|
| [最初 simulator spec](../../superpowers/specs/2026-07-01-npu-simulator-design.md) | 7 月初侧载、单进程原型；完整当前路径见当前架构 |
| [最初实施计划](../../superpowers/plans/2026-07-01-npu-simulator-implementation.md) | 当时任务分解；待办勾选不代表当前完成状态 |
| [MHC spec](../../superpowers/specs/2026-07-01-mhc-real-op-name-capture-design.md) / [实施计划](../../superpowers/plans/2026-07-01-mhc-real-op-name-capture-implementation.md) | 算子真实名称捕获的原始方案；当前 implementation-class shim 见架构 |
| [SMLA spec](../../superpowers/specs/2026-07-01-smla-real-op-name-capture-design.md) / [实施计划](../../superpowers/plans/2026-07-01-smla-real-op-name-capture-implementation.md) | sparse attention/indexer 原始方案；当前 shim 见架构 |
| [AllToAll 缺失分析](../../design/alltoall-capture-analysis.md) | fake 短路与 autograd 根因；已落地桥、FP8 和 SAC 范围见矩阵/配置 |
| [Optimizer 分析](../../design/optimizer-ops-analysis.md) | foreach/fused 差异起因；当前 fused AdamW 捕获见架构/修复记录 |
| [MXFP8 分析](../../design/low-precision-sim-analysis.md) | 低精度接入思路；当前正式 preset 与范围见矩阵 |
| [早期通信层级设计](../../design/schedule-capture-design.md) | 不能继续采用“所有 FSDP 在 L2”或“RESHARD=RS”；用通信归属契约 |
| [复杂 PP 多图方案](../../design/pipeline-multi-graph-capture-design.md) | 从 MB0 到每计算形态首现的方案；当前 comm variants 与 ownership 见架构 |
| [L2/L3 重构原始方案](../../design/l2-l3-schedule-plan-design.md) | lowered plan 为主源的早期阶段；当前语义流为主源，按 schema v2 契约消费 |
| [内存模型原始设计](../../design/simulator-memory-model-design.md) | 生命周期改造动机、扩展设计；当前已落地字段与边界见输出说明 |
| [1F1B 阶段实施指南](../../design/1f1b-schedule-plan-assembly-guide.md) | 首个 1F1B 版本的分阶段目标；不能将“不扩展 DualPipe”作为当前能力限制 |

Kimi fused-op 和 DeepSeek/Megatron 激活推导文档涉及模型/原始分析，本次不改；模型参数目录、安装指南、模型接入与验收规范同样保留。这里只建立 simulator 现状入口，不替代这些原始文档。

## 已整理的漂移

- 替换旧 928 行使用指南，将配置/模型矩阵/输出字段拆出，移除失效的 `deepseek_v4_pro_simulate_*` 启动示例和重复说明。
- 删除旧指南中“默认导出全套文件”“所有卡独立完整捕获”“L0 不含通信”“只捕获 MB0”的绝对表述，按当前实现说明代表捕获、首现模板和 comm variants。
- 删除旧指南中 RESHARD=reduce-scatter、所有 FSDP 属于 L2、用 peer rank 单独配对 PP 的过时消费方法。
- 删除旧指南中未关联当前版本/环境的固定资源与耗时表，以及将单算子 `peak_mem` 累加当作峰值的旧 summary 示例。
- 把参数存储、MXFP8 计算、FP8 transport 和激活 offload 分别说明；指出 Kimi 默认 EP 路径的接入差异、`target_npu_device_type` 未消费和 PP saved-slot replay 边界。
- 依赖/镜像描述以锁定文件与源码为准，不继续承诺历史预构建镜像与当前分支等价。Dockerfile 内仍有旧 preset 帮助注释，本次仅整理文档，未修改构建文件。

本次没有删除独立原始设计或分析文件。无用/重复的运行说明在刷新使用指南时合并删除；保留原始材料可供追溯，不再将其列作现状操作入口。
