# Simulator 图折叠与死代码清理

当前 WorkloadGraph 直接保留捕获的原始算子、通信、view／alias 和零耗时流控制节点。通信归属处理只调整 L1／L2 计费与 action 清单，不克隆、移动或重新编号算子，也不推断新的算子依赖。文件输出仍只有 mem，CLI 参数不变。

## 删除范围

- metadata-view 图折叠及 JSON／HTML／DOT／text 等图导出已在此前删除，本轮继续清理残余路径。
- 删除 FSDP 通信模板变体、按模块和层序推断 prefetch 锚点、复用 AG 模板以及对应的区域划分辅助代码。
- 删除模板改写后恢复丢失通信节点、重新映射 `source_op_id` 和重建全局后继边的补偿逻辑。
- 删除旧 PP P2P fragment 的节点抽取和依赖改写。PP 训练入口仍明确拒绝，独立 IR／调度结构测试继续保留。
- 不再生成只供模板改写使用的 `sim.fsdp_*` 图标记；实际 FSDP alloc／free／staging 驻留事件继续用于内存估算。
- 删除没有生产调用的 dispatch 隐藏接口及 `suppressed_dispatch_counts` 空诊断、恒为 1 的算子重复计数、无人设置的同步序列 fallback 和旧展示标签函数。
- 删除空的 `StepGraph.fused_regions`、`internal_data_passes` 字段，以及 `communication_ownership.generated_l1_templates` 诊断字段。

`communication_ownership.py` 从约 1,460 行缩减为 281 行。旧折叠表示和恢复机制的测试已退役，替换为对原始节点对象、ID、前驱和后继不变的直接检查。

## 保留的执行语义

计算模板中的实际 AG／RS／AR 只计费一次；对应的重复 L2 residency／gradient action 被移除。计算外的真实通信和显式 prefetch intent 保留，CP／TP 通信不会因为同样使用 all-reduce 就被归为梯度归约。

FSDP transition 必须对应实际捕获的 AG；单卡无需通信的情况除外。缺少捕获时报错，不再按 shape、模块名或另一 microbatch 猜测、复用通信。当前完整训练捕获仍限 PP=1、单个 microbatch。

原始 `op_id` 直接用于所有模板之间的引用，无需 `source_op_id` 映射。上游调度依然需要遵守完整的跨模板 `predecessors`，包括 [零耗时控制节点](stream-control-dependencies.md)。L2 action／DataSlot 和 storage 生命周期保留各自的语义，不属于展示图折叠。

## 验证依据

[通信归属单测](../tests/capture/test_communication_ownership.py) 直接检查未消费的预取 AG 不丢失、所有原始节点和边保持不变、归约不重复计费，以及计算外 action 保留。[原生 HSDP 回归](../tests/test_fsdp_stream_capture.py) 检查实际 post-reduce 等待在最终图中约束后续 MM。

本轮重跑 FSDP、HSDP、TP／CP／EP 组合并行以及 HSDP + selective AC + MXFP8，全部 MTP=1；配置沿用 [FSDP 审计](fsdp-dependency-audit.md)。结果如下：

- 1,086 次参数 storage 读取对应的 AG 祖先关系、78 个区间的 AG 延后传播全部通过。
- 35 对 RS→AR，以及归约→clip／每个 optimizer 的依赖全部通过。
- HSDP 探针的 110 次有效等待和 2,961 对同流相邻设备工作均在最终 DAG 可达；11 条 post-backward 等待的 AR 延后会延后后续 MM。
- 去除废弃的无 tensor 标记后，四组原始 tensor 事件的算子、阶段、shape／dtype／storage bytes 逐项一致；通信 payload、梯度参数数量和捕获显存峰值一致。

直接边并非逐项相同：旧推断代码增加的 95 条冗余直接边被移除，但相应依赖仍可传递到达。组合并行还恢复了 10 条原始直接边，涉及预取 AG 的 copy-in、流等待及其后继；旧模板改写及恢复补偿曾遗漏这些依赖。当前归属处理不再改写图，避免再次发生这一问题，详情见证据中的 `dependency-diff.json`。

最终完整回归 `python -m pytest -q tests`：440 passed；容器 Pyrefly：0 errors。完整 `pre-commit run --all-files --show-diff-on-failure` 的格式、Ruff、拼写及架构等检查通过；宿主 Pyrefly 因缺失可执行文件失败，不能称为完整 Lint 通过。OAT 因无暂存文件跳过实际扫描。增量静态审查未发现必修问题。

完整日志与机器结果位于本机 `torchtitan-migration-audit-20261020/folding-cleanup-20261023/`，包括 `comparison.json`、`analysis.json`、`stream-analysis.json` 和检查日志。固定 CANN／PyTorch 软件栈下的 Meta 结果不代表真实 NPU／HCCL 数值或性能验证；host synchronize 与轮询式设备进度仍未建模。
