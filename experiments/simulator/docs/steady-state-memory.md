# 单卡内存结果

mem 汇总保留六项核心预算：四个包含常驻优化器状态的稳态峰值，以及单卡权重、优化器状态各自的大小。
另有一个逐层卸载／加载数据量列表，每层只给一个字节数。没有新增 CLI 参数。

```bash
python -m torchtitan_simulator \
  --config deepseek_v4_smoketest \
  --simulation.output-formats mem \
  --simulation.output-dir /tmp/simulator-steady-memory
```

## 六个字段

`memory/memory_summary.json` 和图内 `annotations["memory_summary"]` 包含以下六项核心字段，单位为 bytes；新增的逐层列表见下节：

| 字段 | 含义 |
| --- | --- |
| `steady_state_active_bytes_peak` | 稳态总峰值，完整 F/B/O 中为下面三个阶段峰值的最大值 |
| `steady_state_forward_active_bytes_peak` | 前向峰值，包含优化器状态 |
| `steady_state_backward_active_bytes_peak` | 反向峰值，包含优化器状态 |
| `steady_state_optimizer_active_bytes_peak` | 优化器阶段峰值，包含优化器状态 |
| `persistent_param_bytes` | 单卡常驻权重 shard 的大小，不含梯度、优化器状态或激活 |
| `optimizer_state_bytes` | 单卡优化器实际状态的大小，不含权重、梯度或临时张量 |

全部数值对应所选 `simulation.logical_rank`，不是全局求和或各 rank 的最大值。权重和优化器状态已计入四个峰值，不能再次相加。
`memory_parameter_storage_dtype` 仅投影权重字节，不改变优化器状态的 dtype 或大小。

终端只显示六行结果，单位为 GiB：

```text
单卡内存（GiB；峰值包含常驻优化器状态）
  稳态总峰值: ...
  前向峰值: ...
  反向峰值: ...
  优化器阶段峰值: ...
  权重大小: ...
  优化器状态大小: ...
```

同进程直接读取：

```python
summary = workload_graph.iteration.schedule.annotations["memory_summary"]
peak_bytes = summary["steady_state_active_bytes_peak"]
weight_bytes = summary["persistent_param_bytes"]
optimizer_bytes = summary["optimizer_state_bytes"]
```

## 逐层卸载／加载数据量

新增字段 `activation_offload_by_layer`，示例省略六项核心预算：

```json
{
  "activation_offload_by_layer": [
    {"layer": "layers.0", "transfer_bytes": 1048576},
    {"layer": "layers.1", "transfer_bytes": 2097152},
    {"layer": "mtp_layers.0", "transfer_bytes": 1048576}
  ]
}
```

每个条目只有层标识和一个 `transfer_bytes`，单位为 bytes。
该值表示该层一次完整保存／恢复所需的单向 payload：D2H 和 H2D 各按此值估算，不将两个方向相加。
不再分别输出 AC 边界、普通 activation、总计、tensor 数等重复口径。

- 只有 `--simulation.memory-offload-ac-saved-tensors` 开启时才报告传输字节，关闭时捕获到的每层均为 0。
- 沿用现有 offload 选择集合，none/full/selective 按各自实际保存值统计；不包含纯重算临时张量、权重或优化器状态。
- 同层多个 view、保存槽、checkpoint 角色共享同一 storage 时只计一次，按观察到的完整 storage 字节计量，不取卸载后为 0 的设备驻留字节。
- 按主模型层数字顺序排列，随后是 MTP 层；没有卸载值的层保留为 0。非层保存值或无法归属的保存值保留其标识，如 `<unattributed>`，不伪装成某一层。
- 列表对应当前 logical rank、捕获的单 microbatch。同一 storage 被不同层使用时，各层有各自的 payload 预算，不能将列表之和解释为全局去重内存或实际总线流量。

这是传输数据量预算，不执行真实 copy，不模拟跨层主机副本复用、重复加载次数和传输时序；后续 DES 可另行处理这些行为。
终端仍只打印六项核心预算，不打印逐层明细。

传输字节沿用捕获时保存激活的实际 dtype/storage 大小，不统一强制为 FP32 或 BF16。
`training.mixed_precision_param` 影响实际计算和保存 dtype，因此可能改变列表中的值；FP32 激活通常按每元素 4 bytes、BF16 按 2 bytes，但层内还可能包含 FP32、整数或 MXFP8 code/scale，不能将整层统一乘除 2。
`training.mixed_precision_reduce` 控制梯度归约，不是激活传输精度开关；`memory_parameter_storage_dtype` 只投影常驻权重，不改变本列表。当前也不额外假设卸载时进行 dtype 转换或压缩。

旧汇总中的捕获步峰值、model peak、storage／保存槽计数、归因统计、策略标签、notes 等已移除；终端不再打印计时表、执行类别、通信归属和两套重复峰值。原始生命周期、事件和图仍保留，用于校验与排查。`memory_trace.json` 的事件轨迹仍是捕获步，metadata 中的预算使用上述六项，并用 `timeline_scope=captured_iteration` 标明轨迹范围。

## 计算口径

状态取自原生 optimizer/container 的公开 `state`，按实际设备和本地 storage 去重；多个 view 共用 storage 时只计一次。将这些实际状态投影为从步首常驻后，重新扫描生命周期计算四个峰值，不按权重倍数猜测状态，也不在 optimizer 阶段重复加算。

这是状态驻留的稳态估算，其他捕获生命周期不变，不额外执行第二步；仍只捕获一个 microbatch，不是完整 GA／PP 回放。数值是逻辑活跃 storage 字节，不含 allocator cache/reserved、碎片或未建模 workspace。

自定义 callback 没有可读取的公开 optimizer owner 时，相关值为 `null`，终端显示「不可用」；不会把未知状态假设成零。已知无状态的优化器报告状态字节为 0。

## 验证记录

底层稳态计算及实际冷／热两步 Meta 对照的冻结证据见 [上一阶段结果](steady-state-memory-results.json)。本次只收敛汇总格式与打印，不改变预算计算；结果不代表真实 NPU allocator 峰值。

此前六项格式收敛阶段：全套回归 455 passed，容器 Pyrefly 0 errors（10 suppressed）；原样 CLI 及 world16/rank15、MTP1、selective AC 验收通过。该阶段日志位于本机 `torchtitan-migration-audit-20261020/compact-memory-summary/`，历史六键约束现已扩展为六项预算加一个列表。

逐层列表的测试入口：[storage 去重与文件导出](../tests/memory/test_layer_offload_summary.py)、[真实模型与 MTP 的 offload 开关](../tests/test_memory_delivery.py)、[闭式内存 oracle](../tests/memory/test_optimizer_state.py) 和 [终端六行格式](../tests/test_step.py)。

2026-10-26 复验：sidecar 全套 489 passed，Pyrefly 0 errors（10 suppressed）。DSV4/MTP 与 Kimi 的 MXFP8 × none/full/selective 覆盖 offload 开关、图／梯度与权重投影隔离；独立 64+32=96 bytes 用例核对共享 storage、view 和多保存角色去重，且不乘双向系数。真实模块 CLI 的 full AC + MTP + offload 导出通过；更新后的 acceptance 在 world16/rank15、TP2/CP2/EP2、selective AC、MTP1 下通过，共 11,047 节点、143 个参数梯度。终端仍为六行，文件／图内汇总为七键。

常规 pre-commit、Ruff 和架构检查通过；全仓检查仍因主机缺少训练依赖而有类型错误，不能标为全绿。日志和导出位于本机 `torchtitan-migration-audit-20261020/layer-offload-summary-20261026/`。本轮不验证真实传输或硬件数值，未重建镜像。
