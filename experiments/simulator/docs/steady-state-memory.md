# 单卡内存结果

mem 汇总只保留六个值：四个包含常驻优化器状态的稳态峰值，以及单卡权重、优化器状态各自的大小。没有新增 CLI 参数。

```bash
python -m torchtitan_simulator \
  --config deepseek_v4_smoketest \
  --simulation.output-formats mem \
  --simulation.output-dir /tmp/simulator-steady-memory
```

## 六个字段

`memory/memory_summary.json` 和图内 `annotations["memory_summary"]` 严格只含以下六个字段，单位为 bytes：

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

旧汇总中的捕获步峰值、model peak、storage／保存槽计数、归因统计、策略标签、notes 等已移除；终端不再打印计时表、执行类别、通信归属和两套重复峰值。原始生命周期、事件和图仍保留，用于校验与排查。`memory_trace.json` 的事件轨迹仍是捕获步，metadata 中的预算使用上述六项，并用 `timeline_scope=captured_iteration` 标明轨迹范围。

## 计算口径

状态取自原生 optimizer/container 的公开 `state`，按实际设备和本地 storage 去重；多个 view 共用 storage 时只计一次。将这些实际状态投影为从步首常驻后，重新扫描生命周期计算四个峰值，不按权重倍数猜测状态，也不在 optimizer 阶段重复加算。

这是状态驻留的稳态估算，其他捕获生命周期不变，不额外执行第二步；仍只捕获一个 microbatch，不是完整 GA／PP 回放。数值是逻辑活跃 storage 字节，不含 allocator cache/reserved、碎片或未建模 workspace。

自定义 callback 没有可读取的公开 optimizer owner 时，相关值为 `null`，终端显示「不可用」；不会把未知状态假设成零。已知无状态的优化器报告状态字节为 0。

## 验证记录

底层稳态计算及实际冷／热两步 Meta 对照的冻结证据见 [上一阶段结果](steady-state-memory-results.json)。本次只收敛汇总格式与打印，不改变预算计算；结果不代表真实 NPU allocator 峰值。

测试入口：[六项汇总与原生 CLI](../tests/test_memory_delivery.py)、[闭式内存 oracle](../tests/memory/test_optimizer_state.py)、[终端格式与冷／热对照](../tests/test_step.py)。本次全套回归 455 passed，容器 Pyrefly 0 errors（10 suppressed）；文档中的 CLI 原样执行及 world16/rank15、MTP1、selective AC 验收均通过，六项预算与格式精简前一致。终端六行与 JSON 严格六键均有测试保护。常规 lint／Ruff／架构检查通过；主机 pre-commit 仍因缺少 Pyrefly 可执行文件未全绿，OAT 无 staged 文件而跳过。日志及当前源码 SHA256 位于本机 `torchtitan-migration-audit-20261020/compact-memory-summary/`；本次未重建镜像。
