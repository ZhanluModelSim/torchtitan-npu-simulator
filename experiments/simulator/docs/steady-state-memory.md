# 单卡内存结果

mem 汇总保留六项核心预算：四个遵循优化器驻留策略的稳态峰值，以及单卡权重、优化器状态各自的大小。
另有一个逐层卸载／加载数据量列表，每层只给一个字节数。优化器卸载使用独立开关，见[优化器卸载](optimizer-offload.md)。

```bash
python -m torchtitan_simulator \
  --config deepseek_v4_smoketest \
  --simulation.output-dir /tmp/simulator-steady-memory
```

CLI 和 `simulate()` 默认仅通过内存对象交付预算，不写文件；显式添加 `--simulation.output-formats mem`（Python 为 `output_formats=["mem"]`）才写入 `<output_dir>/memory/memory_summary.json` 及事件、timeline、生命周期等明细。

## 六个字段

`memory/memory_summary.json` 和图内 `annotations["memory_summary"]` 包含以下六项核心字段，单位为 bytes；新增的逐层列表见下节：

| 字段 | 含义 |
| --- | --- |
| `steady_state_active_bytes_peak` | 稳态总峰值，完整 F/B/O 中为下面三个阶段峰值的最大值 |
| `steady_state_forward_active_bytes_peak` | 前向峰值，未卸载时包含优化器状态 |
| `steady_state_backward_active_bytes_peak` | 反向峰值，未卸载时包含优化器状态 |
| `steady_state_optimizer_active_bytes_peak` | 优化器阶段峰值，未卸载时包含优化器状态 |
| `persistent_param_bytes` | 单卡常驻权重 shard 的大小，不含梯度、优化器状态或激活 |
| `optimizer_state_bytes` | 单卡优化器实际状态的大小，不含权重、梯度或临时张量 |

全部数值对应所选 `simulation.logical_rank`，不是全局求和或各 rank 的最大值。权重已计入四个峰值；优化器状态仅在未卸载时计入，不能再次相加。卸载后 `optimizer_state_bytes` 仍报告真实本地状态总量，不能据此推断设备驻留。
`memory_parameter_storage_dtype` 仅投影权重字节，不改变优化器状态的 dtype 或大小。

显式 mem 导出时终端显示六项结果，单位为 GiB；默认不打印。下例为未卸载优化器状态的标题：

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
- none/full/selective 按各自实际保存值统计；只对有明确层归属或显式 checkpoint 边界归属的保存值投影 offload，不包含纯重算临时张量、权重或优化器状态。head/norm 等层外保存值及无法确定归属的值保留设备驻留，不建模传输。
- 标准捕获中，若一个 storage 的全部保存消费和实际使用都在同一语义前向 F action 内完成（例如 chunked loss 的立即反向），它不参与 offload，按观察到的原始生命周期保留设备字节。此规则同时影响逐层传输列表和四项显存峰值，不是仅过滤报表。仍有后续 B/recompute 使用的共享 storage 不属于此排除项。
- 同层多个 view、保存槽、checkpoint 角色共享同一 storage 时只计一次，按观察到的完整 storage 字节计量，不取卸载后为 0 的设备驻留字节。
- 按主模型层数字顺序排列，随后是 MTP 层；没有卸载值的层保留为 0。当前标准捕获不再为层外／未归属保存值创建 `<unattributed>` 传输项；其来源、保存槽和真实设备生命周期仍保留在诊断明细中，不伪装成某一层，也不从显存预算删除。
- 列表对应当前 logical rank、捕获的单 microbatch。同一 storage 被不同层使用时，各层有各自的 payload 预算，不能将列表之和解释为全局去重内存或实际总线流量。

启用 offload 后，标准入口同时生成激活 H2D/D2H 与等待节点，交由平台计算传输耗时和 DES；本仓不执行真实 copy。静态预算按同一计划补回 storage 去重的生成／卸载与消费／预取驻留，不是动态内存回放。晚产生 payload 保守保留至 F 末，可能超过两层；head 原驻留不重复补入，O 不变。原始 lifetime 保留，新增区间在 `activation_buffer_lifetimes`，详见[执行图与静态预算](activation-offload-design.md)。
终端仍只打印六项核心预算，不打印逐层明细。

历史结果中的 `<unattributed>` 不等于 loss。它还包含 head/norm 等层外保存值。当前这些值不卸载、不计传输，但仍在原始生命周期、alloc/free 时间线和设备峰值中；不同 chunk 的累计字节数不直接相加为峰值。不能仅在下游隐藏旧传输行而沿用旧的零驻留预算，应使用修复后的捕获结果。

传输字节沿用捕获时保存激活的实际 dtype/storage 大小，不统一强制为 FP32 或 BF16。
`training.mixed_precision_param` 影响实际计算和保存 dtype，因此可能改变列表中的值；FP32 激活通常按每元素 4 bytes、BF16 按 2 bytes，但层内还可能包含 FP32、整数或 MXFP8 code/scale，不能将整层统一乘除 2。
`training.mixed_precision_reduce` 控制梯度归约，不是激活传输精度开关；`memory_parameter_storage_dtype` 只投影常驻权重，不改变本列表。当前也不额外假设卸载时进行 dtype 转换或压缩。

旧汇总中的捕获步峰值、model peak、storage／保存槽计数、归因统计、策略标签、notes 等已移除；终端不再打印计时表、执行类别、通信归属和两套重复峰值。原始生命周期、事件和图仍保留，用于校验与排查。`memory_trace.json` 的事件轨迹仍是捕获步，metadata 中的预算使用上述六项，并用 `timeline_scope=captured_iteration` 标明轨迹范围。

## 计算口径

状态取自原生 optimizer/container 的公开 `state`，按实际设备和本地 storage 去重；多个 view 共用 storage 时只计一次。未开启优化器卸载时，将这些实际状态投影为从步首常驻后，重新扫描生命周期计算四个峰值，不按权重倍数猜测状态，也不在 optimizer 阶段重复加算。

这是状态驻留的稳态估算，其他捕获生命周期不变；`simulate()` 默认在[结构预热](steady-state-capture.md)后捕获，低层捕获也可投影首 O 的状态。正式记录仍只有一个 microbatch，不是完整 GA／PP 回放。数值是逻辑活跃 storage 字节，不含 allocator cache/reserved、碎片或未建模 workspace。

自定义 callback 没有可读取的公开 optimizer owner 时，相关值为 `null`，终端显示「不可用」；不会把未知状态假设成零。已知无状态的优化器报告状态字节为 0。

## 验证与排查入口

- [逐层 payload 与 storage 去重](../tests/memory/test_layer_offload_summary.py)：共享 view／多保存角色不能重复计量。
- [真实模型交付](../tests/test_memory_delivery.py)：DSV4／Kimi、MTP、AC、量化、两种卸载以及默认无文件／显式 mem 行为。
- [优化器独立字节预期](../tests/memory/test_optimizer_state.py)：实际本地状态、去重及稳态驻留。
- [F 内交错反向](../tests/memory/test_interleaved_loss_offload.py)：chunk-local loss 不卸载，后续 B／recompute 共享的 storage 仍按真实使用处理。

归属优先使用保存槽的模块路径。明确的 head/norm/root 不借用滞后反向模块路径归到 decoder；没有模块归属才用真实生产／消费事实辅助判断。没有确认层或 checkpoint 绑定的保存值继续驻留，不能通过隐藏 unattributed 报表行修正显存。

排查时比较每个 storage 的原始与建模字节、birth/death、全部保存／消费和相邻窗口，再核对峰值。多个 loss chunk 累计字节不能直接相加为峰值。历史数值不作为新模型固定阈值，独立手算应基于当前实际形状、dtype 与 sharding。
