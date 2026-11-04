# 优化器状态卸载

`SimulationConfig.memory_offload_optimizer_states=True` 或 CLI `--simulation.memory-offload-optimizer-states` 开启 AdamW 本地优化器状态卸载建模，默认关闭。它与激活卸载独立，AdamW 的四种开关组合均可用。Muon 当前明确拒绝该开关，其 momentum 准备和重分布流水不能套用本窗口，见 [Muon 捕获](muon.md)。模型仍执行原生 Meta optimizer 捕获；本功能不执行真实 CPU optimizer 或设备 copy。

## 执行图

`capture/optimizer_transfers.py` 从 `optimizer_work_scope=actual_updated_parameters` 标记选取 O 内原生更新节点，状态量 S 来自实际 optimizer owner 按本地 storage 去重的 `optimizer_state_bytes`。S 不乘 world 或 GA。未知 inventory 报错，已知为零则不添加传输。

共同入口等待更新块的全部外部前驱；原生更新 DAG、H2D 与 D2H 三条路径从入口开始，共同出口等待三者结束，再放行更新块后继。原计算节点及其已有依赖、成本保留，clip、准备和收尾工作不作为更新重叠窗口。支持单更新节点、相互独立或直接依赖的多个更新节点；更新之间夹有非更新准备节点时明确报错，不生成可能成环的共同门控。

这是理想逐 tensor 流水的聚合近似，忽略启动和排空：一次更新窗口为 `max(T_H2D, T_updates, T_D2H)`，不是将两方向时间相加，也不是一份 tensor 可在完成更新前写回的真实执行依赖。若状态 10 GB、每方向带宽 50 GB/s，则传输窗口为 0.2 s。O 只捕获一次；平台负责硬件带宽、传输耗时与 DES，不能再补相同传输的拖尾或修改 AdamW 耗时。

新增 raw name 为 `simulator.optimizer_h2d`、`simulator.optimizer_d2h`，属性为 `optimizer_transfer={direction, bytes}`，两方向各为 S；不伪造 tensor shape/dtype。入口和出口为 `simulator.stream_wait`。节点带 `modeling_source=optimizer_offload_full_duplex_v1`，schedule 的 `optimizer_transfer_plan` 记录更新与新增节点 ID、单向字节和策略。源 `captured_workload` 保持不变。

## 静态显存与交付

F/B/O 均移除 optimizer state 设备驻留；按用户确认的近似，忽略逐 tensor 搬运 buffer 峰值。原始 `optimizer_state_bytes`、数量、raw events 和 tensor lifetimes 保留，便于审计实际传输 inventory。`MemoryPlan.device_budget_lifetimes()` 将状态的建模驻留投影为零，激活卸载的 `activation_buffer_lifetimes` 继续参与预算，两个开关不能互相覆盖。

`memory_summary` 仍为六项预算与逐层激活列表，文件与内存结果一致；最终终端预算也来自投影后的对象。`annotations['memory_configuration']` 从最终 MemoryPlan 返回实际 `offload_activations` 和 `offload_optimizer_states`，false 也明确返回。不开启优化器卸载时继续使用原常驻状态稳态预算。

不建模参数／梯度卸载、CPU 更新、逐 tensor 启动／排空、跨阶段异步传输或动态显存回放。平台的 MBU 分子不包含主机传输；端到端时间改变后分母自然更新。

## 验证

- [窗口与独立字节先验](../tests/capture/test_optimizer_transfers.py)：更新依赖、clip/pre/post 边界、原对象不变、双方向数据量、零／未知状态、夹层准备拒绝、激活 buffer 与状态预算共存。
- [真实 CLI 四组合](../tests/test_memory_delivery.py)：selective AC 实际捕获，状态总量不变，四项峰值逐项减 S，实际开关、最终输出与 validator 一致。
- 跨仓真实 Meta 捕获已验证 DeepSeek-V4 四组合，以及 MTP、Kimi 双开；平台核对双向并行、AdamW 成本不变、O 不乘 GA、CSV/Trace 与内存输出。测试预测耗时仅用于验证调度逻辑，不是硬件精度验证。
