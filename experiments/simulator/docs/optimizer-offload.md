# 优化器状态卸载

`SimulationConfig.memory_offload_optimizer_states=True` 或 CLI `--simulation.memory-offload-optimizer-states` 开启本地优化器状态卸载建模，默认关闭。支持原有 AdamW 路径和 DeepSeek-V4 的 DistMuon＋AdamW 混合路径，与激活卸载独立。模型仍执行原生 Meta optimizer 捕获；本功能不执行真实 CPU optimizer 或设备 copy。

## 执行图与聚合近似

状态量 S 来自实际 optimizer owner 按本地 storage 去重的 `optimizer_state_bytes`，混合优化器同时计入 Muon momentum 和 AdamW fallback 状态。S 不乘 world 或 GA，不包含梯度、参数、Muon runtime 工作缓冲或 Newton–Schulz 临时张量。未知 inventory 报错，已知为零则不添加传输。

`capture/optimizer_transfers.py` 按优化器路径选取重叠窗口：

- AdamW 保留 `optimizer_work_scope=actual_updated_parameters` 对应的原生更新块，不包含 clip、准备和收尾。更新之间夹有非更新准备节点时仍拒绝，避免共同门控成环。
- Muon 使用整个原生混合 optimizer step，包含 momentum 准备、打包、两向 all-to-all、矩阵计算及 AdamW fallback；不包含 step 之前的梯度裁剪和之后的 scheduler。不能只围绕最终参数 `add_` 加传输，因为 momentum 在 prepare 阶段就被读取和更新。

共同入口等待窗口的全部外部前驱。H2D、原生工作 DAG、D2H 从共同入口并行开始，共同出口等待三者结束，再放行窗口后继：

```text
                 ┌─ 全量状态 H2D ───────────────┐
共同入口 ────────┼─ 原生计算、通信及其已有依赖 ─┼─ 共同出口
                 └─ 全量状态 D2H ───────────────┘
```

原计算与通信节点及其内部依赖保留。Muon 原生 bucket 预取、compute-ready、compute-done 和双缓冲复用等待不变；不会因新增传输而将 all-to-all 与矩阵计算串行化。H2D、D2H 使用独立逻辑流，trace 可以表达三条并行路径。

这是理想细粒度状态流水的**整段聚合近似**，忽略状态搬运的启动和排空。它不声称真实状态可以先读取后加载，也不声称同一份状态可在更新前回写。只有上下行资源独立且没有其他争用时，窗口完成时间才为 `max(T_H2D, T_native_DAG, T_D2H)`。`T_native_DAG` 是保留内部依赖和资源约束后的计算＋通信完成时间，不是各算子耗时串行求和。上下行共享资源时由下游调度施加约束，不能强制套用三项最大值。

平台负责硬件带宽、传输耗时与 DES；不能再加相同传输的拖尾，不能将传输时间改写为原生优化器算子耗时。O 只捕获一次，不乘 GA。

新增 raw name 为 `simulator.optimizer_h2d`、`simulator.optimizer_d2h`，属性为 `optimizer_transfer={direction, bytes}`，两方向各为 S；不伪造 tensor shape/dtype。入口和出口为 `simulator.stream_wait`。schedule 的 `optimizer_transfer_plan` 保留策略、窗口及新增节点、单向字节等审计信息。源 `captured_workload` 保持不变。

## 静态显存与交付

F/B/O 均将可卸载 optimizer state 投影为主机驻留；按已确认的聚合近似，暂不计状态搬运 staging buffer 峰值。这不是实际全量状态加载到设备后的精确峰值，也不是逐桶动态内存回放。原始 `optimizer_state_bytes`、数量、raw events 和 tensor lifetimes 保留，便于审计实际状态 inventory。

Muon runtime 的 `storage_exchange`、`compute_exchange`、`compute_scratch` 和 `storage_scratch` 由原生 runtime 跨迭代持有。正式捕获前登记其实际本地 storage，从步首到步尾保留设备驻留；多个 view 和复用槽不能重复计量。它们不属于可卸载状态，不随开关减掉。Newton–Schulz 临时张量继续按捕获生命周期计入。预热消除初始化计算，与登记预热后已分配的常驻内存是两件事。

`MemoryPlan.device_budget_lifetimes()` 仅将 optimizer state 的建模驻留投影为零。激活卸载的 `activation_buffer_lifetimes`、Muon 常驻工作缓冲和其他原生临时量继续参与预算，两个卸载开关不能互相覆盖。

`memory_summary` 仍为六项预算与逐层激活列表，文件与内存结果一致；最终终端预算也来自投影后的对象。`annotations['memory_configuration']` 从最终 MemoryPlan 返回实际 `offload_activations` 和 `offload_optimizer_states`，false 也明确返回。不开启优化器卸载时继续使用包含常驻状态和工作缓冲的稳态预算。

当前不建模参数／梯度卸载、CPU 更新、状态逐桶加载与回写依赖、staging 槽生命周期、跨阶段异步传输或动态显存回放。未来逐桶模式必须同时提供实际状态字节、首读／末写依赖及缓冲复用保护，不能只增加 AllToAll 前继而沿用聚合显存口径。

## 验证

- [窗口与独立字节先验](../tests/capture/test_optimizer_transfers.py)：原生边界、原对象不变、双方向数据量、零／未知状态和状态预算。
- [Muon 聚合卸载](../tests/test_muon_offload.py)：单卡和 world=8、EP=2 的独立 workspace 容量预期、F/B/O 驻留、混合状态字节与双向传输量；卸载前后原生节点属性和内部前驱保持。
- [Muon 捕获](../tests/test_muon_capture.py)、[预取依赖](../tests/test_muon_dependencies.py)：混合状态与原生计算；卸载开关两种状态均核对 all-to-all 必要等待、双缓冲复用以及独立工作的可重叠关系。
- [状态与 workspace 字节](../tests/memory/test_optimizer_state.py)：storage／alias 去重，workspace 初始驻留和卸载后保留。
- [真实 CLI 四组合](../tests/test_memory_delivery.py)：AdamW selective AC 实际捕获、状态 inventory、开关和交付校验。

平台适配 → DES → Trace 的诊断已分别用受控 synthetic 耗时覆盖原生工作主导和传输主导两种情况，核对 H2D／D2H／原生 DAG 三路汇合及 trace 保留。该诊断验证调度关系，不是设备性能测量。

Meta 和受控预测耗时只验证结构、字节口径与调度关系，不证明 NPU 数值、HCCL 传输性能或真实设备显存峰值。
