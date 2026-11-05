# DeepSeek-V4 Muon 捕获

Simulator 可显式选择 DeepSeek-V4 的原生 DistMuon＋AdamW 混合优化器，保留矩阵计算、双向 all-to-all、打包与搬移、状态更新以及 stream/event 依赖。默认仍使用 recipe 的 AdamW。模型参数匹配和 FlexShard compute layout 来自主包的 DSV4 Muon profile，sim 不重新定义算法或参数分组。

## 使用与范围

按 [README](../README.md) 准备并安装训练软件栈、加载 CANN 环境后运行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torchtitan_simulator \
  --config deepseek_v4_smoketest \
  --optimizer.name Muon \
  --simulation.world-size 8 \
  --parallelism.expert-parallel-degree 2
```

Python 入口为 `simulate(SimulationConfig(optimizer_name="Muon", world_size=8, expert_parallel_degree=2))`。
`--optimizer.muon-ns-steps`／`muon_ns_steps` 可覆盖 Newton–Schulz 迭代次数，必须为正整数；不指定时保留原生 recipe 值，当前 DSV4 为 10。其余超参继续来自原生配置工厂。显式 AdamW 不重建原生分组；自定义工厂若已 materialize Muon，不能只改名称伪装成 AdamW。

当前支持 DSV4、TP=CP=PP=1 下的单卡、DP shard、HSDP、EP 及合法组合。小模型回归包含非零 logical rank、MTP、full AC 和激活卸载。Kimi、Muon 量化训练、HSDP replica dedup、原生 swap/virtual optimizer override 均明确拒绝。优化器状态卸载可用独立开关启用，采用下述整段聚合近似。

Muon 在 EP=1 时若需要将实际参数存储 `Shard(1)` 转为 compute layout 的 `Shard(0)`，在模型并行化之后、构建优化器之前抛出 `ValueError`。报错包含参数 FQN、全局 shape、mesh 轴与大小，并建议使用 AdamW 或受支持的 EP/FSDP 配置；不实现该布局转换。检查依据实际 DTensor placement、原生参数分组和 compute layout，不按卡数一概拒绝。debug 模型有 4 个专家，EP=1、FSDP=2/4 仍通过，FSDP=8/16 提前拒绝；将专家数改为 8 后，EP=1、FSDP=8 的 `Shard(0)` 路径也通过。AdamW 和 EP>1 不受该规则影响。

设置 `memory_offload_optimizer_states=True` 或添加 `--simulation.memory-offload-optimizer-states` 可启用整段 H2D／D2H 卸载近似。共同入口覆盖整个混合 optimizer step，包括 momentum 准备、原生 all-to-all、矩阵计算和 AdamW fallback；不包含梯度裁剪与 scheduler。两方向传输与原生工作 DAG 并行，出口共同等待，内部预取与缓冲复用依赖保持。该窗口不能只包围最终参数 `add_`。

这是理想细粒度状态流水的聚合表达，不是精确逐桶依赖。在上下行资源独立、无其他争用时，窗口耗时为 `max(T_H2D, T_native_DAG, T_D2H)`；实际资源约束仍由平台处理。状态 staging buffer 暂按约定忽略，Muon 原生常驻工作缓冲和临时计算量照常保留。完整依赖与显存口径见[优化器卸载](optimizer-offload.md)。

## 原生执行与 Meta 适配

`compat/muon.py` 只在含 DistMuon 的仿真会话中安装可恢复适配：

- 允许全部本地参数为 Meta 的原生 DTensor；其他设备仍调用原校验。
- 保留原生 layout、bucket 构建和梯度／状态布局检查。跨 rank 的计划哈希比较需要真实 peers 和张量数值，单 logical-rank Meta 会话不执行该比较；`runtime_capabilities.muon_peer_plan_validation` 明确标记 `not_performed_single_logical_rank`。这不证明多进程计划一致性。
- 观察原生 `_apply_update`，将实际写入参数 storage 的最终 `add_(direction)` 关联到参数 FQN，不替换更新计算。

同一模型与优化器先完成结构预热，再捕获代表性 F/B/O。Muon momentum 和 AdamW moments 均来自真实 optimizer state，按实际 dtype、设备和本地 storage 容量去重计入静态预算，不按固定参数倍数或全局参数量除以 world 推断。Muon momentum 使用原生梯度 dtype。

预热后，runtime 的 local slot 与两个通信 slot 已经持有工作缓冲。正式捕获将其 `storage_exchange`、`compute_exchange`、`compute_scratch` 和 `storage_scratch` 登记为初始驻留；它们跨 F/B/O 及迭代复用，不能到 O 首次使用才开始计量。按实际 allocation 去重，而非累计各 bucket 的逻辑容量。这些工作缓冲不属于 `optimizer_state_bytes`，状态卸载后仍计入设备峰值；Newton–Schulz 临时张量按原生生命周期处理。

GMM 对应的 routed-expert `w1_EFD / w2_EDF / w3_EFD` 归 DistMuon；更新时批量矩阵 Newton–Schulz 展开为 `bmm/baddbmm`，前后向仍保留原生 GMM。

交付中的 `optimizer_parameter_groups` 记录实际容器的 algorithm 与参数 FQN。更新节点通过 `optimizer_updated_parameter_names` 和 `optimizer_algorithm` 标明真实更新对象；校验要求全部梯度参数恰好更新一次且算法与 owner 相符。梯度裁剪由 `gradient_clip` scope 识别，不把 Muon 的权重衰减 `mul_` 当成裁剪。AdamW 的六组原生输入绑定与原有更新注记保持。

## All-to-all 与计算重叠

FlexShard 对需要重分布的 bucket 执行：

1. transfer stream 准备输入，执行 storage→compute all-to-all，记录 `compute_input_ready`。
2. caller stream 等待该 event，执行需要重分布的矩阵计算，记录 `compute_done`。同 bucket 无需重分布的本地计算可在等待之前执行。
3. transfer stream 等待 `compute_done`，执行 compute→storage all-to-all 和参数更新，记录 `done`。
4. caller stream 在原生 release 点等待 `done`；复用 buffer slot 必须等待前一轮工作完成。

Python 提交顺序不等于跨流执行依赖。下一 bucket 的 inbound all-to-all 与上一 bucket 的矩阵计算，以及上一 bucket 的 outbound all-to-all 与当前 bucket 的矩阵计算，可无相互依赖。捕获按各流前沿和 event record 时的快照保留这些关系，不按全局记录顺序串行化。无需为 Muon 新建调度器或手工补通信节点。

每次 logical-world 会话将显式 Meta 的 `torch.get_device_module` 查询绑定到本轮设备模块，其他设备沿用原查询，退出后恢复。PyTorch 缓存旧 Meta handle 会导致后续会话的原生流上下文与捕获器看到的上下文不同，错误地把通信归入计算流。该适配避免跨会话缓存造成的虚假串行；依赖测试先执行一个真实 Muon 会话，再检查下一会话的重叠关系。

`tests/test_muon_dependencies.py` 对最终交付图验证必要等待、双向不可达的独立工作及 buffer 复用保护。DSV4 debug、world=8、EP=2、rank=0 的测试覆盖 9 个重分布 bucket 和 18 次原生 all-to-all；这些数量是该配置的预期，不是全模型常数。通信保留有序 ranks、split sizes、shape/dtype、流及正式前驱。

## 验证入口与边界

从 `experiments/simulator` 运行：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/test_muon_config.py tests/test_muon_capture.py tests/test_muon_dependencies.py \
  tests/test_muon_offload.py tests/memory/test_optimizer_state.py \
  tests/test_optimizer_contract.py tests/test_dsv4_capture.py tests/test_kimi_capture.py
```

配置测试覆盖显式选择、原生默认值和拒绝组合；捕获测试覆盖混合参数所有权、完整更新、状态字节、异常恢复和并行组合；原有 DSV4/Kimi 与 AdamW 测试保护已有路径。[聚合卸载测试](../tests/test_muon_offload.py) 以单卡和 world=8、EP=2 检查独立 workspace 容量预期、F/B/O 常驻、状态传输量与原生前驱保持。[内存字节测试](../tests/memory/test_optimizer_state.py) 检查 alias 去重和 workspace 卸载后保留。

依赖测试在卸载开关两种状态下证明图允许预取重叠，并不提供真实重叠时长。平台三路汇合与 Trace 诊断的 synthetic 验证边界见[优化器卸载](optimizer-offload.md#验证)。真实设备计算耗时、HCCL 传输及资源竞争由平台和算子后端处理；Meta 捕获不证明 NPU 数值、性能精度或真实跨 rank 计划一致性。

## 原生规划效率与宿主计时

主包扩展在兼容 backport 之后启用等价的 FlexShard 规划优化，正常训练和 simulator 共用同一实现，不额外改变 recipe 或 optimizer 开关。实现位于 `torchtitan_npu/extensions/torchtitan/distributed/flex_shard/`，与上游兼容补丁分开维护。

- 区域先执行完整边界校验；能沿任一轴证明互不相交时使用排序后的区间证明，否则回退通用两两校验。空区域、覆盖关系、重复参与者和非法布局仍须校验。
- 按源／目标 rank 建立 routes 索引，避免每个 rank 重扫所有 routes。
- All2All 打包按 peer 建立索引，并按原参与者、参数、route 顺序输出 splits 和 spans；保留零 split、自复制、replicated source 选择和全组 `has_remote_transfers`。
- mesh 的 rank→坐标映射仅在一次 `build_bucket_plans` 调用内复用；结束或异常时清理，不跨模拟会话缓存 tensor、slot、图事件或 Owned 分配结果。

优化不改变分桶、compute owner、矩阵计算、通信组与有序 ranks、All2All／事件依赖、打包 offset 或 workspace 容量。不通过删除校验、跳过空参与者的 collective 或改变 Newton–Schulz 次数缩短捕获时间。

`simulate()` 成功返回的 `iteration.schedule.annotations["host_timings"]` 记录 `clock`、`scope`、`status` 和 `stages`；每阶段包含 `wall_seconds` 与 `calls`。配置、model build、parallelize、optimizer build、warmup、正式 F/B 捕获、内存规划、FSDP 投影及卸载插入分别记录。`optimizer_planning.*` 包含 build、geometry、routes、validation、packing；这些是**含子调用的墙钟时间，不可直接求和**。`capture_step.*` 兼容保留原 `capture_timings` 的细项。

日志以 INFO 输出单次宿主计时摘要；失败时记录 `status="failed"` 及已执行阶段，不返回伪成功负载。计时用 `time.perf_counter()`，不做设备同步，不作为算子或通信的设备执行耗时，不改变静态显存预算；旧结果缺少此注记时不补造。
