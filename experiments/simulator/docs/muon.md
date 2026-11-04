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

当前支持 DSV4、TP=CP=PP=1 下的单卡、DP shard、HSDP、EP 及合法组合。小模型回归包含非零 logical rank、MTP、full AC 和激活卸载。Kimi、Muon 量化训练、HSDP replica dedup、原生 swap/virtual optimizer override 以及 sim 优化器状态卸载均明确拒绝。

Muon 在 EP=1 时若需要将实际参数存储 `Shard(1)` 转为 compute layout 的 `Shard(0)`，在模型并行化之后、构建优化器之前抛出 `ValueError`。报错包含参数 FQN、全局 shape、mesh 轴与大小，并建议使用 AdamW 或受支持的 EP/FSDP 配置；不实现该布局转换。检查依据实际 DTensor placement、原生参数分组和 compute layout，不按卡数一概拒绝。debug 模型有 4 个专家，EP=1、FSDP=2/4 仍通过，FSDP=8/16 提前拒绝；将专家数改为 8 后，EP=1、FSDP=8 的 `Shard(0)` 路径也通过。AdamW 和 EP>1 不受该规则影响。

优化器状态卸载需要独立设计：Muon momentum 在 bucket 准备阶段即被读取和更新，随后还有 compute-layout 重分布及交错计算。当前 AdamW 卸载窗口只包围 fused update，不能直接套用，也不能只在最终参数 `add_` 附近补传输。

## 原生执行与 Meta 适配

`compat/muon.py` 只在含 DistMuon 的仿真会话中安装可恢复适配：

- 允许全部本地参数为 Meta 的原生 DTensor；其他设备仍调用原校验。
- 保留原生 layout、bucket 构建和梯度／状态布局检查。跨 rank 的计划哈希比较需要真实 peers 和张量数值，单 logical-rank Meta 会话不执行该比较；`runtime_capabilities.muon_peer_plan_validation` 明确标记 `not_performed_single_logical_rank`。这不证明多进程计划一致性。
- 观察原生 `_apply_update`，将实际写入参数 storage 的最终 `add_(direction)` 关联到参数 FQN，不替换更新计算。

同一模型与优化器先完成结构预热，再捕获代表性 F/B/O。Muon momentum 和 AdamW moments 均来自真实 optimizer state，按本地 storage 去重计入静态预算。GMM 对应的 routed-expert `w1_EFD / w2_EDF / w3_EFD` 归 DistMuon；更新时批量矩阵 Newton–Schulz 展开为 `bmm/baddbmm`，前后向仍保留原生 GMM。

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
  tests/test_optimizer_contract.py tests/test_dsv4_capture.py tests/test_kimi_capture.py
```

配置测试覆盖显式选择、原生默认值和拒绝组合；捕获测试覆盖混合参数所有权、完整更新、状态字节、异常恢复和并行组合；原有 DSV4/Kimi 与 AdamW 测试保护已有路径。依赖测试证明图允许预取重叠，并不提供真实重叠时长。真实设备计算耗时、HCCL 传输及资源竞争由平台和算子后端处理；Meta 捕获不证明 NPU 数值、性能精度或真实跨 rank 计划一致性。
