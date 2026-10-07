# Simulator 当前架构与执行路径

> 现状核对：2026-10-07，`feat/npu-simulator`，实现基线 `ecde2f0`。依据源码与 Git 历史静态核对，未执行测试或仿真；后续行为以当前源码为准。

本文描述当前实现。配置与模型支持范围见[使用入口](../user-guides/simulator.md)，原始方案的历史身份见[文档导航](../user-guides/simulator/documentation-map.md)。

## 定位与边界

当前 simulator 是侧载在真实 TorchTitan/NPU 训练路径上的 **meta 训练图捕获、调度结构导出和逻辑内存分析工具**。它复用 ModelSpec、模型结构、converter、parallelize 和 pipeline schedule，执行一个 forward/backward/optimizer step，输出 L0–L3 IR。

它已经超出 7 月初的单进程原型，具备代表 PP 进程捕获、复杂流水调度、通信归属规范化、FSDP 驻留、checkpoint/saved activation 生命周期及若干假设优化开关。

同时需要严格区分：

- 模型参数和激活不分配真实 NPU 显存，但 Python 图对象、CPU mesh、tokenizer/dataloader 和导出文件仍消耗主机资源。
- meta 无数值，因此不能验证 loss、grad_norm、收敛或数值一致性。
- 当前不是已经校准的设备时延仿真器；静态 FLOPs/通信字节是辅助描述，下游 predictor/DES 负责设备性能、资源和网络约束。
- 内存指标描述 active tensor bytes，摘要峰值字段为 `active_bytes_peak` 等；不包含 allocator reserved/cache、碎片、kernel workspace 等真实设备开销。
- “代码支持”表示当前有注册与实现路径，不表示本次运行验收通过，也不保证所有并行/精度/AC 组合均已覆盖。

## 执行路径

```text
scripts/run_simulator_spawn.py
  └─ ConfigManager 解析与 worker 完全一致的 CLI
     └─ resolve_simulation_runtime_from_environment
        ├─ 解析 logical world_size、DP/TP/CP/EP/ETP/PP
        ├─ PP=1 → fake_backend → 1 个真实进程
        └─ PP>1 → multi_proc_meta → PP 个真实进程

torchtitan_npu.entry.main
  ├─ import torchtitan_npu：加载 NPU patches/converters/model injection
  ├─ 再次解析配置，确认最终运行模式与 rank 身份
  └─ config.build() → SimulationTrainer
     ├─ 强制 MoE 均衡；未设置 seed 时补 42；关闭 compile/swap_optimizer
     ├─ 安装 meta 环境与 MHC/RMSNorm/SMLA shim
     ├─ super().__init__：真实模型、converter、并行化、优化器、dataloader
     ├─ 在原有 Kimi 模块上绑定 shim，保留并行 hooks 和参数
     └─ train：读取一批数据，转 meta，捕获单 step
        └─ run_simulation_step
           ├─ checkpoint execution markers / module path / phase
           ├─ TorchDispatchMode + fake collectives + saved_tensors_hooks
           ├─ forward_backward_step
           ├─ optimizer.step + lr_scheduler.step
           ├─ RankTable + 通信域语义解析
           ├─ OpNode，折叠 workload 图中的 metadata views
           ├─ StepGraph 模板
           ├─ semantic action stream → SchedulePlan + DataSlot
           ├─ 通信 ownership plugins → 模板变体 → 结构校验
           ├─ SchedulePlan 投影为兼容 ScheduleGraph
           ├─ WorkloadGraph：一次 train iteration
           ├─ memory schedule replay / estimator / plugins
           └─ 每个 capture process 独立导出
```

入口证据：[launcher](../../scripts/run_simulator_spawn.py)、[entry](../../torchtitan_npu/entry.py)、[trainer](../../torchtitan_npu/simulator/trainer.py)。

`SimulationTrainer.train()` 不使用普通 `Trainer.train_step()` 的 token reduction、loss/grad_norm 数值日志，因为这些路径会读取 meta tensor 的 `.item()`。它以标签静态元素数量提供 token 归一化输入。`training.steps` 不把仿真变成多 step 捕获，当前 L3 固定一个 iteration。

## Rank、并行和四层 IR

### 并行拓扑

```text
world_size = PP × DP_replicate × DP_shard × CP × TP
EFSDP = DP_shard × CP × TP / (EP × ETP)
EDP = DP_replicate × EFSDP
```

EP/ETP 重新解释 dense world 的一部分，不是额外 world_size 乘数。运行时要求 `EP × ETP` 整除 `DP_shard × CP × TP`，ETP 必须为 1 或 TP；模型自己的实现约束还要另行检查。

`DP_shard=-1` 会用最终 world_size 自动解析。world_size 优先级是显式 `--simulation.world-size`、`NGPU`、spawn 内部 `TORCHTITAN_SIM_WORLD_SIZE`、配置可推导值。当前正式 preset 通常是 `DP_shard=-1`，没有固定 `simulation.world_size`，因此需要提供 world_size 来源，不能沿用旧文档的默认 384 卡假设。

多进程下必须区分：

- `capture_process_rank`：真实 Gloo/PP 控制进程；真实进程数量等于 PP。
- `logical_global_rank`：完整逻辑 mesh 的代表 rank，当前为 `capture_rank × world_size / PP`。
- `stage`：模型虚拟 stage；DualPipeV 中一个物理 rank 可能拥有多个 stage。

完整 logical world 从代表模板及 RankTable 展开，并没有为每张逻辑卡启动独立训练进程。非 PP rank 的不同运行时路由数值和负载波动不在当前均衡 meta 捕获范围内。

### 四层 IR

| 层 | 当前职责 |
|---|---|
| L0 `OpNode` | 原始算子名、canonical type、shape/dtype、依赖、phase、execution_kind、通信元数据和辅助 cost |
| L1 `StepGraph` | 各 stage 的 F/B/I/W/optimizer 模板，包含属于计算块的通信；需要时生成不可变通信变体 |
| L2 `SchedulePlan` | 当前权威调度对象：语义 action 发布顺序、template_ref、DataSlot、CommDetail、稳定通信/驻留 ID |
| L2 `ScheduleGraph` | 从 SchedulePlan 投影的兼容对象，不再独立猜测另一套执行顺序 |
| L3 `WorkloadGraph` | 一次训练迭代、microbatch 和数据流，并携带 SchedulePlan |

捕获按 `(stage, comp_type)` 首次出现保留完整算子图，重复 chunk 保留调度与通信事实。`comp_type` 区分 F、完整 B、输入梯度 I、权重梯度 W。通信变化会由 ownership 处理生成 `__comm_v*` 模板，不能简单宣称“只有 MB0 一张图”。

当前捕获具备 1F1B、GPipe、运行时/interleaved 调度和 DualPipeV 语义入口；重点演进与消费文档覆盖 1F1B 和 DualPipeV。DualPipeV 保留 `OVERLAP_F_B` parent/sub_actions、同 rank 的 V 形 local transfer、跨 rank SEND/RECV。这表示调度结构被捕获，不代表 meta 捕获测出了真实双图并发时间。

Schema v2 的权威关系：

- `schedule_order` 是 rank-local 发布顺序；`seq_idx` 只是 L0/内存诊断位置，不是全局时间。
- 数据依赖来自 `producer_action_id → DataSlot → consumer_action_ids`。
- 跨 rank PP 用 `transfer_id` 配对，不用 action ID、CSV 行号或到达先后猜测。
- FSDP transition ID 连接 unshard/all-gather、wait 后真实驻留和实际 reshard 释放。
- 后端 stream/link 资源串行关系不能冒充 tensor data dependency 写回图。

## 插件与适配

当前有三个不同层次的扩展，不应混为一个“插件开关”。

| 层次 | 接入方式 | 当前内容 |
|---|---|---|
| NPU 插件仓 | 包导入 `_apply_patches()`、converter registry、ModelSpec/module injection | 复用 NPU 算子与模型并行实现，模型训练本身仍来自 TorchTitan |
| Simulator 硬件适配 | meta patches、converter 的运行时替换、绑定原模块方法、shape-only autograd bridge | MHC、SMLA/indexer、RMSNorm、RoPE/MoE backward、KDA、MLA core、SiTU-GLU、GMM、fused AdamW |
| 分析插件 | `CommunicationOwnershipPlugin` 和 `MemoryModelPlugin.apply(context)` | 通信归属规范化；内存生命周期的框架语义补充 |

通信插件按 Pipeline P2P、FSDP Stage、marker cleanup、gradient reduction、embedded stage communication 顺序处理。内存插件包括 FSDP full-param residency、可选 FP8 all-gather staging、ActivationCheckpoint、ActivationOffload、MissingParameterGradient。

MHC/SMLA 目前通过替换已注册 converter 的 implementation class 保留真实 Triton/ACLNN 算子名字，不再剥掉 converter 后退化成整个基类图。Kimi shim 在模型构造/并行化之后绑定原模块，避免替换模块丢失 hooks 或分布式参数。

MXFP8 复用真实 NPU quantize/matmul/grouped-MM meta kernel 和 autograd 实现，只在仿真下绕过硬件 capability 检查。它与“假设参数存 FP8”和“假设通信传 FP8”是三个独立维度。

硬件 shim 可以提供真实名字和 shape/dependency，但其保存 tensor 与 backward 仍是显式建模，不能视为真实 kernel workspace 或真实数值执行。

## 通信归属与依赖

| 观察到的通信 | 归属 | 处理方式 |
|---|---|---|
| F/B/I/W 内的 TP/CP/EP、FSDP AG、梯度归约 | `L1_STAGE` | 在引用模板内计费与回放，L2 不重复补通信 |
| PP SEND/RECV | `L2_PIPELINE` | 专用通信 fragment/action，按 transfer_id 跨进程配对 |
| compute 外显式 FSDP prefetch | `L2_PREFETCH` | UNSHARD → param_full → COMPUTE → control → RESHARD |
| compute 外真实梯度归约 | `L2_STANDALONE` | B/W → local gradient → REDUCE_GRAD → reduced gradient → OPTIMIZER |
| 无实际工作或已经在 L1 的调度意图 | 无额外通信 | 删除，不能产生重复成本或阻塞 slot |

RESHARD 是本地 full-parameter 释放，不是 reduce-scatter。FSDP AG 只门控它自己参数组的计算。Prefetch 的源计算与目标 AG 共享源 invocation readiness，二者是 sibling，不能相互串成循环，也不能让 AG 链提前预取未进入的后续层。

FSDP RS 从该参数组的真实梯度生产者得到依赖；HSDP AR 保留对应 RS 的数据路径。观察上的先后顺序不能强行串行无关参数组，AR 也不能无依据门控下一组 RS/AG。CP/TP 的 RS 不能仅凭算子同名被当成 FSDP reduction。

当前权威文档：[通信归属契约](communication-ownership-contract.md)、[PP L2 架构](pp-l2-capture-architecture.md)、[依赖重建契约](schedule-plan-dependency-reconstruction-contract.md)、[DualPipeV 消费指南](dualpipev-schedule-plan-consumer-guide.md)。

## 源码阅读顺序

1. 现状入口与选项：[trainer](../../torchtitan_npu/simulator/trainer.py)、[registry](../../torchtitan_npu/simulator/config_registry.py)、[runtime](../../torchtitan_npu/simulator/utils.py)。
2. 调度结构：PP L2 架构、通信归属契约、依赖重建契约、DualPipeV 消费指南。
3. 捕获实现：[dispatch](../../torchtitan_npu/simulator/capture/dispatch_capture.py)、[communication ownership](../../torchtitan_npu/simulator/capture/communication_ownership.py)、[schedule builder](../../torchtitan_npu/simulator/capture/schedule_builder.py)。
4. meta/硬件边界：[meta environment](../../torchtitan_npu/simulator/meta_env.py)、[hardware shims](../../torchtitan_npu/simulator/hardware_shims)。
5. 模型特例：DeepSeek/Kimi model、parallelize、config_overrides 和参数目录文档。
6. 内存口径：estimator、schedule_replay、records 及各 memory plugin；设计文档提供起因，当前代码确定落地范围。
7. 历史背景：7 月初 simulator spec/plan、AllToAll/optimizer/MXFP8 分析；用 Git 合入历史确认哪些建议已经实现。
