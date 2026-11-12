# DeepSeek-V4 GraphTrainer 编译路径适配

GraphTrainer 是 torchtitan 上游 `experiments/graph_trainer/` 中的编译式分布式训练实验：用 `minimal_fx_tracer` 把 forward、loss、backward（可选 optimizer.step）整体捕获为单个 FX 图，所有优化都以 graph pass 的形式作用于同一张图。本仓保留 DeepSeek-V4（DSV4）的 NPU 图训练实现；TorchTitan 0.4.0 的当前验收范围为文本 eager，图训练仍待迁移和回归。图训练配置位于独立 recipe 模块。

## 上游 GraphTrainer 特性

设计动机见上游 [MANIFESTO](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/graph_trainer/MANIFESTO.md)：CPU 侧 kernel 启动开销随算力增长成为瓶颈，分布式训练需要编译器提供对性能、数值和可调试性的显式控制。核心能力：

- **整步单图捕获**：`minimal_fx_tracer` 基于 `make_fx`，在 FakeTensor 模式下联合 trace forward + loss + backward，不经 AOTAutograd 分割，所有反向计算在图内显式可见。
- **SimpleFSDP**：把 all-gather / reduce-scatter 表达为图内可 trace 的 DTensor 操作，集合通信成为图中的节点，可被 pass 重排、融合与重叠。
- **张量粒度内存策略**：每个激活可独立选择保存、重算或 CPU offload，区别于模块级 eager SAC。
- **Graph pass 流水线**：默认（保数值）pass 与 opt-in 性能 pass 分层，包括 bucketing 通信重叠、异步 TP、regional/full Inductor 编译、CUDA graph、CPU offload、选择性激活重算等。
- **可组合并行**：FSDP、TP、EP 均在图内表达；EP overlap（MoE chunk 级通信重叠）为实验性 pass。

编译模式固定为 `aot_fx_trace`（上游已弃用 `aot` 与 `jit` 两种旧模式）。NPU 上必须禁用 `cudagraph_pass`（CUDA graph capture 无意义且不兼容），本适配的编译配置已内置该禁用项。

## 本仓实现

### 配置入口

`torchtitan_npu_recipes/graph_trainer/deepseek_v4.py` 保存图训练组合，均复用 `torchtitan_npu_recipes.models.deepseek_v4` 中的对应 eager 配置，经 `torchtitan_npu/models/deepseek_v4/graph_trainer.py` 的 `to_graph_trainer_config` 包装为 `GraphTrainer.Config`：

| 配置名 | 对应 eager 配置 |
| --- | --- |
| `graph_trainer_deepseek_v4_debugmodel` | `deepseek_v4_debugmodel` |
| `graph_trainer_deepseek_v4_flash` | `deepseek_v4_flash` |
| `graph_trainer_deepseek_v4_flash_43layers_16experts` | `deepseek_v4_flash_43layers_16experts` |
| `graph_trainer_deepseek_v4_pro` | `deepseek_v4_pro` |
| `graph_trainer_deepseek_v4_pro_61layers_32experts` | `deepseek_v4_pro_61layers_32experts` |

编译配置统一为 `mode="aot_fx_trace"`、`memory_policy="full"`、`disable_passes=["cudagraph_pass"]`。模型通过 `_graph_trainer_model_registry` 包装为 `GraphTrainerDeepSeekV4Model.Config`，并行化函数指向 `parallelize_graph_trainer_deepseek_v4`。

### 模型变体

`GraphTrainerDeepSeekV4Model`（`model.py`）继承 eager 模型，仅在 `init_states` 外包一层 `disable_active_parametrization()`：懒初始化 parametrization（如 RoPE 频率缓冲）必须在 FX tracer 记录图之前物化，否则录制到的图不完整。

### 并行化

`parallelize_graph_trainer_deepseek_v4`（`parallelize.py`）是 DSV4 专属路径——eager 路径复用 DeepSeek-V3 并行化（DSV4 稀疏注意力分片基于它），而编译路径不能把稀疏注意力计算在 FSDP 下重排，故单独实现：

1. 对 MoE 的 dispatch / combine / compute 区域与各子模块 FQN 附加图节点标注（`annotate_moe_ep_regions` + `annotate_module_fqns`），供 bucketing、区域编译等下游 pass 消费。
2. 使用 `spmd_types` 后端或启用 TP / EP 时调用 `model.parallelize(parallel_dims)`；`spmd_types` 即使在 singleton 模型并行轴下也需要类型标注。
3. 无条件应用 `apply_simple_fsdp`：fsdp mesh 在 degree 1 时也存在（`ParallelDims._mesh_exist`），因此单卡下 MixedPrecisionPolicy 的 param_dtype cast 仍然生效。
4. 按模式应用 `apply_graph_trainer_compile`。

支持 `partial_dtensor` 和 `spmd_types` 后端。`spmd_types` 复用上游 DTensor-backed SimpleFSDP 存储，在模型前向和反向计算侧使用本地 tensor 与 SPMD 类型。入口要求每个 DP rank 的 microbatch token 数能被 `parallel_dims.seq_len_divisor` 整除。

### Graph-safe 改造

GraphTrainer 的 trace 约束与 eager 有本质差异：`minimal_fx_tracer` 在 FakeTensor 下联合 trace，数据相关控制流与形状推导无法静态化，aot_autograd recompute 会丢弃 forward 中的 in-place 写。DSV4 稀疏注意力路径中依赖 in-place 写或数据相关形状推断的写法，在编译路径均改为函数式等价实现；注意力元数据 dataclass 注册为 pytree node，使 `minimal_fx_tracer` 能够识别其结构。

### Inductor runtime estimation patch

`patches/torch_npu/inductor_runtime_estimation.py`（新增，包导入时按需生效）：NPU 上以常量 1200 GB/s 代替 Triton CUDA-driver 探测 DRAM 带宽——该值只进入 graph 调度 roofline 估计，不改变生成算子；同时默认开启 `standalone_compile(donate_graph_module=True)`，避免图模块深拷贝开销。仅在 `torch.npu.is_available()` 时应用。

## 激活方式

TorchTitan 0.4.0 适配当前只验收 DeepSeek 文本 eager，GraphTrainer 尚未完成迁移和回归。本页说明保留的实现结构，不作为当前版本可运行的图训练指南。

旧图训练 launcher 位于 `examples/deepseek_v4/` 及其 `debug/` 子目录，模块选择指向 `torchtitan_npu_recipes.graph_trainer.deepseek_v4`。其中的通用字段 CLI 参数仍需迁移为 Python 配置；迁移完成前不要据此启动 0.4.0 图训练。完整训练策略应定义新 recipe，由 shell launcher 选择，组合方式见 [recipe 指南](../../torchtitan_npu_recipes/README.md)。

下表描述保留实现的设计范围；历史验证不代表当前依赖组合已通过验收。

## 支持范围与限制

| 能力 | 状态 |
| --- | --- |
| 编译模式 | `aot_fx_trace`；NPU 上禁用 `cudagraph_pass` |
| 并行组合 | FSDP（含 degree 1）、EP 已验证；TP 要求 `seq_len` 整除 `seq_len_divisor` |
| Context Parallel | 不支持（DSV4 稀疏注意力编译路径未适配 CP） |
| EP overlap（上游实验性 pass） | 已通过patch适配；见 [`graph_trainer_ep_overlap.md`](graph_trainer_ep_overlap.md) |

## 相关文档

- 上游 GraphTrainer 说明与 EP overlap 用法：[`torchtitan/experiments/graph_trainer/README.md`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/graph_trainer/README.md)
- EP overlap 适配：[`graph_trainer_ep_overlap.md`](graph_trainer_ep_overlap.md)
- DSV4 稀疏注意力与 TND 融合路径：[`deepseek_v4_tnd.md`](deepseek_v4_tnd.md)
