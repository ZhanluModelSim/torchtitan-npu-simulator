# DeepSeek-V4 GraphTrainer 编译路径适配

GraphTrainer 使用 `minimal_fx_tracer` 将前向、loss 和反向捕获为联合 FX 图，再通过 graph pass 优化。本仓提供配套 TorchTitan v0.4.0 的 DeepSeek-V4（DSV4）A3 和 A5 GraphTrainer recipe。

## 启动训练

从仓库根目录启动 A3 八卡 Flash 示例。将 `HF_ASSETS_PATH` 指向实际的 tokenizer 目录。

```bash
HF_ASSETS_PATH=/path/to/DeepSeekV4_tokenizer STEPS=20 \
  bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3_graphtrainer.sh
```

| Python recipe（`torchtitan_npu_recipes.examples.deepseek_v4`） | 对应 launcher | 默认步数 |
| --- | --- | ---: |
| `deepseek_v4_flash_8p_cpt_4k_a3_graphtrainer` | [A3 八卡](../../examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3_graphtrainer.sh) | 100 |
| `deepseek_v4_flash_8p_cpt_4k_a5_graphtrainer` | [A5 八卡](../../examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5_graphtrainer.sh) | 20 |
| `deepseek_v4_flash_cpt_4k_a5_graphtrainer` | [A5 多机](../../examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a5_graphtrainer.sh) | 20 |

三个 recipe 均读取 `STEPS` 覆盖训练步数。其他训练字段通过 Python recipe 设置，配置组合见 [recipe 指南](../../torchtitan_npu_recipes/README.md)。TorchTitan v0.4.0 的 SPMD 路径不支持将梯度累积与 EP overlap 或自定义 pass pipeline 组合，示例使用单 microbatch。

## 上游 GraphTrainer 特性

设计动机见上游 [MANIFESTO](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/graph_trainer/MANIFESTO.md)：CPU 侧 kernel 启动开销随算力增长成为瓶颈，分布式训练需要编译器提供对性能、数值和可调试性的显式控制。核心能力：

- **整步单图捕获**：`minimal_fx_tracer` 基于 `make_fx`，在 FakeTensor 模式下联合 trace forward + loss + backward，不经 AOTAutograd 分割，所有反向计算在图内显式可见。
- **SimpleFSDP**：把 all-gather / reduce-scatter 表达为图内可 trace 的 DTensor 操作，集合通信成为图中的节点，可被 pass 重排、融合与重叠。
- **张量粒度内存策略**：每个激活可独立选择保存、重算或 CPU offload，区别于模块级 eager SAC。
- **Graph pass 流水线**：默认（保数值）pass 与 opt-in 性能 pass 分层，包括 bucketing 通信重叠、异步 TP、regional/full Inductor 编译、CUDA graph、CPU offload、选择性激活重算等。
- **可组合并行**：FSDP、TP、EP 均在图内表达；EP overlap（MoE chunk 级通信重叠）为实验性 pass。

NPU recipe 通过 `disable_passes=["cudagraph_pass"]` 禁用 CUDA Graph capture。

## 本仓实现

### 配置入口

图训练 recipe 复用对应的 Flash 配置，经 `to_graph_trainer_config` 转换后返回 `GraphTrainerEx.Config`，模型配置使用 `GraphTrainerDeepSeekV4Model.Config`。三个示例均选择 `mutation-functionalization+npu_auto_overlap`。A5 八卡示例默认使用 `dsv4-mhc-moe-save` 选重策略，其余两个使用 `full`。

### 模型变体

`GraphTrainerDeepSeekV4Model`（`model.py`）继承 eager 模型，仅在 `init_states` 外包一层 `disable_active_parametrization()`：懒初始化 parametrization（如 RoPE 频率缓冲）必须在 FX tracer 记录图之前物化，否则录制到的图不完整。

### 并行化

`parallelize_graph_trainer_deepseek_v4`（`parallelize.py`）是 DSV4 专属路径——eager 路径复用 DeepSeek-V3 并行化（DSV4 稀疏注意力分片基于它），而编译路径不能把稀疏注意力计算在 FSDP 下重排，故单独实现：

1. 为 MoE dispatch、combine、compute 和模块 FQN 添加图节点标注。
2. 在 SPMD 上下文中调用 `model._parallelize`，再应用 SimpleFSDP。
3. 配置启用 EP overlap 时，在 FX 捕获前包装 MoE forward，详见 [EP overlap 适配](graph_trainer_ep_overlap.md)。

SimpleFSDP 使用 DTensor 存储参数，模型计算侧使用本地 tensor 与 SPMD 类型。启用 sequence parallel 时，每个 DP rank 的 microbatch token 数须能被 TP degree 整除。

### Graph-safe 改造

GraphTrainer 的 trace 约束与 eager 有本质差异：`minimal_fx_tracer` 在 FakeTensor 下联合 trace，数据相关控制流与形状推导无法静态化，aot_autograd recompute 会丢弃 forward 中的 in-place 写。DSV4 稀疏注意力路径中依赖 in-place 写或数据相关形状推断的写法，在编译路径均改为函数式等价实现；注意力元数据 dataclass 注册为 pytree node，使 `minimal_fx_tracer` 能够识别其结构。

### Inductor runtime estimation patch

`patches/torch_npu/inductor_runtime_estimation.py`（新增，包导入时按需生效）：NPU 上以常量 1200 GB/s 代替 Triton CUDA-driver 探测 DRAM 带宽——该值只进入 graph 调度 roofline 估计，不改变生成算子；同时默认开启 `standalone_compile(donate_graph_module=True)`，避免图模块深拷贝开销。仅在 `torch.npu.is_available()` 时应用。

## 支持范围与限制

| 能力 | 状态 |
| --- | --- |
| 图捕获 | SPMD 联合 FX 图 |
| 并行组合 | 使用 SimpleFSDP 和 EP，TP 受 microbatch token 对齐约束 |
| Context Parallel | 不支持（DSV4 稀疏注意力编译路径未适配 CP） |
| EP overlap | 在 MoE forward 中切分序列，见 [EP overlap 适配](graph_trainer_ep_overlap.md) |

## 相关文档

- 上游 GraphTrainer 说明与 EP overlap 用法：[`torchtitan/experiments/graph_trainer/README.md`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/graph_trainer/README.md)
- EP overlap 适配：[`graph_trainer_ep_overlap.md`](graph_trainer_ep_overlap.md)
- DSV4 稀疏注意力与 TND 融合路径：[`deepseek_v4_tnd.md`](deepseek_v4_tnd.md)
