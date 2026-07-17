# 自定义 Context Parallel 特性

在分布式训练任务中，上下文并行（Context Parallelism, CP）是突破单卡内存瓶颈、
支持超长序列训练的核心技术。在基于 NPU 硬件生态推进 torchtitan 框架适配时，
现有技术方案暴露出显著的局限性：

1. PyTorch 原生 CP 设计强绑定于标准 SDPA 算子，仅提供 RingAttention 或原生
   AllGatherKV 的特定实现，无法覆盖 DeepSeek-V3.2 DSA
   （DeepSeek Sparse Attention）等稀疏注意力路径。
2. 框架需要允许开发者灵活扩展新的 CP 范式，如 Ulysses CP。

## 实现原理

`torchtitan_npu` 在
`torchtitan_npu/distributed/context_parallel/registry.py` 中维护一个 NPU 自有的
CP 策略注册表，并提供入口函数 `apply_cp_to_attention_module`，在 CP 开启时自动为
attention 模块匹配并应用合适的 CP 策略。目前注册的自定义 CP 策略包括
SDPA Ulysses CP、DSA CP、CompressorAttention CP 和 NPUVarlen Ulysses CP：

### SDPA Ulysses CP

定义在 `torchtitan_npu/distributed/context_parallel/ulysses_cp.py`，为常用的
SDPA 实现 Ulysses 风格的 CP。通过自定义 Context Parallel Context，我们在
基于 `torch_npu` 提供的融合算子的 Attention 计算前后插入 All-to-All 通信算子，
将数据从“序列并行分布”转换为“多头维度分布”，使单计算节点获得完整上下文。

> 注意：`torchtitan_npu` 使用注册机制自动匹配 CP 策略。
> 当 `context_parallel_degree > 1` 时，模块导入即触发 CP 策略注册，
> 无需额外配置开关。

### DSA CP

定义在 `torchtitan_npu/distributed/context_parallel/sparse_attention_cp.py`，为
DeepSeek Sparse Attention 提供 AllGatherKV 风格的 CP。本项目在遵循
torch 原生 CP 设计逻辑的基础上，通过自定义 Context Parallel Context，
将注意力部分的 forward 函数替换为 CP 感知实现，对 KV 相关激活做 CP 域的
AllGather，并在反向传播中完成对应的 ReduceScatter，以确保 DSA 注意力在
CP 场景下的正确性。

关于 DSA CP 的更多原理介绍，参考
[技术文档](https://gitcode.com/cann/cann-recipes-train/blob/master/docs/llm_pretrain/deepseek-v32_pre_train_optimization.md#自定义CP策略)。

### CompressorAttention CP

定义在 `torchtitan_npu/distributed/context_parallel/compressor_attention_cp.py`，为
DeepSeek-V4 Attention 的 compressor 路径提供 CP。该策略在 pre-hook 中通过
P2P window exchange 拼接相邻 CP rank 的重叠窗口，并在 post-hook 中对压缩后的
KV 与 `k_indexer` 做 CP 域 all-gather、因果切片和必要的 padding。

### NPUVarlen Ulysses CP

定义在 `torchtitan_npu/distributed/context_parallel/npu_varlen_cp.py`，为
`NPUVarlenAttention` 提供 Ulysses 风格的 CP。该策略复用 Ulysses all-to-all，
在 forward 前后交换 Q/K/V 和输出的序列、头维分布，并注册 mask handler 以保留
`VarlenMetadata` 文档边界信息。

## 支持范围

| 模型 | CP 实现 | 触发条件 | 说明 |
| --- | --- | --- | --- |
| DeepSeek-V3 / Qwen3 SDPA | Ulysses CP | `context_parallel_degree > 1` | SDPA 路径使用 All-to-All 变换 |
| DeepSeek-V3.2 | DSA CP | `context_parallel_degree > 1` 且启用 `npu_dsa` converter | - |
| DeepSeek-V4 | CompressorAttention CP | `context_parallel_degree > 1` 且启用 `npu_smla` converter | CompressorAttention 路径处理窗口重叠、压缩 KV 和 CP mask |
| Qwen3 TND | NPUVarlen Ulysses CP | `context_parallel_degree > 1` 且配置使用 `NPUVarlenAttention` | TND 路径使用 All-to-All 变换并保留 `VarlenMetadata` |


## 配置选项

Custom CP 配置写在模型的 `config_registry.py` 中。对应字段
`torchtitan_npu.config.configs.ParallelismConfig`。

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `context_parallel_degree` | int | 1 | Context 并行度；大于 1 时启用 CP mesh |

> **注意**：DeepSeek-V3.2 的 DSA CP 依赖 `npu_dsa` converter，DeepSeek-V4 的
> CompressorAttention CP 依赖 `npu_smla` converter。启用对应模型的 CP 时，
> converters 中需要包含匹配的融合算子 converter。

CP类型由框架根据模型及转换器配置**自动选择**，无需手动指定：

- **DeepSeek-V3 / Qwen3 SDPA**：`context_parallel_degree > 1` 时自动使用 Ulysses CP。
- **DeepSeek-V3.2**：当 converters 中包含 `npu_dsa` 且 CP 开启时，自动使用 DSA CP。
- **DeepSeek-V4**：当 converters 中包含 `npu_smla` 且 CP 开启时，自动使用 CompressorAttention CP。
- **Qwen3 TND**：当模型配置使用 `NPUVarlenAttention` 且 CP 开启时，自动使用 NPUVarlen Ulysses CP。

### 配置示例一：SDPA Ulysses CP（DeepSeek-V3）

在 `torchtitan_npu/models/deepseek_v3/config_registry.py` 的配置函数中设置：

```python
from torchtitan_npu.config.configs import ParallelismConfig

parallelism = ParallelismConfig(
    context_parallel_degree=2,
)
```

启动时也可以直接覆盖：

```bash
bash scripts/run_train.sh \
  --parallelism.context_parallel_degree 2
```

### 配置示例二：DSA Context Parallel（DeepSeek-V3.2）

DeepSeek-V3.2 `config_registry.py` 中的
`deepseek_v32_671b_61layers_32k_128die` 配置默认启用 32k 长序列 CP：

```python
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import ParallelismConfig
from torchtitan_npu.converters import get_model_converter_config

model_converters = ModelConvertersContainer.Config(
    converters=[
        get_model_converter_config("npu_dsa"),
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
    ],
)
parallelism = ParallelismConfig(
    context_parallel_degree=8,
)
```

多机长序列训练推荐在 `scripts/run_train_multinodes.sh` 中配置节点 IP、
`NPUS_PER_NODE` 等集群信息后，直接覆盖为 32k 预置配置启动：

```bash
MODULE=torchtitan_npu.models.deepseek_v32 \
CONFIG=deepseek_v32_671b_61layers_32k_128die \
bash scripts/run_train_multinodes.sh
```

调试时也可以基于已有配置覆盖 CP 参数，例如：

```bash
NGPU=16 \
MODULE=torchtitan_npu.models.deepseek_v32 \
CONFIG=deepseek_v32_671b_4layers_debug \
bash scripts/run_train.sh \
  --parallelism.context_parallel_degree 2
```

上述覆盖方式仍要求 DeepSeek-V3.2 配置中保留 `npu_dsa` converter。

## 约束与注意事项

- DeepSeek-V3.2 CP 当前仅支持 `attn_type="sdpa"` 的入口，并在 CP 路径中根据
  `npu_dsa` converter 自动切换到 DSA CP 路由。
- `training.seq_len` 需要能被 TP 与 CP 共同切分。当前实现中会校验
  `training.seq_len % parallel_dims.seq_len_divisor == 0`，报错信息会提示
  `seq_len` 需要能被 `TP degree * 2 * CP degree` 整除。
- DeepSeek-V3.2 DSA CP 会对 `block.attention.inner_attention` 应用 CP patch，
  并将 `model_args`、`tp_mesh` 和 converters 传入自定义 CP 路由。
- 若使用 32k 长序列训练，建议优先从
  `deepseek_v32_671b_61layers_32k_128die` 预置配置开始调整，避免遗漏 DSA
  converter、CP 并行度和长序列 batch 配置。
