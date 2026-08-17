# 低精度训练特性（MXFP8）

在大规模语言模型的分布式训练中，矩阵乘法运算（GEMM）占据了绝大部分计算开销。传统的 BF16/FP16 混合精度训练虽然已大幅降低了显存占用，但在超大规模模型（如 DeepSeek-V3 671B）上仍面临计算效率瓶颈。低精度训练通过将线性层和 MoE 专家层的矩阵乘法降至 8-bit 浮点精度执行，在保持训练收敛性的前提下，显著提升计算吞吐并降低显存消耗。

本特性基于 [torchao](https://github.com/pytorch/ao) 的 MXFP8 训练框架，通过 NPU 侧的 monkey-patch 将 torchao 的 MXFP8 计算路径重定向至 `torch_npu` 原生算子，覆盖普通线性层（nn.Linear）和 MoE 专家层（Grouped MM）两大场景。

## 硬件要求

低精度训练特性仅支持 **Ascend 950 及更高架构**的 NPU 设备。系统在初始化时会通过 `torch_npu.npu.get_device_name()` 进行硬件检测，不满足要求时将抛出异常。

## 实现原理

### 整体架构

本特性采用 **torchao 原生 MXFP8 框架 + NPU 算子替换** 的架构。torchtitan 上游提供 `MXFP8Converter` 作为模型转换入口，torchao 负责量化配置与权重包装，torchtitan-npu 通过 monkey-patch 将 torchao 内部的矩阵乘法调度函数替换为 NPU 实现。相关代码主要分布在以下文件中：

| 文件路径 | 修改作用                                                                    |
| --- |-------------------------------------------------------------------------|
| `torchtitan_npu/experiments/ao_npu/torchao_npu/patches/mx_capability_check.py` | 替换 `has_cuda_capability` 函数，使 MXFP8Converter 在 NPU 上进行硬件校验              |
| `torchtitan_npu/experiments/ao_npu/torchao_npu/patches/mx_linear.py` | 替换 torchao 的 `_to_mxfp8_then_scaled_mm`，将线性层 MXFP8 计算重定向至 NPU 算子        |
| `torchtitan_npu/experiments/ao_npu/torchao_npu/patches/mxfp8_grouped_mm.py` | 替换 torchao 的 `_to_mxfp8_then_scaled_grouped_mm`，将 MoE 分组矩阵乘法重定向至 NPU 算子 |

### 线性层低精度

`MXFP8Converter` 在转换阶段，通过 torchao 的 `quantize_` API 对模型中指定 FQN 的 `nn.Linear` 模块的权重进行包装（`MXFP8TrainingWeightWrapperTensor`）。在前向传播时，权重包装器的 `__torch_function__` 拦截矩阵乘法调用，进入 `_to_mxfp8_then_scaled_mm` 函数。

NPU 侧通过 patch `torchao.prototype.mx_formats.mx_linear._to_mxfp8_then_scaled_mm`，将其替换为调用 NPU 原生算子的 `NpuMXFP8MM`：

- **前向传播**：使用 `torch_npu.npu_dynamic_mx_quant` 对激活和权重分别进行 per-block 量化（block size=32，沿 axis=-1 方向），每 32 个元素共享一个 e8m0 scale，再通过 `torch_npu.npu_quant_matmul` 执行 FP8 矩阵乘法，输出恢复为原始精度（BF16）。
- **反向传播**：输入梯度（dx）和权重梯度（dw）的计算同样在 FP8 精度下完成，其中权重梯度使用 `npu_dynamic_mx_quant` 对权重沿 axis=-2 方向进行 per-block 量化。

### MoE 专家层低精度

对于 MoE（Mixture of Experts）架构中的专家层，torchao 通过 `_to_mxfp8_then_scaled_grouped_mm` 函数调度分组矩阵乘法。

NPU 侧通过 patch `torchao.prototype.moe_training.mxfp8_grouped_mm._to_mxfp8_then_scaled_grouped_mm`，将其替换为调用 NPU 原生算子的 `NpuMXFP8GroupedMM`：

- **前向传播**：使用 `torch_npu.npu_dynamic_mx_quant` 对输入和权重分别进行 per-block 量化（block size=32），再调用 `torch_npu.npu_grouped_matmul` 执行 FP8 分组矩阵乘法。
- **反向传播**：输入梯度使用 `npu_dynamic_mx_quant` + `npu_grouped_matmul` 计算；权重梯度使用 `torch_npu.npu_grouped_dynamic_mx_quant` 对输入和梯度分别进行 per-group 量化后，再调用 `npu_grouped_matmul` 计算。

> **注意**：MoE 低精度功能依赖 `npu_gmm` converter 提供的分组矩阵乘法基础实现，因此在 converters 配置中 `npu_gmm` 必须位于 `MXFP8Converter` 之前。

## 配置选项

低精度训练通过 `ModelConvertersContainer.Config` 的 `converters` 列表启用，使用上游 `MXFP8Converter.Config` 进行配置。

### MXFP8Converter 配置

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `recipe_name` | str | `"mxfp8_rceil"` | 量化 recipe 名称。当前唯一可选值：`"mxfp8_rceil"`（MXFP8 动态量化，scale 计算采用 RCEIL 舍入模式）。 |
| `fqns` | list[str] | [] | 需要启用 MXFP8 量化的模块全限定名（FQN）列表。匹配规则为子字符串包含，例如 `"moe.experts"` 将匹配所有 FQN 中包含该字符串的模块。留空表示不对任何模块启用 MXFP8。 |

### 环境变量配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MXFP8_DUAL_AXIS_FORWARD` | `1`（启用） | 控制 MXFP8 线性层和 MoE 专家层是否启用 forward dual-axis 量化（forward 阶段同时生成 backward 的量化数据），默认启用。设置为 `0` 或 `false` 可关闭，forward 回退到 single-axis 量化行为。无论 `MXFP8_DUAL_AXIS_FORWARD` 是否启用，Linear的 backward grad始终使用 dual-axis 量化。 |

### 配置示例

在模型的 `config_registry.py` 中配置 `model_converters` 并添加 `MXFP8Converter`：

**示例：对指定线性层和 MoE 专家层启用 MXFP8 低精度训练**

```python
from torchtitan.components.quantization.mx import MXFP8Converter
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.converters import get_model_converter_config

model_converters = ModelConvertersContainer.Config(
    converters=[
        # NPU 基础 converter（npu_gmm 必须在 MXFP8Converter 之前）
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_smla"),
        get_model_converter_config("npu_mhc_pre"),
        # MXFP8 低精度训练
        MXFP8Converter.Config(
            recipe_name="mxfp8_rceil",
            fqns=[
                # Attention 线性层
                "pre_attention.wq_a",
                "pre_attention.wq_b",
                "pre_attention.wkv",
                "pre_attention.indexer.wq_b",
                "post_attention.wo_a",
                "post_attention.wo_b",
                # MoE 专家层
                "moe.experts",
                "moe.shared_experts",
            ],
        ),
    ],
)
```

## 验证清单

1. **确认 converter 生效**：启动日志中应出现以下关键字：
   - `MXFP8 MoE training enabled`（来自上游 `MXFP8Converter.__init__`）
   - `Converted layers matching FQNS ... to use dynamic mxfp8_rceil quantization for grouped_mm and linear ops`（来自上游 `MXFP8Converter.convert`）
2. **确认模块替换数量**：日志中转换信息应与配置的 `fqns` 列表匹配。
3. **常见未生效场景排查**：
   - `converters` 顺序错误：`npu_gmm` 未放在 `MXFP8Converter` 之前，导致 MoE 专家层替换失败
   - `fqns` 匹配不到目标模块：检查模块的 FQN 是否包含配置的子字符串（注意大小写敏感）
   - 硬件不满足要求：日志报错 `MXFP8 is only supported on Ascend950 or higher architecture`

## CLI 启动方式

实验目录 `torchtitan_npu/experiments/ao_npu/benchmarks/e2e/dsv4_flash_single_node_train/` 提供 `run_train.sh` 作为低精度训练的启动入口。通过**环境变量**控制量化 recipe，无需修改 Python 代码。该脚本使用 `RecipeQuantizeConverter`，支持多种量化方案一键切换。

```bash
# 进入实验目录
cd torchtitan_npu/experiments/ao_npu/benchmarks/e2e/dsv4_flash_single_node_train/

# 默认启动（mix recipe + MXFP4 QAT 开启）
bash run_train.sh

# 切换量化 recipe
RECIPE=all_mxfp8 bash run_train.sh
RECIPE=all_block_fp8 bash run_train.sh

# 关闭 MXFP4 QAT（routed expert 仅用 BlockFP8，不加 FP4 fake-quant）
RECIPE=mix ENABLE_MXFP4_QAT=false bash run_train.sh

# 完全关闭量化，跑 BF16 基线用于对比
ENABLE_QUANTIZED_TRAINING=false bash run_train.sh
```

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `RECIPE` | `mix` | 量化 recipe：`all_mxfp8`、`mix`、`all_block_fp8` |
| `ENABLE_QUANTIZED_TRAINING` | `true` | 设为 `false` 跳过所有量化 converter，等价于 BF16 训练 |
| `ENABLE_MXFP4_QAT` | `true` | 设为 `false` 关闭 routed expert 的 MXFP4 fake-quantize |
| `DST_TYPE_MAX` | `0.0` | MXFP4 QAT 权重 fake-quantize 的目标 dtype max（0.0 = 自动推断） |

**其他训练参数**通过 tyro CLI flag 覆盖，例如：

```bash
RECIPE=mix ENABLE_MXFP4_QAT=false bash run_train.sh \
  --training.steps 1000 \
  --training.global_batch_size 128 \
  --optimizer.lr 1e-4 \
  --checkpoint.enable --checkpoint.initial_load_path /path/to/ckpt
```

**日志输出**：启动时脚本会打印所有环境变量和解析后的最终值，便于确认量化配置是否按预期生效。

```
==== Benchmark env vars ====
RECIPE                 = mix
Enable Quantized Train = true
Enable MXFP4 QAT       = true
...
============================
```
