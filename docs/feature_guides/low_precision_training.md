# 低精度训练特性（MXFP8 / HiF8）

在大规模语言模型的分布式训练中，矩阵乘法运算（GEMM）占据了绝大部分计算开销。传统的 BF16/FP16 混合精度训练虽然已大幅降低了显存占用，但在超大规模模型（如 DeepSeek-V3 671B）上仍面临计算效率瓶颈。低精度训练通过将线性层和 MoE 专家层的矩阵乘法降至 8-bit 浮点精度执行，在保持训练收敛性的前提下，显著提升计算吞吐并降低显存消耗。

本特性基于 [torchao](https://github.com/pytorch/ao) 的 MXFP8 训练框架，通过 NPU 侧的 monkey-patch 将 torchao 的 MXFP8 计算路径重定向至 `torch_npu` 原生算子，覆盖普通线性层（nn.Linear）和 MoE 专家层（Grouped MM）两大场景。

除 MXFP8 外，torchtitan-npu 还通过实验性插件子包 `torchtitan_npu/experiments/ao_npu/torchao_npu`  提供 HiF8（`torch_npu.hifloat8`）作为另一种 8-bit 精度训练路径，同样覆盖上述两大场景。与 MXFP8 的 per-block（e8m0 microscaling）量化不同，HiF8 是**纯 per-tensor** 动态量化：整个激活/权重张量只计算一个标量 scale。

## 硬件要求

低精度训练特性仅支持 **Ascend 950 及更高架构**的 NPU 设备。MXFP8 在初始化时会通过 `torch_npu.npu.get_device_name()` 进行硬件检测，不满足要求时将抛出异常；HiF8 当前未在 Python 层做显式硬件能力校验，不支持的硬件上会在实际调用 `torch_npu.npu_dynamic_quant`/`npu_quant_matmul` 等算子时由算子层报错。

## 实现原理

### 整体架构

本特性采用 **torchao 原生 MXFP8 框架 + NPU 算子替换** 的架构。torchtitan 上游提供 `MXFP8Converter` 作为模型转换入口，torchao 负责量化配置与权重包装，torchtitan-npu 通过 monkey-patch 将 torchao 内部的矩阵乘法调度函数替换为 NPU 实现。相关代码主要分布在以下文件中：

| 文件路径 | 修改作用                                                                    |
| --- |-------------------------------------------------------------------------|
| `torchtitan_npu/experiments/ao_npu/torchao_npu/patches/mx_capability_check.py` | 替换 `has_cuda_capability` 函数，使 MXFP8Converter 在 NPU 上进行硬件校验              |
| `torchtitan_npu/experiments/ao_npu/torchao_npu/patches/mx_linear.py` | 替换 torchao 的 `_to_mxfp8_then_scaled_mm`，将线性层 MXFP8 计算重定向至 NPU 算子        |
| `torchtitan_npu/experiments/ao_npu/torchao_npu/patches/mxfp8_grouped_mm.py` | 替换 torchao 的 `_to_mxfp8_then_scaled_grouped_mm`，将 MoE 分组矩阵乘法重定向至 NPU 算子 |

### 整体架构（HiF8）

HiF8 不复用上述 `MXFP8Converter` + patch 架构，而是接入 `torchao_npu` 已有的 **ParamSwap 参数级拦截框架**：`NpuQuantizeConverter`（把 `ParamSwapConfig` 转交给 `torchao.quantize_()`）→ `ParamSwapConfig`（prepare/convert 两步生命周期，wrap/unwrap `nn.Parameter`）→ `HiF8TrainingWeightWrapperTensor`（`__torch_function__` 拦截计算类 op）。相关代码主要分布在：

| 文件路径 | 作用 |
| --- | --- |
| `torchtitan_npu/experiments/ao_npu/torchao_npu/quantization/quant_configs.py` | `HiF8QuantizeConfig`：per-tensor 量化配置，唯一字段 `elem_dtype` 锁定为 `torch_npu.hifloat8` |
| `torchtitan_npu/experiments/ao_npu/torchao_npu/wrapper_tensors/hif8_wrapper_tensor.py` | `HiF8TrainingWeightWrapperTensor`：拦截 `mm`/`matmul`/`grouped_mm`/`linear`/`addmm`，换成 HiF8 量化 kernel |
| `torchtitan_npu/experiments/ao_npu/torchao_npu/ops/hif8_ops.py` | `to_hif8_then_mm`/`to_hif8_then_grouped_mm`：per-tensor HiF8 量化的 Linear/Grouped MM 正反向实现 |
| `torchtitan_npu/experiments/ao_npu/torchao_npu/interfaces/torchtitan.py` | `NpuQuantizeConverter`：把 `ParamSwapConfig` 接入 torchtitan 的 `ModelConvertersContainer` |

### 线性层低精度

`MXFP8Converter` 在转换阶段，通过 torchao 的 `quantize_` API 对模型中指定 FQN 的 `nn.Linear` 模块的权重进行包装（`MXFP8TrainingWeightWrapperTensor`）。在前向传播时，权重包装器的 `__torch_function__` 拦截矩阵乘法调用，进入 `_to_mxfp8_then_scaled_mm` 函数。

NPU 侧通过 patch `torchao.prototype.mx_formats.mx_linear._to_mxfp8_then_scaled_mm`，将其替换为调用 NPU 原生算子的 `NpuMXFP8MM`：

- **前向传播**：使用 `torch_npu.npu_dynamic_mx_quant` 对激活和权重分别进行 per-block 量化（block size=32，沿 axis=-1 方向），每 32 个元素共享一个 e8m0 scale，再通过 `torch_npu.npu_quant_matmul` 执行 FP8 矩阵乘法，输出恢复为原始精度（BF16）。
- **反向传播**：输入梯度（dx）和权重梯度（dw）的计算同样在 FP8 精度下完成，其中权重梯度使用 `npu_dynamic_mx_quant` 对权重沿 axis=-2 方向进行 per-block 量化。

### 线性层低精度（HiF8）

`HiF8TrainingWeightWrapperTensor.__torch_function__` 拦截 `torch.nn.functional.linear`/`torch.addmm`/`torch.mm`/`torch.matmul`，转调用 `to_hif8_then_mm`：

- **前向传播**：对激活和权重分别用 `torch_npu.npu_dynamic_quant(quant_mode="pertensor")` 量化为 HiF8 + 一个标量 scale，再用 `torch_npu.npu_quant_matmul` 计算，输出恢复为原始 BF16 精度。
- **反向传播**：直接复用前向已量化的激活/权重及其 scale（转置对 per-tensor scale 无影响）计算输入梯度和权重梯度，只对新出现的梯度现场量化一次，不需要重新量化激活/权重。

### MoE 专家层低精度

对于 MoE（Mixture of Experts）架构中的专家层，torchao 通过 `_to_mxfp8_then_scaled_grouped_mm` 函数调度分组矩阵乘法。

NPU 侧通过 patch `torchao.prototype.moe_training.mxfp8_grouped_mm._to_mxfp8_then_scaled_grouped_mm`，将其替换为调用 NPU 原生算子的 `NpuMXFP8GroupedMM`：

- **前向传播**：使用 `torch_npu.npu_dynamic_mx_quant` 对输入和权重分别进行 per-block 量化（block size=32），再调用 `torch_npu.npu_grouped_matmul` 执行 FP8 分组矩阵乘法。
- **反向传播**：输入梯度使用 `npu_dynamic_mx_quant` + `npu_grouped_matmul` 计算；权重梯度使用 `torch_npu.npu_grouped_dynamic_mx_quant` 对输入和梯度分别进行 per-group 量化后，再调用 `npu_grouped_matmul` 计算。

> **注意**：MoE 低精度功能依赖 `npu_gmm` converter 提供的分组矩阵乘法基础实现，因此在 converters 配置中 `npu_gmm` 必须位于 `MXFP8Converter` 之前。

### MoE 专家层低精度（HiF8）

拦截 `torch._grouped_mm`，转调用 `to_hif8_then_grouped_mm`：

- **前向传播**：激活/权重整体各量化出一个标量 scale，再用 `torch_npu.npu_grouped_matmul` 计算；scale 通过 `.reshape(1).expand(...)` 广播成算子要求的按 token/按专家长度的向量。
- **反向传播**：输入梯度复用前向对激活的量化结果；权重梯度两个 scale 都按专家数广播（而非 token 数），且需要对转置后的权重重新做一次量化，未复用前向的量化结果。

> **注意**：MoE 场景下 HiF8 是纯 per-tensor 量化：同一次 forward/backward 里所有本地专家、所有 token 共享同一个标量 scale；同样依赖 `npu_gmm` converter 提供的分组矩阵乘法基础实现，因此在 converters 配置中 `npu_gmm` 必须位于 `NpuQuantizeConverter` 之前。

## 配置选项

低精度训练通过 `ModelConvertersContainer.Config` 的 `converters` 列表启用。MXFP8 使用上游 `MXFP8Converter.Config` 进行配置；HiF8 使用 `NpuQuantizeConverter.Config` 包一层 `ParamSwapConfig` 进行配置，不使用 `MXFP8Converter.Config`。

### MXFP8Converter 配置

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `recipe_name` | str | `"mxfp8_rceil"` | 量化 recipe 名称。当前唯一可选值：`"mxfp8_rceil"`（MXFP8 动态量化，scale 计算采用 RCEIL 舍入模式）。 |
| `fqns` | list[str] | [] | 需要启用 MXFP8 量化的模块全限定名（FQN）列表。匹配规则为子字符串包含，例如 `"moe.experts"` 将匹配所有 FQN 中包含该字符串的模块。留空表示不对任何模块启用 MXFP8。 |

### HiF8 配置

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `HiF8QuantizeConfig.elem_dtype` | `torch.dtype` | `torch_npu.hifloat8` | 唯一字段，`__post_init__` 强制锁定为该值，实际不可配置，仅为与 `MXQuantizeConfig` 等其他 config 接口保持一致 |
| `NpuQuantizeConverter.Config.filter_fn` | `(nn.Module, str) -> bool` | `_is_linear` | 模块级过滤器，决定哪些模块参与转换 |
| `ParamSwapConfig.params_filter_fn` | `(nn.Parameter, str) -> bool` | `_is_parameter` | 参数级过滤器，决定命中模块下的哪些参数被 wrap |

`ParamSwapConfig` 的 `weight_config`/`activation_config` 必须都传且均为 `HiF8QuantizeConfig`，否则 `HiF8TrainingWeightWrapperTensor.__init__` 会抛出 `ValueError`。

### 环境变量配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MXFP8_DUAL_AXIS_FORWARD` | `1`（启用） | 控制 MXFP8 线性层和 MoE 专家层是否启用 forward dual-axis 量化（forward 阶段同时生成 backward 的量化数据），默认启用。设置为 `0` 或 `false` 可关闭，forward 回退到 single-axis 量化行为。无论 `MXFP8_DUAL_AXIS_FORWARD` 是否启用，Linear的 backward grad始终使用 dual-axis 量化。 |

> HiF8 没有 dual-axis 量化机制（不区分 block/axis），不受 `MXFP8_DUAL_AXIS_FORWARD` 影响，也没有对应的环境变量。

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

**示例：对指定线性层和 MoE 专家层启用 HiF8 低精度训练**

```python
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.converters import get_model_converter_config
from torchtitan_npu.experiments.ao_npu.torchao_npu.configs import ParamSwapConfig
from torchtitan_npu.experiments.ao_npu.torchao_npu.interfaces.torchtitan import (
    NpuQuantizeConverter,
    is_attention,
    is_routed_expert,
    is_shared_expert,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.filters import (
    any_filter,
    match_fqn_suffix,
)
from torchtitan_npu.experiments.ao_npu.torchao_npu.quantization.quant_configs import HiF8QuantizeConfig

model_converters = ModelConvertersContainer.Config(
    converters=[
        # NPU 基础 converter（npu_gmm 必须在 NpuQuantizeConverter 之前）
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
        get_model_converter_config("npu_rope"),
        get_model_converter_config("npu_smla"),
        get_model_converter_config("npu_mhc_pre"),
        # HiF8 低精度训练
        NpuQuantizeConverter.Config(
            base_config=ParamSwapConfig(
                weight_config=HiF8QuantizeConfig(),
                activation_config=HiF8QuantizeConfig(),
            ),
            filter_fn=any_filter(
                # Attention 线性层 + 共享专家 + 路由专家
                is_attention,
                is_shared_expert,
                is_routed_expert,
                match_fqn_suffix(".e_proj", ".h_proj"),
            ),
        ),
    ],
)
```

参照 `torchtitan_npu/experiments/ao_npu/benchmarks/e2e/dsv4_flash_single_node/config_registry.py` 中的 `debug_deepseek_v4_flash_single_node_hif8_qat()`。

## 验证清单

1. **确认 converter 生效**：启动日志中应出现以下关键字：
   - `MXFP8 MoE training enabled`（来自上游 `MXFP8Converter.__init__`）
   - `Converted layers matching FQNS ... to use dynamic mxfp8_rceil quantization for grouped_mm and linear ops`（来自上游 `MXFP8Converter.convert`）
   - HiF8：`Parameter quantize active with base_config=ParamSwapConfig`（来自 `NpuQuantizeConverter.__init__`）及 `Applied parameter quantize wrapping (prepare step)`（来自 `NpuQuantizeConverter.convert`）
2. **确认模块替换数量**：MXFP8 日志中的转换信息应与配置的 `fqns` 列表匹配；HiF8 可在 `filter_fn`/`params_filter_fn` 命中范围内抽查若干参数，确认其 `.data` 类型为 `HiF8TrainingWeightWrapperTensor` 而非普通 tensor。
3. **常见未生效场景排查**：
   - `converters` 顺序错误：`npu_gmm` 未放在 `MXFP8Converter`/`NpuQuantizeConverter` 之前，导致 MoE 专家层替换失败
   - `fqns`（MXFP8）/`filter_fn`（HiF8）匹配不到目标模块：检查模块的 FQN 是否符合配置的匹配条件（注意大小写敏感）
   - 硬件不满足要求：MXFP8 日志报错 `MXFP8 is only supported on Ascend950 or higher architecture`；HiF8 未做提前校验，报错会发生在实际调用 NPU 算子时
   - HiF8 特有：`weight_config`/`activation_config` 未同时提供、或类型不一致，`HiF8TrainingWeightWrapperTensor` 构造时会抛出 `ValueError`

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
