# NPU 融合算子适配

torchtitan-npu 在 `torchtitan_npu/converters/kernels` 下定义了多个 torchtitan ModelConverter 。在启动模型训练任务时，它们会根据用户配置，自动将模型中的原始模块替换为基于 NPU 融合算子的实现，从而实现模型在 NPU 平台上的亲和适配。

## 如何配置
NPU融合算子通过 `ModelConvertersContainer.Config` 的 `converters` 列表启用。
每个 NPU converter 需要通过 `get_model_converter_config()` 转成 Config 对象：

```python
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.converters import get_model_converter_config

model_converters = ModelConvertersContainer.Config(
    converters=[
        get_model_converter_config("npu_dsa"),
        get_model_converter_config("npu_rms_norm"),
        get_model_converter_config("npu_moe_dispatch"),
        get_model_converter_config("npu_gmm"),
    ],
)
```

当前支持以下 ModelConverters ，前往对应章节查看功能介绍及启用方式：
- [NPU 融合算子适配](#npu-融合算子适配)
  - [如何配置](#如何配置)
  - [DSA (DeepSeek Sparse Attention)](#dsa-deepseek-sparse-attention)
  - [SMLA (Sparse Flash MLA)](#smla-sparse-flash-mla)
  - [MHCPre](#mhcpre)
  - [MHCPost](#mhcpost)
  - [GMM（Grouped MatMul）](#gmmgrouped-matmul)
  - [NPU MoE Dispatch](#npu-moe-dispatch)
  - [RMSNorm](#rmsnorm)
  - [RoPE](#rope)

关于本仓库适配的各融合算子的详细说明，请查看对应的 NPU 融合算子开发者文档。

-----------

## DSA (DeepSeek Sparse Attention)

<p align="center">
<img src="../assets/DSA_overview.png" style="width:80%; max-width: 1200px" >
</p>


DSA 是 DeepSeek-V3.2 中引入的一种特殊注意力机制，主要由图中的两个模块构成：**Lightning Indexer** 筛选出少量高价值token的索引；这些索引被用于高效的 **稀疏 Attention 计算** (图中 Multi-Query Attention部分)。

针对 DeepSeek-V3.2 模型的 Attention 模块，将以上两种核心组件替换为对应的 NPU 融合算子。具体对应关系如下：

| DeepSeek V3.2 Attention 组件              | NPU融合算子                                 |
| ----------------------------------------- | ------------------------------------------- |
| Lightning Indexer 前向计算                | `npu_lightning_indexer`                     |
| Lightning Indexer 反向计算（梯度 + Loss） | `npu_sparse_lightning_indexer_grad_kl_loss` |
| 稀疏注意力计算                            | `npu_sparse_flash_attention`                |

**配置示例：**
```python
get_model_converter_config("npu_dsa")
```
**ModelConverter 源码路径：** `torchtitan_npu/converters/kernels/dsa.py` \
**相关 NPU 融合算子开发者文档：** [`npu_lightning_indexer`](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_lightning_indexer.md) [`npu_sparse_lightning_indexer_grad_kl_loss`](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_sparse_lightning_indexer_grad_kl_loss.md)   [`npu_sparse_flash_attention`](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_sparse_flash_attention.md)

-----------

## SMLA (Sparse Flash MLA)

`npu_smla` 面向 DeepSeek-V4 的稀疏 MLA 注意力路径。该 ModelConverter 会根据硬件类型选择实现：
在 A5 上通过 [`cann_ops_transformer`](https://gitcode.com/cann/ops-transformer)
提供的 `sparse_flash_mla`、`lightning_indexer` 及对应 metadata/反向算子执行；
在非 A5 场景下继续使用兼容原有 DeepSeek-V4 sparse attention / LI / LI loss 的 NPU 实现。
A5 路径要求 `cann_ops_transformer` 能够被 Python 导入，缺失时 converter 会直接报错并指出所缺依赖。

**配置示例：**
```python
get_model_converter_config("npu_smla")
```

DeepSeek-V4 使用 `npu_smla` 时，由于LI loss的计算和对应的反向梯度计算整合在了npu_sparse_lightning_indexer_klloss_grad算子中，因此并不会显式计算LI Loss，导致LI loss 的日志记录行为与 DeepSeek-V3.2 默认路径不同。默认日志级别下，SMLA 路径不会额外计算和打印 LI loss，也不会通过 `DSAIndexerLossLoggingHelper.save_loss_to_tracker`记录该值。这样可以避免在 SMLA 融合反向算子之外，为了日志再额外计算一次 loss 带来的性能开销。

如需在调试阶段查看 LI loss，可将 `torchtitan_npu.converters.kernels.npu_smla` logger 打开到 `DEBUG` 级别，例如在调试入口中加入：

```python
import logging

logging.getLogger("torchtitan_npu.converters.kernels.npu_smla").setLevel(
    logging.DEBUG
)
```

开启后，SMLA 反向路径会额外计算 LI loss，并将其写入 `DSAIndexerLossLoggingHelper` tracker，后续可继续按现有 indexer loss 日志链路汇总打印。
该模式仅建议用于问题定位或与 DeepSeek-V3.2 行为对齐验证；正式性能测试和生产训练建议保持默认关闭，否则会引入额外计算、同步和日志记录开销。

**ModelConverter 源码路径：** `torchtitan_npu/converters/kernels/npu_smla.py`

-----------

## MHCPre

`npu_mhc_pre` 面向 DeepSeek-V4 的 MHC pre-processing 路径，将模型中的 `HcPre` 模块替换为 NPU 亲和实现。在 A5 上使用 `cann_ops_transformer.ops.mhc_pre_sinkhorn` 及其反向融合算子；在非 A5 场景下使用已有 Triton 实现。DeepSeek-V4 默认配置中已启用该 converter。

**配置示例**：
```python
get_model_converter_config("npu_mhc_pre")
```

**ModelConverter 源码路径：** `torchtitan_npu/converters/kernels/mhc_prepost.py`

-----------

## MHCPost

`npu_mhc_post` 面向 DeepSeek-V4 的 MHC post-processing 路径，将模型中的 `HcPost` 模块替换为 NPU 亲和实现。在 A5 上使用 `cann_ops_transformer.ops.mhc_post` 及其反向融合算子；在非 A5 场景下会走已有 Triton 实现并替换 `HcHead`。

该 converter 当前未在 DeepSeek-V4 默认 `config_registry.py` 中打开，主要是出于性能考虑：A5 上建议按需打开以使用 MHC post 融合算子；A3 上建议保持关闭，继续使用默认模型路径，避免 converter 替换带来的额外开销或性能退化。

**配置示例**：
```python
get_model_converter_config("npu_mhc_post")
```

**ModelConverter 源码路径：** `torchtitan_npu/converters/kernels/mhc_prepost.py`

-----------

## GMM（Grouped MatMul）

在 MoE 模块中，每个专家执行前馈网络（FFN）运算：如 Swiglu FFN：输入先经过升维变换 `w1`，再通过激活函数，最后经过降维变换 `w2` 得到输出。

由于各专家执行结构相同的矩阵乘法，为了将同类矩阵运算合并为一次算子调用，提升计算效率，本ModelConverter 引入分组矩阵乘法（GMM）算子 `npu_grouped_matmul`。该算子接收 [NPU MoE Dispatch](#npu-moe-dispatch) 模块输出的重排后 token 及对应的专家索引，在一次调用中并行计算所有专家的同一线性层，如所有专家的 `w1`。

> 注：`npu_gmm` 依赖 MoE token 重排能力。DSV3 / DSV32 / DSV4 / Qwen3 MoE 标准 `ExpertParallel` 场景统一使用 `npu_moe_dispatch`。

**配置示例**：
```python
get_model_converter_config("npu_gmm")
```
**ModelConverter 源码路径：** `torchtitan_npu/converters/kernels/gmm.py` \
**相关 NPU 融合算子开发者文档：** [`npu_grouped_matmul`](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_grouped_matmul.md)

-----------


## NPU MoE Dispatch

`npu_moe_dispatch` 面向 DS 系列和 Qwen3 MoE 的标准 `ExpertParallel` 路径。该 ModelConverter 负责接入 NPU MoE dispatch 流程：router 后仍使用 `npu_moe_token_permute` 做第一组 token/expert 聚合，expert 计算后仍使用 unpermute 还原；同时通过并行策略更新器将标准 `ExpertParallel` 替换为 `NpuExpertParallel`。

在 EP all-to-all 之后，标准 dispatch 流程需要把 rank-major token layout 重新整理为 local expert-major layout。这里的 `torch_npu.npu_moe_re_routing` 只优化这一段 local reroute：它替换旧路径中的 `repeat_interleave + npu_moe_token_permute`，减少局部重排热点上的冗余算子。

需要注意，`npu_moe_re_routing` 不替换 MoE router 后的 `npu_moe_token_permute`。router 后的第一组 token permute 负责根据 `selected_experts_indices [T, K]` 生成 top-k expert slots；EP all-to-all 后的本卡重排负责根据 `counts [ep_degree, num_local_experts]` 调整本卡收到的 local buffer，两者输入语义和布局目标不同。模型配置应直接使用 `npu_moe_dispatch`。

```python
get_model_converter_config("npu_moe_dispatch")
```

**ModelConverter 源码路径：** `torchtitan_npu/converters/kernels/moe_dispatch.py`、`torchtitan_npu/converters/kernels/permutation.py`

## RMSNorm
RMSNorm 通过计算输入张量每个样本的平方均值的平方根来稳定深层网络的训练。

本 ModelConverter 将模型中的 RMSNorm 操作替换为基于 `npu_rms_norm` 融合算子的实现。

**配置项**：`npu_rms_norm`\
**配置示例**：
```python
get_model_converter_config("npu_rms_norm")
```
**ModelConverter 源码路径：** `torchtitan_npu/converters/kernels/rms_norm.py` \
**相关 NPU 融合算子开发者文档：** [`npu_rms_norm`](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/（beta）torch_npu-npu_rms_norm.md)

-----------

## RoPE
RoPE 将 token 位置相关的旋转矩阵应用于自注意力机制中的 Query 和 Key 向量，使每对 token 之间的相对位置信息在 Attention 计算中自然包含 Query 和 Key 的乘积。

在模型实现中，通常预先生成每个位置的旋转角度，在 Attention 计算时，即时对 Query 和 Key 进行旋转变换。本 ModelConverter 将这一旋转变换操作替换为基于 `npu_rotary_mul` 融合算子的实现。

**配置示例**：
```python
get_model_converter_config("npu_rope")
```
**ModelConverter 源码路径：** `torchtitan_npu/converters/kernels/rope.py` \
**相关 NPU 融合算子开发者文档：** [`npu_rope`](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_rotary_mul.md)
