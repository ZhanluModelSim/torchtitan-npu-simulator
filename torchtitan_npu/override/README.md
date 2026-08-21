# Override 扩展

`torchtitan_npu.override` 基于 TorchTitan 的配置级 override 机制替换
`Configurable.Config` 节点，用于接入 NPU 兼容实现、AscendC 融合算子和模型数值参考。
它不是算子级 override API；PyTorch backend 缺口及必须随包导入生效的临时适配放在
`torchtitan_npu.patches`。

DeepSeek-V4 的训练示例 wrapper（`examples/deepseek_v4/*.sh`）会按 `USE_GOLDEN`
组装具体的 `--override.imports` 组合；`scripts/run_train.sh` 只是通用单节点 launcher，
负责参数透传，不提供 `TEST_OVERRIDES` 或 `GOLDEN_OVERRIDES`。

## 启用方式

`override.imports` 中的每个条目必须是完整的 `module.function` 路径。一个条目只会启用
对应的工厂函数，不会启用同一模块中的其他 override。

```bash
python -m torchtitan.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_debugmodel \
  --override.imports \
    torchtitan_npu.override.common.rope.workaround \
    torchtitan_npu.override.deepseek_v4.sparse_attn.golden
```

多个无参数条目可以用空格或逗号分隔。工厂函数需要关键字参数时，使用
`target=<JSON object>`，并将整个条目作为一个 shell 参数：

```bash
--override.imports \
  'torchtitan_npu.override.deepseek_v4.sparse_attn.asc={"indexer_loss_coeff": 2.0}'
```

也可以直接设置配置：

```python
cfg.override.imports = [
    "torchtitan_npu.override.common.rope.workaround",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.golden",
]
```

## 应用过程

TorchTitan 在模型配置执行 `update_from_config()` 后、任何组件执行 `build()` 前应用
override：

1. 导入条目对应的模块，触发 `@override` 注册。
2. 按 `module.function` 解析本次启用的工厂函数。
3. 遍历原始 `Trainer.Config` 树，按 `target`、`exact` 和 `fqns` 收集匹配节点。
4. 在修改配置前检查同节点和祖先、后代节点之间的冲突。
5. 调用工厂函数生成 replacement config，再由后续 `build()` 构造组件。

所有匹配都基于修改前的配置树收集。replacement 不会被再次遍历，因此条目顺序不会改变
匹配结果。成功替换后，日志会记录工厂函数、配置节点 FQN 及替换前后的配置类型。

## 目录规则

```text
torchtitan_npu/override/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── optimizer.py
│   ├── profiler.py
│   ├── rms_norm.py
│   ├── rope.py
│   └── token_dispatcher.py
├── deepseek_v3_2/
│   ├── __init__.py
│   └── sparse_attn/
│       ├── __init__.py
│       └── ascendc.py
└── deepseek_v4/
    ├── __init__.py
    ├── mhc/
    │   ├── __init__.py
    │   ├── ascendc.py
    │   └── triton.py
    └── sparse_attn/
        ├── __init__.py
        ├── ascendc.py
        ├── golden.py
        └── pypto.py
```

- `common/` 存放只依赖 TorchTitan 公共组件、不依赖具体模型配置或元数据契约的实现。
- `<model>/` 存放依赖模型专属 target、配置字段、张量布局或元数据契约的实现。
- 模型专属实现不得跨模型目录引用。可复用部分应先下移到 `common/` 或其他公共模块。
- 简单 target 使用单文件，文件名采用 target 的 snake_case 语义，例如
  `RMSNorm -> rms_norm.py`。
- 同一 target 同时包含较大的多后端实现时使用 package。package 名仍表示 target；
  `__init__.py` 只定义稳定的注册入口，具体实现按 `ascendc.py`、`golden.py`、
  `triton.py` 等拆分。
- 各层 `__init__.py` 不批量导入无关注册模块，避免仅导入上层 package 就注册无关
  target。

## 命名规则

稳定入口格式为：

```text
torchtitan_npu.override.<scope>.<target>.<variant>
```

其中 `scope` 为 `common` 或模型名，`target` 表示被替换对象，`variant` 表示实现
或行为：

| Variant | 含义 |
| --- | --- |
| `asc` | 调用 CANN 或 `torch_npu` 的融合计算能力 |
| `cann` | 调用 CANN 或 `torch_npu` API 的非融合能力，如 `profiler.cann` |
| `npu` | NPU runtime 级能力；仅在不能用具体 CANN、Torch 或行为名称表达时使用 |
| `golden` | 模型专属的 eager 数值参考 |
| `torch` | 完全由标准 PyTorch 算子组成的独立实现 |
| `triton` | Triton kernel 实现 |
| `workaround` | 保持原计算语义、仅绕过当前后端兼容问题 |
| 行为名称 | 与计算后端无关的能力，例如 `optimizer.virtual` |

同一 target family 中存在多个同类实现时，在 variant 后增加对象或职责限定，例如
`rope.asc_complex`、`rope.asc_cossin` 和 `sparse_attn.asc_metadata`。不要添加
`_override` 后缀，也不要使用无法说明实现边界的 `ascend` 等泛化名称。

Replacement 类采用「variant + target」命名，并保留标准缩写的大小写：

| 类型 | 示例 |
| --- | --- |
| AscendC 实现 | `AscRMSNorm`、`AscComplexRoPE` |
| Golden 实现 | `GoldenCompressedSparseInnerAttention` |
| Workaround 实现 | `WorkaroundComplexRoPE` |
| 行为实现 | `VirtualOptimizersContainer` |
| 后端协议或元数据 | `AscCompressedVarlenMetadata`、`AscBlockLayoutMetadata` |

不作为公开 override 入口的内部函数和类使用前导下划线。

## 编写 override

最小实现由 replacement 组件和配置变换工厂组成：

```python
from dataclasses import dataclass

import torch
import torch_npu
from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm


class AscRMSNorm(RMSNorm):
    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]


@override(
    target=RMSNorm.Config,
    description="AscendC fused RMSNorm via torch_npu.npu_rms_norm",
)
def asc(cfg: RMSNorm.Config) -> AscRMSNorm.Config:
    return derive(cfg, AscRMSNorm.Config)
```

实现需满足以下约定：

- `target` 必须是 `Configurable.Config` 子类，并优先选择最小且稳定的组件边界。
- replacement 通常继承 target config，并使用 `derive()` 保留共有字段。只有明确改变配置
  契约时才直接构造新配置。
- 默认匹配 target 及其子类；replacement 仅支持具体类型时使用 `exact=True`。
- `fqns` 使用 glob 限定具体配置节点。当前仓库中的入口暂未使用 `fqns`。
- 不同 override 不能同时声明同一节点或互为祖先、后代的节点。
- replacement 必须自行保持输入输出、DTensor、sharding、checkpoint 和
  `torch.compile` 语义。
- 自定义内核应通过 `torch.library` 注册 schema、fake/meta 和 Autograd，再由
  replacement module 调用。

Float8、LoRA 等 converter 在 override 前执行。两者可能修改同一节点时，需要根据
converter 处理后的实际配置类型和 FQN 核对匹配结果。

## 当前入口

### Common

以下入口省略 `torchtitan_npu.override.common.` 前缀：

| 入口 | Target | Replacement | 说明 |
| --- | --- | --- | --- |
| `optimizer.virtual` | `OptimizersContainer.Config` | `VirtualOptimizersContainer.Config` | 将 Adam/AdamW 的 `exp_avg` 和 `exp_avg_sq` 放入 NPU swap memory |
| `optimizer.checkpoint_virtual` | `CheckpointManager.Config` | `VirtualCheckpointManager.Config` | 为 Virtual Optimizer 的同步 native DCP 保存关闭 writer copy-ahead |
| `profiler.cann` | `Profiler.Config` | `CANNProfiler.Config` | 使用 `torch_npu.profiler` 采集 CPU/NPU trace |
| `rms_norm.asc` | `RMSNorm.Config` | `AscRMSNorm.Config` | 使用 `torch_npu.npu_rms_norm` |
| `rope.workaround` | `ComplexRoPE.Config` | `WorkaroundComplexRoPE.Config` | 预展开 cos/sin cache，并使用 PyTorch 小算子计算 interleaved RoPE；仅精确匹配 `ComplexRoPE.Config` |
| `rope.asc_complex` | `ComplexRoPE.Config` | `AscComplexRoPE.Config` | 使用 interleave 模式的 `torch_npu.npu_rotary_mul`；仅精确匹配 |
| `rope.asc_cossin` | `CosSinRoPE.Config` | `AscCosSinRoPE.Config` | 使用 half 模式的 `torch_npu.npu_rotary_mul` |
| `token_dispatcher.npu_all_to_all_token_dispatcher` | `AllToAllTokenDispatcher.Config` | `NPUAllToAllTokenDispatcher.Config` | 使用 `torch_npu.npu_moe_token_permute` `npu_moe_token_unpermute` 融合 MoE dispatch/combine |


`rope.workaround` 与 `rope.asc_complex` 会声明同一 target，不能同时启用。
`AscComplexRoPE` 和 `AscCosSinRoPE` 当前都要求同一 batch 内各行的位置布局一致，
并使用第一行位置构造 batch 共享的 cosine/sine 表。


### Virtual Optimizer 与 checkpoint

Virtual Optimizer 使用 NPU swap memory 保存 Adam/AdamW 的 `exp_avg` 和
`exp_avg_sq`。通过同步 native DCP 保存和恢复完整训练状态时，应同时启用：

```bash
--override.imports \
  torchtitan_npu.override.common.optimizer.virtual,\
  torchtitan_npu.override.common.optimizer.checkpoint_virtual
```

`optimizer.virtual` 负责创建 swap-backed optimizer state，`optimizer.checkpoint_virtual`
负责让同步 native DCP 正确保存这些 live swap tensors。两个入口位于同一模块，但
作用于不同的配置节点；checkpoint 逻辑是 Virtual Optimizer 的保存兼容处理，不是独立
checkpoint 特性。完整的数据流、支持范围和限制见
[Virtual Optimizer 特性说明](../../docs/feature_guides/virtual_optimizer.md)。

### DeepSeek-V3.2

以下入口省略 `torchtitan_npu.override.deepseek_v3_2.` 前缀：

| 入口 | Target | Replacement |
| --- | --- | --- |
| `sparse_attn.asc_metadata` | `MetadataExtension.Config` | `AscVarlenMetadataExtension.Config` |
| `sparse_attn.asc` | `SparseInnerAttention.Config` | `AscSparseInnerAttention.Config` |

TND 稀疏注意力需要同时启用 metadata 扩展和注意力内核：

```text
torchtitan_npu.override.deepseek_v3_2.sparse_attn.asc_metadata
torchtitan_npu.override.deepseek_v3_2.sparse_attn.asc
```

`sparse_attn.asc` 会同时把嵌套的 `SparseIndexerLoss.Config` 派生为
`AscSparseIndexerLoss.Config`。

### DeepSeek-V4

以下入口省略 `torchtitan_npu.override.deepseek_v4.` 前缀：

| 入口 | Target | Replacement |
| --- | --- | --- |
| `sparse_attn.asc_metadata` | `MetadataExtension.Config` | `AscMetadataExtension.Config` |
| `sparse_attn.asc` | `CompressedSparseInnerAttention.Config` | `AscCompressedSparseInnerAttention.Config` |
| `sparse_attn.pypto` (replaces `sparse_attn.asc`) | `CompressedSparseInnerAttention.Config` | `PyPTOCompressedSparseInnerAttention.Config` |
| `sparse_attn.golden` | `CompressedSparseInnerAttention.Config` | `GoldenCompressedSparseInnerAttention.Config` |
| `mhc.asc_hc_pre` | `HcPre.Config` | `AscHcPre.Config` | 使用 `torch_npu.npu_mhc_pre` + `torch_npu.npu_mhc_sinkhorn` |
| `mhc.asc_hc_post` | `HcPost.Config` | `AscHcPost.Config` | 使用 `cann_ops_transformer.ops.mhc_post` |
| `mhc.triton_hc_pre` | `HcPre.Config` | `TritonHcPre.Config` | 使用 `mhc_pre_sinkhorn_op` + `mhc_pre_bmm_op` |
| `mhc.triton_hc_post` | `HcPost.Config` | `TritonHcPost.Config` | 使用 `mhc_post_bmm1_op` + `mhc_post_bmm2_op` |
| `mhc.triton_hc_head` | `HcHead.Config` | `TritonHcHead.Config` | 使用 `mhc_pre_only_sinkhorn_op` + `mhc_pre_bmm_op` |

`sparse_attn.asc_metadata` 无需参数。`sparse_attn.asc` 还支持可选的 `indexer_loss_coeff`，默认值为 `0.0`。
MHC 的 `asc_hc_pre` / `asc_hc_post` 与 `triton_hc_pre` / `triton_hc_post` / `triton_hc_head` 是可选入口（`deepseek_v4/__init__.py`
默认只导入 `sparse_attn`），需要时显式加入 `override.imports`。
推荐直接使用 `examples/deepseek_v4/*.sh` wrapper；单机调试可使用
[deepseek_v4_mini_1p_cpt_2k_a3.sh](../../examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh)，
它会组装 `--override.imports` 并调用 `scripts/run_train.sh`：

```bash
bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh
USE_GOLDEN=1 bash examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh
```

默认路径启用以下组合：

```text
torchtitan_npu.override.common.rms_norm.asc
torchtitan_npu.override.common.rope.asc_complex
torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata
torchtitan_npu.override.deepseek_v4.sparse_attn.asc
```

`USE_GOLDEN=1` 启用以下数值参考组合：

```text
torchtitan_npu.override.common.rope.workaround
torchtitan_npu.override.deepseek_v4.sparse_attn.golden
```

Golden recipe 不替换 RMSNorm 和 MoE：RMSNorm 使用模型配置中的 Torch 实现，MoE 使用
当前模型路径中经过 package patch 修正的 BF16 实现。仓库不再提供独立的 GoldenRMSNorm
或 GoldenMoE 入口。`sparse_attn.golden` 是逐文档 eager 数值参考，其 indexer score 和
gather-matmul 使用 FP32 计算，并用于与 `dsv4-infer-npu` 基线及 AscendC 融合路径比较。

`sparse_attn.asc` 必须与 `sparse_attn.asc_metadata` 配套使用；
`sparse_attn.golden` 使用模型默认的 metadata 构建（无扩展），不能再启用
`sparse_attn.asc_metadata`。两种注意力实现声明同一 target，也不能同时启用。

DeepSeek-V4 当前采用单行 packed container，要求 `local_batch_size == 1`；增加每步
token 数时应调整 `seq_len`。Golden reference 使用包含 reference tier 的
`CompressedVarlenMetadata`，当前仅支持无 context parallel 的连续文档布局。AscendC
路径使用独立的精简 `AscCompressedVarlenMetadata`，只携带 kernel contract 和预计算
的 AscendC metadata，不构造 Golden 路径使用的稠密 mask、文档位置和静态块列表。

TND 数据约定见 [DeepSeek-V4 TND 适配](../../docs/feature_guides/deepseek_v4_tnd.md)。

若使用 PyPTO 的 LI/LIG 算子，`torchtitan_npu.override.deepseek_v4.sparse_attn.pypto` 替换
`torchtitan_npu.override.deepseek_v4.sparse_attn.asc`，未覆盖的 sparse-attention
能力继续复用已验证的实现。使用该入口需要额外安装与当前 CANN 匹配的 `pypto`
runtime，安装方法参见 [PyPTO 安装文档](https://gitcode.com/cann/pypto/blob/master/docs/zh/install/build_and_install.md)，
安装后可用 `python3 -c "import pypto"` 检查。

## Override 与 package patch

| 维度 | 配置级 override | Package patch |
| --- | --- | --- |
| 激活方式 | 写入 `override.imports` | 导入 `torchtitan_npu` |
| 目标 | `Configurable.Config` 节点 | PyTorch backend 或上游 Python 符号 |
| 生效时机 | 配置构造后、组件构建前 | 包导入时 |
| 冲突检查 | 检查同节点及嵌套节点 | 不经过 override registry |

导入任意 `torchtitan_npu.override.*` 子模块时，Python 会先执行
`torchtitan_npu.__init__`，因此 package patch 也会生效。当前 patch 包含
`torch_npu` 算子适配、pinned TorchTitan 的功能回补及少量后端 workaround；详情见
[patches/torchtitan/README.md](../patches/torchtitan/README.md)。导入过程要求运行环境已经
安装匹配版本的 PyTorch、`torch_npu` 和 CANN 依赖。

新增组件替换时优先使用 override。只有配置树无法表达的 backend 缺口，或随上游合入后可
整体删除的临时适配，才放入 `patches/`。

## 常见问题

| 现象 | 检查项 |
| --- | --- |
| 模块导入失败 | 完整入口路径、Python 依赖及 NPU/CANN 环境 |
| target 未注册 | `override.imports` 是否使用准确的 `module.function` |
| 没有匹配节点 | `target`、`exact`、`fqns` 及 converter 后的配置类型 |
| 同节点或嵌套冲突 | 移除互斥入口，或通过 `fqns` 缩小范围 |
| 训练继续但替换未生效 | 检查 `[Override]` 日志和 `Applied N override(s)` |

TorchTitan override 机制、per-entry kwargs、checkpoint 和并行相关的完整说明见上游
`torchtitan/overrides/README.md`。
