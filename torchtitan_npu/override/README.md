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
python -m torchtitan_npu.train \
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

NPU DeepEP dispatcher 与上游 `moe_comm_backend="deepep"` 配置绑定。模型配置选择
`deepep` 后，显式启用以下 override 即可；`hidden_dim` 和
`num_max_tokens_per_rank` 会由 `update_from_config()` 根据模型及训练形状填充。

```bash
--override.imports \
  torchtitan_npu.override.common.token_dispatcher.asc_deepep
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
│   ├── rms_norm.py
│   ├── rope.py
│   ├── token_dispatcher.py
│   └── swiglu_group/
│       ├── __init__.py
│       └── ascendc.py
├── checkpoint/
│   ├── __init__.py
│   ├── checkpoint.py
│   └── validation.py
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
    │   ├── tilelang.py
    │   └── triton.py
    └── sparse_attn/
        ├── __init__.py
        ├── ascendc.py
        ├── golden.py
        └── pypto.py
```

`deepseek_v4_1/` 按 target 拆分，且不带 `__init__.py`（见
[DeepSeek-V4.1](#deepseek-v41)）：

```text
torchtitan_npu/override/deepseek_v4_1/
├── mhc.py
├── vision_language_dataloader.py
├── engram/
│   ├── __init__.py
│   ├── ascendc.py
│   └── mxfp8.py
├── lightning_indexer/
│   ├── __init__.py
│   └── ascendc.py
└── sparse_attn/
    ├── __init__.py
    └── ascendc.py
```

- `common/` 存放只依赖 TorchTitan 公共组件、不依赖具体模型配置或元数据契约的实现。
- `checkpoint/` 存放 checkpoint manager replacement 及其文件级完整性校验逻辑。
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
| `cann` | 调用 CANN 或 `torch_npu` API 的非融合能力 |
| `npu` | NPU runtime 级能力；仅在不能用具体 CANN、Torch 或行为名称表达时使用 |
| `golden` | 模型专属的 eager 数值参考 |
| `torch` | 完全由标准 PyTorch 算子组成的独立实现 |
| `triton` | Triton kernel 实现 |
| `tilelang` | TileLang 融合 kernel 实现 |
| `workaround` | 保持原计算语义、仅绕过当前后端兼容问题 |
| 行为名称 | 与计算后端无关的能力，例如 `optimizer.virtual` |

同一 target family 中存在多个同类实现时，在 variant 后增加对象或职责限定，例如
`rope.asc_complex`、`rope.asc_cossin` 和 `sparse_attn.golden`。不要添加
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

### Checkpoint 完整性校验

`torchtitan_npu.override.checkpoint.npu` 为基础 `CheckpointManager` 增加文件级 SHA-256
manifest。该入口默认不启用校验；使用以下参数后，保存会生成
`_checkpoint_hash_manifest.json`，加载会在物化 checkpoint state 前验证清单：

```bash
--override.imports \
  'torchtitan_npu.override.checkpoint.npu={"verify_hash_manifest":true}'
```

| 入口 | Target | Replacement | 适用范围 |
| --- | --- | --- | --- |
| `torchtitan_npu.override.checkpoint.npu` | `CheckpointManager.Config`（精确匹配） | `NPUCheckpointManager.Config` | 基础 checkpoint manager |
| `torchtitan_npu.override.checkpoint.npu_virtual` | `CheckpointManager.Config`（精确匹配） | `NPUVirtualCheckpointManager.Config` | 同时需要 SHA-256 manifest 和 Virtual Optimizer checkpoint writer |

Virtual Optimizer 组合场景使用单一 checkpoint replacement，不能再同时启用
`optimizer.checkpoint_virtual`：

```bash
--override.imports \
  torchtitan_npu.override.common.optimizer.virtual,\
  'torchtitan_npu.override.checkpoint.npu_virtual={"verify_hash_manifest":true}'
```

两个 checkpoint 入口均使用精确匹配，不会替换 `TorchFTCheckpointManager.Config` 等上游
特化配置；当前不为 TorchFT 提供 SHA-256 manifest variant。基础 `npu` 入口的 manifest I/O
支持本地路径和 TorchTitan 支持的 fsspec 远程 native DCP 路径；`npu_virtual` 仍受 Virtual
Optimizer writer 的同步、本地 native DCP 限制。异步保存只有在 DCP 和 manifest 均成功后
才完成；遗留 pending 标记会使加载失败。没有 manifest 且没有 pending 标记时，checkpoint
按旧格式加载并跳过校验。

清单只接受 checkpoint 目录中的单层普通文件，不读取绝对路径、父目录、符号链接或其他
非普通文件。CPU 单元测试覆盖本地路径、fsspec `memory://`、异步完成与失败传播，以及路径
边界；实际 S3/GCS backend 和 NPU 分布式训练尚未在本特性中完成验证。

### Common

以下入口省略 `torchtitan_npu.override.common.` 前缀：

| 入口 | Target | Replacement | 说明 |
| --- | --- | --- | --- |
| `optimizer.virtual` | `OptimizersContainer.Config` | `VirtualOptimizersContainer.Config` | 将 Adam/AdamW 的 `exp_avg` 和 `exp_avg_sq` 放入 NPU swap memory |
| `optimizer.checkpoint_virtual` | `CheckpointManager.Config` | `VirtualCheckpointManager.Config` | 为 Virtual Optimizer 的同步 native DCP 保存关闭 writer copy-ahead |
| `rms_norm.asc` | `RMSNorm.Config` | `AscRMSNorm.Config` | 使用 `torch_npu.npu_rms_norm` |
| `rope.workaround` | `ComplexRoPE.Config` | `WorkaroundComplexRoPE.Config` | 预展开 cos/sin cache，并使用 PyTorch 小算子计算 interleaved RoPE；仅精确匹配 `ComplexRoPE.Config` |
| `rope.asc_complex` | `ComplexRoPE.Config` | `AscComplexRoPE.Config` | 使用 interleave 模式的 `torch_npu.npu_rotary_mul`；仅精确匹配 |
| `rope.asc_cossin` | `CosSinRoPE.Config` | `AscCosSinRoPE.Config` | 使用 half 模式的 `torch_npu.npu_rotary_mul` |
| `rope.asc_partial` | `SplitComplexRoPEConfig`（精确匹配） | `AscPartialComplexRoPE.Config` | 整宽（prefix + rotary）partial-RoPE 站点的 query/key 各收敛为一次 `cann_ops_transformer` 融合原地旋转；CANN 依赖在首次调用时惰性加载 |
| `token_dispatcher.asc` | `AllToAllTokenDispatcher.Config` | `AscAllToAllTokenDispatcher.Config` | 使用 `torch_npu.npu_moe_token_permute` `npu_moe_token_unpermute` 融合 MoE dispatch/combine |
| `token_dispatcher.asc_deepep` | `DeepEPTokenDispatcher.Config` | `AscDeepEPTokenDispatcher.Config` | 使用 `cann_ops_transformer.ElasticBuffer` 实现训练路径的 MoE DeepEP dispatch/combine；当前要求 `expert_parallel_degree > 1` |
| `swiglu_group.asc` | `GroupedExperts.Config` | `AscGroupedExperts.Config` | 使用 `cann_ops_nn.swiglu_group` 完成路由专家的 SwiGLU 激活及可选路由分数乘法 |
| `swiglu_group.asc_shared_experts` | `FeedForward.Config` | `AscFeedForward.Config` | 使用 `cann_ops_nn.swiglu_group` 完成共享专家的 SwiGLU 激活 |

`rope.workaround` 与 `rope.asc_complex` 会声明同一 target，不能同时启用。
`rope.asc_partial` 精确匹配 split-aware 的 `SplitComplexRoPEConfig`，与其他 rope
入口作用于同一组件，同一站点只能启用其一；其 `cann_ops_transformer` 依赖在首次调用时
惰性加载，导入 `override.common` 或构造 RoPE 模块不需要 NPU 算子包。
`AscComplexRoPE` 和 `AscCosSinRoPE` 当前都要求同一 batch 内各行的位置布局一致，
并使用第一行位置构造 batch 共享的 cosine/sine 表。


### Common SwiGLU

入口前缀为 `torchtitan_npu.override.common.swiglu_group.`，适用于采用公共专家配置和下述 FQN 布局的模型。

`swiglu_group.asc` 和 `swiglu_group.asc_shared_experts` 分别直接匹配
`routed_experts.inner_experts` 与 `shared_experts`，不会声明整个 `MoE.Config` 已被替换，
因此可以和 MoE 中其他子模块的 override 并存。两者只替换 FFN 中间的 gate/up + SwiGLU
激活桥接，路由选择、token dispatch 以及前后分组矩阵乘仍复用现有路径；路由分数作为
`weight` 传给 `SwigluGroup`，由融合算子完成激活阶段的 score absorption。CANN 实现位于
`common/swiglu_group/ascendc.py`。

在 `TrainerEx` 中，低精配置转换先于 override 执行：选中的标准 `GroupedExperts.Config` 或
`AscGroupedExperts.Config` 会转换为 `NpuQuantizedAscGroupedExpertsModule.Config`，复用融合前向，
无需额外启用 `swiglu_group.asc`；显式启用时保留已有量化配置。
其他专家子类保留原有实现；共享专家使用融合激活仍需启用 `swiglu_group.asc_shared_experts`。

启用两个 expert override：

```bash
--override.imports \
  torchtitan_npu.override.common.swiglu_group.asc \
  torchtitan_npu.override.common.swiglu_group.asc_shared_experts
```

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
| `sparse_attn.asc` | `SparseInnerAttention.Config` | `AscSparseInnerAttention.Config` |

TND 稀疏注意力只需要 `sparse_attn.asc`：注意力内核与它消费的 TND metadata 同属一个
backend，metadata 由被替换的 `AscSparseInnerAttention.Config.build_attention_metadata`
直接构建（upstream PR #4897 的 backend-owned 接缝），不再有单独的 metadata 入口。

```text
torchtitan_npu.override.deepseek_v3_2.sparse_attn.asc
```

`sparse_attn.asc` 会同时把嵌套的 `SparseIndexerLoss.Config` 派生为
`AscSparseIndexerLoss.Config`。

### DeepSeek-V4

以下入口省略 `torchtitan_npu.override.deepseek_v4.` 前缀：

| 入口 | Target | Replacement | 说明 |
| --- | --- | --- | --- |
| `sparse_attn.asc_metadata` | `SparseAttentionMetadata.Config` | `SparseAttentionMetadata.Config` | 选择 `AscMetadataExtension` |
| `compressor.asc_metadata` | `CompressorMetadata.Config` | `CompressorMetadata.Config` | 独立预构建 compressor 状态表 |
| `sparse_attn.asc` | `CompressedSparseInnerAttention.Config` | `AscCompressedSparseInnerAttention.Config` | — |
| `sparse_attn.pypto` (replaces `sparse_attn.asc`) | `CompressedSparseInnerAttention.Config` | `PyPTOCompressedSparseInnerAttention.Config` | — |
| `sparse_attn.golden` | `CompressedSparseInnerAttention.Config` | `GoldenCompressedSparseInnerAttention.Config` | — |
| `mhc.asc_hc_pre` | `HcPre.Config` | `AscHcPre.Config` | 使用 `torch_npu.npu_mhc_pre` + `torch_npu.npu_mhc_sinkhorn` |
| `mhc.tilelang_hc_pre` | `HcPre.Config` | `TilelangHcPre.Config` | 使用外部 TileKernels Split/Apply 与 `torch_npu.npu_mhc_sinkhorn` |
| `mhc.asc_hc_post` | `HcPost.Config` | `AscHcPost.Config` | 使用 `cann_ops_transformer.ops.mhc_post` |
| `mhc.triton_hc_pre` | `HcPre.Config` | `TritonHcPre.Config` | 使用 `mhc_pre_sinkhorn_op` + `mhc_pre_bmm_op` |
| `mhc.triton_hc_post` | `HcPost.Config` | `TritonHcPost.Config` | 使用 `mhc_post_bmm1_op` + `mhc_post_bmm2_op` |
| `mhc.tilelang_hc_post` | `HcPost.Config` | `TilelangHcPost.Config` | 通过 compile-safe wrapper 延迟调用外部 TileKernels `mhc_post` |
| `mhc.triton_hc_head` | `HcHead.Config` | `TritonHcHead.Config` | 使用 `mhc_pre_only_sinkhorn_op` + `mhc_pre_bmm_op` |
| `mhc.tilelang_hc_head` | `HcHead.Config` | `TilelangHcHead.Config` | 使用 `mhc_head_compute_mix_tilelang` (TileLang 融合 kernel) |
| `moe.tilelang_swiglu_grouped` | `GroupedExperts.Config` | `TilelangGroupedExperts.Config` | 使用外部 `tile_kernels` SwiGLU forward/backward |
| `moe.tilelang_swiglu_shared` | `FeedForward.Config` | `TilelangFeedForward.Config` | 使用外部 `tile_kernels` SwiGLU forward/backward |
| `moe_router.tilelang` | `HashRouter.Config`（精确匹配） | `TilelangHashRouter.Config` | 非 hash 层使用 TileKernels `topk_gate`；hash 层保持原逻辑 |

`sparse_attn.asc_metadata` 无需参数。`sparse_attn.asc` 还支持可选的 `indexer_loss_coeff`，默认值为 `1.0`。
使用 fused compressor 时，同时启用 `compressor.asc` 和 `compressor.asc_metadata`；稀疏注意力保留 `sparse_attn.asc_metadata`。
CV metadata 入口也可与 `compressor.asc_metadata` 同时启用；它只与 `sparse_attn.asc_metadata` 互斥。
仅启用 sparse-attention metadata 时不分配 compressor 状态表；CP 保持原有交换后构表路径。
MHC 的 `asc_hc_pre` / `asc_hc_post` 与 `triton_hc_pre` / `triton_hc_post` / `triton_hc_head` / `tilelang_hc_head` 是可选入口（`deepseek_v4/__init__.py`
默认只导入 `sparse_attn`），需要时显式加入 `override.imports`。`triton_hc_head` 与 `tilelang_hc_head` 声明同一 `HcHead.Config` 节点，两者互斥，只能启用其一；二者均可与 `asc_hc_pre` / `asc_hc_post` 共存。

`tilelang_hc_post` 与 `asc_hc_post` 声明同一 `HcPost.Config` 节点，两者互斥，只能启用其一；二者均可与 `asc_hc_pre` 共存。
`tilelang_hc_pre` 与 `asc_hc_pre`、`triton_hc_pre` 作用于同一 `HcPre.Config` 节点，只能选择一个；它可以与作用于 `HcPost.Config` 的 `asc_hc_post` 同时启用。

依赖的 tilelang/tile_kernels 暂时未开源，不随默认 requirements 安装。
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
torchtitan_npu.override.common.swiglu_group.asc
torchtitan_npu.override.common.swiglu_group.asc_shared_experts
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
token 数时应调整 `seq_len`。Golden reference 与模型目录默认内核都使用公共的
`CompressedVarlenMetadata`（只含 kernel contract 与 packed 偏移），当前仅支持无
context parallel 的连续文档布局。AscendC 路径使用独立的精简
`AscCompressedVarlenMetadata`，只携带 kernel contract 和预计算的 AscendC metadata。

TND 数据约定见 [DeepSeek-V4 TND 适配](../../docs/feature_guides/deepseek_v4_tnd.md)。

若使用 PyPTO 的 LI/LIG 算子，`torchtitan_npu.override.deepseek_v4.sparse_attn.pypto` 替换
`torchtitan_npu.override.deepseek_v4.sparse_attn.asc`，未覆盖的 sparse-attention
能力继续复用已验证的实现。使用该入口需要额外安装与当前 CANN 匹配的 `pypto`
runtime，安装方法参见 [PyPTO 安装文档](https://gitcode.com/cann/pypto/blob/master/docs/zh/install/build_and_install.md)，
安装后可用 `python3 -c "import pypto"` 检查。

### DeepSeek-V4.1

以下入口省略 `torchtitan_npu.override.deepseek_v4_1.` 前缀：

| 入口 | Target | Replacement | 说明 |
| --- | --- | --- | --- |
| `sparse_attn.asc` | `CompressedSparseInnerAttention2.Config`（精确匹配） | `AscV41SparseAttention.Config` | AscendC attention 前后向；`topk_scores` 作为教师边。该 config 按**自己的 `compress_ratio`** 预计算 attention 与 indexer 的 metadata（每个几何每 microbatch 一次，全层共享同一份 entry），因此没有单独的 metadata 入口；kernel geometry 是该 config 自己的必需字段，由 override 参数一次写入，默认值为 V4.1-Flash 形状。本入口不声明任何量化：量化 selector 的 metadata 属于量化后的 attention backend，它替换本节点并自建 entry（见下文），本入口在那条路径上不生效 |
| `lightning_indexer.asc` | `Selector.Config`（精确匹配） | `AscSelector.Config` | 融合 LI 选分与 SLIKG 反向；它的两条 AI_CPU metadata 由 `sparse_attn.asc` 的几何一并预计算，随同一份 map entry 下发 |
| `mhc.asc_sinkhorn` | `HcPre.Config`（精确匹配） | `AscV41HcPre.Config` | 仅融合 Sinkhorn，保留调用方的 pre mix |
| `mhc.asc_hc_post` | `HcPost.Config`（精确匹配） | `AscV41HcPost.Config` | `cann_ops_transformer.ops.mhc_post` |
| `engram.host_offload` | `HostEngramTable.Config`（精确匹配） | `HostOffloadEngramTable.Config` | CANN EngramFetch/EngramFetchGrad |
| `engram.host_offload_mxfp8` | `HostEngramTable.Config`（精确匹配） | `MXFP8HostOffloadEngramTable.Config` | MXFP8 数据与 E8M0 scale |
| `vision_language_dataloader.sft` | `DeepSeekV41DataLoader.Config`（`fqns=["dataloader"]`） | V4.1 结构化多模态 SFT dataloader | 数据行不覆盖 `thinking_mode` 等策略字段 |

```text
torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc
torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc
torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn
```

`deepseek_v4_1/` 没有 `__init__.py`：导入该 package 不注册任何 V4.1 override，每个入口
只在 `override.imports` 按完整 `module.function` 列名导入时注册。`mhc` 下的两个入口
只替换单个阶段，`HcPost` 的 kernel 调用因此要求显式的 `[B, T, D]` 流；`engram` 下的两个
入口都要求 `num_max_tokens_per_rank`，取值需覆盖该次运行的最大 batch。启用方式见
[V4.1 独立训练基线](../../examples/deepseek_v4_1/readme.md)。

#### `sparse_attn.asc` 的 kernel geometry 是 override 的参数

SMLA 与 indexer 的 metadata 都是 AI_CPU 算子，只依赖本次 forward 的 frames 与静态几何，
所以它们在 `build_attention_metadata`（每 microbatch 一次的 eager 边界）里预计算，随 map entry
下发、全层共享。这份几何**不在参考 config 上**：参考核只读整行的 document 边界，以及后端在
同一处按自己的 `compress_ratio` 折好的 `cu_seq_k`（池化轴），因此 kernel 几何属于端口，由
override 参数声明：

```text
# 默认值就是已发布的 V4.1-Flash 形状，Flash 配方不需要写参数
torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc
# 其他宽度（debug / 自研 config）在这里声明五个数
torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc={"num_heads":8,"head_dim":64,"index_topk":32,"index_num_heads":8,"index_head_dim":32}
```

参数默认值为 `num_heads=64, head_dim=512, index_topk=512, index_num_heads=32, index_head_dim=128`，
即 V4.1-Flash 的发布形状；端口自己的 config 顶层则是**必填、无默认值**，所以一层一定会读到
按它自己那五个数构建的 metadata。**没有 inline 回退**：这些 metadata 是内核的唯一来源，因此

- config 上的五个数是必需的，缺一个就 `TypeError`，不存在「依赖默认值静默按发布形状构建」的
  情况，也不会在层里补一次 AI_CPU 调用；
- metadata 挂在 entry 的 `aicpu_metadata` 上，**按消费它的 kernel 分组、按方向取值**：

  ```text
  aicpu_metadata
  + sparse_attn
    + fwd   -> sparse_flash_mla_metadata
    + bwd   -> sparse_flash_mla_grad_metadata
  + lightning_indexer
    + fwd   -> lightning_indexer_metadata
    + bwd   -> sparse_lightning_indexer_kl_loss_grad_metadata
  ```

  消费者直接读字段（`aicpu_metadata.sparse_attn.fwd`），一条 entry 就是一个几何，没有按 topk
  或按几何查表的步骤；
- 默认值与 config 字段的对应由 `tests/unit_tests/override/deepseek_v4_1/test_fused_geometry.py`
  对着模型自己的取值核对（debug flavor 与默认形状不同，因此它的配方必须显式声明）。

每个 `(config 类, compress_ratio)` 是 map 里的一个 entry，所以一个 40 层、三个几何的栈只建三份
metadata（不是四十份），也不需要任何一层知道整个栈的 ratio 表；CP 下由该 entry 自己的
`prepare_cp_metadata` 按 rank-local frames 重建。

#### 量化路径是另一个 backend：它自己建自己的 entry

量化后的 sparse attention（`torchao_npu` 里 fused AscendC attention 的拷贝）不是这一版的包装，
而是**另一个 backend**，因此它自己拥有 entry：量化适配层在 converter 里安装
`QuantV41SparseAttention.Config`，它覆写 `build_attention_metadata`，用参考 row 与本几何的三个
frame 建出四份 metadata（SMLA 前后向、量化 selector 的两条 schedule）。入口没有量化词汇：`asc`
的参数列表不变，量化与否只决定「哪个 config 类被安装」。

两条路径的 entry 接口是同一个：

```text
lightning_indexer
+ fwd            -> 打分层的 QLI schedule（建池与 pool-free 共用；与池宽度无关，宽度是 launch 参数）
+ fwd_sparse     -> 搜候选池的 QSLI schedule（按几何建；非量化路径暂空，等带 candidate 的非量化算子）
+ sparse_length  -> QSLI 的 per-row 候选数（同上，按几何建）
+ bwd            -> sparse_lightning_indexer_kl_loss_grad（教师边，两条精度路径共用）
```

- 候选池的两个数来自模型 config（每一层 selector 自己的字段），只出现在 launch 参数与候选数推导里；
- 量化 selector 与量化 sparse attention **必须同开**：非量化 attention 的 entry 装的是 BF16
  的 schedule，量化 kernel 读它不是「稍差一点」而是垃圾索引；
- 没有池（`candidate_topk_blocks <= 0`）时 `fwd` 仍然是量化 QLI 的那份（pool-free 层也要跑 QLI），
  只有 `fwd_sparse` / `sparse_length` 为空；
- CP：非量化 entry 由 port 的 `prepare_cp_metadata` 按 rank-local frames 重建；量化 entry 目前
  只支持 CP=1，其余情况显式报错。

#### indexer 只在两个 override 同开时学习

`sparse_attn.asc` 与 `lightning_indexer.asc` 各自替换独立的配置节点，但**必须同时启用**：
配置阶段（`update_from_config`）会检查这一对，只开其中一个直接报 `ValueError`，不会进入
训练。检查按本次运行**请求的 override 列表**判定——该阶段在 `apply_overrides` 之前，能看到的
就是请求本身；两个名字写在 `torchtitan_npu/models/deepseek_v4_1/model.py` 的
`_FUSED_ATTENTION_OVERRIDES` 中，改名需要同步。

原因是 indexer 的训练目标只存在于两个 kernel 反向之间：

- `sparse_attn.asc` 的 SMLAG 反向按每个 key 产生 attention 自身的完整 softmax
  边际质量 `p`（窗口 + 选中压缩条目 + sink），写在 `topk_scores` 的梯度上。
- `lightning_indexer.asc` 的 SLIKG 反向把该梯度当作教师：`dI = Z·Y - p` 由 kernel
  自己施加，据此产出 indexer 的 `q`/`k`/`w` 梯度。
- `topk_scores` 不参与任务损失，除这条边之外没有任何消费者。只开
  `lightning_indexer.asc` 时该梯度为 `None`，SLIKG 反向直接返回零梯度，indexer
  全程拿不到更新。
- 只开 `sparse_attn.asc` 时 reference selector 会接住这条边，但它把
  attention 级别的边际质量加到按 head 求和的 indexer logits 上，两侧不是同一个
  量，得到的训练信号是错的而不是「另一种」。这条组合当前不做修复，由上面的
  构造期检查拦住，使该路径不可达。

V4.1 的 `index_source_layers` 都落在 `compress_ratio` 为 1 或 2 的层上，纯窗口层
不携带 indexer，也不参与这条边。

两个 override 覆盖同一份文档局部 index 坐标：SMLAG 的教师按文档局部位置散布回
`topk_scores`，SLIKG 也按文档局部 index 计算，所以两侧都不做坐标转换。不要用其他
来源写入 `topk_scores` 的梯度：该边被约定为原始 `p`，写入带符号的值会被 SLIKG
当作教师并翻转 indexer 的梯度方向。

CPU 测试用 eager 替身覆盖这条边：整模型前反向跑通后逐个断言 indexer 参数都有有限
梯度，并覆盖只开一侧被拒绝、两侧同开可构建
（`tests/unit_tests/override/deepseek_v4_1/test_lightning_indexer_fused.py`）。替身验证的
是模型侧的搬运契约，SMLAG、SLIKG 和 LI 算子本身没有在 CPU 上执行过；NPU 上的数值和
是否真的产生非零梯度仍需在设备上确认。

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
| V4.1 构建报 `must be enabled together` | `sparse_attn.asc` 与 `lightning_indexer.asc` 必须同开（见 [DeepSeek-V4.1](#deepseek-v41)） |

TorchTitan override 机制、per-entry kwargs、checkpoint 和并行相关的完整说明见上游
`torchtitan/overrides/README.md`。
