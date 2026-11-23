# Override 扩展

`torchtitan_npu.override` 基于 TorchTitan 的配置级 override 机制替换
`Configurable.Config` 节点，用于接入 NPU 兼容实现、AscendC 融合算子和模型数值参考。
它不是算子级 override API；PyTorch backend 缺口及必须随包导入生效的临时适配放在
`torchtitan_npu.patches`。

DeepSeek-V4 的 tests 配方在 `config.override.imports` 中显式选择融合算子；DeepSeek-V4.1
的训练配方（`torchtitan_npu_recipes/models/deepseek_v4_1.py`）按 `USE_GOLDEN` 在融合组合
与数值参考组合间切换。`scripts/run_train.sh` 负责进程启动与原生参数透传。

## 启用方式

CLI 上经 `--override` 传入（TorchTitan `0.4.0` 的 `ConfigLoader` 参数，可重复给出，
条目会追加进 `config.override.imports`），或在 Python recipe 中直接设置。
`override.imports` 中的每个条目必须是完整的 `module.function` 路径。一个条目只会启用
对应的工厂函数，不会启用同一模块中的其他 override。

```bash
python -m torchtitan_npu.train \
  --module torchtitan_npu_recipes.tests.models.deepseek_v4 \
  --config deepseek_v4_debugmodel \
  --override \
    torchtitan_npu.override.common.rms_norm.asc
```

`--override` 可重复传入以启用多个入口；多个无参数条目也可以写在一个 token 里用逗号
分隔。工厂函数需要关键字参数时，使用
`target=<JSON object>`，并将整个条目作为一个 shell 参数：

```bash
--override \
  'torchtitan_npu.override.deepseek_v4.sparse_attn.asc={"num_heads":128}'
```

NPU DeepEP dispatcher 与上游 `moe_comm_backend="deepep"` 配置绑定。模型配置选择
`deepep` 后，显式启用以下 override 即可；`hidden_dim` 和
`num_max_tokens_per_rank` 由 recipe 根据最终 token budget 和 CP×TP 确定，再通过上游
`TokenDispatcherTransform` 在构造阶段填入；容量不足时明确报错。

```bash
--override \
  torchtitan_npu.override.common.token_dispatcher.asc_deepep
```

也可以直接设置配置：

```python
cfg.override.imports = [
    "torchtitan_npu.override.common.rms_norm.asc",
]
```

## 应用过程

recipe 在模型 registry 中完成分片声明，并应用配置变换。上游 Trainer 复制模型配置后、
任何组件执行 `build()` 前应用 override；转换与替换必须保留分片声明和精度字段：

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
├── a3_workaround.py
├── common/
│   ├── __init__.py
│   ├── optimizer.py
│   ├── rms_norm.py
│   ├── rope.py
│   ├── token_dispatcher.py
│   └── swiglu_group/
│       ├── __init__.py
│       └── ascendc.py
├── deepseek_v4/
│   ├── __init__.py
│   ├── chat_dataloader.py
│   ├── common.py
│   ├── compressor/
│   │   ├── __init__.py
│   │   └── ascendc.py
│   ├── lightning_indexer/
│   │   ├── __init__.py
│   │   ├── ascendc.py
│   │   ├── pypto.py
│   │   └── pypto_pro.py
│   ├── metadata.py
│   ├── mhc/
│   │   ├── __init__.py
│   │   ├── ascendc.py
│   │   ├── tilelang.py
│   │   └── triton.py
│   ├── moe.py
│   ├── moe_router.py
│   └── sparse_attn/
│       ├── __init__.py
│       ├── ascendc.py
│       └── native.py
└── qwen3_5/
    ├── fla.py
    ├── gated_delta.py
    ├── parallelize.py
    └── varlen_attention.py
```

`deepseek_v4_1/` 按 target 拆分，且不带 `__init__.py`（见
[DeepSeek-V4.1](#deepseek-v41)）：

```text
torchtitan_npu/override/deepseek_v4_1/
├── engram/
│   ├── __init__.py
│   ├── ascendc.py
│   ├── compile.py
│   └── mxfp8.py
├── lightning_indexer/
│   ├── __init__.py
│   └── ascendc.py
├── metadata.py
├── mhc.py
├── sparse_attn/
│   ├── __init__.py
│   └── ascendc.py
├── tilelang.py
└── vision_language_dataloader.py
```

- `common/` 存放只依赖 TorchTitan 公共组件、不依赖具体模型配置或元数据契约的实现。
- `<model>/` 存放依赖模型专属 target、配置字段、张量布局或元数据契约的实现。
- 模型专属实现不得跨模型目录引用。可复用部分应先下移到 `common/` 或其他公共模块。
- 简单 target 使用单文件，文件名采用 target 的 snake_case 语义，例如
  `RMSNorm -> rms_norm.py`。
- 同一 target 同时包含较大的多后端实现时使用 package。package 名仍表示 target；
  `__init__.py` 只定义稳定的注册入口，具体实现按 `ascendc.py`、`triton.py`
  等拆分。
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
| `torch` | 完全由标准 PyTorch 算子组成的独立实现 |
| `triton` | Triton kernel 实现 |
| `tilelang` | TileLang 融合 kernel 实现 |
| `workaround` | 保持原计算语义、仅绕过当前后端兼容问题 |
| 行为名称 | 与计算后端无关的能力，例如 `optimizer.virtual` |

同一 target family 中存在多个同类实现时，在 variant 后增加对象或职责限定，例如
`rope.asc_complex`、`rope.asc_cossin`。不要添加
`_override` 后缀，也不要使用无法说明实现边界的 `ascend` 等泛化名称。

Replacement 类采用「variant + target」命名，并保留标准缩写的大小写：

| 类型 | 示例 |
| --- | --- |
| AscendC 实现 | `AscRMSNorm`、`AscComplexRoPE` |
| Workaround 实现 | `WorkaroundComplexRoPE` |
| 行为实现 | `VirtualOptimizersContainer` |
| 后端协议或元数据 | `AscV4Metadata`、`AicpuMetadata` |

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
- `fqns` 使用 glob 限定具体配置节点；当前仓库中的部分入口依赖它缩小匹配范围（例如
  `common.swiglu_group.*`、`deepseek_v4.common.*` 与 `qwen3_5.*` 下的入口）。
- 不同 override 不能同时声明同一节点或互为祖先、后代的节点。
- replacement 必须自行保持输入输出、DTensor、sharding、checkpoint 和
  `torch.compile` 语义。
- 自定义内核应通过 `torch.library` 注册 schema、fake/meta 和 Autograd，再由
  replacement module 调用。

Float8、LoRA 等 converter 在 override 前执行。两者可能修改同一节点时，需要根据
converter 处理后的实际配置类型和 FQN 核对匹配结果。

## 当前入口

### Checkpoint 完整性校验（已迁移至 extensions）

Checkpoint manager 的 NPU 替换与文件级 SHA-256 manifest 校验不再是 override 入口：该能力
已迁移到 extensions 侧的
[`torchtitan_npu/extensions/torchtitan/components/checkpoint.py`](../extensions/torchtitan/components/checkpoint.py)
（NPU `CheckpointManager`，`TrainerEx` 的默认 checkpointer 配置节点）与
[`torchtitan_npu/extensions/components/validation.py`](../extensions/components/validation.py)
（manifest 的写入与校验）。

校验经 checkpoint 配置的 `extensions` 字段启用，不经过 `--override`。TorchTitan `0.4.0`
的 `ConfigLoader` 不接受扁平配置 CLI 参数，在 Python recipe 中设置：

```python
config.checkpointer.extensions.verify_hash_manifest = True
```

开启后保存会生成 `_checkpoint_hash_manifest.json`，加载会在物化 checkpoint state 前验证
清单。校验启用时，缺少清单会抛出 `FileNotFoundError`；存在
`_checkpoint_hash_manifest.pending` 标记时会抛出 `CheckpointManifestError`，
不会继续加载该 checkpoint；未开启校验时不读取清单，checkpoint 按旧格式加载。
清单只接受 checkpoint 目录中的单层普通文件，不读取绝对路径、
父目录、符号链接或其他非普通文件。

### Common

以下入口省略 `torchtitan_npu.override.common.` 前缀：

| 入口 | Target | Replacement | 说明 |
| --- | --- | --- | --- |
| `optimizer.virtual` | `OptimizersContainer.Config` | `VirtualOptimizersContainer.Config` / `VirtualHostSparseOptimizersContainer.Config` | 将 Adam/AdamW 的 `exp_avg` 和 `exp_avg_sq` 放入 NPU swap memory；HostSparse 输入派生 `VirtualHostSparseOptimizersContainer.Config` |
| `optimizer.cpu_offload` | `OptimizersContainer.Config` | `CpuOffloadOptimizersContainer.Config` / `CpuOffloadHostSparseOptimizersContainer.Config` | optimizer 规范状态保留在 CPU，更新在 NPU 上执行；要求在 Python recipe 中设置 `config.training.enable_cpu_offload = True`（HostSparse 输入派生 `CpuOffloadHostSparseOptimizersContainer.Config`） |
| `optimizer.swap_optimizer` | `OptimizersContainer.Config` | `OptimizerStateSwapContainer.Config` / `HostSparseOptimizerStateSwapContainer.Config` | 统一 offload 策略入口：已设 `config.training.enable_cpu_offload = True` 时转发 `cpu_offload`，否则将 optimizer state 放入 NPU swap memory |
| `rms_norm.asc` | `RMSNorm.Config` | `AscRMSNorm.Config` | 使用 `torch_npu.npu_rms_norm` |
| `rope.workaround` | `ComplexRoPE.Config` | `WorkaroundComplexRoPE.Config` | 预展开 cos/sin cache，并使用 PyTorch 小算子计算 interleaved RoPE；仅精确匹配 `ComplexRoPE.Config` |
| `rope.asc_complex` | `ComplexRoPE.Config` | `AscComplexRoPE.Config` | 使用 interleave 模式的 `torch_npu.npu_rotary_mul`；仅精确匹配 |
| `rope.asc_cossin` | `CosSinRoPE.Config` | `AscCosSinRoPE.Config` | 使用 half 模式的 `torch_npu.npu_rotary_mul` |
| `rope.asc_partial` | `SplitComplexRoPEConfig`（精确匹配） | `AscPartialComplexRoPE.Config` | 整宽（prefix + rotary）partial-RoPE 站点的 query/key 各收敛为一次 `cann_ops_transformer` 融合原地旋转；CANN 依赖在首次调用时惰性加载 |
| `rope.asc_half_rotation` | `HalfRotation.Config`（精确匹配） | `AscHalfRotation.Config` | 使用 half 模式的 `torch_npu.npu_rotary_mul` 处理 external half-width cos/sin 表（拼接为整宽后一次旋转）；V4.1 视觉侧 RoPE 使用 |
| `token_dispatcher.asc` | `AllToAllTokenDispatcher.Config` | `AscAllToAllTokenDispatcher.Config` | 使用 `torch_npu.npu_moe_token_permute` `npu_moe_token_unpermute` 融合 MoE dispatch/combine |
| `token_dispatcher.asc_deepep` | `DeepEPTokenDispatcher.Config` | `AscDeepEPTokenDispatcher.Config` | 使用 `cann_ops_transformer.ElasticBuffer` 实现训练路径的 MoE DeepEP dispatch/combine；当前要求 `expert_parallel_degree > 1` |
| `swiglu_group.asc` | `SwiGLU.Config`（`fqns=["*.feed_forward.routed_experts.activation_fn"]`） | `AscSwiGLU.Config` | 使用 `cann_ops_nn.swiglu_group` 完成路由专家的 SwiGLU 激活 |
| `swiglu_group.asc_shared_experts` | `SwiGLU.Config`（`fqns=["*.feed_forward.shared_experts.activation_fn"]`） | `AscSwiGLU.Config` | 使用 `cann_ops_nn.swiglu_group` 完成共享专家的 SwiGLU 激活 |

`rope.workaround` 与 `rope.asc_complex` 会声明同一 target，不能同时启用。
`rope.asc_partial` 精确匹配 split-aware 的 `SplitComplexRoPEConfig`，与其他 rope
入口作用于同一组件，同一站点只能启用其一；其 `cann_ops_transformer` 依赖在首次调用时
惰性加载，导入 `override.common` 或构造 RoPE 模块不需要 NPU 算子包。
`AscComplexRoPE` 和 `AscCosSinRoPE` 当前都要求同一 batch 内各行的位置布局一致，
并使用第一行位置构造 batch 共享的 cosine/sine 表。

### Common SwiGLU

入口前缀为 `torchtitan_npu.override.common.swiglu_group.`，适用于采用公共专家配置和下述 FQN 布局的模型。

`swiglu_group.asc` 和 `swiglu_group.asc_shared_experts` 均按 `SwiGLU.Config`（非精确
匹配）经 `fqns` 分别限定到 `routed_experts` 与 `shared_experts` 的 `activation_fn`
节点，只替换
FFN 中间的 gate/up + SwiGLU 激活桥接；`w13`、`w2` 和 dispatcher 配置保留，可与这些组件的
override 并存。CANN 激活消费 gate/up，并保留配方的 `swiglu_limit`；路由专家在激活后、`w2`
之前单独执行 FP32 路由分数乘法，再转回激活 dtype。该分数乘法当前没有与 CANN 激活融合。
CANN 实现位于 `common/swiglu_group/ascendc.py`。

低精转换作用于独立的 `Linear.Config`、`GroupedLinear.Config` 及其参数包装；融合激活
仍通过上述两个 override 显式选择。参数包装与 `activation_fn` 分别保留各自配置。

启用两个 expert override：

```bash
--override torchtitan_npu.override.common.swiglu_group.asc \
  --override torchtitan_npu.override.common.swiglu_group.asc_shared_experts
```

### Virtual Optimizer 与 checkpoint

Virtual Optimizer 使用 NPU swap memory 保存 Adam/AdamW 的 `exp_avg` 和
`exp_avg_sq`，只启用一个入口：

```bash
--override \
  torchtitan_npu.override.common.optimizer.virtual
```

`optimizer.virtual` 负责创建 swap-backed optimizer state。checkpoint 保存兼容已由
extension `CheckpointManager`（`torchtitan_npu/extensions/torchtitan/components/checkpoint.py`）
提供，对同步 native DCP 保存关闭 writer copy-ahead，无需额外的 checkpoint override。
完整的数据流、支持范围和限制见
[Virtual Optimizer 特性说明](../../docs/feature_guides/virtual_optimizer.md)。

### DeepSeek-V4

以下入口省略 `torchtitan_npu.override.deepseek_v4.` 前缀：

| 入口 | Target | Replacement | 说明 |
| --- | --- | --- | --- |
| `common.rms_norm` | `RMSNorm.Config`（`fqns` 限定 compressor 子树之外的 norm 节点） | `AscRMSNorm.Config` | 对 DeepSeek-V4 各 norm 节点统一启用 `common.rms_norm.asc` 的融合实现 |
| `common.rope` | `ComplexRoPE.Config`（精确匹配，`fqns` 限定 attention / indexer rope） | `AscComplexRoPE.Config` | 对 DeepSeek-V4 的 rope 节点统一启用 `common.rope.asc_complex` |
| `chat_dataloader.sft` | `GrainDataLoader.Config`（`fqns=["dataloader"]`，精确匹配） | `GrainDataLoader.Config` | 使用结构化 DeepSeek-V4 chat 数据做 SFT（`thinking_mode` / `drop_thinking` / `add_default_bos_token` / `reasoning_effort` 可配）；要求 CPT 文本数据 recipe 打底 |
| `sparse_attn.asc` | `DSV4InnerAttention.Config` | `AscSlidingWindowAttention.Config` / `AscHeavilyCompressedAttention.Config` / `AscCompressedSparseAttention.Config` | 在原生 TorchTitan DeepSeek-V4 attention（TP=CP=1）内部按子注意力类型使用 SMLA；`selector_backend` 选择 `native`/`asc`，`asc` 选择器自带 indexer 目标，要求 `aux_loss=None` |
| `sparse_attn.asc_legacy` | `CompressedSparseInnerAttention.Config`（精确匹配） | `AscCompressedSparseInnerAttention.Config` | 遗留几何 backend 的 `cann_ops_transformer` TND SparseFlashMLA 与融合 SMLAG 反向；记录几何由本入口的参数声明 |
| `lightning_indexer.asc` | `LightningIndexer.Config`（精确匹配） | `AscLightningIndexer.Config` | 融合 LI 选分与 SLIKG 反向；消费 `topk_scores` 上的 teacher |
| `lightning_indexer.pypto` | `LightningIndexer.Config`（精确匹配） | `PyPTOLightningIndexer.Config` | PyPTO 实现的 LI 选分与 SLIKG 反向；需额外安装与当前 CANN 匹配的 `pypto` runtime |
| `lightning_indexer.pypto_pro` | `LightningIndexer.Config`（精确匹配） | `PyPTOProLightningIndexer.Config` | PyPTO-Pro 实现的 LI 选分与 SLIKG 反向；同样需 `pypto` runtime |
| `compressor.asc` | `Compressor.Config`（精确匹配） | `AscNativeCompressor.Config` | 按 upstream 组件契约使用 CANN 压缩；`fused_norm` / `fused_rope` / `fused_pooling` 可分别关闭 |
| `compressor.legacy_asc` | `CompressorImplementation.Config`（精确匹配） | `AscCompressor.Config` | 遗留 NPU 几何 backend 的 CANN compressor |
| `mhc.asc_hc_pre` | `HcPre.Config` | `AscHcPre.Config` | 使用 `torch_npu.npu_mhc_pre` + `torch_npu.npu_mhc_sinkhorn` |
| `mhc.tilelang_hc_pre` | `HcPre.Config` | `TilelangHcPre.Config` | 使用外部 TileKernels Split/Apply 与 `torch_npu.npu_mhc_sinkhorn` |
| `mhc.asc_hc_post` | `HcPost.Config` | `AscHcPost.Config` | 使用 `cann_ops_transformer.ops.mhc_post` |
| `mhc.triton_hc_pre` | `HcPre.Config` | `TritonHcPre.Config` | 使用 `mhc_pre_sinkhorn_op` + `mhc_pre_bmm_op` |
| `mhc.triton_hc_post` | `HcPost.Config` | `TritonHcPost.Config` | 使用 `mhc_post_bmm1_op` + `mhc_post_bmm2_op` |
| `mhc.tilelang_hc_post` | `HcPost.Config` | `TilelangHcPost.Config` | 通过 compile-safe wrapper 延迟调用外部 TileKernels `mhc_post` |
| `mhc.triton_hc_head` | `HcHead.Config` | `TritonHcHead.Config` | 使用 `mhc_pre_only_sinkhorn_op` + `mhc_pre_bmm_op` |
| `mhc.tilelang_hc_head` | `HcHead.Config` | `TilelangHcHead.Config` | 使用 `mhc_head_compute_mix_tilelang` (TileLang 融合 kernel) |
| `moe.tilelang_swiglu_grouped` | `RoutedExperts.Config` | `activation_fn=TilelangSwiGLU.Config` | 使用 TileLang SwiGLU forward/backward |
| `moe.tilelang_swiglu_shared` | `FeedForward.Config` | `activation_fn=TilelangSwiGLU.Config` | 使用 TileLang SwiGLU forward/backward |
| `moe_router.tilelang` | `TokenChoiceTopKRouter.Config`（精确匹配） | `TilelangRouter.Config` | 动态路由使用 TileKernels `topk_gate` |
| `moe_router.tilelang_v4` | `DeepSeekV4Router.Config`（精确匹配） | `TilelangV4Router.Config` | V4 非 hash 层使用 TileKernels `topk_gate`；hash 层保持原逻辑 |
| `moe_router.tilelang_v41` | `DeepSeekV41Router.Config`（精确匹配） | `TilelangV41Router.Config` | V4.1 动态路由使用 TileKernels `topk_gate` |

DeepSeek-V4 的既有 `moe_router.tilelang` 配置需要改为 `moe_router.tilelang_v4`；V4.1 改用 `moe_router.tilelang_v41`，通用动态 router 使用 `moe_router.tilelang`。

`sparse_attn.asc` 支持记录几何参数（`num_heads`/`head_dim`/`index_num_heads`/`index_head_dim`
默认 `64`/`512`/`64`/`128`，即 V4-Flash 的发布形状；debug/pro 等其它宽度在同一个入口上给出，例如
`sparse_attn.asc={"num_heads":16,"index_num_heads":8}`）。四者是 port config 自己的必需
字段、无默认值——记录由它们定尺寸，未声明就无法构建；`head_dim` 恒为 `512`，因此任何
recipe 都不需要重复它。`window_size` 与 `index_topk` 不在这里：它们是层自己的 model config
字段，由模型声明一次，port 读取即可。teacher 的强度固定：SMLAG 反向把 raw marginal `p`
写到 `topk_scores` 的梯度上（`-1` 槽位清零，并乘一次 `softmax_scale`），没有关闭开关。

`sparse_attn.asc` 与 `lightning_indexer.asc` / `lightning_indexer.pypto` 是两组独立的配置节点，
训练 indexer 时必须成对启用（CANN 侧配 CANN，PyPTO 侧配 PyPTO）：前者产出 teacher，
后者在 `_LightningIndexerTND.backward` 用 `sparse_lightning_indexer_kl_loss_grad` 消费这条边。

metadata 不再是独立入口：每个 microbatch、每个几何一份 entry，由被替换配置自己的
`build_attention_metadata` 直接构建（upstream PR #4897 的 backend-owned 接缝）。
默认 `sparse_attn.asc` 的构建器是 `AscNativeInnerAttentionConfig.build_attention_metadata`
（`sparse_attn/native.py`，native SMLA 适配器当前要求 CP=1）；legacy 几何路径
`sparse_attn.asc_legacy` 的构建器是 `AscCompressedSparseInnerAttention.Config.build_attention_metadata`
（`sparse_attn/ascendc.py`）。两者都在模型目录的 `build_v4_metadata` 给出的整行
document 边界、按 ratio 折好的池网格与三个 kernel frame（`q`/`swa_k`/`cmp_k`）之上
补算子记录。map key 是 `(端口类, compress_ratio)`，每层只读自己 ratio 的
entry，因此没有跨几何查找、也没有 `plan_for` 之类的守卫。LI record 属于 selector：BF16 的
由 port 在同一入口按 `AicpuGeometry` 构建；量化的 selector 是**另一个 backend**
（`torchao_npu.quantized_modules.deepseek_v4.QuantizedV4SparseAttention`），它替换融合核的
config、用 `quant_lightning_indexer_metadata` 构建自己的 entry，并拒绝 CP 重切（其记录契约
不属于本 port）。`override/` 与 `models/` 都不知道量化存在。CP 下由
`AscV4Metadata.prepare_cp_metadata` 用 `doc_chunk_entry` 按 rank-local 行重建 plan，再用
`build_aicpu_metadata` 在其上重填记录（upstream PR #4617，`asc_legacy` 路径）；native
路径的 `NativeAscV4Metadata.prepare_cp_metadata` 直接拒绝 CP；参考契约的 hook 同样拒绝 CP。
compressor 的状态表由**融合 attention 的 metadata 构建器**在 eager 边界写进 plan（表宽随 batch 的
文档走，不能在建图时构），因此选 `sparse_attn.asc` 就带上了它，不再有独立的 compressor 状态表入口。非融合的 attention backend（参考核）不建表，压缩器在 forward 里按当前 batch 自行构表（eager 下逐
batch 正确；只有「参考 attention + 融合 compressor + 图训练」这一组合目前没有 recipe 使用）。
MHC 的 `asc_hc_pre` / `asc_hc_post` 与 `triton_hc_pre` / `triton_hc_post` / `triton_hc_head` / `tilelang_hc_head` 是可选入口（`deepseek_v4/__init__.py`
默认只导入 `compressor`、`lightning_indexer` 与 `sparse_attn`），需要时显式加入
`override.imports`。`triton_hc_head` 与 `tilelang_hc_head` 声明同一 `HcHead.Config` 节点，两者互斥，只能启用其一；二者均可与 `asc_hc_pre` / `asc_hc_post` 共存。

`tilelang_hc_post` 与 `asc_hc_post` 声明同一 `HcPost.Config` 节点，两者互斥，只能启用其一；二者均可与 `asc_hc_pre` 共存。
`tilelang_hc_pre` 与 `asc_hc_pre`、`triton_hc_pre` 作用于同一 `HcPre.Config` 节点，只能选择一个；它可以与作用于 `HcPost.Config` 的 `asc_hc_post` 同时启用。

依赖的 tilelang/tile_kernels 暂时未开源，不随默认 requirements 安装。
单卡调试使用 [DeepSeek-V4 配方](../../torchtitan_npu_recipes/tests/models/deepseek_v4.py)，先在配方中设置资产路径，再运行：

```bash
MODULE=torchtitan_npu_recipes.tests.models.deepseek_v4 \
CONFIG=deepseek_v4_flash_1p_2k NGPU=1 bash scripts/run_train.sh
```

DeepSeek-V4 的 tests 配方在 Python 中显式启用融合组合，不通过 `USE_GOLDEN` 切换；
`deepseek_v4_flash_43layers_16experts_eager_bf16` / `deepseek_v4_flash_1p_2k` 默认启用：

```text
torchtitan_npu.override.common.rms_norm.asc
torchtitan_npu.override.common.rope.asc_complex
torchtitan_npu.override.deepseek_v4.sparse_attn.asc   # selector_backend="asc"
torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre
torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post
torchtitan_npu.override.common.token_dispatcher.asc
torchtitan_npu.override.common.optimizer.swap_optimizer
```

DeepSeek-V4.1 训练配方（`torchtitan_npu_recipes/models/deepseek_v4_1.py` 的 `_overrides()`）
读取 `USE_GOLDEN`。`USE_GOLDEN=1` 时数值内核仅替换 RoPE：

```text
torchtitan_npu.override.common.rope.workaround
```

经 `_full()` 生成的非 LoRA 配方（`deepseek_v4_1_flash_cpt_4k_*` / `..._sft_4k_*`，`lora=False`）
还会在此之上追加 `optimizer.swap_optimizer`；LoRA 配方不追加。`deepseek_v4_1_flash_cpt_1m_a5`
直接写死 override 列表、不读取 `USE_GOLDEN`。

参考 recipe 只替换 RoPE：RMSNorm 使用模型配置中的 Torch 实现，MoE 使用当前模型
路径中经过 package patch 修正的 BF16 实现，attention 使用模型目录默认内核
（Attention Gym 的 `gather_attn`，逐文档 FP32），不再有独立的 attention 数值基线
入口。

DeepSeek-V4 当前采用单行 packed container，要求 `local_batch_size == 1`；增加每步
token 数时应调整 `seq_len`。参考 recipe（模型目录默认内核）使用公共的
`DeepSeekV4Metadata`（每个几何一份：整行边界、该 ratio 的池网格、三个 kernel frame 与查询
位置），当前仅支持无 context parallel 的连续文档布局。AscendC 路径的 entry（`AscV4Metadata`）
在它之上只加按 kernel 分组的预计算记录。

TND 数据约定见 [DeepSeek-V4 TND 适配](../../docs/feature_guides/deepseek_v4_tnd.md)。

PyPTO selector（`torchtitan_npu.override.deepseek_v4.lightning_indexer.pypto`）是同一契约的
独立实现：输入准备、排序、teacher 边与反向都在它自己的模块里，只使用 PyPTO 的 LI/SLIKG 两个
kernel；使用它需要额外安装与当前 CANN 匹配的 `pypto` runtime，安装方法参见
[PyPTO 安装文档](https://gitcode.com/cann/pypto/blob/master/docs/zh/install/build_and_install.md)，
安装后可用 `python3 -c "import pypto"` 检查。

### DeepSeek-V4.1

以下入口省略 `torchtitan_npu.override.deepseek_v4_1.` 前缀：

| 入口 | Target | Replacement | 说明 |
| --- | --- | --- | --- |
| `sparse_attn.asc` | `CompressedSparseInnerAttention2.Config`（精确匹配） | `AscV41SparseAttention.Config` | AscendC attention 前后向；`topk_scores` 作为教师边。该 config 按**自己的 `compress_ratio`** 预计算 attention 与 indexer 的 metadata（每个几何每 microbatch 一次，全层共享同一份 entry），因此没有单独的 metadata 入口；kernel geometry 是该 config 自己的必需字段，由 override 参数一次写入，默认值为 V4.1-Flash 形状。本入口不声明任何量化：量化 selector 的 metadata 属于量化后的 attention backend，它替换本节点并自建 entry（见下文），本入口在那条路径上不生效 |
| `lightning_indexer.asc` | `Selector.Config`（精确匹配） | `AscSelector.Config` | 融合 LI 选分与 SLIKG 反向；它的两条 AI_CPU metadata 由 `sparse_attn.asc` 的几何一并预计算，随同一份 map entry 下发 |
| `mhc.asc_sinkhorn` | `HcPre.Config`（精确匹配） | `AscV41HcPre.Config` | 仅融合 Sinkhorn，保留调用方的 pre mix |
| `mhc.asc_hc_post` | `HcPost.Config`（精确匹配） | `AscV41HcPost.Config` | `cann_ops_transformer.ops.mhc_post` |
| `tilelang.tilelang_hc_pre` | `HcPre.Config`（精确匹配） | `TilelangV41HcPre.Config` | 单趟 mHC pre，由 TileKernels Split/Apply 与 `torch_npu.npu_mhc_sinkhorn` 完成 |
| `tilelang.tilelang_hc_post` | `HcPost.Config`（精确匹配） | `TilelangV41HcPost.Config` | 由外部 TileKernels `mhc_post` 算子完成 mHC post；与 `mhc.asc_hc_post` 同 target，互斥 |
| `engram.host_offload` | `HostEngramTable.Config`（精确匹配） | `HostOffloadEngramTable.Config` | CANN EngramFetch/EngramFetchGrad |
| `engram.host_offload_mxfp8` | `HostEngramTable.Config`（精确匹配） | `MXFP8HostOffloadEngramTable.Config` | MXFP8 数据与 E8M0 scale |
| `engram.shard_over_efsdp` | `HostEngramTable.Config`（精确匹配） | `HostEngramTable.Config` | 将 Host Engram 表与 SparseAdam 状态按 EFSDP 与 EP 联合分片 |
| `vision_language_dataloader.sft` | `GrainDataLoader.Config`（`fqns=["dataloader"]`，精确匹配） | `GrainDataLoader.Config` | V4.1 结构化多模态 SFT dataloader（`image_root` / `thinking_mode` / `drop_thinking` / `add_default_bos_token` / `reasoning_effort` 可配） |

```text
torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc
torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc
torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn
```

`deepseek_v4_1/` 没有 `__init__.py`：导入该 package 不注册任何 V4.1 override，每个入口
只在 `override.imports` 按完整 `module.function` 列名导入时注册。`mhc` 下的两个入口
只替换单个阶段，`HcPost` 的 kernel 调用因此要求显式的 `[B, T, D]` 流；`engram` 下的
`host_offload` / `host_offload_mxfp8` 两个入口要求 `num_max_tokens_per_rank`，取值需覆盖
该次运行的最大 batch，`shard_over_efsdp` 只置分片开关、不要求该参数。启用方式见
[V4.1 独立训练基线](../../torchtitan_npu_recipes/README.md)。

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
配置需保证两者配套；模型构建入口不再检查 teacher 配对。当前模型并行入口仅接受 TP=CP=PP=1。

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

### Qwen3.5

以下入口省略 `torchtitan_npu.override.qwen3_5.` 前缀：

| 入口 | Target | Replacement | 说明 |
| --- | --- | --- | --- |
| `fla.operators` | `InnerGDN.Config`（`fqns=["model.layers.*.attention.inner_attention"]`） | `FLAInnerGDN.Config` / `FLAContextParallelInnerGDN.Config` | 使用官方 FLA 的 GDN 与可选卷积；`context_parallel` / `causal_conv1d` / `force_fp32` 三个关键字参数决定落到哪个 replacement，`context_parallel` 必须与 Qwen 配方的 GDN 布局一致 |
| `fla.context_parallel` | `InnerGDN.Config`（`fqns=["model.layers.*.attention.inner_attention"]`） | `FLAContextParallelInnerGDN.Config` | 官方 FLA GDN + 卷积，带 Qwen CP head exchange；要求 context-parallel GDN 配方 |
| `gated_delta.triton` | `GDNKernel.Config`（精确匹配） | `TritonGatedDeltaKernel.Config` | 使用 Triton-Ascend GDN kernel |
| `gated_delta.npu` | `InnerGDN.Config`（`fqns=["model.layers.*.attention.inner_attention"]`，精确匹配） | `AscInnerGDN.Config` | 使用内置 Triton GDN kernel，卷积换成 NPU 实现；`kernel` 字段与 `gated_delta.triton` 同为 `TritonGatedDeltaKernel.Config` |
| `gated_delta.context_parallel` | `AscContextParallelInnerGDN.Config`（`fqns=["model.layers.*.attention.inner_attention"]`，精确匹配） | `AscContextParallelInnerGDN.Config` | NPU GDN 的全序列 head exchange 形态 |
| `gated_delta.qwen_sft` | `GatedRMSNorm.Config`（`fqns=["model.layers.*.attention.norm"]`，精确匹配） | `QwenSFTGatedRMSNorm.Config` | 保持 Qwen SFT 的 norm 与 gating 精度 |
| `varlen_attention.asc` | `VarlenInnerAttention.Config`（`fqns=["model.layers.*.attention.inner_attention"]`，精确匹配） | `AscVarlenInnerAttention.Config` | 使用 CANN TND attention |
| `varlen_attention.asc_cp` | `UlyssesVarlenInnerAttention.Config`（`fqns=["model.layers.*.attention.inner_attention"]`，精确匹配） | `AscCPVarlenInnerAttention.Config` | 使用 CANN TND attention，带 Qwen CP head exchange |

Qwen 配方默认启用 `gated_delta.npu`；MoE 路径可再叠加
`torchtitan_npu.override.common.token_dispatcher.asc`（见
`torchtitan_npu_recipes/tests/models/qwen3_5.py`）。

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
