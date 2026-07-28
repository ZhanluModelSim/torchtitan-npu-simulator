# Override 机制

`torchtitan_npu.override` 使用 torchtitan 的配置级扩展点替换 `Configurable.Config` 节点，适用于 NPU 兼容实现、融合组件。它不是算子级 override API；ATen backend 缺口和必须随包导入生效的行为放在 `torchtitan_npu.patches`。

## 显式启用

每个 `override.imports` 条目必须是完整的 `module.function`，并且只激活该工厂函数：

```bash
python -m torchtitan.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_debugmodel \
  --override.imports torchtitan_npu.override.common.rms_norm.npu_rms_norm_override
```

多个 target 可以用空格或逗号分隔，也可以单独一行，行末加上 \ 。工厂函数支持关键字参数时，CLI 使用带 JSON 的单个参数：

```bash
--override.imports 'vendor_pkg.overrides.kernel={"block_size":256}'
```

也可以直接设置配置：

```python
cfg.override.imports = [
    "torchtitan_npu.override.common.rms_norm.npu_rms_norm_override",
]
```

## 应用过程

torchtitan 在模型配置执行 `update_from_config()` 后、任何组件构建前处理 override：

1. 导入 target 所在模块，触发 `@override` 注册。
2. 按完整的 `module.function` 找到已启用工厂函数。
3. 遍历原始 `Trainer.Config` 树，按 `target`、`exact` 和 `fqns` 收集匹配节点。
4. 在修改配置前检查同节点和祖先/后代冲突。
5. 调用工厂函数生成替换配置，再由后续 `build()` 构造组件。

所有节点声明都从原始配置树收集，替换结果不会被再次遍历，因此应用顺序不会改变匹配结果。正常替换会记录工厂函数、配置节点 FQN 和替换前后的类型。

## 编写 override

最小工厂函数只声明目标和配置变换：

```python
from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm


@override(
    target=RMSNorm.Config,
    description="NPU fused RMSNorm via torch_npu.npu_rms_norm",
)
def npu_rms_norm_override(cfg: RMSNorm.Config) -> NPURMSNorm.Config:
    return derive(cfg, NPURMSNorm.Config)
```

编写时遵循以下约定：

- `target` 必须是 `Configurable.Config` 子类，优先选择最小稳定组件。
- replacement 通常继承目标 `Config`，并使用 `derive()` 保留共有字段；有意改变契约时才直接构造新配置。
- 默认匹配 `target` 及其子类；replacement 只实现具体类型契约时使用 `exact=True`。
- `fqns` 使用 glob 限定具体配置节点；本仓现有工厂函数暂未设置 `fqns`。
- 同一节点或祖先/后代节点不能被不同 override 同时声明；不相交节点可以使用相同目标类型。
- replacement 必须自行保持输入输出、DTensor、sharding、checkpoint 和 `torch.compile` 语义。
- 自定义内核应通过 `torch.library` 注册 schema、fake/meta 和 Autograd，再由 replacement module 调用。
- 不在 `__init__.py` 中批量导入具体注册模块，避免隐藏替换和互斥冲突。

Float8、LoRA 等 converter 在配置构造期间先执行，override 随后看到 converter 处理后的配置树。若两者可能修改同一节点，需要按实际类型和 FQN 核对匹配结果。

## 目录与常用 target

```text
torchtitan_npu/override/
├── common/           # 模型无关的 NPU 兼容和融合实现
├── deepseek_v3_2/    # DeepSeek-V3.2 模型专属实现
└── deepseek_v4/      # DeepSeek-V4 Golden、RoPE 和 DSA 实现
```

常用的模型无关 target：

| 用途 | Target |
| --- | --- |
| FlexAttention 替换为 SDPA | `torchtitan_npu.override.common.attention.npu_sdpa_override` |
| 优化器状态放入 NPU swap memory | `torchtitan_npu.override.common.optimizer.npu_swap_optimizer_override` |
| Ascend profiler | `torchtitan_npu.override.common.profiler.npu_profiler_override` |
| 融合 RMSNorm | `torchtitan_npu.override.common.rms_norm.npu_rms_norm_override` |
| Complex RoPE 兼容实现 | `torchtitan_npu.override.common.rope.npu_rope_override`<br>`torchtitan_npu.override.common.rope.npu_single_complex_rope_override` |
| 融合 RoPE | `torchtitan_npu.override.common.rope.npu_fused_rope_override`<br>`torchtitan_npu.override.common.rope.npu_fused_single_rope_override`<br>`torchtitan_npu.override.common.rope.npu_cossin_rope_override` |
| 本地 MoE token dispatch（EP=1） | `torchtitan_npu.override.common.token_dispatcher.npu_token_dispatcher_override` |

DeepSeek-V3.2 稀疏注意力需要同时启用：

```text
torchtitan_npu.override.deepseek_v3_2.sparse_attention.kernel
torchtitan_npu.override.deepseek_v3_2.sparse_attention.mask_handler
```

DeepSeek-V4 推荐使用 [`scripts/run_train_dsv4.sh`](../../scripts/run_train_dsv4.sh) 组合 recipe：

```bash
ATTENTION=golden ./scripts/run_train_dsv4.sh
ATTENTION=smla   ./scripts/run_train_dsv4.sh
```

两种 DSA kernel 都必须搭配：

```text
torchtitan_npu.override.deepseek_v4.varlen_dsa.npu_dsv4_packed_mask_handler_override
```

`dsa_sparse_attention_golden` 与 `npu_smla_tnd_override` 声明同一个 `DSAFlexAttention.Config` 节点，不能同时启用。DSV4 compat RoPE 与 fused RoPE、Golden RMSNorm 与通用融合 RMSNorm 也分别互斥。TND 数据约定见 [DeepSeek-V4 TND 适配](../../docs/TND.md)。

## Override 与 package patch

| 维度 | 配置级 override | Package patch |
| --- | --- | --- |
| 激活方式 | 写入 `override.imports` | 导入 `torchtitan_npu` |
| 目标 | `Configurable.Config` 节点 | PyTorch backend 或上游 Python 符号 |
| 生效时机 | 配置构造后、组件构建前 | 包导入时 |
| 冲突检查 | 同节点和嵌套声明会报错 | 不经过 override registry |

导入任意 `torchtitan_npu.override.*` 子模块时，Python 会先执行包入口，因此 `torchtitan_npu.patches` 也会生效。当前 patch 包含 PrivateUse1 算子实现，以及 pinned torchtitan 尚未提供的 trainer、mask、metrics 和 eager FlexAttention 适配。该导入要求完整的 `torch_npu`、CANN 和 HCCL 环境。

新增组件替换时优先使用 override。只有配置树无法表达的 backend 缺口，或随上游合入后可整体删除的临时能力，才放入 `patches/`。

## 常见失败

| 现象 | 检查项 |
| --- | --- |
| 模块导入失败 | target 路径、依赖和 NPU 环境 |
| target 未注册 | 是否使用准确的 `module.function` |
| 没有匹配节点 | `target`、`exact`、`fqns` 和 converter 后的配置类型 |
| 同节点或嵌套冲突 | 移除互斥 target，或缩小 `fqns` |
| 训练继续但替换未生效 | 检查 `[Override]` 日志和 `Applied N override(s)` |

torchtitan 机制的完整设计、per-entry kwargs、checkpoint 和并行示例见上游 `torchtitan/overrides/README.md`；本文只保留本仓使用和开发所需的约定。
