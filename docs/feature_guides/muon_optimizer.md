# DeepSeek Muon 优化器与 NovaSwap

DeepSeek-V4/V4.1 使用 TorchTitan 的 DistMuon/FlexShard 更新指定矩阵，其余参数使用 AdamW。Host Engram 表由独立的 CPU SparseAdam 更新，不进入 NovaSwap。

## 配置与启动

完整训练配置位于 `torchtitan_npu_recipes`。TorchTitan 0.4.0 使用 Python recipe，不再接受 `--optim.optimizer.name` 等通用字段 CLI。模型实现目录的 `config_registry.py` 只保留模型 registry 导出。

在自定义 Python recipe 中选择优化器，例如 `my_recipes.py`：

```python
from torchtitan.components.optim import AdamW

from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu_recipes.models.deepseek_v4 import (
    _dsv4_muon_profile,
    _enable_indexer_aux_loss,
    deepseek_v4_debugmodel,
)


def muon_training():
    config = deepseek_v4_debugmodel()
    _enable_indexer_aux_loss(config.model)
    profile = _dsv4_muon_profile(config.model)
    config.optim.optimizer.optimizers = [
        DistMuon.Config(
            pattern=profile.muon_pattern,
            compute_sharding_by_fqn=profile.compute_sharding_by_fqn,
            bucket_configs=profile.bucket_configs,
            lr=1e-5,
            ns_steps=10,
            adjust_lr_fn="match_rms_adamw",
        ),
        AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-8, fused=True, foreach=False),
    ]
    return config
```

本例使用参考 attention，默认通过 `_enable_indexer_aux_loss()` 为 indexer 配置上游已有的
`SparseIndexerLoss`，系数为 `1e-3`，使参与 Muon 更新的 indexer 参数获得梯度。
V4 的参考 Muon 示例和 checkpoint IT 沿用此配置，不需要冻结 indexer。
使用融合 attention 的 `selector_backend="asc"` 时，其已有独立的 indexer 训练目标，
应保留 `aux_loss=None`，不要叠加上述辅助损失。

启动时选择该工厂。NovaSwap 通过显式 override 启用：

```bash
MODULE=my_recipes CONFIG=muon_training NGPU=2 bash scripts/run_train.sh \
  --override torchtitan_npu.override.common.optimizer.swap_optimizer
```

数据、并行和 checkpoint 策略也由 recipe 配置，见 [Python 配置指南](../developer_guides/python_recipes.md)。关闭 swap 时不传该 override，并确认 recipe 的 `override.imports` 中也没有它。

`optimizers` 列表直接指定最终使用的优化器。超参数分别填写在 `AdamW.Config`、`DistMuon.Config` 中，不再设置外层 `name`、学习率或模式默认表。纯 AdamW 使用单个 `AdamW.Config`，V4 的默认 `eps=1e-6`。混合模式中的 AdamW 使用 `eps=1e-8`，Muon 的 `eps` 在自身 Config 中独立设置。Host Engram 的 SparseAdam 保留独立配置。

## 参数分组与 FlexShard

参数匹配、compute layout 和 bucket 配置由 recipe 定义：V4 位于 `torchtitan_npu_recipes/models/deepseek_v4.py`，V4.1 位于 `torchtitan_npu_recipes/models/deepseek_v4_1.py`。配置辅助函数根据模型结构生成默认规则，不改变优化器实现。

FlexShard 按 `Owned`、`BlockShard` 或专家维分片，把参数的 storage layout 转成 Muon 需要的计算布局。当前上游要求 Muon 的 `ComputeLayout.compute_dtype=torch.bfloat16`。这只规定 Newton–Schulz 的计算精度，不等于把持久动量统一改为 BF16。

V4.1 Host Engram 的表权重使用 SparseAdam，投影等 NPU 参数继续按模型 profile 分配。

## NovaSwap 与 checkpoint

Muon 动量仍按单个 Tensor 注册和搬运。FlexShard 准备重分布 bucket 时提前提交 H2D，消费前等待数据，完成 prepare 和 inbound 通信提交后执行 D2H。无需重分布的参数保持单 Tensor 路径。

首次创建动量后注册 NovaSwap，checkpoint 初始化沿用现有 optimizer state 初始化流程，不改变上游 Newton–Schulz 与参数更新公式。

AdamW 仍使用独立的 flat-state bucket。checkpoint 使用共享 CPU storage 的 view 和全局 shard metadata，保持原有 key 及 optimizer Tensor identity。未注册 NovaSwap 的优化器继续使用普通 checkpoint 路径。

开启 swap 后支持同步和 CPU snapshot 异步 `async`，不支持 `async_with_pinned_mem`。NovaSwap 要求 AdamW `moment_dtype="parameter"`，拒绝会提前创建 state 的 BF16 moment hook，不限制 BF16 前向计算。

`swap_optimizer` 与 `torchtitan_npu.override.common.optimizer.virtual` 替换同一 optimizer container，不能同时启用。

## HSDP replica dedup 与支持边界

在 recipe 的 `DistMuon.Config` 中设置 `enable_hsdp_replica_dedup=True`，可为每个 Muon 参数选择一个 HSDP replica owner。只有 owner 保存和更新动量，随后向其他 replica 广播参数分片。

当前 DistMuon 路径要求 TP=1、PP=1。新依赖和设备要求见 [安装指南](../user-guides/installation.md)。跨版本 checkpoint 兼容性、图捕获以及更复杂的并行组合不由本说明保证，必须使用对应运行时单独验证。
