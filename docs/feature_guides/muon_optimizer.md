# DeepSeek Muon 优化器与 NovaSwap

DeepSeek-V4/V4.1 使用 TorchTitan 的 DistMuon/FlexShard 更新指定矩阵，其余参数使用 AdamW。Host Engram 表由独立的 CPU SparseAdam 更新，不进入 NovaSwap。

## 配置与启动

完整训练配置位于 `torchtitan_npu_recipes`。TorchTitan 0.4.0 使用 Python recipe，不再接受 `--optim.optimizer.name` 等通用字段 CLI。

在自定义 Python recipe 中选择优化器，例如 `my_recipes.py`：

```python
from torchtitan.components.optim import AdamW

from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu_recipes.models.deepseek_v4 import (
    _dsv4_muon_profile,
    _enable_indexer_aux_loss,
)
from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel


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
`SparseIndexerLoss`，使用 `mass_weighted=True`，系数为 `1.0`，使参与 Muon 更新的 indexer 参数获得梯度。
V4 的参考 Muon 示例和 checkpoint IT 沿用此配置。同时启用 `lightning_indexer.asc` 和 `sparse_attn.asc`
的配方也需要配置 `SparseIndexerLoss`。融合 attention 在反向缩放 teacher，经 `topk_scores` 的梯度
传给融合 indexer。

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

Muon 的 `momentum_buffer` 按 FlexShard plan 的消费顺序组织为 group，需要重分布和无需重分布的状态分别分组。多个 Tensor 打包到连续 pinned CPU buffer，通过 byte offset 定位，普通 group 上限为 512 MiB。超过上限的单个本地动量独立成组，后续普通 group 仍按 512 MiB 切分。超大组与相邻组之间不预取，改为消费时按需加载。驻留预算仅在需要时扩大到容纳单个超大组的动量与 H2D staging，随后恢复普通预算，设备仍须具备容纳该 Tensor 及临时存储的空间。

消费前执行 group H2D 并等待数据。组内最后一个非空成员完成 `_prepare_local()` 的动量更新并写入计算输入后，立即提交该组 D2H，再执行后续 FlexShard 重分布、Newton–Schulz 正交化及参数更新。D2H 逐 Tensor 写入同一组的 pinned CPU buffer，H2D 整块搬至 NPU staging 后再逐 Tensor 拷回原动量存储。

首次创建动量后注册 NovaSwap，不改变 FlexShard plan、Newton–Schulz 与参数更新公式。HSDP replica dedup 下只处理当前 replica 拥有的状态，checkpoint 初始化沿用现有 optimizer state 初始化流程。

AdamW 仍使用独立的 flat-state bucket。checkpoint 使用共享 CPU storage 的 view 和全局 shard metadata，保持原有 key 及 optimizer Tensor identity。未注册 NovaSwap 的优化器继续使用普通 checkpoint 路径。

开启 swap 后支持同步和 CPU snapshot 异步 `async`，不支持 `async_with_pinned_mem`。NovaSwap 要求 AdamW `moment_dtype="parameter"`，拒绝会提前创建 state 的 BF16 moment hook，不限制 BF16 前向计算。

`swap_optimizer` 与 `torchtitan_npu.override.common.optimizer.virtual` 替换同一 optimizer container，不能同时启用。

## HSDP replica dedup 与支持边界

在 recipe 的 `DistMuon.Config` 中设置 `enable_hsdp_replica_dedup=True`，可为每个 Muon 参数选择一个 HSDP replica owner。只有 owner 保存和更新动量，随后向其他 replica 广播参数分片。

当前 DistMuon 路径要求 TP=1、PP=1。新依赖和设备要求见 [安装指南](../user-guides/installation.md)。跨版本 checkpoint 兼容性、图捕获以及更复杂的并行组合不由本说明保证，必须使用对应运行时单独验证。
