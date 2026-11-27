# 单副本内 Dense 副本与路由专家分片

在 DeepSeek-V4 / DeepSeek-V4.1 的 HSDP 训练中，设置 `ParallelismConfig(disable_expert_replication=True)`（默认关闭），可在保留 Dense 参数副本的同时，将路由专家权重分片到全部专家数据并行卡上（副本度 = 1）。

## 使用方式

在 Python recipe 中使用 `torchtitan_npu.config.ParallelismConfig` 开启该选项，详见 [Python 训练配置](../developer_guides/python_recipes.md)。

以单机 8 卡、Dense 4 分片 × 2 副本、路由专家 EP4 × EFSDP2 为例：

```python
from torchtitan_npu.config import ParallelismConfig
from torchtitan_npu_recipes.tests.models.deepseek_v4 import (
    deepseek_v4_flash_43layers_16experts_eager_bf16,
)


def deepseek_v4_8p_expert_no_replica():
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=4,
        data_parallel_replicate_degree=2,
        expert_parallel_degree=4,
        fsdp_reshard_after_forward="always",
        disable_expert_replication=True,
    )
    return deepseek_v4_flash_43layers_16experts_eager_bf16(
        parallelism=parallelism,
        hf_assets_path="/path/to/DeepSeekV4_tokenizer",
    )
```

从仓库根目录启动：

```bash
MODULE=my_recipes CONFIG=deepseek_v4_8p_expert_no_replica NGPU=8 bash scripts/run_train.sh
```

DeepSeek-V4.1 传入相同的 `ParallelismConfig` 即可（例如 `deepseek_v4_1_debugmodel_text(parallelism=parallelism)`）。

| 参数范围 | 存储布局（8 卡，`dp_shard=4`、`dp_replicate=2`、`EP=4`） |
| --- | --- |
| Dense 参数（Attention、Embedding、LM head、router、shared experts） | `["dp_replicate", "dp_shard"]`（4 分片 × 2 副本） |
| 路由专家（`w13`、`w2`） | `["dp_replicate_edp_shard", "ep"]`（EP4 × EFSDP2，无额外副本） |
| V4.1 Host Engram 表 | 默认按 `EP` 切分；开启 `engram.shard_over_efsdp` 时同步折叠 `dp_replicate`（EP4 × EFSDP2，无额外副本） |

## 实现原理

开启后，`torchtitan_npu/models/common/fsdp.py` 的 `disable_expert_replication_dims()` 将路由专家的 `dp_replicate` 轴并入 `edp_shard` 分片轴并取消专家副本轴（有效专家 FSDP 度为 `dp_replicate * dp_shard / EP`），由 FSDP2 原生展平为 `dp_replicate_edp_shard` 完成 all-gather 与 reduce-scatter，不改变模型计算与梯度归一化公式；V4.1 的 Host Engram 表在开启 `shard_over_efsdp=True` 时同步将 `dp_replicate` 并入联合表分片组（`engram_shard`）并取消跨副本稀疏梯度同步。

## 常见约束

- **并行拓扑**：要求 `dp_replicate > 1`、`EP > 1`、`TP=CP=PP=1`，且 `dp_shard * dp_replicate <= num_experts`（超出时校验直接报错）；开启 `engram.shard_over_efsdp` 时还要求 Engram 表行数能被 `dp_shard * dp_replicate` 整除。
- **模型与训练器**：仅支持 DeepSeek-V4 / DeepSeek-V4.1 的 eager 路径；不支持 `GraphTrainer` 与其他模型。
- **优化器互斥**：不支持同时开启 `DistMuon.Config(enable_hsdp_replica_dedup=True)`。
- **未验证组合**：`FaultTolerantTrainerEx`（仅 DeepSeek-V4）已放行校验但未实机验证；V4.1 多模态、`torch.compile`、A5 低精度及切换并行布局后的 checkpoint 续训尚未验证。
