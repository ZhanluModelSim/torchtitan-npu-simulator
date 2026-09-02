# DeepSeek-V4 Muon 优化器

本文说明当前 `master` 分支中的 Muon 方案、使用方法和能力边界。
实现以 TorchTitan 上游 `DistributedMuon`/FlexShard 为核心；上游 FlexShard 的
`ComputeLayout`、`Owned`、`BlockShard`、bucket 和 storage-to-compute 语义，参见
[TorchTitan FlexShard README](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/flex_shard/README.md)。

## 当前方案

DSV4 使用一个混合优化器容器：匹配 Muon 规则的矩阵参数交给
`DistributedMuon`，其余参数交给 AdamW。配置入口是
`torchtitan_npu/models/deepseek_v4/config_registry.py` 中的
`_dsv4_muon_profile()`。该 profile 只承载模型参数匹配、FlexShard compute
layout 和 bucket 元数据；所有可调优化器标量由 CLI schema 提供。

Muon 参数主要包括：

- attention 投影、压缩器和 indexer 投影；
- shared experts、routed experts 和 router；
- mHC 的 `hc_fn` 以及全局 `hc_head.hc_fn`。
- 主干 compressor 与 indexer compressor 的二维 `ape` 参数。

未匹配的参数（例如 embedding、输出头、归一化参数和其他 1D 参数）落入 AdamW
组。Muon 使用上游 Newton-Schulz 实现；DSV4 recipe 的默认 CLI 值为：

```text
momentum=0.95
weight_decay=0.1
ns_steps=10
adjust_lr_fn="match_rms_adamw"
foreach=False
```

`match_rms_adamw` 使 Muon 更新幅度与 AdamW 超参数处于相近尺度；当前实现没有在
DSV4 配置中暴露独立的 `muon_lr` 或 `hybrid_ns` 开关。

## 如何使用 Muon

使用常规 DSV4 配方，并显式选择 Muon。例如 8 卡 DSV4 Flash（43 层、16
experts）：

```bash
MODULE=torchtitan_npu.models.deepseek_v4 \
CONFIG=deepseek_v4_flash_43layers_16experts \
NGPU=8 \
bash scripts/run_train.sh \
  --hf-assets-path tests/assets/deepseek_v3 \
  --dataloader.dataset c4_test \
  --dataloader.dataset-path tests/assets/c4_test \
  --parallelism.spmd-backend spmd_types \
  --parallelism.data-parallel-shard-degree 8 \
  --parallelism.data-parallel-replicate-degree 1 \
  --parallelism.expert-parallel-degree 8 \
  --parallelism.tensor-parallel-degree 1 \
  --parallelism.context-parallel-degree 1 \
  --parallelism.pipeline-parallel-degree 1 \
  --training.local-batch-size 1 \
  --training.global-batch-size -1 \
  --training.seq-len 4096 \
  --training.steps 100 \
  --optimizer.name Muon \
  --optimizer.lr 2.2e-4 \
  --optimizer.weight_decay 0.1 \
  --optimizer.muon_momentum 0.95 \
  --optimizer.muon_enable_nesterov \
  --optimizer.muon_ns_steps 10 \
  --optimizer.muon_adjust_lr_fn match_rms_adamw \
  --debug.no-moe-force-load-balance \
  --checkpoint.no-enable \
  --override.imports \
    torchtitan_npu.override.common.rms_norm.asc \
    torchtitan_npu.override.common.rope.asc_complex \
    torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata \
    torchtitan_npu.override.deepseek_v4.sparse_attn.asc \
    torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post \
    torchtitan_npu.override.common.token_dispatcher.npu_all_to_all_token_dispatcher
```

`--optimizer.name` 默认为 `native`，保持常规 recipe 原有的 AdamW 行为；设为
`Muon` 时才生成 DistributedMuon 与 AdamW fallback 两组。不存在 `_muon` 专用
recipe。DSV4 Muon 当前要求：

- `tensor_parallel_degree=1`；
- `pipeline_parallel_degree=1`；
- optimizer 使用 `DistributedMuon` 的 FlexShard compute layout；
- routed experts 的布局同时声明 DP shard、EFSDP 和 EP，以覆盖当前 EP=8/EP=1
  的存储 mesh；
- AdamW 组保持 `foreach=False`。Ascend 上 `foreach=True` 可能触发
  `aclnnForeachLerpScalar` 不支持错误。

## 如何开启 swap

swap 是显式 opt-in 的 override，不是当前 recipe 中的 `swap_optimizer=true` 字段。
在同一条命令的 `--override.imports` 末尾增加：

```text
torchtitan_npu.override.common.muon_state_swap.muon_state_swap
torchtitan_npu.override.common.muon_state_swap.muon_state_swap_checkpoint
```

完整示例（省略与上例相同的模型、数据和并行参数）：

```bash
--override.imports \
  ... \
  torchtitan_npu.override.common.muon_state_swap.muon_state_swap \
  torchtitan_npu.override.common.muon_state_swap.muon_state_swap_checkpoint
```

启用后：

- Muon 的 `momentum_buffer` 在首次创建后注册为 NovaSwap tensor，并在 step 前
  H2D、step 后 D2H；
- AdamW 的 `exp_avg` 和 `exp_avg_sq` 在 step 前换入、step 后换出；
- optimizer state 仍由 PyTorch optimizer 懒创建，swap 不会在模型初始化阶段提前
  生成完整 state storage；
- AdamW 和 Muon 使用不同的唯一 swap name，避免多个 optimizer 实例互相覆盖；
- Muon 使用 FlexShard 的 compute layout 和 per-layer bucket，swap 只包裹 state
  tensor 的生命周期，不改变参数 storage layout。

推荐同时关闭 checkpoint：

```bash
--checkpoint.no-enable
```

当前 `muon_state_swap_checkpoint` 会拒绝 checkpoint save/load，以及
`checkpoint.initial_load_path`。因此当前 swap 方案不能用于带 optimizer state 的
断点续训；需要 checkpoint 互操作时，应先关闭该 swap override 并使用普通 optimizer
路径。

## DSV4 与上游方案的差异

上游 FlexShard README 描述的是通用 optimizer-compute 基础设施和 Kimi 集成：
`ComputeLayout` 可以表达 `Owned`、`BlockShard`、多 mesh axis 的 storage-to-compute
重分布，`BucketConfig` 用于打包通信并与 optimizer compute 重叠。当前 DSV4 复用这些
上游 API，但配置和运行边界更窄：

| 维度 | 上游 FlexShard / DistMuon | 当前 DSV4 适配 |
|---|---|---|
| 参数选择 | 由具体模型 registry 定义 | 固定 DSV4 attention、experts、router、mHC 参数正则 |
| compute layout | 支持 `Owned`、`BlockShard`、per-head 等通用布局 | 主要使用 `Owned`，attention `wq_b/wo_a` 使用 per-head `Shard(0)`，routed experts 使用 DP/EFSDP/EP `Shard(0)` |
| bucket | 通用 bucket API | 每层一个 bucket，`hc_head` 单独一个 bucket |
| TP/PP | 由上游集成决定 | 当前明确拒绝 TP>1 或 PP>1 |
| swap | 上游 FlexShard 不提供 NPU state offload | 由 `muon_state_swap` + `extension/novaswap` 额外提供，且暂不支持 optimizer checkpoint |
| AdamW | 上游 Muon 集成通常只描述参数路由 | 当前同时对混合容器中的 AdamW moments 做 whole-step swap |
| 迭代步数 | 由上游 recipe 决定 | DSV4 默认 `ns_steps=10`，可用 CLI 覆盖 |

## 能力边界

当前方案适合 DSV4 单机 8 卡、TP/PP=1、EP/DP-shard 并行的实验和训练。以下能力
尚未由当前实现证明：

- TP>1 或 PP>1 的 DistributedMuon；
- swap 开启时的 optimizer checkpoint 保存、加载和断点续训；
- 长程收敛与无 swap 基线的数值等价；
- all-rank profiling skew 和多机 HCCL 场景；
- Muon 独立学习率或 bucket 合并策略。

关闭 swap 时，仍可使用同一 DSV4 recipe 的 Muon CLI 选择，移除上述两个
`muon_state_swap.*` override 即可。
