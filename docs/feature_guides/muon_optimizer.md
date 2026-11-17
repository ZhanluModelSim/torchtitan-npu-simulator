# DeepSeek-V4 Muon 优化器

本文说明当前实现中的 Muon 方案、使用方法和能力边界。实现以 TorchTitan 上游
`DistMuon`/FlexShard 为核心；上游的 `ComputeLayout`、`Owned`、`BlockShard`、bucket
和 storage-to-compute 语义参见 [TorchTitan FlexShard README](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/flex_shard/README.md)。

## 当前方案

DSV4 使用混合优化器容器：匹配 Muon 规则的二维矩阵参数交给 `DistMuon`，其余参数交给
AdamW。`torchtitan_npu/models/deepseek_v4/config_registry.py` 中的
`_dsv4_optimizer_config()` 提供常规 DSV4 optimizer schema；
`_dsv4_muon_profile()` 提供参数匹配、FlexShard compute layout 和 bucket 元数据。
`--optimizer.name Muon` 才 materialize 出 DistMuon 与 AdamW fallback 两组。

Muon 覆盖 attention 投影、compressor/indexer 投影、shared/routed experts、router、
mHC 的 `hc_fn` 和全局 `hc_head.hc_fn`，以及对应的二维 `ape` 参数。embedding、输出头、
归一化和其他 1D 参数落入 AdamW 组。

DSV4 recipe 的默认 Muon 参数为：

```text
momentum=0.95
weight_decay=0.1
ns_steps=10
adjust_lr_fn="match_rms_adamw"
foreach=False
```

Muon 与 AdamW fallback 当前共用 `--optimizer.lr`，没有独立 `muon_lr` 字段；
`--optimizer.muon_momentum`、`--optimizer.muon_ns_steps` 和
`--optimizer.muon_adjust_lr_fn` 可以分别覆盖。

## 如何使用 Muon

使用常规 `deepseek_v4_flash_43layers_16experts` recipe，并显式选择 Muon。以下是单机
8 卡的直接启动命令；swap override 也是显式列出，而非 recipe 隐式启用：

```bash
MODULE=torchtitan_npu.models.deepseek_v4 \
CONFIG=deepseek_v4_flash_43layers_16experts \
NGPU=8 \
bash scripts/run_train.sh \
  --hf-assets-path /path/to/DeepSeekV4_tokenizer \
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
    torchtitan_npu.override.common.token_dispatcher.asc \
    torchtitan_npu.override.common.optimizer.swap_optimizer
```

DeepSeek-V4 示例脚本默认使用 Muon，并固定同一组 Muon 与 swap 参数。

当前只验证并支持 TP=1、PP=1；routed experts 的 layout 同时声明 DP shard、EFSDP 和
EP，以兼容当前 EP=8/EP=1 的 storage mesh。

## HSDP replica dedup

HSDP 的 `dp_replicate` 轴保存相同的本地参数分片。开启 replica dedup 后，DistMuon 为每个
参数 FQN 在该轴上选择一个确定的 owner；只有 owner 保留并更新该参数的 Muon momentum，随后将
更新后的本地参数分片异步 broadcast 给同一 replica group 的其他坐标。该路径减少重复的
storage-to-compute、Newton--Schulz 和参数更新计算，同时不改变模型参数的 storage layout。

该功能仅作用于 `DistMuon`，默认关闭。使用 HSDP 且显式启用 Muon 时，在启动参数中增加：

```text
--optimizer.muon-enable-hsdp-replica-dedup
```

不使用 HSDP，或未启用该开关时，DistMuon 保持原有行为。DCP 保存和恢复沿用常规 checkpoint
配置，无需增加 dedup 专用 checkpoint 参数。

## 如何开启 swap

swap 是显式 opt-in override，不存在 `swap_optimizer=true` 配置字段。在同一条命令的
`--override.imports` 末尾增加：

```text
torchtitan_npu.override.common.optimizer.swap_optimizer
```

`swap_optimizer` 与
`torchtitan_npu.override.common.optimizer.virtual`（Virtual Optimizer）都替换
`OptimizersContainer.Config`，不能同时启用。使用 Muon state swap 时，
`override.imports` 中不得包含 Virtual Optimizer；需要 Virtual Optimizer 时则移除
`swap_optimizer`。

swap 将 optimizer state 置于 NovaSwap 管理的 CPU/NPU 生命周期中。PyTorch 仍在首次梯度
更新时懒创建 optimizer state；swap 不会在模型初始化阶段预分配完整 state storage。Muon
和 AdamW 使用不同的唯一 swap name，因此多个 optimizer 实例不会互相覆盖 state。

Muon 的 `momentum_buffer` 首次创建后注册为 NovaSwap tensor 并执行 D2H。后续 step 按
FlexShard bucket 调度：

1. FlexShard 准备执行一个 redistributed bucket 的 storage-to-compute enqueue 时，swap
   在 prefetch stream 上为该 bucket 的 `redistributed_items` 和 `unredistributed_items`
   提交 H2D。
2. 每个 tensor 到达 `_prepare_local()` 时仍执行 `WAIT_DEVICE`，随后由上游更新 momentum。
   H2D 提前提交不改变 tensor 的消费依赖。
3. transfer stream 完成该 bucket 的 prepare、pack 和 inbound `all_to_all_single` 提交后，
   swap 在同一 transfer stream 上提交该 bucket 已更新 redistributed momentum 的 D2H。
   NovaSwap 的 offload stream 因而等待 transfer-stream event；D2H 不会抢在该 bucket 的
   inbound A2A 前提交。
4. caller stream 等待 `compute_input_ready` 后执行 Newton--Schulz。FlexShard 可以在
   caller stream 计算前一个 bucket 时预取后一个 bucket 的 H2D 和 inbound A2A。

`unredistributed_items` 不进入 storage-to-compute A2A。它们在 caller stream 上执行
`_prepare_local()`、Muon 计算和参数更新，随后沿现有单 tensor 路径提交 D2H。


AdamW 的 `exp_avg`、`exp_avg_sq`（AMSGrad 时还包括 `max_exp_avg_sq`）按 bucket 拼成连续
flat state。每个 bucket 执行 H2D、`WAIT_DEVICE`、上游 AdamW 更新和 D2H，并在当前 bucket
更新前提交下一个 bucket 的 H2D。

开启 swap 之后，checkpoint 的存储和加载支持同步模式和基于 CPU snapshot 的异步模式 `async`，暂不支持
`async_with_pinned_mem`。


## 能力边界

当前方案适合 DSV4 单机 8 卡、TP/PP=1、EP/DP-shard 并行的实验和训练。尚未证明：

- TP>1 或 PP>1 的 DistMuon；
- `async_with_pinned_mem`、跨 world size 恢复及更复杂并行拓扑下的 swap checkpoint；
