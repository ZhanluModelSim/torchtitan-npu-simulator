# 低精度训练

torchtitan-npu 通过 TorchAO-NPU 为 DeepSeek-V4 和 DeepSeek-V4.1 提供 MXFP8、Block FP8 低精度训练，以及 MXFP4 量化感知训练（QAT）。量化配置复用现有预训练和续训练循环，可按需选择线性层、专家权重和 attention Q/KV 的量化方式。

## 环境要求

- 当前低精度训练面向 A5（Ascend 950）硬件，需要配套的 torch_npu、CANN、HCCL 和融合算子环境。
- 按[软件安装](../user-guides/installation.md)准备基础环境，并安装其中的可选依赖 TorchAO-NPU。仓内适配包依赖 `torchao==0.17.0`，也可按安装文档通过 `PYTHONPATH` 加载源码。
- 数据集、Tokenizer、CANN 环境变量及单机启动命令见[快速上手](../user-guides/quickstart.md)。本文中的参数组合用于追加到对应训练命令。

配置默认关闭量化训练，DeepSeek-V4/DeepSeek-V4.1 的 A5 示例脚本会显式启用量化。

## 当前支持的数据格式

当前实现中的量化数据和 scale 表示如下。不同量化路径分别使用低精度计算或 fake quant，并通过对应的反向实现传递梯度。

| 路径 | 量化数据 | scale 表示 | 分组方式 |
| --- | --- | --- | --- |
| MXFP4 | FP4（e2m1，pack uint8） | E8M0 | 每 32 个元素一个 scale |
| MXFP8 | FP8（E4M3） | E8M0 | 每 32 个元素一个 scale |
| Block FP8 | FP8（E4M3） | E8M0（二维 block 路径保存两组 scale） | 每 32×32 block 一个 scale |

Block FP8 的二维 block 路径中，两组 scale 分别覆盖最后一维和倒数第二维，scale 张量采用带 pack-2 末维的 E8M0 布局。MXFP4 此处主要用在 SFT-QAT 训练中， MXFP4 是训练时的 fake quant 约束，参数和 matmul 数据仍由 Block FP8 wrapper 保存和计算。

## MXFP8 与 Block FP8

`--extension.quantization.recipe` 决定匹配模块的量化方式：

| recipe | attention linear 和 shared expert | routed grouped experts |
| --- | --- | --- |
| `all_mxfp8` | MXFP8 | MXFP8 |
| `mix` | MXFP8 | Block FP8 |
| `all_block_fp8` | Block FP8 | Block FP8 |

recipe 按模型配置中的模块路径匹配 Attention linear、shared expert 和 routed expert，将匹配上的模块转化为对应的量化实现。MXFP8 路径对权重和激活使用 MX 量化；Block FP8 路径对权重使用 Block FP8 量化，激活使用 MX 量化。

## MXFP4 QAT

`--extension.quantization.enable-mxfp4-qat` 为 routed expert 的 Block FP8 权重增加 MXFP4 fake quant 数值约束，先将权重量化至 MXFP4 格式然后再反量化至 Block FP8，用于在训练中模拟 MXFP4 量化误差。它仅对 `mix` 和 `all_block_fp8` 中的 routed expert 生效，默认关闭。

在训练命令中使用以下参数组合：

```bash
--extension.quantization.enable-quantized-training \
--extension.quantization.recipe all_block_fp8 \
--extension.quantization.enable-mxfp4-qat
```

该选项不会将参数持久化为 4-bit，也不会将计算替换为原生 A8W4 GEMM。`all_mxfp8` 不包含 Block FP8 权重，启用该项 QAT 约束无效。

QAT 下做 MXFP4 量化时可以使用 `--extension.quantization.dst_type_max` 传参来指定量化时目标域的最大值。参数默认值为 0.0 对应 `Amax(Dtype)`；可以传参 7.0，以防止量化数据中的最大值被截断而产生较大精度损失。

DeepSeek-V4/V4.1 的完整 QAT 启动方式见[快速上手](../user-guides/quickstart.md#deepseek-v41-torchao-npu-低精度训练)。V4.1 QAT 脚本同时启用 routed expert MXFP4 fake quant、Lightning Indexer 和稀疏 attention/KV source Compressor 量化，相关自定义算子依赖见下一节。

## Attention 量化

### DeepSeek-V4 KV Cache 量化训练

DeepSeek-V4 的 KV Cache 量化训练针对 attention 的 nope 部分进行 MXFP8 fake quant，从而在训练阶段模拟 KV Cache 的量化误差。实现通过替换指定 norm 节点的配置，在 norm 输出上插入量化，量化对象是 norm 输出中的 nope 通道。该配置不适用于 DeepSeek-V4.1 的稀疏 attention 路径。A5 的 DeepSeek-V4 启动脚本已包含以下默认配置：

```bash
--extension.quantization.kv-norm-quantization.format mxfp8 \
--extension.quantization.kv-norm-quantization.fqns .attention.kv_norm,.attention.compressor.norm \
--extension.quantization.kv-norm-quantization.block-size 64
```

三个参数分别指定量化格式、需要替换的配置树节点后缀和 MX block size。`fqns` 中的两个节点分别对应 attention 的 `kv_norm` 与 compressor 的 `norm`，这些节点是量化插入位置。配置字段默认不启用（`format=None`，节点列表为空）；启用时需同时打开 `enable-quantized-training`。当前唯一支持的格式是 `mxfp8`，block size 由消费该配置的 MX 量化实现校验。

该路径使用 8 bit E4M3 的 MXFP8 数据和 8 bit E8M0 scale；默认脚本把 block size 设为 64，因此每 64 个 nope 通道共享一个 scale。量化结果随后反量化回原始高精度 dtype 参与训练，梯度通过 fake quant 的 STE 传递。

### DeepSeek-V4 LI Q/K 量化

Lightning Indexer（LI）的 Q/K 量化通过 `li-quantization` 单独选择，需要同时启用低精度训练。例如，在 DeepSeek-V4 命令中追加：

```bash
--extension.quantization.enable-quantized-training \
--extension.quantization.li-quantization mxfp4
```

| `li-quantization` | CANN `quant_mode` | Q/K 量化方式 |
| --- | ---: | --- |
| `fp8` | 1 | FP8 per-token-head |
| `mxfp8` | 3 | MXFP8 |
| `mxfp4` | 5 | MXFP4 |
| `hif8` | 4 | HiFloat8 per-tensor |

LI 各模式的实现细节不同：`fp8` 使用 8 bit E4M3 数据，按 token-head 动态量化，scale 以 FP32 保存；`mxfp8` 使用 8 bit E4M3 数据和 E8M0 scale；`mxfp4` 使用 4 bit E2M1 数据和 E8M0 scale；`hif8` 使用 8 bit HiFloat8 数据，按 tensor 动态量化，scale 以 FP32 保存。CANN 的 `quant_mode` 由这些数据格式和量化粒度共同决定。

`li-quantization` 默认值为 `None`，即不启用 LI 量化。本节表中的四种格式适用于 DeepSeek-V4；DeepSeek-V4.1 当前仅支持 `mxfp4`，并使用下一节说明的独立 QLI/QSLI 实现。

### DeepSeek-V4.1 LI 与稀疏 attention 量化

QLI（Quantized Lightning Indexer，量化索引器）与 QSLI（Quantized Sparse Lightning Indexer，量化稀疏索引器）负责为每个 query 选择参与稀疏 attention 的 Top-K key。在当前 V4.1 实现中，两者都将 indexer 的 Q/K 量化为 MXFP4，主要区别是选分的搜索范围及是否生产候选池：

| 实现 | 搜索范围 | 候选池行为 | 使用位置 |
| --- | --- | --- | --- |
| QLI（`quant_lightning_indexer`） | 当前 query 因果可见的全部 key | 候选源层除输出自身 Top-K 外，还生成候选块表供后续层使用；未启用候选池的层只输出 Top-K | 候选源层，以及候选池之外的选分层 |
| QSLI（`quant_sparse_lightning_indexer`） | 传入候选块表限定的可见 key | 消费候选源层生成的块表，重新选出本层 Top-K，并将原块表继续向后传递 | 使用候选池的 Reindex 层 |

候选池保存 key 的块索引，每块包含 `candidate_block_size` 个位置，最多保留 `candidate_topk_blocks` 个块。它与最终的 `index_topk` 个 key 是两级筛选：QLI 候选源层先确定候选块，后续 QSLI 层使用各自的 query 和权重在这些块内重新选分。Reuse 层直接沿用已有选择，不调用 QLI 或 QSLI。

QLI/QSLI 的 metadata（一条 QLI schedule、一条 QSLI schedule 以及 QSLI 被约束的 per-row 候选数
`sparse_length`）在每 microbatch 的 eager 边界预计算，随 attention metadata 下发、全层共享；候选数
按本 rank 的 frames 推导而不是回读 QLI 的输出，所以这些 metadata 在 forward 之前就绪，层内不再
调用任何 metadata 算子。量化后的 sparse attention 是**另一个 backend**：`li-quantization` 只在
`enable-sparse-attention-quantization` 同开时可用（converter 直接报错），由量化适配层
（`torchtitan_npu/extensions/adaptor/torchao/converter.py`）安装它自己的 attention config，该 config
用 `torchao_npu` 的 provider（`quantized_modules/deepseek_v4_1/attention_metadata.py`）建整份 entry
——SMLA 前后向、QLI schedule、QSLI schedule + 候选数、BF16 SLIKG 的 backward metadata，字段分组与
非量化路径同一套（`sparse_attn` / `lightning_indexer`，各按 `fwd` / `bwd` 取值）。候选池的容量与块大小
取自模型 config（`candidate_topk_blocks` / `candidate_block_size`，参考 selector 建池、搜池用的就是它们），
只作为 **launch 参数**与候选数推导的输入；QLI metadata 本身与池宽度无关（实测三组宽度逐字节相同），
所以一个几何只有一条 QLI schedule，建池层和 pool-free 层共用它。`sparse_attn.asc` 上没有任何池参数，
在这条路径上也不生效；只跑 BF16 selector 的配方不装量化 attention，`fwd_sparse` / `sparse_length`
保持为空（字段存在，值 `None`）。

`li-quantization=mxfp4` 同时启用这两种实现，由模型层的角色和候选池配置自动选择，无需分别指定。两者输出的索引交给 sparse attention 计算注意力；训练反向均使用 BF16 SLIKG 消费教师信号，QLI/QSLI 本身不替代 sparse attention。

V4.1 的 indexer 训练目标通过 sparse attention 反向提供 teacher（教师信号），再由 selector 的 SLIKG 反向消费。目标侧可以通过 `enable-sparse-attention-quantization` 单独量化；选分侧则不能单独量化，因为量化 selector 启动的是量化 attention 建的 schedule（单独开启 `li-quantization` 会被 converter 拒绝）：

- `--extension.quantization.li-quantization mxfp4`：将 BF16 LI selector 替换为 MXFP4 QLI/QSLI，SLIKG 反向保持 BF16；必须与下一条同开。
- `--extension.quantization.enable-sparse-attention-quantization`：同时替换 KV source Compressor 和 sparse attention；sparse attention 反向继续提供 indexer teacher。该开关本身也决定 entry 由哪个 backend 建——开着就是量化 attention 自建，关着就是 `sparse_attn.asc` 那份。

稀疏 attention 量化路径中，swa_kv 使用 `mxfp8_bf16`，cmp_kv 使用 `mxfp4_bf16`。该路径对 KV 量化再反量化后调用 sparse attention 算子，以引入对应的量化误差。

该缓存布局使用按组保存的 BF16 scale：滑窗 KV 为 8 bit E4M3 数据，每 32 个元素共享一个 BF16 scale；压缩 KV 为 4 bit E2M1 数据，每 16 个元素共享一个 BF16 scale。MXFP4 数据以 `uint8` 打包，每个字节包含两个 4 bit 值。V4.1 的反向计算使用量化反量化的 `swa_kv` 和 `cmp_kv`，并结合量化前向的输出和 LSE 计算梯度。

> [!NOTE]
> 启用 DeepSeek-V4.1 LI 量化（`li-quantization=mxfp4`）前，需参考 [cannbot-dsl 算子编译安装说明](https://gitcode.com/cann/cannbot-dsl/blob/master/net/native_package/README.md)，编译并安装 QLI 和 QSLI 算子。
>
> 启用稀疏 attention 量化前，需参考 [Ascend C 自定义算子编译安装说明](https://gitcode.com/cann/cann-recipes-infer/blob/master/ops/ascendc/README.md)，编译并安装 `kv_compress_epilog_v2`，并配置自定义算子环境变量。

在 V4.1 的 CPT 命令中追加以下参数，可以同时启用两侧量化；QAT 脚本已包含这些开关：

```bash
--extension.quantization.enable-quantized-training \
--extension.quantization.li-quantization mxfp4 \
--extension.quantization.enable-sparse-attention-quantization
```

未启用时，recipe 仍会量化匹配的线性层和专家模块，稀疏 attention 保持原始路径。此开关仅用于 V4.1，若没有对应的匹配节点，会记录警告并跳过替换。

## FSDP 权重预量化

`--extension.quantization.enable-fsdp-prequantize` 将 Block FP8 权重的量化提前到 FSDP all-gather 之前：各 rank 先量化本地权重分片，再传输 FP8 权重与 scale，前向和反向在同一次 unshard 周期内复用量化结果。相较于先聚合高精度权重、再在各 rank 上量化完整权重，该方式减少了参数通信量和重复量化计算，主要用于 attention 等 dense 权重的 FP8 训练优化。

### 工作原理

FSDP 的参数通信和梯度通信是两个阶段：参数在模块计算前通过 all-gather 恢复，反向结束后通过 reduce-scatter 规约并重新分片。预量化只改变参数 all-gather 的 payload，不改变梯度 reduce-scatter 使用的 `reduce_dtype`：

```text
本地权重 shard
    │ fsdp_pre_all_gather()
    ├─ 满足条件：cast 到 param_dtype → Block MX 量化
    │              → B_q（FP8）+ B_s1/B_s2（scale）
    ├─ 结构性 no-op（不在白名单 / 无 FSDP 分片 / 本地存储未分配）：
    │              回退普通参数通信（高精度 + 运行时量化），不报错
    └─ 白名单命中但对齐失败（axis=-2 非 64 倍数 / 末维非 32 倍数）：
                   抛 ValueError 立即终止，不静默回退
    │
    ├─ FSDP 分别 all-gather B_q、B_s1、B_s2
    │
    └─ fsdp_post_all_gather()
       重建 wrapper：逻辑参数接口保持 BF16，内部保存 FP8 权重和 scale
       → 前向/反向复用预量化 payload
       → 梯度按 reduce_dtype reduce-scatter
```

`fsdp_pre_all_gather()` 只处理当前 rank 持有的本地分片；量化结果由 FP8 权重 `B_q` 和两个方向的 scale `B_s1`、`B_s2` 组成，FSDP 对三类数据分别执行 all-gather。通信完成后，`fsdp_post_all_gather()` 首次创建 wrapper，后续更新已分配的 wrapper。wrapper 对 PyTorch、FSDP 和 autograd 暴露 shape 不变的逻辑参数，量化 MatMul 则直接读取内部的 `B_q + scale`，避免在同一次 unshard 周期的每个 MatMul 前重复量化权重。

这里的逻辑 dtype 与物理 payload 是两个概念：逻辑参数仍用于梯度路由和 FSDP 状态管理，实际通信和量化计算使用 FP8 权重及 scale。量化 wrapper 的自定义反向将梯度转换回逻辑权重，再交给 FSDP 按 `reduce_dtype` 做 reduce-scatter；优化器参数和 checkpoint 不会因此永久变成 FP8。

该选项默认关闭，需要同时启用低精度训练。它只作用于 `mix` 中的 routed expert 和 `all_block_fp8` 中匹配的 Block FP8 模块；`all_mxfp8`/`all_hif8` 没有任何预量化 scope，与该开关同开属于矛盾配置，转换阶段直接抛出 `ValueError`（提示改用 `all_block_fp8`/`mix` 或关闭开关），不会被静默忽略：

```bash
--extension.quantization.enable-quantized-training \
--extension.quantization.recipe all_block_fp8 \
--extension.quantization.enable-fsdp-prequantize
```

哪些权重保留预量化由 `--extension.quantization.fsdp-prequantize-fqns` 白名单控制：未设置时使用 recipe 默认白名单（全部 Block FP8 投影），显式空表与开关同开在配置校验时报 `ValueError`。白名单命中后，退出路径分为两类，行为刻意不同：

- **结构性 no-op（回退，不报错）**：没有 FSDP 分片（mesh size 1，如 EFSDP=1 的 MoE）、本地存储未分配（反向重建阶段）或本地形状即全局形状时，回退为高精度通信加运行时量化。这是并行拓扑决定的正常路径，不是错误。
- **对齐失败（报错，不回退）**：分片后的量化维（axis=-2）须为 64 的倍数、最后一维须为 32 的倍数，不满足时抛 `ValueError` 立即终止，报错信息给出本地/全局分片形状与 mesh size，并提示从 `fsdp-prequantize-fqns` 移除该权重或调整并行度。按专家轴分片（Shard(0)）的 3D 专家权重本地量化维完整，不受 64 对齐约束。静默回退会让训练行为偏离配置，因此对齐失败是硬错误。

典型示例：默认白名单包含 `.attention.wkv`（dim0=512）。`dp_shard=8` 时每卡分片 64 行，满足对齐、正常预量化；`dp_shard=16` 时每卡 32 行，启动即报错，需把 `.attention.wkv` 从 `fsdp-prequantize-fqns` 中排除（列出默认白名单的其余模式）。更高并行度会成批触发：如 `dp_shard=128` 时要求 dim0 为 8192 的倍数，均分切分的 `wq_a`/`wkv`/`wo_b`/shared experts（dim0 为 512–4096）全部违规，仅按 head/group 块切分的 `wq_b`/`wo_a`/`indexer.wq_b` 与按专家轴切分的 routed experts 仍可预量化。

### MoE 专家低精度通信

专家存在 EFSDP 分片时，也支持 FP8 或 MXFP4 参数通信。该能力主要用于功能支持，不以增加 EFSDP 并行度来获取训练加速为目标。
专家参数 all-gather 在 `EFSDP>1` 时发生，EP 切分本身不产生该通信。

默认专家通信格式为 `fp8`。在 `all_block_fp8` recipe 下，以下配置将启用 MXFP4 QAT 的 routed experts 切换为 MXFP4 通信，同时保留 attention 等 dense 权重的 FP8 通信：

```bash
--extension.quantization.enable-quantized-training \
--extension.quantization.recipe all_block_fp8 \
--extension.quantization.enable-mxfp4-qat \
--extension.quantization.enable-fsdp-prequantize \
--extension.quantization.fsdp-prequantize-dtype mxfp4
```

`enable-fsdp-prequantize` 启用参数通信预量化；`fsdp-prequantize-dtype mxfp4` 只将 routed experts 的通信格式切换为 MXFP4，
attention 等 dense 权重仍使用 FP8。专家在 `[E, N, K]` 布局下沿 K 轴量化，通信完成后转换为计算所需的 Block FP8；梯度通信仍使用 FSDP 原有配置。

该模式要求 routed experts 启用 MXFP4 QAT，并使用均匀的 `Shard(0)` 分片、2D/3D 权重和 block size 32；不支持非均匀分片或 `Shard(1)`。
未显式设置 `fsdp-prequantize-fqns` 时使用 recipe 默认白名单；若只指定 `.moe.routed_experts.inner_experts`，则只启用专家预量化通信，attention 的 FP8 预量化通信也会关闭。白名单不影响原有计算量化。

该功能基于 FSDP2，当前不支持 FSDP1 低精度通信。`EFSDP=1` 时不会进入专家预量化通信路径。

## 配置与生效检查

以下字段均位于 `--extension.quantization` 下，表中省略公共前缀：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `enable-quantized-training` | `False` | 低精度训练总开关 |
| `recipe` | `all_block_fp8` | 选择 `all_mxfp8`、`all_hif8`、`mix` 或 `all_block_fp8` |
| `enable-mxfp4-qat` | `False` | Routed expert 的 MXFP4 fake quant 约束 |
| `li-quantization` | `None` | LI Q/K 量化；V4 可选 `fp8`、`mxfp8`、`mxfp4`、`hif8`，V4.1 仅支持 `mxfp4` |
| `enable-sparse-attention-quantization` | `False` | V4.1 KV source Compressor 与稀疏 attention 混合量化 |
| `dst-type-max` | `0.0` | 用于 MXFP4 量化的目标类型最大值，可取值 0.0, 6.0-12.0 |
| `enable-fsdp-prequantize` | `False` | 启用 FSDP 参数 all-gather 前预量化 |
| `fsdp-prequantize-dtype` | `fp8` | Routed experts 的预量化通信格式；`mxfp4` 要求专家启用 MXFP4 QAT。`all_block_fp8` 下 attention 等 dense 权重仍使用 FP8 通信，启用范围由白名单决定 |
| `fsdp-prequantize-fqns` | `None` | 预量化权重 FQN 后缀白名单；`None` 用 recipe 默认白名单（全部 Block FP8 投影），显式空表与开关同开报 `ValueError`，命中但对齐失败的权重启动即报错 |
| `kv-norm-quantization.format` | 未设置 | V4 nope KV Cache 量化格式，目前为 `mxfp8`（8 bit E4M3） |
| `kv-norm-quantization.fqns` | 空列表 | V4 的量化插入位置，脚本使用 `.attention.kv_norm,.attention.compressor.norm` |
| `kv-norm-quantization.block-size` | `32` | V4 KV Cache MXFP8 的 block size；A5 脚本使用 `64` |

布尔开关可通过对应的 `no-` 形式显式关闭，例如 `--extension.quantization.no-enable-mxfp4-qat`、`--extension.quantization.no-enable-fsdp-prequantize`。使用 `--extension.quantization.no-enable-quantized-training` 可关闭量化训练总开关。

启动日志中的 `Applied TorchAO-NPU recipe=...` 表示 recipe 已应用；同时检查 `mxfp4_qat`、`li_quantization`、`sparse_attention_quantization` 和 `enable_fsdp_prequantize` 是否符合预期。预量化白名单的生效来源由 `fsdp_prequantize whitelist = recipe default ...`（默认白名单）或 `fsdp_prequantize whitelist = user ...`（用户白名单）日志行确认。该日志用于确认配置应用情况，不能替代训练精度和性能验证。

需要采集性能信息时，可增加 `--profiler.enable-profiling`。示例脚本中的 `USE_GOLDEN=1` 用于选择数值基线 attention override，默认使用 Ascend 融合路径；该变量不控制低精度训练总开关。

## 实现与验证入口

量化入口根据模型类型选择匹配规则，通过 converter 转换模型配置节点，并在模块构建时应用 TorchAO-NPU 量化。本文依据当前配置、转换逻辑和启动脚本整理，未在本文档重构中执行 NPU 训练；不同并行配置的精度与性能需在目标环境中验证。

- [量化配置字段](../../torchtitan_npu/config/configs.py)
- [TorchAO-NPU converter 与 recipe](../../torchtitan_npu/extensions/adaptor/torchao/converter.py)
- [Block FP8 权重与 FSDP 预量化实现](../../experiments/torchao-npu/torchao_npu/wrapper_tensors/block_mx_wrapper_tensor.py)
- [V4.1 Lightning Indexer 量化实现](../../experiments/torchao-npu/torchao_npu/quantized_modules/deepseek_v4_1/lightning_indexer.py)
- [V4.1 稀疏 attention 量化实现](../../experiments/torchao-npu/torchao_npu/quantized_modules/deepseek_v4_1/sparse_attention.py)
