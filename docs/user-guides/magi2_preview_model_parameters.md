# MAGI-2-preview 模型参数与训练指南

MAGI-2-preview（sandai/magi-2-preview，114B 视频+音频扩散 MoE Transformer）的训练支持位于
`torchtitan_npu/models/magi2_preview/`。本文档汇总模型 flavor、关键可调字段、
可用 NPU 算子 converter 与当前能力边界。训练数据准备见
[MAGI-2-preview 数据管线](magi2_preview_data_pipeline.md)。

## Flavors 与内置配置

| flavor | 训练配置函数 | 层数 | hidden | 专家数 | 用途 |
|--------|-------------|------|--------|--------|------|
| `debug` | `magi2_preview_smoketest` | 4（mm=[0,3]，MoE=[1,2]） | 512 | 8 | 本地/CI 冒烟（约 17.5M 参数） |
| `full` | `magi2_preview_baseline_bf16` | 40（mm=[0,1,38,39]，MoE=2..37） | 3072 | 256 | 官方 114B 结构 |

仿真入口同名配置：`--module torchtitan_npu.simulator --config magi2_preview_smoketest`。
另有 `magi2_preview_latent_smoketest`（使用离线 latent 数据集，需 `--dataloader.data_path`）。

所有模型字段可通过 `--model-overrides.<字段名>` 覆盖（CLI 中下划线换连字符），
字段与 `Magi2PreviewModel.Config` 一一对应，解析阶段执行整除关系、
`moe_top_k` 范围、`mm_layers`/`moe_layers` 索引合法性与互斥校验。

## 关键字段

| 字段 | 默认值（full） | 说明 |
|------|---------------|------|
| `num_layers` | 40 | Transformer 层数 |
| `hidden_size` / `head_dim` | 3072 / 128 | 需满足 `hidden_size % head_dim == 0` |
| `num_stream` | 4 | MHC hyper-connect 流数（算子 converter 当前硬编码 4） |
| `mm_layers` | [0,1,38,39] | 多模态条件层（3 专家分组投影 + 稠密 MLP） |
| `moe_layers` | 2..37 | Multi-Head MoE 层，与 mm_layers 必须互斥 |
| `moe_num_heads` / `num_experts` / `moe_top_k` | 12 / 256 / 6 | 每头独立路由；需 `hidden_size % moe_num_heads == 0` |
| `expert_intermediate_size` / `shared_expert_intermediate_size` | 1280 / 1280 | 路由专家 / 共享专家中间维 |
| `route_scale` | 4.9 | sigmoid 路由概率 L1 归一后的缩放 |
| `sink_token_num` | 1 | 每段注意力学习式 sink 数 |
| `attn_backend` | `sdpa` | 注意力后端：`sdpa`（分段 softmax，保底）或 `flex`（见下） |
| `text_in_channels` | 5120（debug 为 64） | 文本嵌入通道（对应文本编码器输出维） |

## 注意力后端（`attn_backend`）

- `sdpa`：分段 softmax，学习式 sink 以"零值附加键 + 每头学习 logit"实现，
  任意环境可用，是数值参考实现。
- `flex`：单算子路径。加速器设备上使用 `flex_attention` + `create_block_mask`
  （段内全连接 + sink 列），CPU 上等价退化为带 sink 扩展键的掩码 SDPA
  （torch 2.12 CPU eager 不支持 flex_attention 反向）。两种机制与 `sdpa`
  前反向数值等价（单测覆盖），长序列下替换逐段 Python 循环。

## NPU 算子 converter

通过训练配置的 `model_converters` 启用（与现有模型的 `npu_*` converter 用法一致）：

| 注册名 | 作用 | 约束 |
|--------|------|------|
| `npu_magi2_mhc` | 层内 MHC 混合（norm+phi 投影、pre/post 混合、Sinkhorn）替换为 `MHCPreTriton`/`MHCPostTriton` 融合算子 | 需要 triton-ascend 环境；`num_stream=4`；参数与 checkpoint 键不变 |
| `npu_rms_norm` | `MultiModalityRMSNorm` 替换为 `npu_rms_norm`（多模态版本按段调用） | 无 |

注意：`hyper_connect` 组合约定与 dsV4 MHC 转置相反，converter 内部已做
`h_res` 转置适配（与官方 `_hyper_connect_fwd_kernel` 对齐）。仿真
（`torchtitan_npu.simulator`）下 triton 类 converter 不可用，会保持纯 torch 路径。

## Checkpoint 加载

官方 `sand-ai/MAGI-2-preview` checkpoint 键名与内部模块路径完全一致，
`Magi2PreviewStateDictAdapter` 为显式恒等映射：`from_hf` 按模型配置过滤键集合
（丢弃 VAE/文本编码器等无关键），`to_hf` 直通。路由专家张量
（`block.layers.{i}.mlp.moe_mlp.{gate,W_gate,W_up,W_down}`）以
`(moe_num_heads * num_experts, ...)` 专家主序堆叠存储。

## 当前能力边界

- 并行：
  - FSDP2 + activation checkpoint 为默认路径。
  - EP：head-parallel MoE（`expert_parallel_degree > 1` 时沿头轴按整头切分
    各层路由专家，要求 `moe_num_heads % expert_parallel_degree == 0`；token
    全复制、本地头计算后零填充 + all-reduce 组合，checkpoint 键不变、全量
    权重经 DTensor Shard(0) 分发）。EP+FSDP 组合（eFSDP）尚未在真实硬件
    联调。
  - CP：Ulysses 上下文并行（序列按原始 token 顺序分片，注意力内部做序列↔头
    all-to-all，出口 autograd all-gather 并自动做 `1 / cp_degree` 梯度补偿，
    训练损失无需改动；要求序列长度与注意力头数均可被 `cp_degree` 整除）。
    CP 与 EP 暂不支持同时启用（显式报错）。
  - TP：v1 为序列复制式张量并行（注意力头/专家头按 TP 切分，稠密与共享
    专家按中间维切分并保持 swiglu7 gate/up 配对，行切分输出 all-reduce；
    分组权重中无法用单一 DTensor 放置表达的部分以本地切片保存）。
    TP+CP、TP+EP 暂不支持同时启用（显式报错）。
  - PP：`pipeline_magi2` 级切分（stage 0 持 pre_adapter，末级持
    post_adapter，flow-matching 损失在末级计算）。v1 限制：单微批
    （`local_batch_size == pipeline_parallel_microbatch_size`），pp>1
    使用 GPipe 调度，`num_layers % pp == 0`；PP+CP/TP/EP 暂不支持。
  - 组合限制外的其他并行组合会显式报错说明。
- 数据：合成 latent（冒烟）与离线真实 latent 数据集；在线编码不支持。
- MoE 专家计算为按专家分段的纯 torch 实现（正确但非融合内核），全量 12×256
  规模建议配合 converter/EP 交付后再上真实集群。
