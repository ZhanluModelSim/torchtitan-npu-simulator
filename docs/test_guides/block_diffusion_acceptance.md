# Block Diffusion 接入验收记录

本文按[新模型最终验收规范](model_acceptance.md)记录 `block_diffusion` 的可复现
证据。验收对象包含 full-causal Attention 计算代理、sliding-window corruption、
masked-only loss 和 simulator 的结构级/meta 训练接入；没有外推到原始 workload
未提供的 checkpoint 数值等价、真实 NPU 数值结果或 DLM 推理调度。

## 被测版本和环境

- 仓库：`/home/c00856591/third_party_torchtitan-npu-simulator`
- 分支：`feat/npu-simulator-local`
- 基线：`feat/npu-simulator@3eeed771d70b733f4f7eadd60ccd38bb77ac3aea`
- 被测版本：上述基线上的当前工作区改动；合入前应提交并把 commit hash 回填到
  本节，作为不可变验收锚点。
- 容器：`titan-sim-PG`
- 镜像：`torchtitan-npu-simulator:v1.0`
- Python：3.12.13
- PyTorch：2.12.0+cpu
- torch-npu：2.12.0.rc1
- CANN：9.1.0-beta.1

所有 simulator 运行均在同一容器内执行，仓库 bind mount 到 `/workspace`。输出
目录通过 `--simulation.output_dir` 分场景隔离；下表记录执行时的摘要值。

## 静态契约

正式 `full` flavor 的参数量按模型定义逐项计算为
`10,052,615,823,360`，与 raw workload 的 10.05T 目标一致。主要正式 local shape：

| 参数 | 全局 shape |
| --- | --- |
| routed expert `w1` | `[1024, 4096, 8192]` |
| attention `q_proj` | `[8192, 8192]` |
| output projection | `[262144, 8192]` |

正式训练输入为 4096 tokens；计算代理把它作为单条 causal 序列，并以 alpha 只折算
Attention FLOPs。数据侧仍把最后 256-token block 作为 canvas。单测同时覆盖四种
flavor、未知 flavor 拒绝、block aligned sequence 约束、full-causal 等价、alpha
透传与成本折算、masked-only corruption/loss、TP/CP
head 整除、expert degree 约束，以及 raw schema state-dict 双向 round trip。
此外，使用真实 `c4_test` dataloader batch 在 CPU 上完成 debug MoE 数值前向、
masked-only loss 和反向：输出 shape 为 `[1, 32, 2048]`，15 个有效目标，所有
trainable parameters 均获得梯度。

## 执行矩阵

`persistent/active/model` 均为 simulator 报告的峰值字节数。`debug` 为 2 层、8
experts、top-2；`reduced` 为 8 层、32 experts、top-4。
“全长 causal + alpha”是本轮计算代理的复验数据；其余行是接入阶段的并行结构
证据，算子数仍对应旧 prefix/canvas Attention，需在需要更新精确账本时重新执行。

| 场景 | flavor / 并行 | 结果 | ops / comm | persistent / active / model peak |
| --- | --- | --- | --- | --- |
| 全长 causal + alpha | reduced, seq256/block64, 单卡 | 通过 | 1564 / 0 | 本轮关闭内存跟踪 |
| 单卡 | debug, seq32/block16 | 通过 | 403 / 0 | 6,433,280 / 25,733,228 / 12,883,328 |
| FSDP2 | debug, DP shard=2 | 通过 | 508 / 12 | 3,216,640 / 9,650,028 / 5,961,024 |
| TP2 | debug | 通过 | 434 / 5 | 3,222,016 / 13,412,460 / 6,985,088 |
| EP2 | debug, expert domain=2 | 通过 | 593 / 24 | 3,216,640 / 9,650,028 / 5,538,720 |
| ETP2 | debug | 通过 | 434 / 5 | 3,222,016 / 13,412,460 / 6,985,088 |
| CP2 | reduced, seq256/block64, Ulysses | 通过 | 2903 / 126 | 990,414,848 / 2,971,244,940 / 1,229,893,632 |
| AC full | debug | 通过 | 556 / 0 | 6,433,280 / 25,733,228 / 12,883,328 |
| PP2 | debug, 2 microbatches | 通过 | stage0 203/4；stage1 221/4 | stage0 active 12,865,588；stage1 active 12,867,640 |
| 核心组合 | reduced, DP shard2 + TP2 + CP2 + EP2 + AC | 通过 | 3477 / 225 | 247,743,488 / 743,230,860 / 366,780,096 |
| 最终组合 | reduced, PP2 + DP shard2 + TP2 + CP2 + EP2 + AC | 通过 | stage0 1733/247；stage1 1750/312 | stage0 active 430,810,816；stage1 active 431,338,176 |
| 正式容量 | full, seq4096/block256, DP shard64 + TP8 + EP512 + AC | 通过 | 35,971 / 1,173 | 78,580,647,424 / 235,741,946,940 / 83,832,417,816 |

最终组合的 logical world size 为 16，pipeline 两个 stage 分别持有 layers 0--3
和 layers 4--7，完成两个 microbatch 的 forward/backward 通信。TP/ETP 路径使用
仓库 grouped-mm bridge；CP 路径使用 Ulysses all-to-all；EP 路径包含 token
dispatch/combine collectives。EP 与 ETP 分别运行通过，但当前公共 ExpertParallel
不支持二维 expert mesh，因此 `EP > 1 && ETP > 1` 会在配置更新时明确 fail fast。

activation checkpoint 场景捕获到 154 个 recompute calls；reduced 核心组合捕获到
1,008 个 recompute calls。说明 AC 已进入实际反向重算路径，而不是只完成配置解析。
正式容量配置使用 logical world size 512，在约 43 秒内完成结构级 meta
forward/backward/recompute/optimizer 和内存报告导出；其中 forward、backward、
recompute 分别捕获 11,874、12,153、9,797 calls。该结果验证 10.05T/4K 规格的结构和
并行账本可运行，不代表单个进程真实分配了 10.05T 权重。

## Attention 算子闭环

针对 Zhanlu 在 `ZHANLU_DISABLE_PRUNE_OP=1` 下报告的 `SafeSoftmax`、`Tril`、
`ScalarTensor` 和 `WhereSelf`，新增 `npu_block_diffusion_attention` converter。该
converter 没有使用模糊算子别名，而是把整条序列转换为一个 causal 融合
Attention kernel；显式 autograd bridge 在 backward 调用
`npu_fusion_attention_grad`，避免裸算子缺少 Autograd dispatch 时静默丢梯度。

计算代理不再复现 prefix/canvas mask。`block_diffusion_reduced` 的 seq 256 作为一条
完整自回归序列进入 causal FlashAttention，不再拆成 Q/KV 长度为 192/192 和
64/256 的两个算子。forward、recompute 和 backward 每层各只有一个融合 Attention
边界。

复验发现外部 Zhanlu FlashAttention cost model 虽能找到融合算子模型，但不支持
四维 BSND 输入，导致 32 个 forward/recompute 和 16 个 grad 在模型内部断言后返回
零成本。为兼容该真实消费端，converter 进一步将 Q/K/V 从 BSND 仅作 metadata
reshape 后，以三维 BSH 调用同一个 NPU kernel，输出再恢复 BSND；这不改变 GQA 或
causal attention 数学。

reduced 每层输入统一为 `Q=[1,256,1024], K/V=[1,256,512]`。内部 CostModel 先按
全长 Attention 计算，再仅对 Attention FLOPs 乘 `attention_compute_alpha=0.8125`；
投影、MoE、Norm、loss 和优化器不参与折算。full 的默认 alpha 为 `0.94140625`。
alpha 通过模拟融合算子的 metadata 传给仓库内 CostModel；真实 Zhanlu 若忽略未知
metadata，需要在其汇总侧对 Attention 项后处理。真实命中率与折算结果仍需在其
运行镜像中重新执行并归档；本记录不以 meta 结果替代外部 cost-model 验收。

本轮单卡 reduced 复验捕获 1564 ops / 0 comm。8 层分别产生 8 个 original-forward、
8 个 recompute 和 8 个 backward 融合 Attention；每个算子均为上述 256-token BSH
shape、`sparse_mode=2`、`compute_alpha=0.8125`，单算子 FLOPs 从未折算的
268,435,456 降为 218,103,808。复验同时修正了 capture 在构建节点时丢弃 synthetic
op attrs 的问题，否则 alpha 虽出现在 JSON 中却不会进入成本计算。

为满足外部 FlashAttention cost model 的特征提取，融合算子同时显式携带
`head_num/num_heads=16`、`num_kv_heads=8`、`head_dim=64`、`input_layout/layout=BSH`、
`q_seq_len=kv_seq_len=256`、scale、causal window 和 sparse-mode 参数。真实 torch-npu
接口使用其原生拼写 `pre_tockens/next_tockens`；synthetic metadata 同时提供
`pre_tokens/next_tokens`，兼容下游成本模型的标准拼写。

## 回归测试

在容器内执行：

```bash
pytest -q \
  tests/unit_tests/models/test_block_diffusion.py \
  tests/unit_tests/simulator/capture/test_op_mapping.py \
  tests/unit_tests/simulator/cost/test_op_cost_model.py \
  tests/unit_tests/simulator/test_trainer.py
# 54 passed

pytest -q \
  tests/unit_tests/simulator/test_trainer.py \
  tests/unit_tests/simulator/capture/test_comm_group_resolver.py \
  tests/unit_tests/simulator/test_selective_ac.py
# 22 passed

pytest -q \
  tests/unit_tests/converters/test_moe_dispatch.py \
  tests/unit_tests/converters/test_model_custom_frame.py \
  tests/unit_tests/converters/test_registry.py
# 15 passed
```

额外执行的 simulator config-registry smoke suite 中，本次新增配置均可加载；suite
整体另有 4 个既有失败：测试期望 `ValueError`，当前 Tyro 将 DeepSeek V4 MXFP8
参数校验转换为 `SystemExit`。失败不经过 `block_diffusion` 代码路径，故记录为基线
回归项，没有在本次模型接入中改写异常语义。

`git diff --check` 和 Python `compileall` 通过。运行环境未安装 ruff、black、pyrefly，
因此这三项没有伪造通过记录。

## 能力边界与结论

结论：**Conditionally Ready（结构级/meta simulator 范围）**。

已满足原生 ModelSpec、配置注册、模型构建、参数公式、state-dict schema 闭环、
sliding-window corruption、full-causal Attention 计算代理、masked-only loss、CPU 前后向、核心
并行、PP 最终组合、AC 与内存/通信捕获。以下能力明确不属于本结论：

- 真实 NPU 数值训练与单步 loss/梯度参考对齐；
- 与未提供的权威训练实现对齐 mask-ratio 分布、loss weighting 和 timestep 条件；
- 多轮 confidence commit、causal boundary pass、跨 block KV cache；
- 未提供公开契约的 Hugging Face checkpoint/tokenizer 兼容性；
- EP 与 ETP 同时大于 1 的二维 expert mesh。

若要把结论提升为完整 Ready，需要先冻结 commit，再补充权威 checkpoint/数据与
loss 契约，在真实 NPU 上完成数值前后向、梯度和并行等价验收。
