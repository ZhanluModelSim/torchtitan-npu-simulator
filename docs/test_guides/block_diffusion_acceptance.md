# Block Diffusion 接入验收记录

本文按[新模型最终验收规范](model_acceptance.md)记录 `block_diffusion` 的可复现
证据。验收对象包含 prefix/canvas attention、sliding-window corruption、
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

正式训练输入为 4096 tokens：前 3840 tokens 使用 causal attention，最后 256-token
canvas 对完整输入双向可见。单测同时覆盖四种 flavor、未知 flavor 拒绝、block
aligned sequence 约束、显式参考 mask 等价、masked-only corruption/loss、TP/CP
head 整除、expert degree 约束，以及 raw schema state-dict 双向 round trip。
此外，使用真实 `c4_test` dataloader batch 在 CPU 上完成 debug MoE 数值前向、
masked-only loss 和反向：输出 shape 为 `[1, 32, 2048]`，15 个有效目标，所有
trainable parameters 均获得梯度。

## 执行矩阵

`persistent/active/model` 均为 simulator 报告的峰值字节数。`debug` 为 2 层、8
experts、top-2；`reduced` 为 8 层、32 experts、top-4。

| 场景 | flavor / 并行 | 结果 | ops / comm | persistent / active / model peak |
| --- | --- | --- | --- | --- |
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
converter 没有使用模糊算子别名，而是把 prefix 和 canvas 分别保留为两个融合
Attention kernel；显式 autograd bridge 在 backward 调用
`npu_fusion_attention_grad`，避免裸算子缺少 Autograd dispatch 时静默丢梯度。

单卡 `block_diffusion_reduced`（8 layers、seq 256、block 64、full AC）的本仓库
meta 复验捕获到：原始 forward 16 个融合 Attention，recompute 16 个融合
Attention，backward 16 个融合 Attention grad。Attention 路径中不再出现 SDPA、
`Tril`、`WhereSelf` 或 mask 构造算子；剩余 forward softmax 为每层一个 MoE router
softmax，另有 loss 的 log-softmax，不属于 Attention 分解。该次结构复验捕获
1660 ops / 0 comm，较融合前的未剪枝算子数量不可直接作性能比较。

内部 CostModel 按 layout 读取 key sequence length：prefix 使用 `P x P`，canvas
使用 `B x S`，避免把 BSND 的 head 维误当成 sequence，也避免把矩形 canvas
Attention 错算成 `B x B`。真实 Zhanlu 的算子命中率仍需在其运行镜像中重新执行
并归档；本记录不以 meta 结果替代 Zhanlu cost-model 验收。

## 回归测试

在容器内执行：

```bash
pytest -q \
  tests/unit_tests/models/test_block_diffusion.py \
  tests/unit_tests/simulator/capture/test_op_mapping.py \
  tests/unit_tests/simulator/cost/test_op_cost_model.py \
  tests/unit_tests/simulator/test_trainer.py
# 50 passed

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
sliding-window corruption、prefix/canvas mask、masked-only loss、CPU 前后向、核心
并行、PP 最终组合、AC 与内存/通信捕获。以下能力明确不属于本结论：

- 真实 NPU 数值训练与单步 loss/梯度参考对齐；
- 与未提供的权威训练实现对齐 mask-ratio 分布、loss weighting 和 timestep 条件；
- 多轮 confidence commit、causal boundary pass、跨 block KV cache；
- 未提供公开契约的 Hugging Face checkpoint/tokenizer 兼容性；
- EP 与 ETP 同时大于 1 的二维 expert mesh。

若要把结论提升为完整 Ready，需要先冻结 commit，再补充权威 checkpoint/数据与
loss 契约，在真实 NPU 上完成数值前后向、梯度和并行等价验收。
