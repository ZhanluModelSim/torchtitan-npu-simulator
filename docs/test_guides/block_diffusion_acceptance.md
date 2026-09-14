# Block Diffusion 接入验收记录

本文按[新模型最终验收规范](model_acceptance.md)记录 `block_diffusion` 的可复现
证据。验收对象是 simulator 的结构级/meta 训练接入；没有外推到原始 workload
未提供的 checkpoint 数值等价、真实 diffusion 训练目标或 DLM 推理调度。

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

单测同时覆盖四种 flavor、未知 flavor 拒绝、一个 canvas 的 sequence 约束、TP/CP
head 整除、expert degree 约束、双向 SDPA，以及 raw schema state-dict 双向 round
trip。

## 执行矩阵

`persistent/active/model` 均为 simulator 报告的峰值字节数。`debug` 为 2 层、8
experts、top-2；`reduced` 为 8 层、32 experts、top-4。

| 场景 | flavor / 并行 | 结果 | ops / comm | persistent / active / model peak |
| --- | --- | --- | --- | --- |
| 单卡 | debug | 通过 | 343 / 0 | 6,433,280 / 25,733,228 / 12,875,008 |
| FSDP2 | debug, DP shard=2 | 通过 | 436 / 12 | 3,216,640 / 9,650,028 / 5,652,352 |
| TP2 | debug | 通过 | 374 / 5 | 3,222,016 / 13,412,460 / 6,976,768 |
| EP2 | debug, expert domain=2 | 通过 | 521 / 24 | 3,216,640 / 9,650,028 / 5,143,872 |
| ETP2 | debug | 通过 | 374 / 5 | 3,222,016 / 13,412,460 / 6,976,768 |
| CP2 | reduced, Ulysses | 通过 | 2455 / 126 | 990,414,848 / 2,971,244,940 / 1,229,697,024 |
| AC full | debug | 通过 | 468 / 0 | 6,433,280 / 25,733,228 / 12,875,008 |
| PP2 | debug, 2 microbatches | 通过 | 两个 stage 各 5 个模板 | stage0 active 12,865,588；stage1 active 12,867,640 |
| 核心组合 | reduced, DP shard2 + TP2 + CP2 + EP2 + AC | 通过 | 3029 / 225 | 247,743,488 / 743,230,860 / 359,738,688 |
| 最终组合 | reduced, PP2 + DP shard2 + TP2 + CP2 + EP2 + AC | 通过 | stage0 1509 / 247；stage1 1526 / 312 | stage0 active 423,769,408；stage1 active 424,296,768 |
| 正式容量 | full 10.05T, DP shard64 + TP8 + EP512 + AC | 通过 | 30,636 / 1,173 | 78,580,647,424 / 235,741,946,940 / 83,416,681,984 |

最终组合的 logical world size 为 16，pipeline 两个 stage 分别持有 layers 0--3
和 layers 4--7，完成两个 microbatch 的 forward/backward 通信。TP/ETP 路径使用
仓库 grouped-mm bridge；CP 路径使用 Ulysses all-to-all；EP 路径包含 token
dispatch/combine collectives。EP 与 ETP 分别运行通过，但当前公共 ExpertParallel
不支持二维 expert mesh，因此 `EP > 1 && ETP > 1` 会在配置更新时明确 fail fast。

activation checkpoint 场景捕获到 124 个 recompute calls；reduced 核心组合捕获到
856 个 recompute calls。说明 AC 已进入实际反向重算路径，而不是只完成配置解析。
正式容量配置使用 logical world size 512，在约 35 秒内完成结构级 meta
forward/backward/recompute/optimizer 和内存报告导出；其中 forward、backward、
recompute 分别捕获 9,934、10,213、7,857 calls。该结果验证 10.05T 规格的结构和
并行账本可运行，不代表单个进程真实分配了 10.05T 权重。

## 回归测试

在容器内执行：

```bash
pytest -q \
  tests/unit_tests/models/test_block_diffusion.py \
  tests/unit_tests/simulator/capture/test_comm_events.py
# 38 passed

pytest -q \
  tests/unit_tests/simulator/test_trainer.py \
  tests/unit_tests/simulator/capture/test_comm_group_resolver.py \
  tests/unit_tests/simulator/test_selective_ac.py
# 22 passed

pytest -q \
  tests/unit_tests/converters/kernels/test_moe_dispatch.py \
  tests/unit_tests/models/test_model_custom_frame.py \
  tests/unit_tests/config_manager/test_registry.py
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

已满足原生 ModelSpec、配置注册、模型构建、参数公式、state-dict schema 闭环、CPU
前后向、核心并行、PP 最终组合、AC 与内存/通信捕获。以下能力明确不属于本结论：

- 真实 NPU 数值训练与单步 loss/梯度参考对齐；
- mask corruption、diffusion target 和 loss 的训练语义；
- 多轮 confidence commit、causal boundary pass、跨 block KV cache；
- 未提供公开契约的 Hugging Face checkpoint/tokenizer 兼容性；
- EP 与 ETP 同时大于 1 的二维 expert mesh。

若要把结论提升为完整 Ready，需要先冻结 commit，再补充权威 checkpoint/数据与
loss 契约，在真实 NPU 上完成数值前后向、梯度和并行等价验收。
