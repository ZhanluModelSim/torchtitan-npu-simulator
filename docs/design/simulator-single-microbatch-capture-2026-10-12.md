# 非 PP 单 microbatch 捕获

2026-10-12。用户明确要求先处理非 PP，PP 的执行次数优化另行处理。以下行为只在 `parallel_dims.pp_enabled=False` 时生效。

## 行为

非 PP 固定读取一个本地批次，执行一次 F/B、梯度裁剪、optimizer 和 LR 更新。原始算子、内存事件、L1 图、L2 执行实例和 L3 iteration 都只覆盖一个 MB，不随配置 GA 展开。模板名保持 `s0_F`、`s0_B`、`s0_OPTIMIZER`。

原配置不改写；SchedulePlan、ScheduleGraph 和内存 summary 的 `capture` 元数据保存：

- `scope=single_microbatch`、`captured_microbatches=1`。
- `configured_microbatches` / `configured_gradient_accumulation`：原始 GA。
- `configured_global_batch_size` / `configured_local_batch_size`：原始训练配置。
- `microbatch_batch_size`：实际捕获的本地批次大小。

下游据此折算多个 MB 的 F/B 时间，optimizer 仍是一次。内存值是单 MB 加 optimizer 的逻辑内存。独立调用 `run_simulation_step` 时只使用 `microbatches[0]`；显式 token count 必须对应这一个 MB。

PP 恢复并保留原调度，不截断成 MB0，不改 dataloader 大小，不强制所有 rank/stage 只有一份模板。用户已有 `pp_loss_normalize.py` patch 原样保留且不纳入本次提交。此前试验性的 PP 单 MB helper 及对应测试已撤回，PP 的进一步执行优化暂停。

## 验证

现有 `gmm-fix-validation` 容器、当前工作区、同一依赖环境。20T BF16、8192 logical cards、EP128、FSDP1024、DP replicate8、local batch1、seq4096、full AC、MTP1，顺序运行，无并发测试。

| 配置 GA | GBS | 仿真运行时间 | Python RSS 峰值 | 捕获算子数 | 通信事件数 | 原始内存事件数 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 8192 | 15.75s | 1.870GiB | 84802 | 1666 | 152407 |
| 64 | 524288 | 15.69s | 1.870GiB | 84802 | 1666 | 152407 |

时间来自相同 RSS probe，覆盖 Python 导入/初始化、捕获、建图、估算与导出，不包含外层 Docker exec 和加载 CANN shell。两次单次对照不是统计均值。两份 memory_summary 仅 `capture` 内的原配置字段不同，其余所有字段完全一致。

DeepSeek smoketest 的非 PP GA=1/4/64 同样分别只捕获5636个原始内存事件，逻辑峰值216361552 bytes，仿真运行约3.3–3.7s。

CPU/meta、配置 GA=1/2/3、内存跟踪开/关：一次 F/B/O、三个模板、一份原始 MB 事件。CPU loss、梯度和参数更新与只执行首个批次的原生参考实现逐位一致；梯度范数和一次性清梯度/裁剪/更新也有对应测试。本验证不再宣称等价于完整 GBS 的真实梯度累积训练。

`tests/unit_tests/simulator tests/smoke_tests/simulator`：533 passed、1 skipped、3 failed。三项失败已在未修改的基线复现（GMM helper 返回值解包、旧 P2P role 测试、旧 LiLoss placeholder 测试）。全量 `tests/ -x` 的现有阻塞仍为 CPU-only 环境的 Qwen3 save/load CUDA 初始化失败。无真实 NPU 数值验收。

运行产物位于 `simulator_output/deepseek_v4_20t_timing_20261012/nonpp_20t_ga{1,64}*`，以及 `singlemb_ga_{1,4,64}*`。该目录是本地未跟踪产物，不提交运行日志。

## 代码检视

按 torchtitan-npu-code-reviewer 检查：读取根目录及 `.agents/AGENTS.md`；修改为 simulator 与文档/单测，没有匹配生产专项路径。保留 storage 来源验证和算子依赖；集合索引、schema 缓存、capture 索引释放只改变捕获效率。审计模板 builder、synthetic comm、configured MB helper 的所有内部调用点。PP 单 MB 执行和 virtual-stage 验证器的试验修改已撤回。修正新增的 optional schedule_plan 类型诊断，未发现新增明确 S1/S2 问题。

`git diff --check` 通过。局部 pre-commit 已执行，但被仓库已有 Ruff / 全仓 pyrefly 报错阻塞；未为通过检查而重排用户 PP patch 或修复无关生产模块。全量 `pre-commit run --all-files` 已在隔离源码副本执行，exit=1，仍有仓库已有 Ruff / pyrefly 报错；没有为通过检查而修改无关模块。用户在知悉这些检查限制后明确要求先提交非 PP 修复。
