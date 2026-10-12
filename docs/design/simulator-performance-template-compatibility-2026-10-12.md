# Simulator 性能与多 microbatch 模板兼容性修复

本文记录第一阶段的性能与兼容性验收。随后用户明确区分 PP 与非 PP：非 PP 的执行及内存固定为一个 MB，PP 保留原流水线策略。下文的非 PP 逐 MB 内存捕获是当时行为，当前语义见 [非 PP 单 microbatch 捕获](simulator-single-microbatch-capture-2026-10-12.md)。

## 问题与根因

2026-10-12，在现有 `gmm-fix-validation` 容器、当前挂载工作区上验证。对照版本为 `ab77fa4`，通过 Git archive 放在容器 `/tmp/dsv4-before-opt`，没有切换或清理当前工作区。用户已有 `pp_loss_normalize.py` patch 和对应未跟踪单测原样保留。

`0a3dc00` 为持久参数排除增加了真实 storage 来源验证，但每个输入/输出都线性扫描参数列表。20T 配置采样得到 169689 次检查、476995779 次逐项比较，内存估算约 10 秒。该复杂度随模型参数张量数量增加，不能用恢复旧版按 shape 猜参数身份来规避。

同一提交还在非 PP 梯度累积中为每个 MB 生成 `s0_F_mb0`、`s0_B_mb0` 等 L1 图，破坏原有下游模板命名契约。实际梯度累积与保存激活需要逐批执行，但这不要求逐批保留 L1 计算图。

## 实现

- 参数 storage key 在每次估算时建立集合，一次查询代替扫描完整参数列表；保留来源及别名的正确性判定。
- 缓存不可变算子 schema 的 scalar argument 描述，不缓存实际参数值。
- 内部 `_RawEvent` 使用 slots；L0 节点建好后释放 capture 的原始算子列表及不再需要的索引，不释放内存事件。
- DAG 检查使用 deque；同步通信后继扫描不复制节点列表后缀。
- PP 重复 chunk 在执行算子后直接跳过 capture 的元数据处理，保持原有代表 microbatch 配对逻辑。
- 非 PP 的首个 microbatch 生成一份 F/B 模板。后续 MB 只记录实际内存/保存信息和轻量执行跨度；L2 实例引用相同 `s0_F` / `s0_B`，优化器仍为单次 `s0_OPTIMIZER`。不增加公共配置开关。
- 合成通信返回实际捕获的 op ID，以便未保留 L1 节点的内存事件仍能对应各次通信。

原始内存跟踪仍随实际 GA 执行数量增长；这里只消除重复计算图与不必要的捕获元数据。PP 的通信状态变体 `__comm_vN` 保留，未按 MB 新建数学计算模板。PP 多 MB 加 optimizer 的完整内存回放支持限制未改变。

## 性能与实际主机内存

同一容器、同一依赖、顺序运行，未并发执行其他测试。配置为 DeepSeek V4 20T BF16、EP128、FSDP1024、local batch1、seq4096、full AC、MTP1。8192卡为 DP replicate8 / GBS8192；65536卡为 DP replicate64 / GBS65536。只输出默认内存 summary。

每50ms采集容器 cgroup memory.current / anon / file 与 Python RSS，使用 Linux ru_maxrss 记录精确进程高水位。表中的时间为同一 probe 启动到 launcher 完成的单次端到端对照，包含 Python 导入/初始化、捕获、建图、内存估算与输出，不包含外层 Docker exec 和加载 CANN shell 的启动开销；不是重复测量的统计均值。

| logical world | 优化前时间 | 优化后时间 | 时间减少 | 优化前进程 RSS 峰值 | 优化后进程 RSS 峰值 |
|---|---:|---:|---:|---:|---:|
| 8192 | 25.53s | 15.62s | 38.8% | 1.983GiB | 1.872GiB |
| 65536 | 29.52s | 19.86s | 32.7% | 2.223GiB | 2.113GiB |

8192卡内存估算阶段由 9.70s 降为 0.82s；65536卡由 10.18s 降为 0.83s。两组 memory_summary.json 与优化前逐字段完全一致，包括逻辑峰值、各阶段峰值、分类计数、保存激活清单与说明。

原始文件位于 `simulator_output/deepseek_v4_20t_timing_20261012/`：`baseline_w8192.log`、`optimized_w8192.log`、`baseline_w65536.log`、`optimized_w65536.log`，以及对应 `*_rss.json`。该目录为本地未跟踪运行产物。

## 模板与数值验证

- GA=1/2/3、CPU/meta、内存跟踪开/关：始终只有 `s0_F`、`s0_B` 和 `s0_OPTIMIZER` 三份模板，每个 MB 的 L2 F/B 动作引用统一模板。CPU loss、累积梯度、更新后的参数与原生 AdamW 参考逐位一致；已有 clip grad norm 测试也核对参考范数与参数。
- DeepSeek smoketest、world8、GBS32（GA4）：导出三个统一模板，计算图仍为3002个节点，与GA1相同；与优化前 GA4 的 memory summary 逐字段一致。
- 同配置增加 EP2：优化前9份带MB后缀模板，优化后只有3份 canonical 模板；memory summary 仍逐字段一致。
- PP2、4个microbatch：两个 worker 正常结束，保留代表MB计算模板和必要通信变体。该验收关闭内存跟踪，因为完整多MB优化器内存回放原本显式不支持。
- `tests/unit_tests/simulator tests/smoke_tests/simulator`：533 passed、1 skipped、3 failed；3个失败均在未修改的 `ab77fa4` archive 中独立复现，分别是 GMM helper 返回值解包、旧 P2P role 构造用例、旧 LiLoss placeholder 用例。
- `tests/ -x`：9 passed、29 skipped，停止于现有 Qwen3 save/load 的 CPU-only 环境 CUDA 初始化错误。
- `pre-commit run --files ...` 在隔离源码副本执行，避免 hook 批量重排当前文件及触碰用户PP patch；仍被已有 Ruff/全仓 pyrefly 错误阻塞。新引入的 optional TensorMeta 类型诊断已修复；`git diff --check` 通过。
- 未运行真实 NPU 训练；上述数值一致性为 CPU 测试证据，meta 仿真不提供真实 NPU loss/grad norm 证明。

## 代码检视检查结果

未发现新增明确、可验证的 S1/S2 问题。

- 项目规范：读取根目录及 `.agents/AGENTS.md`。
- 专项规则：修改路径位于 simulator，未匹配生产 model/converter/patch/config 专项路径；参考分布式规则核对 PP、DP/FSDP、EP、TP 的调用边界。
- 设计与架构：复用现有 L0/L1 模板、L2 ActionSpec、内存事件及 estimator，不引入模型专属 wrapper 或实验开关。
- 上游交互：不修改上游源码，不改变生产训练入口、模型注册、checkpoint key 或 tensor placement。
- 兼容性：审计 build_step_graphs / build_schedule_plan / record_synthetic_op 的调用点，移除按MB后缀命名路径，保持已有 canonical 模板及 CSV schema。
- 验证范围：CPU/meta 回归、两种 logical world 的20T容器仿真、多MB导出与PP多进程冒烟、基线失败复现、局部 pre-commit 与 diff检查。

本节保留第一阶段的验收记录；当前非 PP 捕获语义见单 microbatch 报告。
