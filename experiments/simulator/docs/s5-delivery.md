# DeepSeek-V4 仿真交付与复现

> 历史验收记录：本文命令和输出对应当时源码。精度后缀 preset 已移除；当前按模型结构选择 preset，并以独立参数开启量化；验收脚本也使用显式 `--enable-quantized-training`。当前已移除 mem 以外的图导出、CSV rank 参数和 view 折叠；当前入口见 [接入指南](upstream-integration.md)，当前验证见 [mem 与精度审计](mem-only-precision.md)。

本交付使用固定的 TorchTitan v0.3.0 和 torchtitan-npu master 快照，在完整官方 CANN 环境中
执行 DeepSeek-V4（DSV4）的 Meta 前向、反向、梯度裁剪和优化器更新。真实 NPU 数值、
收敛、耗时和 allocator 峰值不属于这些 Meta 结果的证明范围。

S5 的 21 项全规模 Meta 验收已通过。逐项结果见能力矩阵、`s5-results.json` 和各次运行的 `acceptance.json`。

## 环境与安装

固定版本见 [迁移状态](migration-status.md)。CANN 使用完整官方镜像，镜像内应已安装
配套的 torch、torch-npu、TorchTitan、torchtitan-npu 和独立 torchao-npu。
在 `experiments/simulator/` 中安装 sidecar：

```bash
source /usr/local/Ascend/cann/set_env.sh
python -m pip install --no-deps --no-build-isolation .
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

该环境不要求可见的 NPU。simulator 通过显式入口启用，普通训练导入不激活 sidecar。

## 单次完整模型执行

以下命令保留 Pro 20T 的 96 层、12288 隐藏维和 2048 专家几何，启用原生 MXFP8、
MTP=1、Selective AC，逻辑 rank 15 承载组合并行捕获。输入为 B=1、S=512 的合成单文档；
EP 路由采用导出中明确记录的均衡假设。此命令不使用 `--build-only` 或缩减模型几何。

```bash
python -m torchtitan_simulator \
  --config deepseek_v4_pro_20t_baseline_mxfp8 \
  --simulation.world-size 16 \
  --simulation.logical-rank 15 \
  --parallelism.data-parallel-replicate-degree 2 \
  --parallelism.expert-parallel-degree 2 \
  --parallelism.tensor-parallel-degree 2 \
  --parallelism.context-parallel-degree 2 \
  --training.local-batch-size 1 \
  --training.seq-len 512 \
  --training.num-mtp-modules 1 \
  --activation-checkpoint.mode selective \
  --simulation.csv-max-ranks 1 \
  --simulation.output-formats text csv mem \
  --simulation.output-dir ./pro_20t_output
```

20T 是旧 simulator 几何的原生配置树适配，不是新主线提供的官方 preset。
旧配置的压缩比 `0`（不建立压缩流）映射为新主线的 `1`；20T 最后一层保留窗口注意力语义。
smoketest 使用目标原生 debugmodel；其参数规模与旧版 smoketest 不同。

## 七个 preset 的验收入口

从 sidecar 根目录执行：

```bash
python scripts/acceptance.py --output-dir ./dsv4_acceptance
```

默认顺序执行 7 个 preset × AC none/full/selective，共 21 个独立子进程。覆盖 smoketest，
Flash/Pro/Pro 20T 的 BF16/MXFP8，均为 MTP=1、B=1、S=512、组合并行、逻辑 rank 15。
smoketest 输出六种格式；大模型输出 text/csv/mem，限制 CSV 展示 rank 数，避免按逻辑 world
复制所有文件。父进程在每项结束后写入 `manifest.json`，子进程日志、可复现命令和
`acceptance.json` 保留。任何失败都会使入口最终返回非零状态。输出目录必须不存在，避免旧结果混入。

只验收指定配置时可以使用：

```bash
python scripts/acceptance.py \
  --presets deepseek_v4_flash_baseline_bf16 deepseek_v4_flash_baseline_mxfp8 \
  --modes none full selective \
  --output-dir ./flash_acceptance
```

每项检查 L0 跨模板 DAG、L1 标识与 L2/L3 引用、DataSlot 双向引用、通信源唯一归属、
实际梯度及 optimizer 更新集合、保存张量的真实来源、FSDP 释放，以及按生命周期区间独立求得的
逻辑内存峰值。运行报告还保留原生几何、参数 local/global 布局、量化参数数量、实际算子计数、
捕获进程峰值 RSS、捕获及导出耗时和导出文件哈希。RSS 来自 `RUSAGE_SELF`，不含 JIT 编译子进程或整个 VM。
这些主机资源指标不是模型设备内存或 NPU 性能。

## 输出契约与旧命令迁移

| 旧用法或含义 | 新用法或解释 |
| --- | --- |
| `torchtitan_npu.simulator` 模块入口 | `python -m torchtitan_simulator` 或 `torchtitan-simulate`；先独立安装 sidecar |
| 七个 DSV4 preset 名称 | `--config deepseek_v4_smoketest` 或 `deepseek_v4_{flash,pro,pro_20t}_baseline_{bf16,mxfp8}` |
| 旧平铺模型字段 | `--model-overrides.*`，按完整原生配置树重建；未迁移字段明确报错 |
| 旧 MXFP8 FQN 复写 | 移除；使用原生 `all_mxfp8` recipe，实际选择结果见 JSON 的 quantization_configuration |
| 构建诊断 | `--build-only`，不代表图、梯度或内存验收 |
| CSV rank 展开 | `--simulation.csv-max-ranks N`，默认 1；仅限制展示，不修改 world 或 mesh |
| 参数存储 dtype | `--simulation.memory-parameter-storage-dtype bfloat16`，仅投影本地常驻 shard 字节 |
| 保存激活卸载 | `--simulation.memory-offload-ac-saved-tensors`，驻留假设，不执行设备卸载或预测传输性能 |
| GBS/GA | 原生批配置保留；只捕获一个 microbatch，不声称完整累积训练峰值 |
| PP、FP8 AG/A2A、A5 | 入口明确拒绝；不隐式回退到其他路径 |

`export_manifest.json` 的 schema_version=2 标明输出范围。JSON 字段名沿用既有结构；
`step_templates` 的 key 与 `StepGraph.step_id` 现统一为稳定的 `s0_F`、`s0_B`、`s0_OPTIMIZER`，
消费者应使用它们关联 `template_ref`，不再使用随机 step_id。
计划层 `capture_process_rank` 表示捕获 worker，`logical_rank`／action 的 `rank` 表示模拟逻辑 rank；
单 worker 捕获 rank 15 时分别为 0 和 15。早期审计导出中计划层曾混用这两个字段，读取这些历史数据时应以
`capture_coordinates` 为准；该标注修正不改变已有计算事件、mesh、梯度或内存区间。
DOT 保留跨模板边；metadata view 的省略仍只收缩别名路径，不删除其生产者与消费者之间的依赖。

`schedule_plan.csv` 是权威 SchedulePlan 的导出，包含 Actions 和以 `# DataSlots` 分隔的
DataSlots 两段；消费时不能把第二段当作 action 行。L0 CSV 和 kernel summary 用于算子明细，
不能据其行顺序另推调度。kernel summary 的其他 rank 行是同一个代表性捕获模板的展示投影，
不代表逐 rank 独立执行；实际捕获坐标以 manifest/JSON 为准。

`memory/memory_summary.json` 的峰值为逻辑活跃 storage。保存槽数、去重保存 storage 数、
offload inventory 累计字节均不等于峰值；allocator reserved/cache、碎片和未建模 workspace 不计入。
未知算子成本保留显式标记，不以零值冒充已知硬件耗时。

## 质量检查与遗留项

```bash
TORCHTITAN_SOURCE=/path/to/torchtitan \
TORCHTITAN_NPU_SOURCE=/path/to/torchtitan-npu \
bash scripts/validate.sh
```

完整仓库 pre-commit 与 sidecar 的独立类型检查分别记录。既有全仓类型错误或主机工具限制不能
写成全仓 CI 通过。实际命令、运行结果和安装镜像证据见能力矩阵与 `s5-results.json`。

CANN metadata 是否恢复普通 dispatcher 入口已独立记录在 [遗留问题](open-questions.md)。
这条路径暂未调整，不能把 metadata 准备阶段的成本计作已经捕获。
