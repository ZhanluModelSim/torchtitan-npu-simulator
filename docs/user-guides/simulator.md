# Simulator 使用指南

> 现状核对：2026-10-07，`feat/npu-simulator`，实现基线 `ecde2f0`。依据源码与 Git 历史静态核对，未执行测试或仿真；后续行为以当前源码为准。

Simulator 复用真实 TorchTitan/NPU 模型与并行化，在 meta device 上捕获一次 forward/backward/optimizer，导出 L0–L3 图、调度和逻辑内存。完整逻辑 world 通过代表进程、模板和 RankTable 展开；不为每张逻辑卡单独运行数值训练。

## 阅读入口

| 需求 | 文档 |
|---|---|
| 启动、环境和基本输出 | 本文 |
| 当前模型、精度和并行支持边界 | [模型与特性矩阵](simulator/model-matrix.md) |
| 核心开关、SAC、FP8、offload、模型覆盖 | [配置参考](simulator/configuration.md) |
| 输出目录、CSV/JSON 和内存字段 | [输出与内存口径](simulator/outputs.md) |
| 完整执行路径、四层 IR、插件、通信归属 | [当前架构](../design/simulator-architecture.md) |
| 历史问题、根因与已合入优化 | [历史修复与当前边界](../design/simulator-fixes-and-limits.md) |
| 当前契约与原始方案的阅读关系 | [文档导航与历史方案索引](simulator/documentation-map.md) |

## 环境前提

无需真实 NPU 硬件，但需要 torch_npu 的 meta kernel 注册与 CANN 动态库。Python 图对象、CPU mesh、tokenizer/dataloader 仍使用主机资源。依赖以 [requirements.txt](../../requirements.txt) 为准：当前锁定 torch `2.12.0+cpu`、torch_npu `2.12.0rc1`、torchao `0.17.0`、triton-ascend `3.2.1` 和 torchtitan `ac13e536c84e7f6647b14fa9375c3c8a8a2b8578`。

仓库 [Dockerfile](../../Dockerfile) 使用 CANN `9.1.0-beta.1-950` 基础镜像。安装方式参见[安装指南](installation.md)；若使用已有镜像，须核对安装代码与挂载工作区是否对应当前分支，不能把历史镜像版本视作当前源码版本。

进入已有 CANN 环境后按其安装位置加载环境，例如：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

本文命令按当前源码静态核对，本次文档整理未实际运行，也不记录新的环境验收结果。

## 快速开始

从仓库根目录执行。launcher 会先解析最终 CLI，再按 PP degree 选择模式和真实进程数。PP=1 使用 `fake_backend`；PP>1 使用 `multi_proc_meta`，真实进程数等于 PP。

DeepSeek V4 小规模配置：

```bash
python3 scripts/run_simulator_spawn.py \
    --config deepseek_v4_smoketest \
    --simulation.world-size 8 \
    --hf-assets-path ./tests/assets/tokenizer/deepseekv3_tokenizer \
    --simulation.output-formats text html csv mem
```

Kimi K3 小规模配置：

```bash
python3 scripts/run_simulator_spawn.py \
    --config kimi_k3_smoketest \
    --simulation.world-size 8 \
    --hf-assets-path ./tests/assets/tokenizer/deepseekv3_tokenizer \
    --simulation.output-formats text html csv mem
```

DeepSeek PP 示例使用 4 层 smoketest、2 个 PP 进程和 4 个 microbatch：

```bash
python3 scripts/run_simulator_spawn.py \
    --config deepseek_v4_smoketest \
    --simulation.world-size 8 \
    --parallelism.pipeline-parallel-degree 2 \
    --parallelism.pipeline-parallel-microbatch-size 1 \
    --training.local-batch-size 4 \
    --training.num-mtp-modules 0 \
    --hf-assets-path ./tests/assets/tokenizer/deepseekv3_tokenizer \
    --simulation.output-formats text csv mem
```

Token IDs 仍由真实 dataloader/tokenizer 产生。仓库跟踪了 `deepseekv3_tokenizer` 资产；部分 DeepSeek preset 默认的 V4 tokenizer 路径未随仓库提供，可通过上述路径覆盖以捕获 meta 结构。此做法不构成真实训练 tokenizer 或数值行为的等价性验证。

正式 baseline 配置见[模型矩阵](simulator/model-matrix.md)。它们默认 `DP_shard=-1`，需要显式 world_size 来源；默认 EP、batch 和模型约束必须一起满足。DeepSeek baseline 默认 MTP=1，开启 PP 时须显式改为 0。Kimi 当前不支持 PP。

PP=1 也可直接进入模块：

```bash
NGPU=8 python3 -m torchtitan_npu.entry \
    --module torchtitan_npu.simulator \
    --config deepseek_v4_smoketest \
    --hf-assets-path ./tests/assets/tokenizer/deepseekv3_tokenizer \
    --simulation.output-formats text
```

不要将 `training.steps` 解释为捕获 iteration 数；当前 simulator 始终捕获单 step。

## 输出与排查

preset 默认写到 `simulator_output/<preset>/`。PP 模式每个 capture process 写入 `rank_N/`，不自动合并。当前 `output_formats=[]`；内存跟踪默认开启，因此默认仅输出内存 summary，完整产物需显式选择格式。

常见排查顺序：确认当前 registry 中有配置工厂、tokenizer 路径可读、world/EP/TP/CP 整除关系正确、模型支持所选并行方式、CANN 动态库可加载。需要详细内存时选择 `mem`；需要调度关系时选择 `csv` 或 `json`。部分 `inspect_*` 脚本会启动捕获，不应把其名称理解为只读操作。

数值一致性、真实设备时延与 allocator/workspace 开销不由 meta 图证明。内存 summary、保存激活清单和单算子成本的口径见[输出说明](simulator/outputs.md)。原始需求与设计不在本次整理中编辑或移除，历史阅读关系见[文档导航](simulator/documentation-map.md)。
