# 调试支持特性

torchtitan-npu 目前提供多种调试特性支持，帮助开发者定位分布式训练中的各类问题，包括内存问题和性能瓶颈等。以下是常见使用场景和对应功能的快速参考：

| 使用场景 | 对应功能 |
|---------|---------|
| 分析 OOM 和内存泄漏 | [Memory Snapshot](#memory-snapshot) |
| 定位性能瓶颈和优化性能 | [Profiling](#profiling) |

---

## Memory Snapshot

内存快照功能用于捕获和记录训练过程中的内存使用情况，包括内存分配、显存占用、张量生命周期等信息。通过命令行参数进行定时内存快照收集，本功能生成的`.pickle`格式内存快照文件可通过[memory_viz](https://docs.pytorch.org/memory_viz)工具进行解析和可视化查看。

### 使用场景

- 训练过程中出现 OOM（Out of Memory）错误，需要分析内存占用情况
- 怀疑存在内存泄漏，需要追踪内存分配和释放情况
- 需要优化显存使用，了解框架不同模块的内存占用

### 配置选项

torchtitan 原生提供内存快照功能，使用以下 CLI 参数配置：

| CLI 参数 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `--profiler.enable-memory-snapshot` | bool | false | 是否启用内存快照功能。 |
| `--profiler.save-memory-snapshot-folder` | str | "profiling/memory_snapshot" | 内存快照文件保存目录。 |
| `--profiler.profile-freq` | int | 10 | 每隔多少个训练步骤收集一次内存快照。 |

torchtitan 原生内存快照功能会按照 `--profiler.profile-freq` 指定的频率定期收集内存快照，并在发生 OOM 错误时自动转储当前内存快照。收集到的内存快照将保存到 `--profiler.save-memory-snapshot-folder` 指定的目录中。

### 配置示例

通过训练命令的分层 CLI 参数启用内存快照：

```bash
torchrun --nproc_per_node=2 -m torchtitan.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_debugmodel \
  --profiler.enable-memory-snapshot \
  --profiler.save-memory-snapshot-folder profiling/memory_snapshot \
  --profiler.profile-freq 10
```

---

## Profiling

性能分析是优化训练性能的关键工具。torchtitan-npu 对性能分析功能进行了 NPU 适配，支持详细的性能数据收集和分析。系统使用 `torch_npu.profiler` 提供的原生性能分析器，能够追踪 CPU 和 NPU 的活动，记录内存使用情况、调用栈信息、张量形状等详细数据，并提供 AI 算力利用率指标。

当前分支使用时需通过 CLI 的 `--override.imports` 参数启用
`torchtitan_npu.override.common.profiler.cann`。

### 使用场景

- 需要分析训练过程中的性能瓶颈
- 需要对比不同配置或优化方案的性能表现
- 需要定位训练过程中的性能异常或退化

### 配置选项

性能分析配置使用 TorchTitan 的分层 CLI 参数，并通过
`--override.imports` 启用 CANN profiler。rank 筛选和 CANN 专属选项
通过 override 条目附带的 JSON 参数传入。

#### torchtitan 原生 CLI 配置选项

| CLI 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--profiler.enable-profiling` | bool | false | 是否启用性能分析功能。 |
| `--profiler.save-traces-folder` | str | "profiling/traces" | 性能分析结果的保存目录路径。 |
| `--profiler.profile-freq` | int | 10 | 周期模式下每隔多少步采集一次。 |
| `--profiler.profiler-warmup` | int | 3 | 性能分析器的预热步数。 |
| `--profiler.profiler-active` | int | 1 | 性能分析器的采集步数。 |
| `--profiler.profiler-repeat` | int | null | 周期模式重复采集的次数；设置为 `1` 时采集一次后停止。 |
| `--profiler.profiler-skip-first` | int | 0 | 开始第一个采集周期前跳过的训练步数。 |

CLI 中的布尔参数是开关形式：启用时直接写参数名，不要在后面追加
`true`；需要关闭已启用的布尔项时使用对应的 `--profiler.no-...` 参数。

#### torchtitan-npu override 扩展选项

下表中的字段名是 override JSON 键名，通过 `--override.imports` 参数传入。

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `profile_ranks` | list[int] | [-1] | 需要进行性能分析的 rank 列表，例如 [0, 1, 2]。使用 [-1] 表示对所有 rank 进行分析。 |
| `profile_with_memory` | bool | false | 是否在性能分析期间记录内存使用情况。 |
| `profile_with_stack` | bool | false | 是否在性能分析期间记录调用栈信息。 |
| `enable_online_parse` | bool | true | 是否启用性能分析数据的在线解析。设置为 `false` 时仅落盘原始数据，需在训练结束后离线解析（见[离线解析](#离线解析)）。 |

当前分支的对应情况如下：

| 配置项 | 当前分支对应方式 |
| --- | --- |
| `profile_ranks` | 只在指定 rank 上启动 profiler。`[-1]` 表示所有 rank，未选中的 rank 正常训练但不采集。 |
| `profile_with_memory` | 通过 override 参数控制，默认值为 `false`。 |
| `profile_with_stack` | 通过 override 参数控制，默认值为 `false`。 |
| `enable_online_parse` | 通过 override 参数控制，默认值为 `true`。 |

#### 离线解析

当 `enable_online_parse=False` 时，性能分析仅将原始数据转储到
`save_traces_folder` 指定的目录，不进行在线解析。
训练结束后，在 `save_traces_folder/profiling_data/` 目录下会生成以 `{hostname}_{pid}_{timestamp}_ascend_pt` 命名的子目录，包含原始 profiling 数据。多 rank 场景下，每个 rank 会生成独立的 `*_ascend_pt` 子目录。使用 `scripts/parse_profiling_data.py` 对其进行离线解析：

```bash
# 解析单个 *_ascend_pt 目录
python3 scripts/parse_profiling_data.py path/to/xxx_ascend_pt

# 解析父目录下所有 *_ascend_pt（支持多 rank 场景）
python3 scripts/parse_profiling_data.py path/to/save_traces_folder
```

脚本接受单个 `*_ascend_pt` 目录或其父目录（自动扫描 `profiling_data/*_ascend_pt` 和 `*_ascend_pt` 两种布局）。解析完成后，会在每个 `*_ascend_pt/ASCEND_PROFILER_OUTPUT/` 子目录下生成以下文件：

- `kernel_details.csv`：NPU kernel 级别的耗时统计
- `api_statistic.csv`：PyTorch API 调用统计
- `ascend_pytorch_profiler_0.db`：可用 MindStudio Insight 或 Chrome Tracing 打开的 SQLite 格式 trace
- `trace_view.json`：完整的 trace 视图数据

### 配置示例

使用 CLI 启用 CANN profiler，并配置原生的周期采集模式：

```bash
torchrun --nproc_per_node=2 -m torchtitan.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_debugmodel \
  --profiler.enable-profiling \
  --profiler.save-traces-folder profiling/traces \
  --profiler.profile-freq 10 \
  --profiler.profiler-warmup 3 \
  --profiler.profiler-active 1 \
  --override.imports \
  torchtitan_npu.override.common.profiler.cann
```

按绝对步数和 rank 采集时，原生配置仍使用 CLI 参数，CANN 扩展选项作为
`cann` override 的 JSON 参数传入：

```bash
torchrun --nproc_per_node=2 -m torchtitan.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_debugmodel \
  --profiler.enable-profiling \
  --profiler.save-traces-folder profile_traces \
  --profiler.profile-freq 4 \
  --profiler.profiler-warmup 3 \
  --profiler.profiler-active 1 \
  --profiler.profiler-repeat 1 \
  --profiler.profiler-skip-first 1 \
  --override.imports \
  'torchtitan_npu.override.common.profiler.cann={"profile_ranks":[0],"profile_with_memory":true,"profile_with_stack":false,"enable_online_parse":true}'
```

上述配置只在 rank 0 采集第 5 步。若使用
`scripts/run_train.sh`，应将 profiler override 与脚本中已有的 override 统一放入
同一个 `--override.imports` CLI 参数中；不要重复传入该参数，因为它只保留最后一次
出现的值。

使用仓库脚本时，性能分析默认关闭：脚本默认传入 `--profiler.no-enable-profiling`，
需要手动设置环境变量 `ENABLE_PROFILING=1` 开启（此时脚本自动传入
`--profiler.enable-profiling`）。开启后可以通过环境变量让脚本自动把绝对步数转换
为原生 profiler 参数，并将 CANN 配置追加到现有 override 列表：

```bash
# 默认关闭（脚本传入 --profiler.no-enable-profiling），无需额外参数：
NGPU=2 ./scripts/run_train.sh

# 手动开启并按绝对步数与 rank 采集：
ENABLE_PROFILING=1 PROFILE_START=5 PROFILE_END=6 PROFILE_WARMUP=3 \
PROFILER_OVERRIDE='torchtitan_npu.override.common.profiler.cann={"profile_ranks":[0],"profile_with_memory":true,"profile_with_stack":false,"enable_online_parse":true}' \
NGPU=2 ./scripts/run_train.sh \
  --profiler.save-traces-folder profile_traces
```
