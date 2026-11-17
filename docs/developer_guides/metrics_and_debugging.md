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
torchrun --nproc_per_node=2 -m torchtitan_npu.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_debugmodel \
  --profiler.enable-memory-snapshot \
  --profiler.save-memory-snapshot-folder profiling/memory_snapshot \
  --profiler.profile-freq 10
```

---

## Profiling

性能分析是优化训练性能的关键工具。torchtitan-npu 对性能分析功能进行了 NPU 适配，支持详细的性能数据收集和分析。系统使用 `torch_npu.profiler` 提供的原生性能分析器，能够追踪 CPU 和 NPU 的活动，记录内存使用情况、调用栈信息、张量形状等详细数据，并提供 AI 算力利用率指标。

### 使用场景

- 需要分析训练过程中的性能瓶颈
- 需要对比不同配置或优化方案的性能表现
- 需要定位训练过程中的性能异常或退化

### 配置选项

性能分析配置使用 TorchTitan 的分层 CLI 参数。NPU 专属选项放在
`profiler.extension.*` 命名空间中。

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

#### torchtitan-npu profiler 扩展选项

下表中的字段通过 `profiler.extension.*` CLI 参数传入。

| 配置项 | CLI 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- | --- |
| `profile_ranks` | `--profiler.extension.profile-ranks` | list[int] | [-1] | 需要进行性能分析的 rank 列表。使用 [-1] 表示所有 rank。 |
| `profiler_start` | `--profiler.extension.profiler-start` | int \| None | None | 绝对步数采集窗口的起始步，包含该步。需与 `profiler_end` 同时设置。 |
| `profiler_end` | `--profiler.extension.profiler-end` | int \| None | None | 绝对步数采集窗口的结束步，不包含该步。需与 `profiler_start` 同时设置。 |
| `profile_with_memory` | `--profiler.extension.profile-with-memory` | bool | false | 是否记录内存使用情况。 |
| `profile_with_stack` | `--profiler.extension.profile-with-stack` | bool | false | 是否记录调用栈信息。 |
| `enable_online_parse` | `--profiler.extension.enable-online-parse` | bool | true | 是否在线解析；使用 `--profiler.extension.no-enable-online-parse` 可关闭。 |

#### 离线解析

当 `--profiler.extension.no-enable-online-parse`（即
`enable_online_parse=False`）时，性能分析仅将原始数据转储到
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

#### 周期采集

使用 CLI 启用 CANN profiler，并配置周期采集模式：

```bash
torchrun --nproc_per_node=2 -m torchtitan_npu.train \
  --module torchtitan_npu.models.deepseek_v4 \
  --config deepseek_v4_debugmodel \
  --profiler.enable-profiling \
  --profiler.save-traces-folder profiling/traces \
  --profiler.profile-freq 10 \
  --profiler.profiler-warmup 3 \
  --profiler.profiler-active 1
```

#### 绝对步数采集

使用 CLI 启用 CANN profiler，并配置绝对步数采集模式，可以直接用 `profiler-start/end` 指定采集窗口：

```bash
NGPU=2 ./scripts/run_train.sh \
  --profiler.enable-profiling \
  --profiler.extension.profiler-start 5 \
  --profiler.extension.profiler-end 6 \
  --profiler.save-traces-folder profile_traces \
  --profiler.extension.profile-ranks 0 \
  --profiler.extension.profile-with-memory
```

也可以通过原生周期参数显式拼接出相同的采集窗口：

```bash
NGPU=2 ./scripts/run_train.sh \
  --profiler.enable-profiling \
  --profiler.profile-freq 4 \
  --profiler.profiler-warmup 3 \
  --profiler.profiler-active 1 \
  --profiler.profiler-repeat 1 \
  --profiler.profiler-skip-first 1 \
  --profiler.save-traces-folder profile_traces \
  --profiler.extension.profile-ranks 0 \
  --profiler.extension.profile-with-memory
```

设置绝对步数窗口后，窗口范围以 `profiler-start/end` 为准，扩展会自动推导
`profiler-active`、`profiler-repeat` 和 `profiler-skip-first`。

断点续训时，绝对步数窗口会根据恢复后的全局步数继续对齐；如果窗口已经结束，
Profiler 不会重复启动。
