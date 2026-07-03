# 调试支持特性

torchtitan-npu 目前提供多种调试特性支持，帮助开发者定位分布式训练中的各类问题，包括通信故障、内存问题和性能瓶颈等。以下是常见使用场景和对应功能的快速参考：

| 使用场景 | 对应功能 |
|---------|---------|
| 排查 HCCL 通信超时问题 | [HCCL Flight Recorder](#hccl-flight-recorder) |
| 分析 OOM 和内存泄漏 | [Memory Snapshot](#memory-snapshot) |
| 定位性能瓶颈和优化性能 | [Profiling](#profiling) |

---

## HCCL Flight Recorder

在大规模分布式训练任务因通信算子超时而异常退出时，往往因为缺少详尽的通信轨迹及对应代码位置信息，难以快速定位问题的根本原因。HCCL Flight Recorder 通过自动配置 HCCL 相关环境变量，在分布式训练过程中实时监控 HCCL 通信操作，并记录通信算子的下发调用栈、时间戳和数据量大小等关键信息。当检测到通信超时异常时，系统会自动将缓冲区中的追踪信息转储到文件。

### 使用场景

- 训练过程中出现 HCCL 通信超时错误
- 需要排查 HCCL 通信算子的调用顺序和参数

### 实现原理

由于启用 HCCL Flight Recorder 功能所需的环境变量配置与 NCCL 存在差异，因而 torchtitan-npu 在 `torchtitan_npu/tools/flight_recorder.py` 中对 `torchtitan.distributed.utils.init_distributed` 函数进行了拦截，自动配置 HCCL Flight Recorder 相关的环境变量，使能了通信超时的自动检测和轨迹转储功能。在分布式初始化阶段，系统会根据配置自动设置以下环境变量：

1. **`TORCH_HCCL_TRACE_BUFFER_SIZE`**：设置 HCCL 追踪缓冲区大小，控制记录的通信轨迹数据量。
2. **`HCCL_ASYNC_ERROR_HANDLING`**：启用 HCCL 异步错误处理机制，创建检测通信问题的 HCCL Watchdog 线程。
3. **`TORCH_HCCL_ENABLE_MONITORING`**：启用 HCCL 监控功能，实时检测通信状态。
4. **`TORCH_HCCL_DUMP_ON_TIMEOUT`**：配置在检测到超时时自动转储追踪信息。
5. **`TORCH_HCCL_DEBUG_INFO_TEMP_FILE`**：指定追踪文件的存储路径和文件名前缀。

### 配置选项

HCCL Flight Recorder 配置写在模型的 `config_registry.py` 中，
对应 `CommConfig` 配置类。

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `trace_buf_size` | int | 0 | HCCL 追踪缓冲区大小。设置为 0 时不启用追踪，设置为大于 0 的值时启用自动追踪和超时转储。 |
| `save_traces_folder` | str | "comm_traces" | 追踪文件的保存目录路径，相对于训练输出目录。 |
| `save_traces_file_prefix` | str | "rank_" | 追踪文件的文件名前缀。 |

需要注意的是，HCCL 通信超时的阈值目前只能通过环境变量 `HCCL_EXEC_TIMEOUT` 来配置。设置此环境变量可以确保 HCCL 本身的其他调试信息能够被正确收集。例如，在启动训练脚本中设置：

```bash
export HCCL_EXEC_TIMEOUT=120  # 设置超时阈值为 120 秒
```

### 配置示例

在模型 `config_registry.py` 的配置函数中设置：

```python
from torchtitan.config import CommConfig

comm = CommConfig(
    trace_buf_size=2000,
    save_traces_folder="hccl_traces",
    save_traces_file_prefix="rank_",
)
```

或通过命令行覆盖：

```bash
bash scripts/run_train.sh \
  --comm.trace_buf_size 2000 \
  --comm.save_traces_folder hccl_traces
```

当训练过程中出现 HCCL 通信超时或错误时，系统会自动将追踪信息转储到 `hccl_traces/rank_*` 文件中，开发者可以使用这些文件进行问题诊断和分析，相关指导可以参考 PyTorch 社区关于 Flight Recorder 功能的相关[说明文档](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html)。

---

## Memory Snapshot

内存快照功能用于捕获和记录训练过程中的内存使用情况，包括内存分配、显存占用、张量生命周期等信息。通过 torchtitan 配置项进行定时内存快照收集，本功能生成的`.pickle`格式内存快照文件可通过[memory_viz](https://docs.pytorch.org/memory_viz)工具进行解析和可视化查看。

### 使用场景

- 训练过程中出现 OOM（Out of Memory）错误，需要分析内存占用情况
- 怀疑存在内存泄漏，需要追踪内存分配和释放情况
- 需要优化显存使用，了解框架不同模块的内存占用

### 配置选项

torchtitan 原生提供内存快照功能，相关配置项位于 `ProfilingConfig` 类，支持的配置选项如下：

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `enable_memory_snapshot` | bool | false | 是否启用内存快照功能。 |
| `save_memory_snapshot_folder` | str | "memory_snapshot" | 内存快照文件保存目录。 |
| `profile_freq` | int | 10 | 每隔多少个训练步骤收集一次内存快照。 |

torchtitan 原生内存快照功能会按照 `profile_freq` 指定的频率定期收集内存快照，并在发生 OOM 错误时自动转储当前内存快照。收集到的内存快照将保存到 `save_memory_snapshot_folder` 指定的目录中。

### 配置示例

在模型 `config_registry.py` 的配置函数中设置：

```python
from torchtitan_npu.config.configs import ProfilingConfig

profiling = ProfilingConfig(
    enable_memory_snapshot=True,
    save_memory_snapshot_folder="memory_snapshot",
    profile_freq=10,
)
```

---

## Profiling

性能分析是优化训练性能的关键工具。torchtitan-npu 对性能分析功能进行了 NPU 适配，支持详细的性能数据收集和分析。系统使用 `torch_npu.profiler` 提供的原生性能分析器，能够追踪 CPU 和 NPU 的活动，记录内存使用情况、调用栈信息、张量形状等详细数据，并提供 AI 算力利用率指标。

### 使用场景

- 需要分析训练过程中的性能瓶颈
- 需要对比不同配置或优化方案的性能表现
- 需要定位训练过程中的性能异常或退化

### 配置选项

性能分析配置写在模型的 `config_registry.py` 中，相关配置定义在 `ProfilingConfig` 类。

#### torchtitan 原生配置选项

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enable_profiling` | bool | false | 是否启用性能分析功能。 |
| `save_traces_folder` | str | "profile_traces" | 性能分析结果的保存目录路径。 |
| `profile_freq` | int | 10 | 性能分析的采样频率，每隔多少步采集一次。 |
| `profiler_warmup` | int | 1 | 性能分析器的预热步数。 |
| `profiler_active` | int | 3 | 性能分析器的采集步数。 |

#### torchtitan-npu 扩展配置选项

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `profile_step_start` | int | 0 | 开始性能分析的步数。需与 `profile_step_end` 同时设置为大于 0 的值，才使用基于步数的分析模式。 |
| `profile_step_end` | int | 0 | 结束性能分析的步数。需与 `profile_step_start` 同时设置为大于 0 的值，才使用基于步数的分析模式。 |
| `profile_ranks` | list[int] | [-1] | 需要进行性能分析的 rank 列表，例如 [0, 1, 2]。使用 [-1] 表示对所有 rank 进行分析。 |
| `profile_record_shapes` | bool | true | 是否在性能分析期间记录张量形状。 |
| `profile_with_memory` | bool | false | 是否在性能分析期间记录内存使用情况。 |
| `profile_with_stack` | bool | false | 是否在性能分析期间记录调用栈信息。 |
| `enable_online_parse` | bool | true | 是否启用性能分析数据的在线解析。设置为 `false` 时仅落盘原始数据，需在训练结束后离线解析（见[离线解析](#离线解析)）。 |

#### 离线解析

当 `enable_online_parse=False` 时，性能分析仅将原始数据转储到 `save_traces_folder` 指定的目录，不进行在线解析。
训练结束后，在 `save_traces_folder/profiling_data/` 目录下会生成以 `{hostname}_{pid}_{timestamp}_ascend_pt` 命名的子目录，包含原始 profiling 数据。多 rank 场景（`profile_ranks` 含多个 rank 或为 `[-1]`）下，每个 rank 会生成独立的 `*_ascend_pt` 子目录。使用 `scripts/parse_profiling_data.py` 对其进行离线解析：

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

在 `config_registry.py` 的配置函数中设置，启用性能分析功能。

使用 torchtitan 原生的基于频率的分析模式：

```python
from torchtitan_npu.config.configs import ProfilingConfig

profiling = ProfilingConfig(
    enable_profiling=True,
    save_traces_folder="profile_traces",
    profile_freq=10,
    profiler_warmup=1,
    profiler_active=3,
)
```

使用 torchtitan-npu 扩展的基于步数的分析模式：

```python
from torchtitan_npu.config.configs import ProfilingConfig

profiling = ProfilingConfig(
    enable_profiling=True,
    save_traces_folder="profile_traces",
    profile_step_start=5,
    profile_step_end=6,
    profile_ranks=[0],
    profile_record_shapes=True,
    profile_with_memory=False,
    profile_with_stack=False,
    enable_online_parse=True,  # 设为 False 启用离线模式，训练后用 parse_profiling_data.py 解析
)
```
