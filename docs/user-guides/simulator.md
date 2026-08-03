# Simulator 使用指南

torchtitan-npu Simulator 是一个**侧载式（side-loaded）**的计算图捕获工具，它在不依赖真实 NPU 硬件、不分配真实显存的前提下，完整捕获一个训练 step 中**所有卡**的计算图，并输出四层 IR（L0 OpNode → L1 StepGraph → L2 ScheduleGraph → L3 WorkloadGraph）及可视化文件。

## 工作原理

Simulator 通过以下机制实现"零硬件捕获"：

1. **Meta Device 替身**：将 `torchtitan.tools.utils.device_type` 猴子补丁为 `"meta"`，模型全程在 `torch.device("meta")` 上构建和运行，不分配真实显存。
2. **Fake Process Group**：两种模式——`fake_backend`（单进程，`FakeProcessGroup` 模拟全部 rank）和 `multi_proc_meta`（多进程，PP 维度用 gloo 真实多进程，CP/TP/EP/FSDP 维度用 `FakeProcessGroup` 模拟）。
3. **集合通信拦截**：拦截 `torch.distributed.*` 和 `_functional_collectives.*` 的所有集合通信调用（含 autograd 变体和 P2P），在 meta 张量上不做真实通信，只记录通信事件（op 类型、group、shape、bytes、PP context）。
4. **MoE 强制负载均衡**：强制 `debug.moe_force_load_balance=True`，路由逻辑退化为与输入数值无关的 round-robin，保证 meta 张量下 shape 推断正确。
5. **算子级捕获**：通过 `TorchDispatchMode` 拦截每个 aten/npu 算子调用，记录 op 类型、输入输出 shape、producer/consumer 依赖关系、PP stage/microbatch 上下文。
6. **Microbatch 分层捕获**：首个 microbatch 完整捕获 L0/L1 算子图，后续 microbatch 跳过 L0/L1（pass-through 模式），只捕获 L2 调度时序和 L3 通信事件。L0/L1 捕获开销与 microbatch 数量无关。

## 环境准备

Simulator 需要 `torch` 和 `torch_npu`（用于 NPU 自定义算子的 meta 核注册），但**不需要真实 NPU 硬件**（`torch.npu.is_available()` 可以为 `False`）。

### 方式一：使用预构建镜像（推荐）

已将调试环境打包为容器镜像并上传至华为云 SWR，内含 CANN 9.1.0-beta.1-950、Python 3.12.13、torch 2.12.0+cpu、torch_npu 2.12.0.rc1、torchtitan 0.2.2、torchtitan_npu 0.2.2.post1，开箱即用。

```bash
# 1. 拉取镜像
sudo docker pull swr.cn-north-4.myhuaweicloud.com/zhanlu/torchtitan-npu-simulator:v1.0

# 2. 克隆项目代码（如已有可跳过）
git clone https://github.com/ZhanluModelSim/torchtitan-npu-simulator.git
cd torchtitan-npu-simulator
git checkout feat/npu-simulator

# 3. 启动容器，挂载项目目录
sudo docker run -d --name titan-sim \
  -v $(pwd):/workspace \
  -w /workspace \
  swr.cn-north-4.myhuaweicloud.com/zhanlu/torchtitan-npu-simulator:v1.0 \
  sleep infinity

# 4. 进入容器
sudo docker exec -it titan-sim bash

# 5. 加载 CANN 环境变量（每次进入容器都需要执行）
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 6. 验证环境
python3 -c "import torch; import torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu available', torch.npu.is_available())"
# 预期输出：
#   torch 2.12.0+cpu
#   torch_npu 2.12.0.rc1
#   npu available False   ← 正常，simulator 不需要真实硬件
```

> [!TIP]
> 镜像约 24.5GB，首次拉取需要较长时间。如果网络较慢，可先用 `--quiet` 模式后台拉取。
>
> 每次重新进入容器都需要执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 加载 CANN 环境变量。也可以将其写入 `~/.bashrc` 一劳永逸：
> ```bash
> echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null' >> ~/.bashrc
> ```

### 方式二：使用 Dockerfile 自行构建镜像（支持 ARM64）

项目根目录提供 `Dockerfile`，可在 x86_64 和 ARM64（aarch64）架构下自行构建镜像。CANN 基础镜像、torch、torch_npu、triton-ascend 均已提供 ARM64 wheel，无需额外适配。

```bash
# 1. 克隆项目代码
git clone https://github.com/ZhanluModelSim/torchtitan-npu-simulator.git
cd torchtitan-npu-simulator
git checkout feat/npu-simulator

# 2. 构建镜像（自动识别当前架构）
sudo docker build -t torchtitan-npu-simulator:v1.0 .

# 3. 启动容器，挂载项目目录
sudo docker run -d --name titan-sim \
  -v $(pwd):/workspace \
  -w /workspace \
  torchtitan-npu-simulator:v1.0 \
  sleep infinity

# 4. 进入容器（CANN 环境变量已写入 bashrc，自动加载）
sudo docker exec -it titan-sim bash

# 5. 验证环境
python3 -c "import torch; import torch_npu; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu available', torch.npu.is_available())"
```

> [!NOTE]
> Dockerfile 会自动完成以下操作：
> - 以 CANN 9.1.0-beta.1-950 镜像为基础（提供 `libhccl.so` 等 torch_npu 依赖库）
> - 安装 `torch==2.12.0+cpu`（CPU 版，无需 GPU/NPU 硬件）
> - 安装 `torch_npu==2.12.0rc1`（提供 NPU 算子 meta 核注册）
> - 从 gitcode 拉取并安装 torchtitan（pinned commit `ac13e536`）
> - 安装 `requirements.txt` 中的全部依赖（numpy、triton-ascend、scipy 等）
> - 克隆并安装 torchtitan_npu（simulator 分支，editable 模式）
> - 将 `source set_env.sh` 写入 `/etc/bash.bashrc`，进入容器自动加载 CANN 环境
>
> 构建耗时约 10-30 分钟（取决于网络速度），镜像约 25GB。

### 方式三：使用基础 CANN 镜像手动搭建

如果需要完全自定义环境，可从基础 CANN 镜像开始逐步安装：

```bash
# 拉取基础 CANN 镜像
docker pull quay.m.daocloud.io/ascend/cann:9.1.0-beta.1-950-ubuntu22.04-py3.12

# 启动容器
docker run -d --name titan-sim \
  -v $(pwd):/workspace \
  -w /workspace \
  quay.m.daocloud.io/ascend/cann:9.1.0-beta.1-950-ubuntu22.04-py3.12 \
  sleep infinity

# 进入容器，加载 CANN 环境并安装依赖
docker exec -it titan-sim bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install torch==2.12.0+cpu torch_npu==2.12.0.rc1 -f https://download.pytorch.org/whl/cpu/torch_stable.html
cd /workspace && pip install -e .
```

### 方式四：本地环境

确保已安装 `torch` 和 `torch_npu`（版本以 `requirements.txt` 锁定为准），并已 source CANN 环境变量。

## 快速开始

以下命令均在**容器内**执行（通过 `docker exec -it titan-sim bash` 进入后，先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`）。

### 1. 切换到 simulator 分支

```bash
git checkout feat/npu-simulator
```

### 2. 准备 Tokenizer

DeepSeek-V4-Pro 仿真配置默认引用 `./tests/assets/tokenizer/deepseek_v4_pro_tokenizer`，该目录**未随仓库提交**。Simulator 只在 meta device 上运行，tokenizer 仅用于 dataloader 生成 token ID（不影响捕获的 shape/结构），因此可以用仓库内置的 `deepseekv3_tokenizer` 替代：

```bash
# 方式一：用 --hf_assets_path 覆盖（推荐，无需修改代码）
# 在运行仿真命令时追加 --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer

# 方式二：创建软链接
mkdir -p tests/assets/tokenizer/deepseek_v4_pro_tokenizer
ln -s ../deepseekv3_tokenizer/tokenizer.json tests/assets/tokenizer/deepseek_v4_pro_tokenizer/
ln -s ../deepseekv3_tokenizer/tokenizer_config.json tests/assets/tokenizer/deepseek_v4_pro_tokenizer/
```

> [!NOTE]
> 仓库内置的 tokenizer 资产位于 `tests/assets/tokenizer/`，已 git 跟踪的有 `deepseekv3_tokenizer`、`qwen3-tokenizer`、`vlm_tokenizer`。Simulator 不关心词表内容（meta tensor 无数值），任何能正常 encode 的 tokenizer 均可使用。

### 3. 运行仿真

Simulator 推荐使用 `run_simulator_spawn.py`。启动器会先完成与 worker 相同的配置解析和 CLI 覆盖，再根据最终 PP degree 自动选择通信模式和真实进程数：

```bash
# 16 层小规模验证（PP=1，自动使用 fake_backend）
python3 scripts/run_simulator_spawn.py \
    --config deepseek_v4_pro_simulate_16_layers \
    --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
```

```bash
# 61 层验收配置（384 die，PP=1）
python3 scripts/run_simulator_spawn.py \
    --config deepseek_v4_pro_simulate_61_layers \
    --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
```

```bash
# 千卡规模（模拟 2048 die，自动启动 16 个 PP 进程）
python3 scripts/run_simulator_spawn.py \
    --config deepseek_v4_pro_simulate_61_layers_pp16_tp8_cp4_ep128 \
    --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
```

PP=1 时也可以直接用 `python3 -m` 调用，通信模式仍会自动解析：

```bash
NGPU=384 python3 -m torchtitan_npu.entry \
    --module torchtitan_npu.simulator \
    --config deepseek_v4_pro_simulate_61_layers \
    --training.steps=1 \
    --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
```

> [!NOTE]
> - `simulation.world_size` 指定模拟的**总卡数**。优先级为显式 `--simulation.world-size` > 启动环境 `NGPU` > 配置函数默认值；解析结果会在构建 mesh 前写回配置。
> - 最终 PP=1 时自动使用 `fake_backend`；最终 PP>1 时自动使用 `multi_proc_meta`。
> - `--training.steps=1` 是默认值，Simulator 只捕获一个 step。
> - `--hf_assets_path` 指定 tokenizer 路径，用仓库内置的 `deepseekv3_tokenizer` 即可（见上一步说明）。
> - CLI 对 PP/TP/CP/EP/DP 的覆盖会在选择模式、计算进程数和构建 mesh 前统一生效。

### 4. 获取输出文件

仿真输出默认写入项目目录下的 `simulator_output/<配置名>/`。由于启动容器时已将项目目录挂载到 `/workspace`，输出文件会直接同步到宿主机，无需额外拷贝：

```bash
# 在宿主机上直接查看（无需进入容器）
ls simulator_output/deepseek_v4_pro_61_layers/
# summary.txt  trace.html  compute_graph.dot  simulation_result.json

# 查看摘要
cat simulator_output/deepseek_v4_pro_61_layers/summary.txt

# 在浏览器中打开可视化页面
# macOS:  open simulator_output/deepseek_v4_pro_61_layers/trace.html
# Linux:  xdg-open simulator_output/deepseek_v4_pro_61_layers/trace.html
```

> [!TIP]
> 如果容器启动时未挂载项目目录（`-v` 参数），输出文件只在容器内。可通过 `docker cp` 拷出到宿主机：
> ```bash
> sudo docker cp titan-sim:/workspace/simulator_output ./simulator_output
> ```

## 内置仿真配置

所有仿真配置定义在 `torchtitan_npu/simulator/config_registry.py` 中，通过 `--config <函数名>` 选择：

| 配置函数名 | 模型 | 层数 | 并行策略 | world_size | 模式 | 说明 |
|-----------|------|------|---------|------------|------|------|
| `deepseek_v4_pro_simulate_16_layers` | DeepSeek-V4-Pro | 16 | PP=1/TP=1/CP=1/EP=16 | 384 | fake_backend | 快速验证 |
| `deepseek_v4_pro_simulate_16_layers_cp4` | DeepSeek-V4-Pro | 16 | PP=1/CP=4/EP=16 | 16 | fake_backend | CP 通信捕获验证 |
| `deepseek_v4_pro_simulate_16_layers_pp4_cp4` | DeepSeek-V4-Pro | 16 | PP=4/CP=4/DP=4 | 64 | multi_proc_meta | PP+CP+FSDP 全通信捕获 |
| `deepseek_v4_pro_simulate_61_layers` | DeepSeek-V4-Pro | 61 | PP=1/TP=1/CP=1/EP=192 | 384 | fake_backend | 验收配置 |
| `deepseek_v4_pro_simulate_61_layers_pp16_tp8_cp4_ep128` | DeepSeek-V4-Pro | 61 | PP=16/TP=8/CP=4/EP=128/FSDP=-1 | 2048 | multi_proc_meta | 千卡规模（16 个 PP 进程） |

每个仿真配置内部复用对应的真实训练配置（如 `deepseek_v4_pro_debug_61_layers_4k_384die()`），原样继承 model_spec、并行度、optimizer 等全部字段。运行时统一处理：

- 根据最终 PP degree 推导 `comm.mode`
- `compile.enable = False`（捕获需要 eager dispatch）
- `debug.moe_force_load_balance = True`（MoE 强制负载均衡）

### Activation checkpoint

可通过 CLI 选择 activation checkpoint 模式：

```bash
--activation-checkpoint.mode none       # 不重计算，保留普通 forward 激活
--activation-checkpoint.mode full       # 按 transformer block 完整重计算
--activation-checkpoint.mode selective  # op 级 selective activation checkpoint
```

`selective` 使用 TorchTitan 的 op SAC policy：保存部分计算/通信结果，其余算子在
backward 中重放。可通过
`activation_checkpoint.per_op_sac_force_recompute_mm_shapes_by_fqns`
指定需要强制重计算的 `nn.Linear` FQN。`memory_budget` 依赖
`torch.compile` 的图分区器，而 simulator 必须使用 eager TorchDispatch 捕获，因此
当前明确不支持。

## 输出文件

### 单进程模式（fake_backend）

仿真完成后，输出文件默认写入 `simulator_output/<配置名>/` 目录：

| 文件 | 格式 | 内容 |
|------|------|------|
| `summary.txt` | 纯文本 | 摘要报告：各阶段 op 数、FLOPs、内存、通信量统计 |
| `trace.html` | 自包含 HTML | 可视化页面：L3→L2→L1→L0 层级展开 |
| `compute_graph.dot` | Graphviz | 算子依赖图 |
| `simulation_result.json` | JSON | 完整四层 IR 结构化数据 |
| `kernel_summary/` | CSV 目录 | 按 Rank 拆分的算子汇总 |
| `ir_export/` | CSV 目录 | 各层级 IR 导出（见下文） |
| `memory/` | JSON/CSV 目录 | 默认包含内存摘要；显式启用 `mem` 后还包含 Perfetto trace、事件时间线和 tensor 生命周期 |

内存计算由 `simulation.enable_memory_tracking` 控制。只要内存跟踪开启，
`memory/memory_summary.json` 就会默认导出，不依赖 `output_formats`。
`simulation.output_formats` 中的 `mem` 只控制体积较大的详细 trace 和 CSV。例如
导出完整内存产物：

```bash
--simulation.output_formats mem
```

`memory/` 中的主要文件：

| 文件 | 内容 |
|------|------|
| `memory_summary.json` | 默认导出；参数常驻、前向/反向/optimizer 峰值及 checkpoint 聚合 |
| `memory_events.csv` | `mem`；未折叠的算子输入输出事件，含 PP stage/microbatch/comp_type |
| `memory_timeline.csv` | `mem`；alloc/free 后的 active tensor bytes 曲线 |
| `tensor_lifetimes.csv` | `mem`；每个 tensor 的 birth、last consumer、death、逻辑大小、建模驻留大小和分类 |
| `checkpoint_tensors.csv` | `mem`；AC wrapper 边界输入/输出及 selective 内部保存 tensor 元数据；存在 AC 边界时生成 |
| `activation_offload_tensors.csv` | `mem`；不属于 AC wrapper 记录、但被统一激活卸载策略覆盖的 tensor 及逐层归属 |
| `memory_actions.csv` | `mem`；PP 调度 action 与展开后内存事件区间的映射；非 PP 不生成 |
| `memory_trace.json` | `mem`；可由 Chrome Trace 或 Perfetto 打开的阶段趋势图 |

可通过两个独立开关调整内存建模假设：

```bash
# 参数仍保留捕获到的原始 dtype/num_bytes，但静态本地参数分片按 BF16 计入设备驻留。
--simulation.memory-parameter-storage-dtype bfloat16

# 所有前向保存激活仍保留元数据，但不计入设备驻留和峰值。
--simulation.memory-offload-ac-saved-tensors
```

参数 dtype 覆盖仅作用于静态本地参数分片，不改变梯度、optimizer state 或 FSDP
all-gather 全参数的 dtype。激活卸载作用于所有“forward 产生且正常 backward
仍需使用”的 tensor，因此对 `activation_checkpoint.mode=none/full/selective`
使用同一建模假设。full/selective 的 checkpoint 内部重计算临时值不是保存激活，
仍按其 forward/recompute 生命周期建模。freqs、mask 等无梯度外部上下文不会被
当作保存激活卸载。两个开关默认均关闭。

selective checkpoint 保存的内部 tensor 在 `tensor_lifetimes.csv` 中标记为
`checkpoint_saved_for_recompute`。summary 中的
`checkpoint_recompute_saved_{tensor_count,logical_bytes,modeled_bytes}`
分别给出数量、逻辑大小和建模驻留大小；全量明细中对应的
`checkpoint_tensors.csv` 记录使用 `role=recompute_saved`。

`memory_summary.json` 的 `checkpoint_saved_activations` 按稳定的
`checkpoint_id` marker 聚合保存激活。例如：

```json
{
  "checkpoint_saved_activations": {
    "part0:layers.0": {
      "marker_kind": "module",
      "logical_bytes_per_instance": 262144,
      "modeled_bytes_per_instance": 0,
      "instance_count": 2,
      "logical_bytes_total": 524288,
      "modeled_bytes_total": 0,
      "pp_stages": [0],
      "microbatches": [0, 1],
      "size_variants": [
        {
          "logical_bytes": 262144,
          "modeled_bytes": 0,
          "tensor_count": 1,
          "instance_count": 2
        }
      ]
    }
  }
}
```

`logical_bytes_per_instance` 可直接作为后续一次恢复/load 的大小。若同一个 marker
存在动态 shape，该字段为 `null`，具体大小和出现次数记录在 `size_variants`。
marker 当前使用 AC wrapper 模块路径且 `marker_kind="module"`；该结构不依赖“层”的
概念，后续 selective AC 可以使用 op 级 marker 和 `marker_kind="op"` 并复用相同格式。

`checkpoint_prefetch` 保留 AC 专用的边界输入与 selective 保存值统计。评估三种
AC 模式下的统一激活回捞带宽时，应使用 `activation_prefetch`：

```json
{
  "activation_offload_tensor_count": 567,
  "activation_offload_logical_bytes": 32992845480,
  "activation_offload_modeled_bytes": 0,
  "activation_prefetch_logical_bytes": 32992845480,
  "activation_prefetch": {
    "part0:layers.0": {
      "marker_kind": "layer",
      "logical_bytes_per_instance": 1771982464,
      "modeled_bytes_per_instance": 0,
      "tensor_count_per_instance": 29,
      "instance_count": 1,
      "logical_bytes_total": 1771982464,
      "modeled_bytes_total": 0,
      "pp_stages": [0],
      "microbatches": [0],
      "size_variants": [
        {
          "logical_bytes": 1771982464,
          "modeled_bytes": 0,
          "tensor_count": 29,
          "instance_count": 1
        }
      ]
    }
  }
}
```

none 模式按正常 backward consumer 的 `layers.N` 归档保存激活；full/selective
复用 checkpoint marker，并合并该层其他保存激活。没有唯一层归属的 loss、logits
等记录在 `part0:<unattributed>`，其总量同时写入
`activation_prefetch_unattributed_{tensor_count,logical_bytes}`。

AC 专用的 `checkpoint_prefetch` 结构如下：

```json
{
  "checkpoint_prefetch": {
    "part0:layers.0": {
      "marker_kind": "module",
      "boundary_logical_bytes_per_instance": 262144,
      "recompute_saved_logical_bytes_per_instance": 1048576,
      "prefetch_logical_bytes_per_instance": 1310720,
      "boundary_modeled_bytes_per_instance": 0,
      "recompute_saved_modeled_bytes_per_instance": 0,
      "modeled_bytes_per_instance": 0,
      "boundary_tensor_count_per_instance": 1,
      "recompute_saved_tensor_count_per_instance": 3,
      "tensor_count_per_instance": 4,
      "instance_count": 2,
      "prefetch_logical_bytes_total": 2621440,
      "modeled_bytes_total": 0,
      "pp_stages": [0],
      "microbatches": [0, 1],
      "size_variants": [
        {
          "boundary_logical_bytes": 262144,
          "boundary_modeled_bytes": 0,
          "boundary_tensor_count": 1,
          "recompute_saved_logical_bytes": 1048576,
          "recompute_saved_modeled_bytes": 0,
          "recompute_saved_tensor_count": 3,
          "prefetch_logical_bytes": 1310720,
          "modeled_bytes": 0,
          "tensor_count": 4,
          "instance_count": 2
        }
      ]
    }
  }
}
```

`prefetch_logical_bytes_per_instance` 是一次执行该 checkpoint 前需要恢复的总字节数：
wrapper 边界保存输入与 selective AC 内部保存结果之和。full AC 通常只有前者；
selective AC 两部分都可能存在。`prefetch_logical_bytes_total` 是该 marker 在整个
训练 step 中所有实例的总回捞流量。开启 AC offload 后，logical bytes 保持不变，
而 modeled bytes 为 0，表示这些 tensor 不计入设备驻留。

同一 marker 存在动态 shape 时，所有 `*_per_instance` 字段为 `null`，应从
`size_variants` 读取各尺寸及出现次数。若 selective 保存值无法可靠匹配 checkpoint
边界，`checkpoint_prefetch_unattributed_{tensor_count,logical_bytes}` 会非零；
此时分层结果不完整，不应直接作为总带宽结论。

### 多进程模式（multi_proc_meta）

每个 PP stage 进程独立输出到 `rank_N/` 子目录，**不合并到 Rank 0**：

```
simulator_output/<配置名>/
├── rank_0/                    # PP Stage 0
│   ├── summary.txt
│   ├── trace.html
│   ├── simulation_result.json
│   ├── kernel_summary/
│   ├── memory/
│   └── ir_export/
│       ├── rank_schedule.csv          # L3: inter-rank schedule
│       ├── l1_schedule/               # L2: per-stage L1 schedule
│       │   └── stage_0_l1_schedule.csv
│       ├── step_forward_l0_ops.csv    # L1: forward L0 ops (MB 0 only)
│       ├── step_backward_l0_ops.csv   # L1: backward L0 ops (MB 0 only)
│       └── step_optimizer_l0_ops.csv  # L1: optimizer L0 ops
├── rank_1/                    # PP Stage 1
│   └── ...
├── rank_2/
│   └── ...
└── rank_3/
    └── ...
```

> [!NOTE]
> 各 stage 的 IR 独立输出，不合并。跨 stage 的调度关系（如 PP P2P send/recv 配对）可通过各 stage 的 `execution_timeline` 中的 `comm_peer_rank` 字段自行关联。

### kernel_summary/ 说明

将 L0 算子按 **Rank 拆分为独立文件**（`rank_0.csv`、`rank_1.csv`、...），每个文件内按**拓扑序排列**（拓扑序相同时以 op_id 升序），每行一个算子。multi_proc 模式下位于 `rank_N/kernel_summary/` 目录中。

| 列 | 说明 |
|----|------|
| `rank` | 逻辑 rank 编号（0 ~ world_size-1） |
| `step_type` | 步骤类型：`forward` / `backward` / `optimizer` |
| `step_id` | 步骤模板 ID |
| `topo_order` | 在该步骤模板内的拓扑序（从 0 开始，Kahn 算法） |
| `op_id` | 算子唯一 ID |
| `op_type` | 规范化算子类型（如 `matmul`、`rms_norm`），未映射的显示原始算子名 |
| `raw_op_type` | 原始 dispatcher 算子名（如 `aten.addmm.default`、`npu.npu_rms_norm.default`） |
| `inputs_shape` / `outputs_shape` | 输入/输出张量 shape，格式 `[d0,d1];[d0,d1]` |
| `inputs_dtype` / `outputs_dtype` | 输入/输出 dtype |
| `flops` / `peak_mem` / `param_mem` / `comm_bytes` | 成本估算 |
| `repeat_count` | 去重折叠的重复次数 |
| `module_path` | 算子所属模块路径（如 `layers.2._checkpoint_wrapped_module.moe`） |
| `phase` | 捕获阶段：`forward` / `backward` / `optimizer` |
| `execution_kind` | 实际执行类型：`original_forward` / `recompute` / `backward` / `optimizer` |
| `is_recompute` | 当前算子是否由 activation checkpoint 在 backward 中实际重放 |
| `group_name` | 解析后的通信维度名（如 `tp`、`ep`、`fsdp`）；无法唯一解析时回退为框架原始组 ID |
| `raw_group_name` | 框架生成的原始 ProcessGroup 名称（如 `3713`） |
| `comm_dim` | `group_name` 的兼容别名 |
| `comm_ranks` | 通信域包含的 Rank 列表（仅通信算子有值，如 `0,1,2,3,...,15` 表示这 16 个 rank 属于同一通信组） |

> [!NOTE]
> 通信算子（`allgather`、`allreduce`、`reduce_scatter`、`all_to_all`、`broadcast`）在 L0 图中以 `comm.*` 前缀的 `raw_op_type` 注册，同时在 L2 ScheduleGraph 中生成 DataPass。`group_name` 标识通信所属的并行维度，`comm_ranks` 列出参与该次通信的具体 Rank。原始组名、Rank 集合和通信原语仍无法唯一确定维度时，`group_name` 保留 `raw_group_name`，不会猜测。

> [!NOTE]
> activation checkpoint 重放仍属于 `phase=backward`，但会被精确标记为
> `execution_kind=recompute`。该标记来自 PyTorch checkpoint 的重放上下文，
> selective checkpoint 中直接读取已保存结果、没有实际重放的算子不会被标记。
> selective checkpoint 内部被缓存并在重计算阶段复用的 forward tensor 会保留
> 原始 use-def 生命周期，并标记为 `checkpoint_saved_for_recompute`；它们不会按
> full checkpoint 的内部临时值提前释放。
>
> `module_path` 同时覆盖正常 forward、checkpoint recompute 和模块内部的
> backward 算子。backward 路径由模块输出的梯度边界确定，因此可用于按层汇总
> 算子时间；loss、跨层通信等没有唯一模块归属的全局算子可能保持为空。

> [!IMPORTANT]
> 大规模仿真（如 2048 die）时，全量展开所有 rank 的 CSV 会非常大（每 rank 数万行）。可通过配置中的 `simulation.csv_max_ranks` 限制展开的 rank 数量：
>
> ```python
> # 在 config_registry.py 的仿真配置函数中，或运行时覆盖：
> config.simulation.csv_max_ranks = 4  # 只展开前 4 个 rank
> ```
>
> 也可通过命令行跳过 CSV 输出：`--simulation.output_formats text html dot json`（不含 `csv`）。

### summary.txt 示例

```
Workload: 85f74af2e243 (train)
Iterations: 1 (warmup=0)

[forward] step=forward nodes=70656
  total_flops=91237669536768  total_peak_mem_bytes=48222492672  total_comm_bytes=0
  is_acyclic=True

[optimizer] step=optimizer nodes=112
  total_flops=0  total_peak_mem_bytes=0  total_comm_bytes=0
  is_acyclic=True

Schedule: 4096 instances, 3416184 data passes
  dp_degree=16 tp_degree=8 pp_degree=16 pipeline_schedule=1F1B
  comm[allgather] total_bytes=22314758805504
  comm[allreduce] total_bytes=552960
  comm[reduce_scatter] total_bytes=143655866204160
```

## 四层 IR 结构

输出遵循 [workload-model-platform spec](https://github.com/ZhanluModelSim/workload-model-platform/tree/master/spec) 定义的四层 IR。通信算子按其语义归属不同层级（详见 [L0-L3 层级归属设计](../design/schedule-capture-design.md)）。

| 层级 | 类名 | 文件 | 描述 |
|------|------|------|------|
| L0 | `OpNode` | `ir/op_node.py` | 纯计算算子调用：op_type、输入输出 TensorMeta、FLOPs/mem 估算、前后依赖、`pp_stage`/`pp_mb_idx`。**不含通信算子** |
| L1 | `StepGraph` | `ir/step_graph.py` | 一个 microbatch 的一个 phase 的计算 DAG（仅 MB 0 捕获）。包含计算算子 + **CP 通信**（attention 内部的 P2P/allgather，作为内部 DataPass） |
| L2 | `ScheduleGraph` | `ir/schedule_graph.py` | L1 StepGraph 间的调度依赖：microbatch 排队时序、**PP P2P**（stage 间传递）、**FSDP 通信**（unshard/reshard/allreduce）、依赖关系 |
| L3 | `WorkloadGraph` | `ir/workload_graph.py` | 顶层容器：迭代语义、DataFlow（dataloader 输入）、跨迭代参数传递 |

### 通信算子层级归属

通信算子按**调用路径上下文**归属不同层级（不依赖名称匹配，详见 [L0-L3 层级归属设计](../design/schedule-capture-design.md)）：

| 通信类型 | 调用入口 | 归属 | 原因 |
|----------|----------|------|------|
| CP P2P (`_WindowExchange`) | `CompressorAttentionCP._pre_hook` | **L1** | attention forward 的一部分 |
| CP allgather (`_allgather_seq`) | `CompressorAttentionCP._post_hook` | **L1** | attention forward 的一部分 |
| PP P2P (`forward_send/recv`) | `PipelineSchedule._step_microbatches` | **L2** | stage 间调度依赖 |
| FSDP allgather (unshard) | `FSDPParamGroup.pre_forward` | **L2** | forward 前参数 unshard |
| FSDP reduce_scatter (reshard) | `FSDPParamGroup.post_backward` | **L2** | backward 后参数 reshard |
| FSDP allreduce | 梯度同步 | **L2** | optimizer 梯度同步 |

判定方式：在 `meta_env.py` 中维护 `_comm_layer` 上下文变量，由各调用入口的 patch 设置（`"L1"` 表示模型计算内部，`"L2"` 表示框架调度层）。`_record_comm` 读取该变量为每个 CommEvent 标注层级。

### L2 execution_timeline 说明

`execution_timeline` 只包含 **L2 级别**的事件（不含 L0 计算算子和 CP 通信）：

| 字段 | 说明 | 来源 |
|------|------|------|
| `seq_idx` | 全局执行序号 | captured |
| `op_id` | L0 OpNode ID（MB 0 的计算算子为 -1，因为 L0 不在 L2 timeline 中） | — |
| `rank` | 进程 rank | captured |
| `pipeline_stage` | PP stage | captured |
| `micro_batch_idx` | microbatch 序号 | captured |
| `phase` | forward / backward / optimizer | captured |
| `action` | `forward_one_chunk` / `backward_one_chunk` / `comm` | captured |
| `comm_type` | PP/FSDP 通信类型（如 `forward_send`、`allgather`、`reduce_scatter`） | captured |
| `comm_peer_rank` | P2P peer rank | captured |

### L2 DataPass 说明

DataPass 只从 **PP/FSDP 通信**生成（CP 通信在 L1 内部）：

| DataPass 类型 | src_instance | dst_instance | comm_group_ranks | 语义 |
|---------------|-------------|-------------|------------------|------|
| PP P2P | `rank{src_stage}` | `rank{dst_rank}` | `[]` | stage 间数据传输 |
| FSDP 集合 | `rank{caller}` | `group:{dim}` | `[[0,1,...],...]` | 参数 unshard/reshard/梯度同步 |

## 自定义仿真配置

要为其他模型或并行策略创建仿真配置，在 `torchtitan_npu/simulator/config_registry.py` 中添加新函数：

```python
def my_model_simulate() -> SimulationTrainerConfig:
    # 1. 调用目标模型的真实训练配置
    base_config = my_model_debug_config()

    # 2. 如需修改并行策略，用 dataclasses.replace
    base_config = dataclasses.replace(
        base_config,
        parallelism=dataclasses.replace(
            base_config.parallelism,
            pipeline_parallel_degree=4,
            tensor_parallel_degree=2,
            # ...
        ),
    )

    # 3. 只声明模拟总规模；通信模式和其余 degree 在 CLI 解析后生成
    return _to_simulation_config(
        base_config,
        output_dir="./simulator_output/my_model",
        world_size=64,
    )
```

关键约束：

- 内存跟踪默认开启。可通过 CLI 设置 `--simulation.no-enable-memory-tracking` 关闭；关闭后不记录内存事件、tensor 生命周期和 FSDP 参数驻留，也不执行静态内存估算。开启时始终导出 `<output_dir>/memory/memory_summary.json`；只有 `output_formats` 包含 `mem` 时才导出详细 trace 和 CSV。
- `simulation.memory_parameter_storage_dtype` 可指定静态本地参数分片的建模驻留 dtype（例如 `bfloat16`）；原始 dtype 和逻辑大小仍保留在生命周期记录中。
- `simulation.memory_offload_ac_saved_tensors` 可将所有 forward-to-backward 保存激活按零设备驻留建模，覆盖 AC `none/full/selective`；逐层回捞量见 `activation_prefetch`，详细非 checkpoint 记录见 `activation_offload_tensors.csv`。
- `simulation.world_size` 必须等于 `dp_replicate × dp_shard × cp × tp × pp`；EP/ETP 不额外乘入 dense world size。`data_parallel_shard_degree=-1` 时由 simulator 根据最终 CLI 配置计算。
- `pipeline_parallel_degree > 1` 时，DeepSeek-V4 不支持 MTP，需设 `num_mtp_modules=0`。
- `pipeline_parallel_degree > 1` 时，`local_batch_size` 需 ≥ `pp_degree`（1F1B 调度需要足够 microbatch）。
- PP > 1 时自动使用 `multi_proc_meta`；PP=1 时自动使用 `fake_backend`，配置函数无需编码 mode。
- `simulated_parallel_degrees` 是解析后的兼容快照，由 simulator 自动生成，不应在新配置中手写。

## 多进程仿真模式（multi_proc_meta）

### 背景：单进程模式的局限

`fake_backend` 单进程模式通过 `FakeProcessGroup` 在一个进程内模拟全部 rank，能正确捕获 TP/CP/EP/FSDP 的集合通信，但 **PP（Pipeline Parallel）的调度逻辑无法真实复刻**——`PipelineScheduleSingle`（1F1B）要求每个进程只运行一个 stage，单进程内无法模拟多 stage 间的真实 P2P 调度时序。

### 方案：PP 用真实多进程，其他维度用 Fake PG

`multi_proc_meta` 模式的核心思路是**分层处理**：

| 维度 | 处理方式 | 进程数 | ProcessGroup 类型 |
|------|----------|--------|-------------------|
| PP | 真实多进程 | PP degree（如 4） | gloo（真实 rendezvous） |
| CP | Fake PG | — | FakeProcessGroup（size=cp_degree） |
| FSDP | Fake PG | — | FakeProcessGroup（size=fsdp_degree） |
| TP | Fake PG | — | FakeProcessGroup（size=tp_degree） |
| EP | Fake PG | — | FakeProcessGroup（size=ep_degree） |

具体来说：
- 用 `torchrun --nproc_per_node=PP` 启动 PP 个 gloo 进程，每个进程运行一个 PP stage
- `init_distributed` 返回**完整模拟 world_size**（如 PP=4, CP=4, DP=4 → 64），使 `ParallelDims._validate()` 通过
- `init_device_mesh` 创建 size=64 的 world_mesh 时，因 64 > gloo_ws(4)，自动切换到 fake backend
- world_mesh 的 default group 是一个 size=64 的 `FakeProcessGroup`
- 从 world_mesh unflatten 出的子组（CP/FSDP/TP/EP）通过 `new_group(backend="fake")` 创建，各自有正确的 size 和 rank

### 关键技术点

#### 1. 绕过 `new_group` 的 size 检查

PyTorch 的 `_new_group_with_tag` 检查 `group_world_size <= global_world_size`。当 gloo world_size=4 但需要创建 size=16 的 FSDP 子组时，此检查会失败。

`_patch_new_group_for_fake_backend()` 在 `_is_meta_simulation=True` 且调用方明确请求 `backend="fake"` 时，如果检测到 `group_ws > gloo_ws` 或 `rank >= gloo_ws`，直接调用 `_new_process_group_helper` 创建并注册 `FakeProcessGroup`，只绕过逻辑 rank 超出真实 gloo world 的检查。

#### 2. Fake mesh 的 device handle

当 `device_type="fake"` 时，`_get_device_handle("fake")` 返回 `getattr(torch, "fake", None)`。Simulator 将 `torch.fake` 指向与 `torch.meta` 相同的 `_MetaDeviceModule` stub，使 `device_count()`、`current_device()` 等调用返回安全默认值。

同时 patch 了：
- `DeviceMesh._setup_world_group_and_device`：跳过 fake mesh 的真实 world-size 限制；各维度 FakeProcessGroup 随后由 `_init_process_groups()` 正式创建和注册
- `FSDP._get_device_from_mesh`：对 fake mesh 返回 `torch.device("meta")`
- `DTensor._random._resolve_device`：对 fake mesh 返回 `torch.device("meta")`

#### 3. CP P2P 通信捕获

DeepSeek-V4 的 CP 使用 `CompressorAttentionCP`，包含两类通信：
- `_WindowExchange`：P2P `c10d.isend/irecv` 在相邻 CP rank 间交换序列窗口
- `_allgather_seq`：`funcol.all_gather_tensor_autograd` 聚合压缩后的 KV

`_meta_safe_forward` patch 在短路 P2P 前，通过 `get_active_recorder()` 获取当前 `CommEventRecorder`，调用 `_record_comm_with_l0()` 记录 `cp_forward_send/recv` 和 `cp_backward_send/recv` 事件，确保 CP P2P 通信出现在 L0 图中。

#### 4. PP 元数据交换

PP stage 间的 tensor shape/dtype 元数据通过 `PipelineStage._send_meta/_recv_meta` 交换。在 multi_proc 模式下，这些使用 gloo 的 `send_object_list/recv_object_list` 真实传递（不被 comm_events 拦截器 no-op），保证 DYNAMIC 模式的 shape 推断正确。

### 运行方式

```bash
# PP=4 + CP=4：解析最终配置后自动启动 4 个 gloo 进程，模拟 64 rank
python3 scripts/run_simulator_spawn.py \
    --config deepseek_v4_pro_simulate_16_layers_pp4_cp4 \
    --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
```

高级用法可以直接使用 `torchrun`，但进程数在 Python 解析配置前已经确定，因此必须由调用方保证 `--nproc_per_node` 等于 CLI 覆盖后的最终 PP degree。

> [!IMPORTANT]
> - `--nproc_per_node` 必须等于 PP degree（如 PP=4 则 4 个进程）
> - 外部 gloo 进程的 `RANK` 必须连续取 `0..PP-1`，且 `WORLD_SIZE=PP`。例如 PP=4 时使用
>   `RANK=0,1,2,3`；`0,16,32,48` 这类值只是完整逻辑 mesh 中各 PP stage 的代表 rank，
>   由 simulator 内部映射，不能作为 gloo 的进程 `RANK`
> - `simulation.world_size`（或作为回退的 `NGPU`）必须等于完整模拟 world_size（如 PP=4 × CP=4 × DP=4 = 64）
> - 配置中 `context_parallel_degree` 等设为真实值（非 1），`data_parallel_shard_degree=-1` 自动计算
> - `simulated_parallel_degrees` 和 `TORCHTITAN_SIM_WORLD_SIZE` 由解析后的 runtime 自动生成
> - `run_simulator_spawn.py` 只会在全部 PP rank 成功退出、各自抓到本 stage 的前反向 template 并完成输出后报告成功；
>   任一 rank 失败、缺失或重复都会使启动器返回失败

### 输出

每个 PP stage 进程独立捕获自己的 L0-L3 IR，**直接以 rank ID 输出**到 `rank_N/` 目录，不合并到 Rank 0。各 stage 的 IR 独立完整，包含：

- `step_forward_l0_ops.csv` / `step_backward_l0_ops.csv` / `step_optimizer_l0_ops.csv`：L0 算子（仅 MB 0 捕获）
- `l1_schedule/stage_N_l1_schedule.csv`：L2 调度时序（所有 microbatch）
- `rank_schedule.csv`：L3 inter-rank 调度关系
- `simulation_result.json`：完整四层 IR（含 `pp_stage`/`pp_mb_idx` annotations）

### 通信捕获验证

PP=4+CP=4 配置下，每个 stage 的通信捕获情况：

| 通信类型 | 来源 | 每 stage 数量 | 说明 |
|----------|------|--------------|------|
| CP allgather | `_allgather_seq` | 20-39 (fwd) + 8-14 (bwd) | kv_compress + k_indexer 的 all_gather |
| CP P2P send/recv | `_WindowExchange` | 3-22 (fwd) + 4-11 (bwd) | 相邻 CP rank 间的序列窗口交换 |
| FSDP allgather | FSDP2 unshard | 含在 allgather 中 | forward + pre-backward 参数 unshard |
| FSDP reduce_scatter | FSDP2 reshard | 4-5 (bwd) | post-backward 参数 reshard |
| FSDP allreduce | 梯度同步 | 1 (opt) | optimizer 阶段梯度 allreduce |
| PP P2P | 1F1B 调度 | 含在 P2P 中 | stage 间 activation/gradient 传递 |

### Microbatch 分层捕获

| Microbatch | L0 (OpNode) | L1 (StepGraph) | L2 (Timeline) | L3 (CommEvent) |
|------------|-------------|----------------|---------------|-----------------|
| MB 0 | ✅ 完整捕获 | ✅ 从 L0 构建 | ✅ 完整 timeline | ✅ 完整捕获 |
| MB 1+ | ❌ 跳过 (pass-through) | ❌ 跳过 | ✅ 只捕获调度事件 | ✅ 完整捕获 |
| Optimizer | ✅ 完整捕获 | ✅ 从 L0 构建 | ✅ 完整 timeline | ✅ 完整捕获 |

L0/L1 捕获开销从 O(PP × num_mb × ops_per_mb) 降为 O(ops_per_mb)，与 microbatch 数量无关。

PP 内存估算不会只使用 MB 0 的曲线。它以完整捕获的 `(stage, comp_type)` raw memory
events 为模板，按照 L2 `SchedulePlan` 的通用 action 顺序实例化每个 microbatch；支持
F/B、I/W、`OVERLAP_F_B` 等 action 组合，不依赖特定 1F1B 名称。每个实例获得独立的
非持久 tensor ID，参数和 buffer 保持单实例，FSDP 显式 unshard/reshard 驻留事件也不会
按 microbatch 重复克隆。L0/L1 展示图仍保持折叠，因此不会因 microbatch 数量显著膨胀。

`ScheduleAction.schedule_order` 表示 pipeline plan 的逻辑顺序；`seq_idx` 保留其抓图来源
位置。两者不能混排，尤其是 DualPipeV 等包含通信、虚拟 stage 和 overlap action 的调度。

### 资源消耗

| 配置 | 进程数 | 每进程内存 | 总内存 | 耗时 |
|------|--------|-----------|--------|------|
| PP=4, CP=4, 16层 | 4 | ~200MB | ~1GB | ~4s |
| PP=16, 61层 | 16 | ~300MB | ~5GB | ~30s |
| PP=16, TP=8, CP=4, EP=128, 61层 | 16 | ~500MB | ~8GB | ~2min |

> [!TIP]
> 模型全程在 meta device 上运行，零显存分配。内存主要消耗在 Python 对象（OpNode、CommEvent 等），与模型层数和 op 数量成正比。

## 工作目录结构

```
torchtitan_npu/simulator/
├── config_registry.py        # 仿真配置工厂函数
├── trainer.py                # SimulationTrainer：一步训练 + 捕获 + 导出
├── meta_env.py               # Meta device 替身层 + multi_proc patch + PP context
├── moe_force_balance.py      # 强制 MoE 负载均衡
├── rank_table.py             # 从 ParallelDims/DeviceMesh 展开通信域（RankTable）
├── ir/                       # 四层 IR dataclass
│   ├── op_node.py            #   L0 OpNode (含 pp_stage/pp_mb_idx)
│   ├── step_graph.py         #   L1 StepGraph (MB 0 only)
│   ├── schedule_graph.py     #   L2 ScheduleGraph (TimelineEntry/DataPass/StepInstance)
│   └── workload_graph.py     #   L3 WorkloadGraph
├── capture/                  # 捕获管线
│   ├── dispatch_capture.py   #   TorchDispatchMode 算子级捕获 (含 _capture_l0 pass-through)
│   ├── comm_events.py        #   集合通信拦截 + CommEvent + timeline_events
│   ├── step_boundary.py      #   forward/backward/optimizer 边界识别
│   ├── schedule_builder.py   #   组装 L2 (从 OpNode+CommEvent 直接映射，无推断)
│   └── workload_builder.py   #   组装 L3 WorkloadGraph
├── cost/
│   └── op_cost_model.py      # NPU 算子 FLOPs/mem/comm_bytes 估算
├── hardware_shims/           # 硬件依赖算子的 meta 影子实现
│   ├── mhc_converter.py      #   MHC 算子（hc_pre/hc_head/hc_post）
│   └── smla_converter.py     #   SMLA 算子（sparse_attn/lightning_indexer）
└── viz/                      # 可视化导出
    ├── json_export.py        #   JSON (orjson 加速)
    ├── html_export.py        #   HTML L3→L2→L1→L0 层级展开
    ├── dot_export.py
    └── text_summary.py
```

## 常见问题

### Q: 仿真报错 FileNotFoundError: tokenizer path 相关

DeepSeek-V4-Pro 配置默认引用 `./tests/assets/tokenizer/deepseek_v4_pro_tokenizer`，该目录未随仓库提交。解决方法：运行时追加 `--hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer`，用仓库内置的 v3 tokenizer 替代（Simulator 在 meta device 上运行，不关心词表内容，详见"快速开始 → 准备 Tokenizer"章节）。

### Q: 仿真报错 "inductor_npu_ext is not available"

仿真配置已自动强制 `compile.enable=False`，如果仍报此错，检查是否在 `config_registry.py` 之外手动传了 `--compile.enable=True`。

### Q: 仿真报错 "narrow unexpectedly changed concrete size"

这是 DTensor 重分布在 meta device 下的 shape 不一致问题，`meta_env.py` 中的 `_patch_redistribute_local_tensor_for_meta` 已处理。如果仍出现，确认 `patch_device_type_to_meta()` 已被调用（`SimulationTrainer.__init__` 会自动调用）。

### Q: PP > 1 时仿真报错 "Expected _StageBackwardMeta from P2P"

Pipeline 并行下，`meta_env.py` 中的 `_patch_pipeline_stage_meta_exchange_for_fake_pg` 会将 PP 元数据交换从 P2P 改为进程内共享缓冲区，并强制 STATIC 推断模式。确认该 patch 已生效。

### Q: 如何查看特定 rank 的通信详情

multi_proc 模式下，各 stage 的 `rank_N/ir_export/rank_schedule.csv` 和 `rank_N/simulation_result.json` 中的 `data_passes` 列表包含每条通信的 `src_instance`/`dst_instance`/`comm_primitive`/`comm_group_ranks`，可按 rank ID 过滤查看。跨 stage 的 P2P 配对可通过 `comm_peer_rank` 字段关联。

### Q: forward 和 backward 的算子数量差异很大

这是正常的。DeepSeek-V4 的 MoE 模型在 forward 中有大量的 shape 操作（view、unsqueeze、slice 等）用于 attention 和 MoE 的 reshape，而 backward 不需要重复所有 shape 操作。此外，activation checkpointing 的 recomputation 算子已正确放在 backward template 中（通过 `_pp_context["phase"]` 检测）。

### Q: 仿真耗时过长

61 层 / 2048 die 的完整仿真（含 JSON 导出）约需 2-3 分钟。如果只需摘要和 HTML，可在配置中设置 `simulation.output_formats=["text", "html"]` 跳过 JSON 导出（JSON 文件在千卡规模下可达 GB 级）。Microbatch 分层捕获已将 L0 开销降为与 microbatch 数量无关。
