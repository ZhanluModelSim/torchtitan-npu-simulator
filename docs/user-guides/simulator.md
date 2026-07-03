# Simulator 使用指南

torchtitan-npu Simulator 是一个**侧载式（side-loaded）**的计算图捕获工具，它在不依赖真实 NPU 硬件、不分配真实显存的前提下，完整捕获一个训练 step 中**所有卡**的计算图，并输出四层 IR（L0 OpNode → L1 StepGraph → L2 ScheduleGraph → L3 WorkloadGraph）及可视化文件。

## 工作原理

Simulator 通过以下机制实现"零硬件捕获"：

1. **Meta Device 替身**：将 `torchtitan.tools.utils.device_type` 猴子补丁为 `"meta"`，模型全程在 `torch.device("meta")` 上构建和运行，不分配真实显存。
2. **Fake Process Group**：强制 `comm.mode=fake_backend`，单进程内模拟全部 rank 的 `world_size`，`ParallelDims.build_mesh()` 正常建立完整 mesh。
3. **集合通信拦截**：拦截 `torch.distributed.*` 和 `_functional_collectives.*` 的所有集合通信调用（含 autograd 变体和 P2P），在 meta 张量上不做真实通信，只记录通信事件（op 类型、group、shape、bytes）。
4. **MoE 强制负载均衡**：强制 `debug.moe_force_load_balance=True`，路由逻辑退化为与输入数值无关的 round-robin，保证 meta 张量下 shape 推断正确。
5. **算子级捕获**：通过 `TorchDispatchMode` 拦截每个 aten/npu 算子调用，记录 op 类型、输入输出 shape、producer/consumer 依赖关系。

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

### 方式二：使用基础 CANN 镜像手动搭建

如果需要自定义环境，可从基础 CANN 镜像开始：

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

### 方式三：本地环境

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

Simulator 复用现有的 `scripts/run_train.sh` 脚本和 `torchtitan_npu.entry` 入口，只需将 `MODULE` 改为 `torchtitan_npu.simulator`，`CONFIG` 改为对应的仿真配置函数名，并设置 `COMM_MODE=fake_backend`：

```bash
# 16 层小规模验证（快速跑通，约 1 分钟）
MODULE=torchtitan_npu.simulator \
CONFIG=deepseek_v4_pro_simulate_16_layers \
COMM_MODE=fake_backend \
NGPU=384 \
bash scripts/run_train.sh --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
```

```bash
# 61 层验收配置（384 die，PP=1/TP=1/CP=1/EP=192）
MODULE=torchtitan_npu.simulator \
CONFIG=deepseek_v4_pro_simulate_61_layers \
COMM_MODE=fake_backend \
NGPU=384 \
bash scripts/run_train.sh --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
```

```bash
# 千卡规模（2048 die，PP=16/TP=8/CP=4/EP=128/FSDP auto）
MODULE=torchtitan_npu.simulator \
CONFIG=deepseek_v4_pro_simulate_61_layers_pp16_tp8_cp4_ep128 \
COMM_MODE=fake_backend \
NGPU=2048 \
bash scripts/run_train.sh --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
```

也可以直接用 `python3 -m` 调用（与 `run_train.sh` 中 `COMM_MODE` 分支等价）：

```bash
NGPU=384 LOCAL_RANK=0 python3 -m torchtitan_npu.entry \
    --module torchtitan_npu.simulator \
    --config deepseek_v4_pro_simulate_61_layers \
    --comm.mode=fake_backend \
    --training.steps=1 \
    --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
```

> [!NOTE]
> - `NGPU` 环境变量指定模拟的**总卡数**（world_size），必须与配置中的并行度乘积一致。
> - `--comm.mode=fake_backend` 是必须的，启用 fake Process Group 实现单进程模拟全部 rank。
> - `--training.steps=1` 是默认值，Simulator 只捕获一个 step。
> - `--hf_assets_path` 指定 tokenizer 路径，用仓库内置的 `deepseekv3_tokenizer` 即可（见上一步说明）。
> - 不需要 `torchrun`，单进程 `python3 -m` 即可运行。

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

| 配置函数名 | 模型 | 层数 | 并行策略 | world_size | 说明 |
|-----------|------|------|---------|------------|------|
| `deepseek_v4_pro_simulate_16_layers` | DeepSeek-V4-Pro | 16 | PP=1/TP=1/CP=1/EP=16 | 384 | 快速验证 |
| `deepseek_v4_pro_simulate_61_layers` | DeepSeek-V4-Pro | 61 | PP=1/TP=1/CP=1/EP=192 | 384 | 验收配置 |
| `deepseek_v4_pro_simulate_61_layers_pp16_tp8_cp4_ep128` | DeepSeek-V4-Pro | 61 | PP=16/TP=8/CP=4/EP=128/FSDP=-1 | 2048 | 千卡规模 |

每个仿真配置内部复用对应的真实训练配置（如 `deepseek_v4_pro_debug_61_layers_4k_384die()`），原样继承 model_spec、并行度、optimizer 等全部字段，仅强制以下三项：

- `comm.mode = "fake_backend"`（fake PG）
- `compile.enable = False`（捕获需要 eager dispatch）
- `debug.moe_force_load_balance = True`（MoE 强制负载均衡）

## 输出文件

仿真完成后，输出文件默认写入 `simulator_output/<配置名>/` 目录（可通过配置中的 `simulation.output_dir` 自定义）：

| 文件 | 格式 | 内容 |
|------|------|------|
| `summary.txt` | 纯文本 | 摘要报告：各阶段 op 数、FLOPs、内存、通信量统计、未识别算子列表 |
| `trace.html` | 自包含 HTML | 可视化页面：L3 卡片 + L2 RankTable 网格与调度泳道 + L1 步骤汇总 + L0 算子 DAG |
| `compute_graph.dot` | Graphviz | 算子依赖图，可用 `dot -Tsvg compute_graph.dot -o graph.svg` 渲染 |
| `simulation_result.json` | JSON | 完整四层 IR 结构化数据，供程序化消费 |

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

输出遵循 [workload-model-platform spec](https://github.com/ZhanluModelSim/workload-model-platform/tree/master/spec) 定义的四层 IR：

| 层级 | 类名 | 文件 | 描述 |
|------|------|------|------|
| L0 | `OpNode` | `ir/op_node.py` | 单个算子调用：op_type、输入输出 TensorMeta、FLOPs/mem/comm_bytes 估算、前后依赖 |
| L1 | `StepGraph` | `ir/step_graph.py` | 一个 forward/backward/optimizer 步骤的 DAG：节点集合、entry/exit 节点、拓扑校验 |
| L2 | `ScheduleGraph` | `ir/schedule_graph.py` | 并行调度编排：StepInstance（每 rank 每模板一个）、DataPass（跨 rank 通信）、RankTable |
| L3 | `WorkloadGraph` | `ir/workload_graph.py` | 顶层容器：迭代语义、DataFlow（dataloader 输入）、跨迭代参数传递 |

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

    # 3. 转为仿真配置（自动强制 fake_backend / compile=False / moe_force_load_balance）
    return _to_simulation_config(base_config, output_dir="./simulator_output/my_model")
```

关键约束：

- `world_size`（`NGPU` 环境变量）必须等于 `dp_replicate × dp_shard × cp × tp × pp`。`data_parallel_shard_degree=-1` 时由 torchtitan 自动计算。
- `pipeline_parallel_degree > 1` 时，DeepSeek-V4 不支持 MTP，需设 `num_mtp_modules=0`。
- `pipeline_parallel_degree > 1` 时，`local_batch_size` 需 ≥ `pp_degree`（1F1B 调度需要足够 microbatch）。

## 工作目录结构

```
torchtitan_npu/simulator/
├── config_registry.py        # 仿真配置工厂函数
├── trainer.py                # SimulationTrainer：一步训练 + 捕获 + 导出
├── meta_env.py               # Meta device 替身层 + 集合通信/P2P/PP 元数据拦截
├── moe_force_balance.py      # 强制 MoE 负载均衡
├── rank_table.py             # 从 ParallelDims/DeviceMesh 展开通信域（RankTable）
├── ir/                       # 四层 IR dataclass
│   ├── op_node.py            #   L0 OpNode
│   ├── step_graph.py         #   L1 StepGraph
│   ├── schedule_graph.py     #   L2 ScheduleGraph
│   └── workload_graph.py     #   L3 WorkloadGraph
├── capture/                  # 捕获管线
│   ├── dispatch_capture.py   #   TorchDispatchMode 算子级捕获
│   ├── comm_events.py        #   集合通信拦截 + 通信事件记录
│   ├── step_boundary.py      #   forward/backward/optimizer 边界识别
│   ├── schedule_builder.py   #   组装 L2 ScheduleGraph
│   └── workload_builder.py   #   组装 L3 WorkloadGraph
├── cost/
│   └── op_cost_model.py      # NPU 算子 FLOPs/mem/comm_bytes 估算
├── hardware_shims/           # 硬件依赖算子的 meta 影子实现
│   ├── mhc_converter.py      #   MHC 算子（hc_pre/hc_head/hc_post）
│   └── smla_converter.py     #   SMLA 算子（sparse_attn/lightning_indexer）
└── viz/                      # 可视化导出
    ├── json_export.py
    ├── html_export.py
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

`simulation_result.json` 中的 `iteration.schedule.data_passes` 列表包含每条通信的 `src_instance`/`dst_instance`/`comm_primitive`/`volume_bytes`，可按 rank ID 过滤查看。

### Q: 仿真耗时过长

61 层 / 2048 die 的完整仿真（含 JSON 导出）约需 2-3 分钟。如果只需摘要和 HTML，可在配置中设置 `simulation.output_formats=["text", "html"]` 跳过 JSON 导出（JSON 文件在千卡规模下可达 GB 级）。
