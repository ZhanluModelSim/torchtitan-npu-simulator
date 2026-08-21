# Checkpoint 使用指南

本文说明如何使用 TorchTitan 的 checkpoint 机制保存、加载
`torchtitan-npu` 训练状态，以及如何加载或保存 Hugging Face safetensors 权重。

当前实现基于上游 `torchtitan.components.checkpoint.CheckpointManager`：

- DCP（Distributed Checkpoint）用于保存和加载分布式训练状态。
- Hugging Face safetensors 只用于模型权重的加载或保存，不能加载优化器和训练步数。

本文不覆盖权重转换工具。需要转换 Hugging Face 权重时，使用上游仓库提供的转换工具。

## 当前配置入口

训练配置由以下注册表提供：

| 模块 | 配置入口 | checkpoint 默认值 |
| --- | --- | --- |
| `torchtitan_npu.models.deepseek_v4.config_registry` | `deepseek_v4_debugmodel`、`deepseek_v4_flash`、`deepseek_v4_pro` | `enable=False`，`interval=100` |
| `torchtitan_npu.models.deepseek_v3_2.config_registry` | `deepseek_v3_2_debugmodel` | `enable=False`，`interval=10`，`last_save_model_only=False` |

`enable=False` 时既不会保存 checkpoint，也不会执行 checkpoint 加载。训练命令需要显式
传入 `--checkpoint.enable`，或在 Python 配置中设置 `checkpoint.enable = True`。

## checkpoint 格式与目录

### DCP

DCP 是 PyTorch Distributed Checkpoint 格式，可以保存模型、优化器、学习率调度器、
dataloader 和训练状态，适合断点续训以及改变并行切分后的加载。

启用 `checkpoint.folder=checkpoint` 时，目录结构通常为：

```text
<dump_folder>/checkpoint/step-<step>/
├── .metadata
└── *.distcp
```

其中 `<dump_folder>` 来自 `--dump-folder`，`step-<step>` 是训练步数。DCP 加载会检查
`.metadata`，并按 `--checkpoint.load-step` 选择指定步数；默认值 `-1` 表示选择最新步数。

### Hugging Face safetensors

Hugging Face 权重通常包含 `model.safetensors` 或
`model.safetensors.index.json`，也应准备对应的 tokenizer 和 `config.json` 等配置文件。
该格式只表示模型权重，不包含优化器、学习率调度器和训练步数。

## 常用配置项

配置类型来自上游 `torchtitan.config.Checkpoint`，Python 配置使用下划线命名，命令行
使用 tyro 的连字符命名：

| 配置项 | 作用 |
| --- | --- |
| `enable` | 启用 checkpoint 保存和加载。默认 `False`。 |
| `folder` | checkpoint 子目录名，默认 `checkpoint`；最终路径为 `{dump_folder}/{folder}`。 |
| `interval` | 保存 DCP checkpoint 的步数间隔。 |
| `load_step` | 要加载的步数；`-1` 表示最新 checkpoint。 |
| `initial_load_path` | 当前输出目录没有 checkpoint 时使用的初始 checkpoint 路径，必须是绝对路径或远程 URI。 |
| `initial_load_model_only` | 初始加载是否只加载模型权重，默认 `True`。设为 `False` 才会尝试加载完整训练状态。 |
| `initial_load_in_hf` | 将初始路径按 Hugging Face safetensors 读取。HF 加载只能加载模型权重。 |
| `initial_load_in_hf_quantized` | 从 HF 量化权重加载；使用前必须启用 `initial_load_in_hf`。 |
| `last_save_model_only` | 最后一步是否只保存模型权重，默认 `True`。设为 `False` 才保存完整训练状态。 |
| `last_save_in_hf` | 最后一步是否以 Hugging Face safetensors 格式保存。必须同时使用模型权重保存模式。 |
| `export_dtype` | 保存时模型权重导出的 dtype，可用 `float16`、`bfloat16`、`float32`。 |
| `async_mode` | 保存方式：`disabled`、`async` 或 `async_with_pinned_mem`。 |
| `keep_latest_k` | 保留最近的 checkpoint 数量；`0` 表示全部保留，不能设置为 `1`。 |
| `exclude_from_loading` | 从 DCP 加载时排除状态，例如 `optimizer,lr_scheduler,dataloader`。 |
| `enable_first_step_checkpoint` | 是否在第一个训练 step 后立即保存一次 checkpoint。 |
| `create_seed_checkpoint` | 不应用并行切分，创建可供后续任务重分片加载的 seed checkpoint。 |
| `load_only` | 只加载、不保存 checkpoint，适合验证或调试。 |
命令行布尔字段使用反向选项关闭，例如 `--checkpoint.no-enable`；不要写成
`--checkpoint.enable false`。列表字段使用英文逗号分隔，不要用空格拆成多个 token。

## 保存 DCP checkpoint

以下命令使用当前脚本默认的 `deepseek_v3_debugmodel` 配置。运行前准备 Ascend/CANN 环境：

```bash
NGPU=1 \
bash scripts/run_train.sh \
  --hf-assets-path tests/assets/deepseek_v3 \
  --dump-folder ./outputs/dsv3_checkpoint \
  --checkpoint.enable \
  --checkpoint.folder checkpoint \
  --checkpoint.interval 100 \
  --training.steps 200
```

训练过程中会生成：

```text
./outputs/dsv3_checkpoint/checkpoint/step-100/
./outputs/dsv3_checkpoint/checkpoint/step-200/
```

也可以在配置注册表中设置：

```python
from torchtitan.components.checkpoint import CheckpointManager

checkpoint = CheckpointManager.Config(
    enable=True,
    folder="checkpoint",
    interval=100,
    keep_latest_k=5,
    async_mode="disabled",
)
```

## 加载 DCP checkpoint

### 自动加载最新 checkpoint

使用相同的 `--dump-folder` 和 `--checkpoint.folder` 重新启动，并启用 checkpoint：

```bash
NGPU=1 \
HF_ASSETS_PATH=/path/to/dsv4_tokenizer \
bash scripts/run_train.sh \
  --dump-folder ./outputs/dsv4_checkpoint \
  --checkpoint.enable \
  --checkpoint.folder checkpoint
```

当输出目录中存在可用的 `step-*` checkpoint 时，框架会默认加载最新一步。

### 指定加载步数

```bash
bash scripts/run_train.sh \
  --dump-folder ./outputs/dsv4_checkpoint \
  --checkpoint.enable \
  --checkpoint.load-step 100
```

### 从其他目录初始化

使用新的输出目录，并通过 `--checkpoint.initial-load-path` 指向旧 checkpoint 的完整
step 目录。该参数必须使用绝对路径或远程 URI；相对路径会触发上游的 `ValueError`：

```bash
NGPU=1 \
HF_ASSETS_PATH=/path/to/dsv4_tokenizer \
bash scripts/run_train.sh \
  --dump-folder ./outputs/dsv4_new_job \
  --checkpoint.enable \
  --checkpoint.initial-load-path /absolute/path/to/dsv4_checkpoint/checkpoint/step-100
```

默认只加载模型权重。若要加载优化器、学习率调度器和训练状态，应显式关闭
`initial_load_model_only`：

```bash
bash scripts/run_train.sh \
  --dump-folder ./outputs/dsv4_new_job \
  --checkpoint.enable \
  --checkpoint.initial-load-path /absolute/path/to/dsv4_checkpoint/checkpoint/step-100 \
  --checkpoint.no-initial-load-model-only
```

如果 `{dump_folder}/{checkpoint.folder}` 已经存在可用 checkpoint，框架会优先从该目录
优先从该目录加载，并忽略 `initial_load_path`。从新权重启动实验时，应使用新的 `dump-folder` 或
清理旧的 checkpoint 目录。

如只需要模型和部分训练状态，可以排除不需要的键：

```bash
bash scripts/run_train.sh \
  --dump-folder ./outputs/dsv4_model_only \
  --checkpoint.enable \
  --checkpoint.initial-load-path /absolute/path/to/dsv4_checkpoint/checkpoint/step-100 \
  --checkpoint.exclude-from-loading optimizer,lr_scheduler,dataloader
```

## 加载 Hugging Face 权重

从 HF safetensors 初始化时，需要启用 checkpoint、设置
`--checkpoint.initial-load-in-hf`，并确保 `--checkpoint.initial-load-model-only` 保持
为 `True`：

```bash
NGPU=1 \
HF_ASSETS_PATH=/path/to/dsv4_tokenizer \
bash scripts/run_train.sh \
  --dump-folder ./outputs/dsv4_from_hf \
  --checkpoint.enable \
  --checkpoint.initial-load-in-hf \
  --checkpoint.initial-load-path /absolute/path/to/checkpoint/DeepSeek-V4
```

如果不传 `--checkpoint.initial-load-path`，checkpoint 管理器会尝试使用模型配置中的
`hf_assets_path`。`initial_load_path` 优先级高于 `hf_assets_path`。HF 权重不能加载
优化器或训练步数。

## 保存 Hugging Face 权重

训练最后一步可以直接保存 HF safetensors。该模式只保存模型权重，不能作为完整训练
断点使用：

```bash
NGPU=1 \
HF_ASSETS_PATH=/path/to/dsv4_tokenizer \
bash scripts/run_train.sh \
  --dump-folder ./outputs/dsv4_hf \
  --checkpoint.enable \
  --checkpoint.last-save-in-hf \
  --checkpoint.export-dtype bfloat16 \
  --training.steps 100
```

`last_save_in_hf` 需要模型提供 state-dict adapter，并且必须保持
`last_save_model_only=True`。`deepseek_v3_2` 和 `deepseek_v4` 模型均提供
对应的 state-dict adapter；具体可用性仍取决于模型配置和权重格式。

## 创建 seed checkpoint

seed checkpoint 用于先创建未应用并行切分的模型状态，再由多卡任务通过 DCP 重分片加载。
创建时使用单卡，并将各并行度设为 `1`：

```bash
NGPU=1 \
HF_ASSETS_PATH=/path/to/dsv4_tokenizer \
bash scripts/run_train.sh \
  --dump-folder ./outputs/dsv4_seed \
  --checkpoint.enable \
  --checkpoint.create-seed-checkpoint \
  --parallelism.data-parallel-replicate-degree 1 \
  --parallelism.data-parallel-shard-degree 1 \
  --parallelism.tensor-parallel-degree 1 \
  --parallelism.pipeline-parallel-degree 1 \
  --parallelism.context-parallel-degree 1 \
  --parallelism.expert-parallel-degree 1
```

生成的 `step-0` 目录可以作为后续任务的 `checkpoint.initial_load_path`。

## 常见限制与验证范围

- 需要可用的 `torch`、`torch_npu`、TorchTitan、CANN 和 Ascend NPU 环境；本文命令未在当前
  Windows 环境执行。
- DCP 适用于完整训练状态加载；HF safetensors 只适用于模型权重。
- 文档中的模型入口、配置字段和命令来自源码与脚本的静态核对；多卡训练、权重格式兼容性
  和 NPU 性能未在本次文档修改中重新实测。

## 相关文档

- [快速上手](../user-guides/quickstart.md)
- [安装指南](../user-guides/installation.md)
- [训练启动脚本](../../scripts/run_train.sh)
- [DeepSeek-V4 checkpoint 配置](../../torchtitan_npu/models/deepseek_v4/config_registry.py)
