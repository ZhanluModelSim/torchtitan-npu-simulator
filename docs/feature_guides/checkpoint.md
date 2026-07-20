# Checkpoint 使用指南

本文参考 torchtitan 上游 `docs/checkpoint.md`，说明`torchtitan-npu` 训练中常见的 checkpoint 保存、恢复和
Hugging Face 权重加载方式。

## Checkpoint 格式

`torchtitan-npu` 主要涉及两类权重格式：

| 格式 | 典型文件 | 适用场景 | 是否包含优化器等训练状态 |
| --- | --- | --- | --- |
| DCP | `.metadata`、`*.distcp` | 训练断点恢复、不同并行切分间重分片加载 | 可以包含 |
| Hugging Face | `model.safetensors`、`model.safetensors.index.json` | 从开源权重初始化、导出给推理或其他生态使用 | 仅模型权重 |

DCP 是 PyTorch Distributed Checkpoint 格式。它可以保存分布式训练下的模型、
优化器、学习率调度器、dataloader 进度和 `train_state`，因此最适合断点续训。
保存目录通常形如：

```text
outputs/<job_name>/checkpoint/step-1000/
```

其中 `outputs/<job_name>` 来自 `--dump_folder`，`checkpoint` 来自
`checkpoint.folder`，`step-1000` 是训练步数。

Hugging Face safetensors 是模型权重格式。它不保存优化器、学习率调度器和训练步数，
通常用于从预训练权重启动训练，或在训练结束后导出模型权重。

## 常用配置项

checkpoint 配置由 `torchtitan.components.checkpoint.CheckpointManager.Config`
提供，torchtitan-npu 通过 `CheckpointConfig` 补充 NPU 写盘和缓存控制项。
这些配置在模型的 `config_registry.py` 中设置，也可以通过 CLI 覆盖。

| 配置项 | 说明 |
| --- | --- |
| `enable` | 是否启用 checkpoint。**加载和保存都需要打开**。 |
| `folder` | checkpoint 子目录名，最终路径为 `{dump_folder}/{folder}`。 |
| `interval` | 每隔多少 step 保存一次 DCP checkpoint。 |
| `load_step` | 指定加载某一步。为 `-1` 时自动加载最新 checkpoint。 |
| `initial_load_path` | 当前输出目录没有 checkpoint 时，从该路径初始化。 |
| `initial_load_model_only` | 从 `initial_load_path` 加载时是否只加载模型权重，默认 `True`。 |
| `initial_load_in_hf` | 将 `initial_load_path` 或 `hf_assets_path` 当作 HF safetensors 权重加载。 |
| `last_save_model_only` | 训练最后一步是否只保存模型权重，默认 `True`。 |
| `last_save_in_hf` | 训练最后一步是否保存为 HF safetensors。 |
| `export_dtype` | `last_save_model_only=True` 时最终模型权重导出的 dtype。 |
| `load_only` | 只加载不保存，适合验证或调试。 |
| `keep_latest_k` | 只保留最近 k 个 checkpoint。设置为 `0` 表示全部保留。 |
| `exclude_from_loading` | 从 DCP 加载时排除部分状态，例如 `optimizer,lr_scheduler,dataloader`。CLI 传值必须用英文逗号分隔，不能空格分隔。 |
| `async_mode` | DCP 异步保存方式，常用值为 `disabled`、`async`。 |
| `sync_files` | 是否在写入 checkpoint 后同步文件，默认 `True`。设为 `False` 可降低保存耗时，但异常退出时落盘可靠性更弱。 |
| `drop_page_cache_after_save` | 保存后尝试释放 checkpoint 文件占用的 Linux page cache，默认 `False`。不删除文件，只影响主机页缓存。 |
| `empty_cache_after_save` | 保存后清理 NPU allocator cache，默认 `True`，用于减少 checkpoint 临时缓存占用。 |
| `create_seed_checkpoint` | 创建 seed checkpoint 模式，详见下文「创建 seed checkpoint」。 |
| `enable_first_step_checkpoint` | 是否在 step 1 强制保存一次 checkpoint，常用于校验保存路径连通性。 |
| `initial_load_in_hf_quantized` | 从 HF 量化权重加载，使用前需先开启 `initial_load_in_hf`。 |
| `enable_ft_dataloader_checkpoints` | TorchFT 场景下是否独立保存 per-replica dataloader 状态。 |

> 注意：如果 `checkpoint.enable=False`，即使配置了 `initial_load_path` 或
> `initial_load_in_hf=True`，也不会执行 checkpoint 加载逻辑。

### CLI 使用约定

- **布尔字段反转**：tyro 用 `--checkpoint.no-<field>` 把 bool 置 False，不是
  `--no-checkpoint.<field>` 也不是 `--checkpoint.<field> false`。例如要把 `load_only` 关掉，写
  `--checkpoint.no-load-only`。
- **列表字段**：`exclude_from_loading` 这类列表用英文逗号串成一个参数传入
  （如 `--checkpoint.exclude_from_loading optimizer,lr_scheduler,dataloader`）。
  以空格分隔多个 token 会被 tyro 当成子命令报 *Unrecognized options*。
- **覆盖 registry 默认值**：模型的 `config_registry.py` 已经设过的字段（如 dsv32
  4 层 debug config 把 `load_only=True`、`initial_load_path` 写死成某个路径），
  CLI 必须显式覆盖才能生效；只追加 `--checkpoint.enable` 不会把这些字段清掉。
  使用前可以先 `grep checkpoint=` 看一眼模型 registry 默认值。

## 保存 DCP checkpoint

### CLI 启动

以 DeepSeek-V3.2 4 层配置为例，启用 DCP 保存：

```bash
NGPU=16 \
MODULE=torchtitan_npu.models.deepseek_v32 \
CONFIG=deepseek_v32_671b_16npus \
bash scripts/run_train.sh \
  --dump_folder ./outputs/dsv32_4layers \
  --checkpoint.enable \
  --checkpoint.folder checkpoint \
  --checkpoint.interval 100
```

训练过程中会在 `./outputs/dsv32_4layers/checkpoint/step-100`、
`step-200` 等目录保存 DCP checkpoint。中间 checkpoint 包含完整训练状态，
可用于断点续训。

### 在 config_registry.py 中设置

```python
from torchtitan.components.checkpoint import CheckpointManager

checkpoint=CheckpointManager.Config(
    enable=True,
    folder="checkpoint",
    interval=100,
    keep_latest_k=5,
    async_mode="disabled",
)
```

如果希望最后一步也保留完整训练状态，而不是只保存模型权重，将
`last_save_model_only` 设为 `False`：

```python
checkpoint=CheckpointManager.Config(
    enable=True,
    interval=100,
    last_save_model_only=False,
)
```

如果只关心最终模型权重，可以保留默认的 `last_save_model_only=True`，并指定导出精度：

```python
checkpoint=CheckpointManager.Config(
    enable=True,
    interval=500,
    last_save_model_only=True,
    export_dtype="bfloat16",
)
```

## 读取 DCP checkpoint

### 自动续训

使用相同的 `--dump_folder` 和 `--checkpoint.folder` 重新启动训练时，框架会自动从
最新的 `step-*` DCP checkpoint 恢复：

```bash
NGPU=16 \
MODULE=torchtitan_npu.models.deepseek_v32 \
CONFIG=deepseek_v32_671b_16npus \
bash scripts/run_train.sh \
  --dump_folder ./outputs/dsv32_4layers \
  --checkpoint.enable \
  --checkpoint.folder checkpoint
```

如需指定某一步：

```bash
bash scripts/run_train.sh \
  --dump_folder ./outputs/dsv32_4layers \
  --checkpoint.enable \
  --checkpoint.load_step 500
```

### 从另一个 DCP 目录初始化

当希望从旧任务的 checkpoint 启动一个新任务时，使用新的 `dump_folder`，并将
`checkpoint.initial_load_path` 指向旧 checkpoint 的 step 目录：

```bash
NGPU=16 \
MODULE=torchtitan_npu.models.deepseek_v32 \
CONFIG=deepseek_v32_671b_16npus \
bash scripts/run_train.sh \
  --dump_folder ./outputs/dsv32_new_job \
  --checkpoint.enable \
  --checkpoint.initial_load_path ./outputs/dsv32_4layers/checkpoint/step-500
```

默认只加载模型权重。如果要从这个 DCP 完整恢复优化器、学习率调度器和训练步数，
在 `config_registry.py` 中设置：

```python
checkpoint=CheckpointManager.Config(
    enable=True,
    initial_load_path="./outputs/dsv32_4layers/checkpoint/step-500",
    initial_load_model_only=False,
)
```

如果只想加载模型和部分状态，可以排除不需要的键：

```bash
bash scripts/run_train.sh \
  --checkpoint.enable \
  --checkpoint.initial_load_path ./outputs/dsv32_4layers/checkpoint/step-500 \
  --checkpoint.exclude_from_loading optimizer,lr_scheduler,dataloader
```

可排除的常用键包括 `optimizer`、`lr_scheduler`、`dataloader` 和 `train_state`。

> 注意：如果 `{dump_folder}/{checkpoint.folder}` 已经存在可用 checkpoint，
> 框架会优先从该目录续训，并忽略 `initial_load_path`。从新权重启动实验时，
> 请使用新的 `dump_folder` 或清理旧的 checkpoint 目录。

## 读取 Hugging Face 权重

从 HF safetensors 权重初始化时，需要启用 checkpoint，并设置
`initial_load_in_hf=True`。路径应包含 `model.safetensors` 或
`model.safetensors.index.json`，同时建议包含 tokenizer、`config.json` 等 HF 资产。

```bash
NGPU=16 \
MODULE=torchtitan_npu.models.deepseek_v32 \
CONFIG=deepseek_v32_671b_128npus \
bash scripts/run_train.sh \
  --checkpoint.enable \
  --checkpoint.initial_load_in_hf \
  --checkpoint.initial_load_path ./checkpoint/DeepSeek-V3.2
```

如果不显式传 `checkpoint.initial_load_path`，框架会尝试使用顶层
`hf_assets_path`：

```bash
bash scripts/run_train.sh \
  --checkpoint.enable \
  --checkpoint.initial_load_in_hf \
  --hf_assets_path ./checkpoint/DeepSeek-V3.2
```

`checkpoint.initial_load_path` 的优先级高于 `hf_assets_path`。
HF 权重只能作为模型权重加载，不能恢复优化器或训练步数。

## 保存 Hugging Face 权重

如果希望训练最后一步直接导出 HF safetensors 权重，可以同时启用
`last_save_in_hf` 和 `last_save_model_only`：

```bash
NGPU=16 \
MODULE=torchtitan_npu.models.deepseek_v32 \
CONFIG=deepseek_v32_671b_128npus \
bash scripts/run_train.sh \
  --dump_folder ./outputs/dsv32_hf_export \
  --checkpoint.enable \
  --checkpoint.interval 500 \
  --checkpoint.last_save_in_hf \
  --checkpoint.export_dtype bfloat16
```

最终权重会保存在类似下面的目录中：

```text
./outputs/dsv32_hf_export/checkpoint/step-1000/
```

该目录用于模型权重导出，不适合作为完整训练断点恢复。保存 HF 权重时需要经过
state_dict 转换和 safetensors 合并，耗时通常高于 DCP 保存。

## 创建 seed checkpoint

seed checkpoint 用于先在单卡 CPU 初始化模型，再让多卡任务通过 DCP 重分片加载。
创建时需要关闭并行切分，并使用 `NGPU=1`：

```bash
NGPU=1 \
MODULE=torchtitan_npu.models.deepseek_v32 \
CONFIG=deepseek_v32_671b_16npus \
bash scripts/run_train.sh \
  --checkpoint.enable \
  --checkpoint.create_seed_checkpoint \
  --parallelism.data_parallel_replicate_degree 1 \
  --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.pipeline_parallel_degree 1 \
  --parallelism.context_parallel_degree 1 \
  --parallelism.expert_parallel_degree 1
```

生成的 `step-0` checkpoint 后续可作为 `checkpoint.initial_load_path` 使用。

## 离线转换 DCP 为 Hugging Face 权重

训练过程中保存的 DCP checkpoint 可以通过独立脚本
`scripts/checkpoint_conversion/convert_to_hf.py` 离线转换为 Hugging Face
safetensors 格式。适用于训练结束后需要将权重用于推理、评测或发布的场景，
无需重新启动训练流程。

与训练配置中 `--checkpoint.last_save_in_hf` 的区别在于，该脚本可以对
**已有的任意 step 的 DCP checkpoint** 进行转换，而不依赖训练最后一步的自动导出。

**基本用法**

```bash
python scripts/checkpoint_conversion/convert_to_hf.py \
  ./outputs/dsv32_4layers/checkpoint/step-1000 \
  ./outputs/dsv32_4layers/hf_output \
  --model_name deepseek_v32 \
  --model_flavor deepseek_v32_671b_16npus \
  --hf_assets_path ./assets/hf/DeepSeek-V3.2 \
  --export_dtype bfloat16
```

转换完成后，输出目录结构如下：

```text
./outputs/dsv32_4layers/hf_output/
├── model-00001-of-00001.safetensors
└── model.safetensors.index.json
```

> 注意：`input_dir` 必须是完整的 DCP step 目录（包含 `.metadata` 文件）
> `--export_dtype` 可选 `float16`、`bfloat16`、`float32`
