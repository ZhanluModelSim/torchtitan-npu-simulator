# 快速上手

参考 [软件安装](./installation.md) 准备环境后，进入 `torchtitan-npu` 仓库根目录。除另有说明外，本文中的相对路径和命令均以仓库根目录为基准。本文先以 DeepSeek V3 说明通用启动方式，再给出 DeepSeek-V4/V4.1 多卡训练和 TorchAO-NPU 低精度训练入口。

## 数据准备

1. 使用仓库预置的 DeepSeek V3 Tokenizer，目录为 `tests/assets/deepseek_v3/`，单卡默认命令会直接使用该目录，无需额外下载。

```text
tests/assets/deepseek_v3/
├── tokenizer.json
└── tokenizer_config.json
```

如需使用其他 Tokenizer，可通过 `HF_ASSETS_PATH` 指定：

```bash
export HF_ASSETS_PATH=/path/to/tokenizer
```

2. 准备数据集。

已在 `tests/assets/c4_test/` 中预置 `c4_test` 测试数据集，示例 wrapper 默认使用该目录，无需额外下载。

使用其他数据集时，需同时指定数据集名称和目录：

```bash
export DATASET=dataset_name
export DATASET_PATH=/path/to/dataset
```

## 配置 CANN 环境变量

当 CANN 安装在其他目录时，推荐通过 `ASCEND_SET_ENV_PATH` 指定 `set_env.sh` 的路径，并将其传给脚本：

```bash
ASCEND_SET_ENV_PATH=/path/to/ascend-toolkit/set_env.sh \
  bash scripts/run_train.sh
```

未设置该变量时，`scripts/run_train.sh` 会自动按以下顺序查找可用的 `set_env.sh`：

```text
/usr/local/Ascend/cann/set_env.sh
/usr/local/Ascend/ascend-toolkit/set_env.sh
/home/developer/Ascend/ascend-toolkit/set_env.sh
```

无需在当前 shell 中重复执行 `source`。

## 启动训练任务

DeepSeek V3 单卡训练任务可直接使用 `scripts/run_train.sh` 启动。`scripts/run_train.sh` 默认使用 1 张 NPU 和 `deepseek_v3_debugmodel` 配置，并把额外命令行参数原样透传给训练入口。

### 单卡训练任务

使用默认配置启动训练：

```bash
NGPU=1 \
bash scripts/run_train.sh \
  --hf-assets-path tests/assets/deepseek_v3 \
  --dataloader.dataset c4_test \
  --dataloader.dataset-path tests/assets/c4_test \
  --training.local-batch-size 1 \
  --training.seq-len 2048 \
  --training.steps 5
```

> [!NOTE]
> 示例 wrapper / 底层 launcher 配置项说明：
> - `ASCEND_SET_ENV_PATH`：可选，自定义 CANN `set_env.sh` 路径；设置后优先加载该文件。
> - `MODULE`：模型 Python 模块，默认为 `torchtitan.models.deepseek_v3`。
> - `CONFIG`：`torchtitan/models/deepseek_v3/config_registry.py` 中注册的配置函数，默认为 `deepseek_v3_debugmodel`。
> - `NGPU`：当前节点参与训练的 NPU 数量，默认为 `1`。
> - `HF_ASSETS_PATH`：Tokenizer 目录，默认为仓内 `tests/assets/deepseek_v3`。
> - `DATASET` 和 `DATASET_PATH`：数据集名称和目录，默认使用仓内的 `tests/assets/c4_test/`。
> - 脚本后的其他参数会原样传给 `torchtitan_npu.train`，可用于覆盖配置函数中的训练和并行参数。


### 单机 8 卡 EP8 训练任务

直接复用 DeepSeek-V4 单机 8 卡示例脚本。该脚本默认使用 8 卡、EP8/DP8 并行配置和 `deepseek_v4_flash_43layers_16experts` 模型配置。运行前需准备与 DeepSeek-V4 配套的 Tokenizer，并将下面的 `HF_ASSETS_PATH` 替换为其实际目录：

```bash
HF_ASSETS_PATH=/path/to/DeepSeekV4_tokenizer \
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --training.steps 5
```

DeepSeek-V4 的 SMLA 融合路径和 TND 数据约定见 [DeepSeek-V4 TND 适配](../feature_guides/deepseek_v4_tnd.md)。

### DeepSeek-V4 TorchAO-NPU 低精度训练

先按[软件安装](./installation.md#4-安装-torchao-npu可选)安装仓内适配包，或将
`torchao_npu` 源码的父目录加入 `PYTHONPATH`。随后在普通训练命令后显式增加量化 CLI：

> [!NOTE]
> 当前低精度训练仅支持 A5（Ascend 950）硬件。

```bash
HF_ASSETS_PATH=/path/to/DeepSeek-V4-Flash \
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --training.steps 5 \
  --extension.quantization.enable-quantized-training \
  --extension.quantization.recipe all_block_fp8
```

源码方式示例：

```bash
python3 -m pip install torchao==0.17.0
export PYTHONPATH="/path/to/custom/parent${PYTHONPATH:+:${PYTHONPATH}}"
```

自定义目录必须直接包含 `torchao_npu/__init__.py`；使用仓内源码时，对应目录为
`<torchtitan-npu>/experiments/torchao-npu`。单机和多机示例分别调用
`scripts/run_train.sh` 和 `scripts/run_train_multinodes.sh`，脚本只透传量化 CLI。
未设置
`--extension.quantization.enable-quantized-training` 时，仍使用高精度训练。

该入口复用 torchtitan 的预训练/续训练循环，并不表示已经提供 SFT 专用数据处理或训练入口。

启用低精度训练时，每个节点需使用
A5（Ascend 950）硬件，安装相同版本的 `torchao_npu` 及其依赖，并执行相同命令；
`NODE_IPS` 的顺序决定节点 rank：

```bash
NODE_IPS=your_ip1,your_ip2,... \
HF_ASSETS_PATH=/path/to/DeepSeekV4_tokenizer \
CKPT_SAVE_LOAD_PATH=/path/to/save_ckpt \
CKPT_INIT_LOAD_PATH=/path/to/init_load_ckpt \
bash examples/deepseek_v4/deepseek_v4_flash_cpt_4k_a3.sh \
  --extension.quantization.enable-quantized-training \
  --extension.quantization.recipe all_block_fp8 \
  --training.steps 5
```

常用低精度配置如下，均位于 `--extension.quantization` 下；表中参数和布尔开关省略该公共前缀：

| 参数 | 选项 | 用途 |
| --- | --- | --- |
| `enable-quantized-training` | `--enable-quantized-training` / `--no-enable-quantized-training` | 启用或关闭低精度训练，默认关闭。 |
| `recipe` | `all_mxfp8`、`mix`、`all_block_fp8`、`all_hif8` | 选择全 MXFP8、混合 MXFP8/Block FP8、全 Block FP8 或全 HiF8。 |
| `enable-mxfp4-qat` | `--enable-mxfp4-qat` / `--no-enable-mxfp4-qat` | 为 routed expert 启用或关闭 MXFP4 QAT 约束，默认关闭。 |
| `li-quantization` | `fp8`、`mxfp8`、`mxfp4`、`hif8` | 选择 LI Q/K 的量化类型，默认不启用；DeepSeek-V4.1 仅支持 `mxfp4`。 |
| `kv-norm-quantization.format` | `mxfp8` | 启用 DeepSeek-V4 KV Cache 的 MXFP8 量化。 |
| `enable-fsdp-prequantize` | `--enable-fsdp-prequantize` / `--no-enable-fsdp-prequantize` | 在 FSDP all-gather 前预量化 Block FP8 权重，减少通信量，默认关闭。 |
| `fsdp-prequantize-fqns` | FQN 后缀模式列表（默认 `None`） | 仅在 `enable-fsdp-prequantize` 下生效。白名单内的权重保留预量化、其余显式关闭；`None` 用 recipe 默认白名单，显式空表与开关开启冲突会报 `ValueError`。命中权重的分片若不满足对齐约束，运行时立即报错终止（不静默回退），并给出移除该 FQN 或调整并行度的提示。 |
| `enable-hif8-save-quant-codes` | `--enable-hif8-save-quant-codes` / `--no-enable-hif8-save-quant-codes` | 把 HiF8 量化矩乘算子（`npu_quantize`/`npu_dynamic_quant`/`npu_grouped_matmul`/`npu_quant_matmul`）结果保留在 selective activation checkpointing 保存边界内，backward 不再重新推导；仅在 `recipe=all_hif8` 时生效，默认关闭。 |
| `save-block-ops-level` | `0`-`3`（默认 `0`） | 把 attention / mHC / MoE-routing 相关 NPU 自定义算子加入 selective activation checkpointing 的 MUST_SAVE 列表，用显存换取 backward 更少重复计算。 |

### DeepSeek-V4.1 TorchAO-NPU 低精度训练

DeepSeek-V4.1 复用上述 TorchAO-NPU 低精度入口，当前面向 A5（Ascend 950）硬件。Lightning Indexer 与稀疏 attention 量化的配置、组合、依赖和限制见[低精度训练特性指南](../feature_guides/low_precision_training.md#deepseek-v41-li-与稀疏-attention-量化)。

DeepSeek-V4.1 使用 `li-quantization=mxfp4` 启用 Lightning Indexer量化，使用
`enable-sparse-attention-quantization` 启用 KV source Compressor 与稀疏 attention 量化。两项可在
标准 A5 融合路径提供 BF16 fallback 的前提下独立选择；`USE_GOLDEN=1` 不提供这些
fallback。启动日志中出现
`Applied TorchAO-NPU recipe=...` 表示 recipe 已应用。

DeepSeek-V4.1 模型做 Block FP8 低精度预训练可运行以下脚本：

```bash
HF_ASSETS_PATH=/path/to/DeepSeek-V41_tokenizer \
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh
```

DeepSeek-V4.1 模型做 QAT 训练可运行以下脚本。该入口同时启用 routed expert MXFP4
fake quant、Lightning Indexer 和稀疏 attention/KV source Compressor 量化：

```bash
HF_ASSETS_PATH=/path/to/DeepSeek-V41_tokenizer \
bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_qat_4k_a5.sh
```

### 排查启动报错：查看更多 rank 日志

> [!TIP]
> `scripts/run_train.sh` 默认只在控制台打印 `LOG_RANK=0`，即 rank 0 的日志。多卡任务异常退出但控制台没有具体 Python 报错时，可指定需要打印的 rank 后重新运行：
>
> ```bash
> export LOG_RANK=0,1,2,3
> ```
>
> 排查完成后，可执行 `unset LOG_RANK` 恢复默认设置。
