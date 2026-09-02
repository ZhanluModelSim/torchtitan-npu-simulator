# 快速上手

参考 [软件安装](./installation.md) 准备环境后，进入 `torchtitan-npu` 仓库根目录。除另有说明外，本文中的相对路径和命令均以仓库根目录为基准。本文先以 DeepSeek V3 说明通用启动方式，再给出 DeepSeek-V4 多卡训练和 TorchAO-NPU 低精度训练入口。

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

直接复用 DeepSeek-V4 单机 8 卡示例脚本。该脚本默认使用 8 卡、EP8/DP8 并行配置和 `deepseek_v4_flash_43layers_16experts` 模型配置：

```bash
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
未设置 `--extension.quantization.enable-quantized-training` 时，配置树保持高精度版本。

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
  --extension.quantization.no-enable-mxfp4-qat \
  --training.steps 5
```

两份 A3 示例均默认运行高精度训练。只有显式传入
`--extension.quantization.enable-quantized-training` 时，配置构建阶段才会导入并应用 TorchAO-NPU Converter。

低精度 recipe 的目标范围如下：

| `RECIPE` | Attention 和 shared expert | Routed grouped experts |
| --- | --- | --- |
| `all_mxfp8` | MXFP8 | MXFP8 |
| `mix`（默认） | MXFP8 | Block FP8 |
| `all_block_fp8` | Block FP8 | Block FP8 |

量化相关 CLI 参数：

- `--extension.quantization.enable-quantized-training` 与 `--extension.quantization.no-enable-quantized-training`：选择低精度或高精度通路；默认使用高精度，只有显式传入 enable 开关才启用低精度。
- `--extension.quantization.recipe`：选择 `all_mxfp8`、`mix` 或 `all_block_fp8`，默认使用 `mix`。
- `--extension.quantization.enable-mxfp4-qat` 与 `--extension.quantization.no-enable-mxfp4-qat`：控制 routed expert 的 Block FP8 weight 是否增加 MXFP4 QAT fake quant 数值约束，默认关闭，仅对包含 Block FP8 的 recipe 生效。该选项不是持久化 4-bit 参数训练，也不会把算子替换为原生 A8W4 GEMM。
- `--extension.quantization.dst-type-max`：MXFP4 fake quant 的目标数据类型最大值，默认 `0.0`，由数据类型自动推导。
- `--profiler.enable-profiling`：启用 profiler；如需 CANN profiler override，还需将 `torchtitan_npu.override.common.profiler.cann` 加入 override imports。
- `USE_GOLDEN`：设为 `1` 时选择 golden attention override；默认使用 Ascend 融合算子路径。

启动日志中应出现 `Applied TorchAO-NPU recipe=...` 和
`Converted ... config node(s) for torchao-npu`，分别表示 recipe 已读取以及目标模型配置已转换。

### 排查启动报错：查看更多 rank 日志

> [!TIP]
> `scripts/run_train.sh` 默认只在控制台打印 `LOG_RANK=0`，即 rank 0 的日志。多卡任务异常退出但控制台没有具体 Python 报错时，可指定需要打印的 rank 后重新运行：
>
> ```bash
> export LOG_RANK=0,1,2,3
> ```
>
> 排查完成后，可执行 `unset LOG_RANK` 恢复默认设置。
