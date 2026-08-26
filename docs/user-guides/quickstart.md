# 快速上手

参考 [软件安装](./installation.md) 准备环境后，进入 `torchtitan-npu` 仓库根目录。除另有说明外，本文中的相对路径和命令均以仓库根目录为基准。以下步骤以 DeepSeek V3 模型为例，在 NPU 平台上运行 torchtitan-npu。

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

### 排查启动报错：查看更多 rank 日志

> [!TIP]
> `scripts/run_train.sh` 默认只在控制台打印 `LOG_RANK=0`，即 rank 0 的日志。多卡任务异常退出但控制台没有具体 Python 报错时，可指定需要打印的 rank 后重新运行：
>
> ```bash
> export LOG_RANK=0,1,2,3
> ```
>
> 排查完成后，可执行 `unset LOG_RANK` 恢复默认设置。
