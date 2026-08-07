# 快速上手

参考[软件安装](./installation.md)准备环境后，以 DeepSeek-V4 模型为例，按照以下步骤在 NPU 平台上运行 torchtitan-npu。

## 数据准备

1. 从 [DeepSeek-V4 模型页面](https://huggingface.co/deepseek-ai/DeepSeek-V4/tree/main)下载 Tokenizer。

Tokenizer 目录中需包含以下文件：

```text
dsv4_tokenizer/
├── tokenizer.json
└── tokenizer_config.json
```

`scripts/run_train.sh` 默认从示例路径 `/path/to/dsv4_tokenizer` 读取 Tokenizer。运行前需将该路径替换为实际目录，或通过 `HF_ASSETS_PATH` 指定：

```bash
export HF_ASSETS_PATH=/path/to/dsv4_tokenizer
```

2. 准备数据集。

已在 `tests/assets/c4_test/` 中预置 `c4_test` 测试数据集，`scripts/run_train.sh` 默认读取该目录，无需额外下载。

使用其他数据集时，需同时指定数据集名称和目录：

```bash
export DATASET=dataset_name
export DATASET_PATH=/path/to/dataset
```

## 配置 CANN 环境变量

`scripts/run_train.sh` 会自动加载以下 CANN 环境脚本，无需在当前 shell 中重复执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
```

CANN 安装在其他目录时，需将 `scripts/run_train.sh` 中的路径改为实际的 `set_env.sh` 路径。

## 启动训练任务

DeepSeek-V4 训练任务使用 `scripts/run_train.sh` 启动。脚本默认使用 1 张 NPU、`deepseek_v4_debugmodel` 配置和 `TEST_OVERRIDES`。

### 单卡训练任务

使用默认配置启动训练：

```bash
bash scripts/run_train.sh
```

以下命令使用 Golden 基线配置，并覆盖本地 batch size、序列长度和训练步数：

```bash
USE_GOLDEN=1 \
bash scripts/run_train.sh \
  --training.local-batch-size 1 \
  --training.seq-len 4096 \
  --training.steps 5
```

> [!NOTE]
> `scripts/run_train.sh` 配置项说明：
> - `MODULE`：模型 Python 模块，默认为 `torchtitan_npu.models.deepseek_v4`。
> - `CONFIG`：`config_registry.py` 中的配置函数，默认为 `deepseek_v4_debugmodel`。
> - `NGPU`：当前节点参与训练的 NPU 数量，默认为 `1`。
> - `HF_ASSETS_PATH`：Tokenizer 目录，默认为示例路径 `/path/to/dsv4_tokenizer`，运行前需替换为实际目录。
> - `DATASET` 和 `DATASET_PATH`：数据集名称和目录，默认使用仓内的 `tests/assets/c4_test/`。
> - `GOLDEN_OVERRIDES`：固定的 Golden 参考配置，无需修改。
> - `TEST_OVERRIDES`：待测试实现的 override 列表，可按测试需求修改。
> - `USE_GOLDEN`：设置为 `1` 时使用 `GOLDEN_OVERRIDES`；未设置时使用 `TEST_OVERRIDES`。
> - 脚本后的其他参数会原样传给 `torchtitan.train`，可用于覆盖配置函数中的训练和并行参数。


### 单机 4 卡 EP2 训练任务

```bash
NGPU=4 \
bash scripts/run_train.sh \
  --training.local-batch-size 1 \
  --training.seq-len 4096 \
  --training.steps 5 \
  --parallelism.data-parallel-shard-degree 4 \
  --parallelism.context-parallel-degree 1 \
  --parallelism.expert-parallel-degree 2 \
  --checkpoint.no-enable
```

DeepSeek-V4 的 SMLA 融合路径和 TND 数据约定见 [DeepSeek-V4 TND 适配](./TND.md)。

### 排查启动报错：查看更多 rank 日志

> [!TIP]
> `scripts/run_train.sh` 默认只在控制台打印 `LOG_RANK=0`，即 rank 0 的日志。多卡任务异常退出但控制台没有具体 Python 报错时，可指定需要打印的 rank 后重新运行：
>
> ```bash
> export LOG_RANK=0,1,2,3
> ```
>
> 排查完成后，可执行 `unset LOG_RANK` 恢复默认设置。
