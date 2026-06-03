# 快速上手

参考 [软件安装](./installation.md) 准备环境后，以DeepSeek-V3.2模型为例，按照如下步骤在 NPU 平台上运行
torchtitan-npu。

## 数据准备

1. 准备 Tokenizer [（以 DeepSeekV3.2 网络为例）](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/tree/main)。

新建 "deepseekv3.2-tokenizer" 目录，将 `tokenizer.json` 和 `tokenizer_config.json` 文件下载至该目录。

也可以通过以下方式下载 tokenizer：

```bash
# 从huggingface下载 DeepSeek V3.2 tokenizer https://huggingface.co/settings/tokens
python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.2 --assets tokenizer
```

`torchtitan_npu/models/deepseek_v32/config_registry.py` 中的 `hf_assets_path` 默认指向
`./assets/hf/DeepSeek-V3.2`，使用上面的下载命令时无需修改。如果手动下载到 `deepseekv3.2-tokenizer`，
则将实际使用的配置函数中的路径改为：

```python
hf_assets_path="./deepseekv3.2-tokenizer",
```

测试场景可直接使用项目预置在 `tests/assets/tokenizer/deepseekv3_tokenizer/` 的 tokenizer，无需重复下载：

```bash
bash scripts/run_train.sh --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
```

2. 下载数据集。以 [enwiki 数据集](https://huggingface.co/datasets/lsb/enwiki20230101)
   为例：

```bash
cd ./tests/assets
hf download lsb/enwiki20230101 --repo-type=dataset --local-dir .
cd ../..
```

## 配置 CANN 环境变量

```bash
source /usr/local/Ascend/cann/set_env.sh
```

如果环境中使用的是其他 CANN 安装路径，请按实际路径调整。

## 启动训练任务

启动 torchtitan-npu 训练任务时，推荐使用以下脚本：单机环境使用 `scripts/run_train.sh`，多机环境使用
`scripts/run_train_multinodes.sh`。以下展示了一些常见任务的启动方式。

### 单机训练任务

默认配置，以 16 NPU 启动 DeepSeek-V3.2 4层debug模型训练任务：
```bash
bash scripts/run_train.sh
```

自定义配置，调整训练步数和全局 batch size：

```bash
bash scripts/run_train.sh \
  --training.steps 100 \
  --training.global_batch_size 32
```

> [!NOTE]
> * `MODULE`: 指定模型 Python 模块，例如 `torchtitan_npu.models.deepseek_v32`。
> * `CONFIG`: 指定 `config_registry.py` 中的配置函数，例如 `deepseek_v32_671b_4layers_debug`。
> * `NGPU`: 指定单节点参与训练的 NPU 数量，`scripts/run_train.sh` 默认值为 16。
> * `--training.steps` 与 `--training.global_batch_size`: 动态覆盖 registry 配置中的字段。


### 多机训练任务

在执行启动命令前，按照集群的实际情况编辑 `scripts/run_train_multinodes.sh` 文件中的网络与节点配置：

```toml
# TODO change to your network interface
Network_Interface=enp23s0f3 # 填入 ifconfig 的驱动名
...
# TODO change to your device ips
IPs=('192.168.xxx.xxx' '192.168.xxx.xxx') # 填入集群的所有IP
# TODO change 192.168 to your local IP
LOCAL_HOST=`ifconfig|grep "inet 192.168"| awk '{print $2}'` # 将 "192.168" 替换为当前 IP
```

在所有参与训练的节点上同时执行 `scripts/run_train_multinodes.sh`，以启动多机预训练任务。以 DeepSeek-V3.2 完整模型为例：
```bash
bash scripts/run_train_multinodes.sh
```

> [!NOTE]
> * 脚本会自动通过 `LOCAL_HOST` 匹配 `IPs` 数组以推导当前机器的 `NODE_RANK`。若提取规则错误导致未匹配成功，脚本将报错退出。
> * 多机通信依赖相应的端口开放，请确保 `MASTER_PORT` (默认 6300) 以及 HCCL 通信基础端口 (默认 30000) 不被防火墙拦截。


### 排查启动报错:查看更多 rank 日志

> [!TIP]
> 默认启动脚本只在控制台打印 `LOG_RANK=0`(即 rank 0)的日志。若任务异常退出、但控制台没有具体的Python报错,真正报错可能记录在其他rank。此时可临时通过 `LOG_RANK` 指定需要打印日志的rank(用逗号分隔)后重新运行,以定位报错。例如同时查看本机 8 个 rank:
>
> ```bash
> LOG_RANK=0,1,2,3,4,5,6,7 bash scripts/run_train.sh
> ```
>
> > 排查时建议覆盖本机全部rank。多机任务中,各节点日志会自动写入本节点 `logs/` 目录，需登录对应节点查看(同样默认只记录local rank 0)。

### torchtitan 仓库内置训练任务
除了 torchtitan-npu 已经适配的模型外，还可以直接下载 torchtitan 代码，使用原生配置启动训练任务：

1. 拉取 torchtitan 代码

```bash
git clone https://github.com/pytorch/torchtitan.git
git checkout ac13e536c84e7f6647b14fa9375c3c8a8a2b8578
```

2. 将 torchtitan 源代码移动至 torchtitan-npu 项目中

```bash
cp ./torchtitan/torchtitan ./torchtitan-npu/ -r
```

3. 在 torchtitan-npu 项目中，使用 torchtitan 原生 toml 配置文件，启动训练。以 llama3 的 debug_model 配置为例:

```bash
NGPU=8 \
TRAIN_FILE=torchtitan_npu.entry \
MODULE=llama3 \
CONFIG=llama3_debugmodel \
bash scripts/run_train.sh
```
