# SFT 指令微调

当前针对 Deepseek-V4 模型提供全参数 SFT 支持, 训练特性支持与Pretrain训练保持一致。

## 快速开始：Deepseek-V4-Flash SFT

### 前置条件

- 准备 Deepseek-V4-Flash-BF16 的 HuggingFace 预训练权重, 参考[Deepseek-V4训练指导中的模型权重准备](https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md#%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E5%87%86%E5%A4%87)部分
- 准备训练数据集, 参考 tests/assets/sft_test/README.md 文档下载样例数据验证, 或参考[数据准备](#数据准备)

### 单机减层微调启动

```bash
CONFIG_FILE=torchtitan_npu/models/deepseek_v4/train_configs/deepseek_v4_285b_4layers_debug_sft.toml bash scripts/run_train.sh
```

### 8机全参数微调启动

参考[Deepseek-V4训练指导中的启动训练](https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md#%E5%90%AF%E5%8A%A8%E8%AE%AD%E7%BB%83)部分, 修改 run_train_multinodes.sh 脚本

```bash
CONFIG_FILE=torchtitan_npu/models/deepseek_v4/train_configs/deepseek_v4_285b_43layers_4k_128die_sft.toml bash scripts/run_train_multinodes.sh
```

## 数据预处理

### 支持的数据集文件格式

由于采用 [datasets](https://pypi.org/project/datasets/) 库中 load_dataset 函数对数据集进行加载, 支持的文件格式请参考库文档。

### 已支持样本的数据格式
1. QA格式
```python
{
    "query": "用户问题",
    "answer": "助手回答"
}
```

2. Alpaca格式
```python
{
    "instruction": "系统提示",
    "input": "用户问题",
    "output": "助手回答"
}
```

### 支持情况说明

- 当前Dataloader仅支持单轮对话数据, 多轮对话数据目前还未支持。
- 样本数据格式除QA格式和Alpaca格式外均未支持, 如想要支持自定义格式, 请修改 torchtitan_npu/models/deepseek_v4/text_datasets.py 中的 process_sample 函数, 实现自定义加载逻辑
- 当前不支持数据packing。如果数据超长则在右侧截断, 如果长度不足则补充eos。
