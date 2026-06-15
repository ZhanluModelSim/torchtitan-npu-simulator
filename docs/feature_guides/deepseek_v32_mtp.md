# 多Token预测特性(Multi Token Prediction, MTP)

在大规模语言模型的训练与推理优化中，MTP 通过单次前向传播同时预测多个连续目标 Token，大幅提升模型训练效率与数据利用率。传统单 Token 预测仅能学习逐词依赖关系，MTP 则通过扩展预测长度、引入辅助损失函数，显著加速模型收敛速度，尤其在长文本、代码、多轮对话等任务上效果显著。
torchtitan-npu 在 DeepSeek-V3.2 模型的基础上进一步适配了MTP训练特性，实现了可配置长度的MTP训练，同时支持FSDP2/EP/TP等分布式训练。

## 实现原理

参考[DeepSeek-V3的技术报告](https://arxiv.org/pdf/2412.19437)，我们在 DeepSeek-V3.2 模型代码中引入了`MTPModule`类的定义，其继承于原有的`TransformerBlockV32`类，并在此基础上新增了MTP模块所需的额外结构与参数。此外，为最大化复用原有 Transformer 层成熟的分布式训练实现，我们在模型顶层定义中，将标准 Transformer Layer 与 MTP Layer 统一封装至`model.layers`列表中，实现与原有FSDP2、EP、TP等分布式并行逻辑的无缝兼容。相关代码见`torchtitan_npu/models/deepseek_v32/model.py`

为了实现 MTP 模块的有效训练，我们新增了适配 MTP 模块的训练损失函数。具体而言，每个MTPModule都会独立计算对应的交叉熵损失；在此基础上，模型总训练损失被定义为主损失与 MTP 损失的加权和。相关代码实现见`torchtitan_npu/patches/torchtitan/loss.py`

## 配置选项

`num_mtp_modules` 和 `mtp_loss_weight` 已作为 NPU 扩展字段定义在
`torchtitan_npu.config.configs.TrainingConfig` 中，在模型
`config_registry.py` 中配置。

| 配置项 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `num_mtp_modules` | int | 0(不开MTP) | MTP预测的token个数 |
| `mtp_loss_weight` | float | 0.3 | MTP训练损失的权重，total_loss = main_loss + mtp_loss_weight * mtp_loss |

### 配置示例

```python
from torchtitan_npu.config.configs import TrainingConfig

training = TrainingConfig(
    local_batch_size=4,
    seq_len=2048,
    num_mtp_modules=1,
    mtp_loss_weight=0.3,
)
```

启动时也可以通过命令行覆盖：

```bash
bash scripts/run_train.sh \
  --training.num_mtp_modules 1 \
  --training.mtp_loss_weight 0.3
```
