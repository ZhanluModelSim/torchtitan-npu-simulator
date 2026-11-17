# 精度对齐开发指南

本文档定义模型、算子、训练流程和并行策略变更时统一使用的精度对齐方法。

## 1. 核心原则

- 先明确对齐目标，再选择基线和指标。
- 比较双方必须使用 `相同权重、输入、随机状态和运行模式`。
- 基线必须说明来源和适用范围。
- 先检查 `局部模块` 前向和反向，再检查 `模型整网` 单步前反向和多步训练。

## 2. 基线与对齐目标

### 2.1 基线

| 基线 | 说明 |
|---|---|
| 推理基线 | 官方发布的 HuggingFace 推理脚本或者 Transformers 推理库作为推理基线 |
| 小算子基线(golden) | 仓内高精度小算子实现，为融合算子、模型组件或训练路径提供稳定参考 |
| 修改前基线 | 判断改动是否改变模型或训练行为，修改前后精度需要对齐 |

### 2.2 对齐目标

#### 比特一致

适用情况：模型重构、权重或 checkpoint 格式转换、metadata、layout 或 packed 等只改变存储形式的变更、其他明确要求的场景。

#### 数值近似

适用情况：不同算子实现、不同归约顺序、不同并行策略以及混合精度等之间的比较。

#### 精度指标

比较前先确认两组张量的 `shape` 相同、张量非空，且有效元素不包含 NaN 或 Inf。使用平均绝对误差（Mean Absolute Error，MEAN_AE）、最大相对误差（Max Relative Error，MAX_RE）、平均相对误差（Mean Relative Error，MEAN_RE）和余弦相似度（Cosine Similarity，COS）作为精度指标。

MEAN_AE 提供稳定的绝对误差尺度，尤其适用于 `loss` 和 `grad_norm` 等标量；MEAN_RE 提供相对误差约束，COS 反映整体方向一致性。

设参考张量为 $r$，被测张量为 $a$，两者均包含 $N$ 个元素。令 $\rho = 10^{-4}$ 表示相对误差下限，用于避免接近零的参考值放大逐元素相对误差；$\varepsilon$ 表示避免范数除零的浮点极小值。

| 指标 | 计算公式 | 比特一致阈值 | `logits` 数值近似阈值 | `loss` 数值近似阈值 | `grad_norm` 数值近似阈值 |
|---|---|---:|---:|---:|---:|
| MEAN_AE | $\displaystyle \frac{1}{N}\sum_{i=1}^{N}\lvert a_i-r_i \rvert$ | `0` | `<= 0.02` | `<= 0.02` | `<= 0.02` |
| MAX_RE | $\displaystyle \max_{1 \le i \le N}\frac{\lvert a_i-r_i \rvert}{\max(\lvert r_i \rvert,\rho)}$ | `0` | — | `<= 0.005` | `<= 0.005` |
| MEAN_RE | $\displaystyle \frac{1}{N}\sum_{i=1}^{N}\frac{\lvert a_i-r_i \rvert}{\max(\lvert r_i \rvert,\rho)}$ | `0` | `<= 0.02` | `<= 0.005` | `<= 0.005` |
| COS | $\displaystyle \frac{a \cdot r}{\max(\lVert a \rVert_2,\varepsilon)\max(\lVert r \rVert_2,\varepsilon)}$ | `1` | `>= 0.9999` | — | — |

## 3. 精度对齐流程

非数值改动，例如纯文档、日志文案或不参与计算的重构，可以在 PR 中说明后跳过。
凡是可能改变模型输出、梯度、loss、参数更新或随机行为的改动，都必须按需完成以下验证。

### 3.1 推理基线 vs 小算子基线

目的：使小算子基线与推理基线始终保持 `比特一致` 对齐。

对比方法：

- 选取典型 case，使用相同的权重、输入，并开启确定性计算；
- 运行一次推理和小算子基线的一次前向计算；
- 判断推理和训练 logits 是否 `比特一致`。

示例：[DeepSeek-V4 推理基线与小算子基线前向对齐方法](../../scripts/numerical_alignment/README.md#步骤三transformers-和-torchtitan-前向对比)。权重准备与转换见同一文档的步骤一、二。

DeepSeek-V4 的 Transformers 基线固定为 `transformers==5.13.0`，默认启用数值对齐补丁。验收记录必须区分原生与打补丁的 Transformers。
脚本在前向计算前调用 `set_allow_hf32(False)`，关闭 NPU matmul、conv 和 ACLNN 的 HF32。
Routed expert 在 `w2` 投影前乘 router score，并回到 BF16；将乘法延后到投影后会改变舍入位置。

### 3.2 改动局部精度对比

目的：确认被修改的算子或模块本身没有公式、边界、索引或梯度错误。

对比方法：

- 抽取改动的部分模块固定输入，覆盖正常 shape、最小 shape、边界 shape；
- 比较小算子基线和被测实现模块输出精度，至少需要满足 `数值近似` 的要求（参考 `logits` 数值近似阈值）；
- 存在反向时，还需对比梯度。

示例：[局部模块前向和反向精度对比方法](../../scripts/numerical_alignment/README.md#局部模块对比)，使用相同输入、权重和上游梯度比较 Torch RMSNorm 与 AscRMSNorm。

### 3.3 小算子基线 vs 被测实现

目的：确认融合算子、自定义算子或其他实现接入模型后，其数值结果相对于小算子基线满足对应的精度要求。

对比方法：

- 选取典型 case，使用相同的权重、输入，并开启确定性计算；
- 分别运行小算子基线和被测实现的前向计算；
- 判断两种实现的 `logits` 和 `loss` 是否满足对应的精度要求，若被测实现为对小算子路径的修改，需满足 `比特一致`。

示例：[DeepSeek-V4 小算子基线与被测实现训练指标对齐方法](../../scripts/numerical_alignment/README.md#数值近似对比)，通过 runner 的 `USE_GOLDEN` 比较 Golden 和融合实现的 `loss`、`grad_norm`。

训练对比默认使用 `seq_len=4096`；其他长度通过 `--seq-len` 统一指定。

### 3.4 修改前后训练精度对比

目的：发现单步指标无法暴露的误差累积、梯度偏移和 optimizer update 异常等。

对比方法：

- 使用修改前后 commit，选取典型 case，使用相同的权重、输入，并开启确定性计算；
- 至少运行 `100` 个训练 step；
- 声明不改变计算结果时，`loss` 和 `grad_norm` 必须 `比特一致`；涉及计算变化时，需满足 `数值近似`。

示例：[修改前后训练精度对齐方法](../../scripts/numerical_alignment/README.md#步骤四训练-loss-和-grad_norm-对比)，通过两个 Git ref 指定比较对象，使用 `--steps 100` 运行两侧训练。

## 4. 验证范围

以 PR 的文件变更列表确定验证范围。

| 改动目录或文件 | 必做验证 |
|---|---|
| `torchtitan_npu/**` | `3.1`、`3.2`、`3.3`、`3.4` |
| `torchtitan_npu/override/common` | `3.1`、`3.2`、`3.4` |
| `torchtitan_npu/override/<golden.py>` | `3.1`、`3.2`、`3.4` |
| `torchtitan_npu/override/<非common\非golden.py>` | `3.2`、`3.3`、`3.4` |
| `scripts/checkpoint_conversion/**` | `3.1`、`3.4` |
| `scripts/numerical_alignment/**` | `3.1`、`3.2`、`3.3` |
| `scripts/run_train.sh` | `3.4` |
| `docs/**`、README和其他文档类 | 可跳过 |
