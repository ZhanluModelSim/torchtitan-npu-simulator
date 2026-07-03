# Config 矩阵选择指南

基于 Step 0 审计中收集的 `git diff <branch_a>..<branch_b>` 变更范围，与用户逐轴确认需要验证的 config 组合。

## 各轴说明

| 轴 | 为什么重要 | 至少需要 |
| --- | --- | --- |
| **TP degree** | TP>1 引入 all-gather/reduce-scatter，改变归约顺序和 broadcasting 语义 | 1 个 TP=1 + 1 个 TP>1 |
| **torch.compile** | 编译器融合改变算子执行顺序，`a+(b+c)` vs `(a+b)+c` 即可导致浮点偏差 | 若改动涉及被编译的模块，需 compile=on + compile=off 各一 |
| **Activation Checkpointing** | recompute 改变前向/反向执行边界，影响中间值保存精度 | 若改动在 AC 包裹的层内，需 `full` + `selective`（或 `off`）各一 |
| **Optimizer** | 不同优化器更新规则不同（AdamW 的 eps、momentum 累积） | 若改动涉及 optimizer 或 loss scale，需覆盖受影响的 optimizer |
| **Expert Parallelism** (MoE) | EP 改变 expert 路由和 all-to-all 通信 | MoE 模型至少 1 个 EP>1 config |
| **Context Parallelism** | CP 改变序列划分和 attention mask 计算 | 若改动涉及 attention 或序列处理，需 CP>1 |
| **ModelConverters** | 改动可能涉及启用或关闭部分 modelconverter 的场景 | 若改动涉及某些 modelconverters，或 models 里面可能被 modelconverters 覆盖的场景，需要验证开启或关闭该 modelconverter 的场景。 |

## 讨论流程

1. 根据 Step 0 的 diff 结果，列出变更涉及的文件和模块。
2. 逐轴与用户确认：该轴是否受变更影响？需要几组 config？
3. 汇总 config 矩阵（每个组合 = 一次训练运行），确认 NPU 资源和时间是否可接受。
4. 优先选择已有 `config_registry.py` 配置函数，避免从头新增配置。

> [!TIP]
> 最小可行矩阵：1 个 TP=1 + 1 个 TP>1，其余轴使用默认值。
> 变更范围越大，矩阵越大。若矩阵过大无法全部跑完，优先保留 TP>1 的 config。

## 确认话术示例

> 根据 Step 0 diff，变更涉及 `converters/kernels/rms_norm.py`（算子替换）和 `model.py`（RMSNorm 调用）。
> 建议验证：
> - `deepseek_v32_671b_4layers_debug`（TP=1，覆盖基本路径）
> - `deepseek_v32_671b_61layers_4k_128die`（TP=4，覆盖 TP>1 路径）
>
> compile / optimizer 轴不受影响，不额外覆盖。总计 2 config × 2 分支 = 4 次运行。是否合理？

> [!IMPORTANT]
> **Config 修改必须确认。** 选定 config 矩阵后，可能需要修改 config 文件才能实际运行（如调整 converter 开关、修改并行度、改 tokenizer 路径等）。**任何对 config 文件的修改，必须在执行前逐条列出并征得用户同意。** 不要静默修改 config —— 改动 converter 或并行策略会直接影响验证的代码路径，用户必须知情。
