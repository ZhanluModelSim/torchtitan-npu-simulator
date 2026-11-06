# 通用 Loop 抽象与单文件验收

日期：2026-11-05。分支：`feat/npu-simulator-local`。

## 抽象结果

Raw bundle v3 包含标准 block diffusion v1 与通用 LOOP v1。`model.py` 不再
包含 ALoDLM 专属类型、Gate、双流数据、训练目标或退出／提交策略。

`LoopedModel.begin/step` 负责共享层执行、pre-coda carry、可微外部反馈以及
按需 Coda/readout。固定深度 dense/MoE 循环可以独立使用，不需要 Gate 或任何
论文配方。读出可为最后一步、每一步或不读出；也可只生成 features 而跳过词表投影。

ALoDLM 的具体策略迁入 `scripts/block_diffusion/recipes/alodlm.py`，复用同一
Loop 机制。其 Transformer、RoPE、carry normalization 和 Coda 计算没有复制。
Checkpoint 转换明确处理 `backbone.*` 到 `loop.backbone.*`，且不重复注册模块。

`ModelConfig.qk_norm` 为通用 backbone 选项，默认 False 保持标准 v1 行为；
参数与推理语义统计已同步考虑该选项。Native 标准训练对照仍限定在已验证的
原有配置，不自动声明 Loop 或新增架构选项已接入。

## 验收

环境：Python 3.11.16、PyTorch 2.12.0+cpu、真实 TorchTitan
`ac13e536c84e7f6647b14fa9375c3c8a8a2b8578`。

```bash
python -B -m pytest -q --import-mode=prepend \
  --confcutdir=tests/unit_tests/models/block_diffusion_reference \
  tests/unit_tests/models/block_diffusion_reference
```

主 agent 复跑：**230 passed, 14 warnings in 10.58s，无跳过**。Warning 来自
上游 TorchScript 弃用提示。

| 项目 | 结果 |
| --- | --- |
| 原有标准模型、ALoDLM 配方与 native 标准对照 | 原有 208 项保持通过 |
| 通用 Loop 与单文件交付 | 20 项新增，覆盖无论文策略的 dense/MoE、读出成本、共享参数、carry、反馈、CPU/meta 和隔离导入 |
| QK norm 负载统计 | 2 项新增，参数字节与实际模型一致，TP 下 norm 权重复制及事件形状正确 |

测试确认 raw 中没有论文专属接口；Loop 不自动创建 gate 参数，增加执行深度
不增加物理 backbone 参数。独立的非 ALoDLM 客户端验证这是真实的复用机制。

## 重构前后数值对照

基线为 `c614e8a` 的 raw `model.py`，临时冻结加载，不将旧实现复制进仓库。
按 QK norm 开／关、carry norm 开／关、两种反馈规则组合为 8 种配置。
使用同一组权重及非零 gate 权重，固定随机种子，核对：

- 随机退出轨迹、每轮 logits／gate、final carry。
- 所有 practical objective 分量和全部参数梯度。
- 两次 SGD 更新后的每个参数。
- 更新后的推理结果、提交状态和两种反馈行为。

主 agent 与独立验收均确认全部 **逐位一致（atol=0、rtol=0）**。ALoDLM
数据、Gate、目标函数和决策规则只迁移位置，没有改变其数学定义。

## 单文件对接

隔离测试只复制 canonical `model.py` 到临时目录，并以全新 `python -I`
进程导入该副本。在 CPU/meta 上执行标准 prefill／denoise／commit 和固定深度
Loop，确认未加载 `torchtitan`、`torchtitan_npu`、`raw_reference` 或 `recipes`。
副本与 canonical 文件逐字节一致。

这说明模型定义只依赖 PyTorch 与标准库，适合独立交付。公开仓中的 native
路径是训练适配；推理平台的注册、调度、kernel 映射和集成验证不属于该测试。

## 检查与边界

按仓库 reviewer skill 和 codebase-design 原则审查，未发现明确 S1/S2；
修改范围的 Ruff、格式与 scoped Pyrefly 检查通过（0 errors）；修改文件的
codespell、OAT 等 hook 通过。全仓 pre-commit 仍有范围外格式／拼写及 project
类型检查问题，自动格式化的 91 个无关文件已恢复，未混入本次提交。

Loop 当前为完整窗口、无持久 KV 的逻辑参考，feedback mask 不裁剪计算。
它不内置深度缓存布局、token pruning 或服务调度；具体策略和后端需独立适配。
本记录不声明真实 NPU 性能、分布式执行或推理平台集成通过。
