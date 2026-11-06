# ALoDLM raw 首版验收

本记录保留对应历史实现阶段的证据。当前 raw 已抽象为通用 Loop，ALoDLM 配方
位于 `scripts/block_diffusion/recipes/alodlm.py`；新接口、迁移和单文件交付验证见
[Loop 重构验收](block_diffusion_loop_refactor.md)。公开 native 路径仅为训练适配，
不把这些结果视为外部推理引擎已接入。

日期：2026-11-05。开发分支：`feat/npu-simulator-local`。

## 范围与依据

Raw bundle 升为 v2，新增 ALoDLM 子契约 v1；标准 block diffusion 子契约仍为
v1。实现仍位于 raw 的 `README.md` 与 `model.py`，测试和 loader 位于外部目录。
没有修改 native 生产实现，也没有将其旧版对照结果视为 ALoDLM 适配通过。

依据 [论文 v1](https://arxiv.org/html/2610.04198v1) 和作者代码
[`fe225a99`](https://github.com/amazon-science/ALoDLM/tree/fe225a99e0b5aeff351804e871d7581155d7f27d)
核对数学行为，在本仓 backbone 上独立实现。未复制作者代码或下载模型权重。
作者论文与发布代码的反馈规则不同，因此本参考显式区分 cumulative 和
newly-committed 规则，不宣称整个引擎等价或 checkpoint 兼容。

本轮范围为 **uncached dense**：共享 core、pre-coda carry、每轮 coda、detached
gate、双流 batch、固定轨迹回放、practical loss 和单步决策。运行时 cache、
streaming、packed 文档、activation checkpointing 及 native 适配不在本轮。

## 执行结果

环境：Python 3.11.16、PyTorch 2.12.0+cpu、pytest 9.2.0；原有 native 回归继续
使用真实 TorchTitan 基线 `ac13e536c84e7f6647b14fa9375c3c8a8a2b8578`。

```bash
python -B -m pytest -q --import-mode=prepend \
  --confcutdir=tests/unit_tests/models/block_diffusion_reference \
  tests/unit_tests/models/block_diffusion_reference
```

主 agent 最终复跑：**208 passed, 14 warnings in 6.87s，无跳过**。
新增 ALoDLM 45 项，原有 163 项全部保留通过。Warning 为上游 TorchScript
弃用提示。Ruff 检查及格式检查通过；修改生产模块 scoped Pyrefly 为 0 errors。

修改文件的 pre-commit 检查中，空白、格式、codespell 与 OAT 等通过；project
类型 hook 和全仓检查仍因范围外的既有问题失败。全仓检查自动修改的 91 个
无关文件已恢复，没有混入本次提交；PR 继续保持 Draft。

| 验收项 | 证据 |
| --- | --- |
| 共享物理权重与层调用 | Prelude 一次、Core/Coda 每轮；114 次层调用／84 个逻辑 KV 对公式核对；不将提交深度当成实际工作量 |
| 隐状态与反馈 | 独立逐层算式和手工展开参考；pre-coda carry、轮间 norm、标签只在退出读出后回填 |
| 两种推理反馈 | 前两轮一致、后续 latent 行为按规则区分；新 outer 调用重新初始化 |
| 双流训练 | 2S 输入、原位置重排、可见性独立 oracle，禁止读取同块 clean target |
| 梯度 | 全部 backbone 参数与独立参考一致；Gate 特征隔离主干梯度 |
| Practical 目标 | categorical-product／sequence-credit 独立公式核对 loss 及 logits/gate 梯度；AR shift、深度平均及 /2 口径正确 |
| 概率和标签 | 末轮质量吸收、detached denoising weights、逆 mask 权重、零监督及自定义 ignore_index |
| 决策 | Confidence 与 halting 分开；进度 fallback；词表 -inf 屏蔽保持有限熵 |
| CPU/meta | 固定形状回放、目标函数反向、make_fx 不读取张量值；配置／轨迹／普通 KV cache 明确拒绝非法输入 |

功能指南中的最小例子已在 CPU 和 meta 实际执行，输出形状为 `[3,1,8,16]`，
目标函数与 backward 均完成。

## 审查与修复

独立审查按论文／发布源码数学规则与仓库 reviewer skill 执行。发现并关闭：

- 合法词表屏蔽 `[0,-inf]` 导致 `0*log(0)` 的 NaN 熵；改用安全的 `xlogy`，
  同时拒绝 NaN、正无穷和全无效分布。
- Batch 自定义 ignore index 与目标函数独立默认值可能不一致；现在从 batch
  继承同一口径，显式冲突配置被拒绝。

## 解释边界

这是范式的独立稠密参考，不是发布引擎的性能复现：推理 heads 覆盖全部输入
行，denoising CE 先计算全部 2S 行再屏蔽，loss reduction 使用 FP32，并保留
完整读出历史用于验收。作者优化实现的行选择、低精度及存储寿命可能不同。

`logical_buffer_count()` 仅描述潜在缓存布局，当前实现不分配或读取这些
cache。论文中的 deepest-available fallback 与 shared-coda cache 是 serving
近似，不能直接复用标准 KV cache，也不保证等价于全深度前缀重计算。

固定退出轨迹不等于从当前 gate 采样；practical 目标也不冒用基础 joint-KL
estimator 的完整无偏性结论。未验证论文 benchmark、训练收敛、真实 NPU、
ALoDLM native 后端或生产服务性能。
