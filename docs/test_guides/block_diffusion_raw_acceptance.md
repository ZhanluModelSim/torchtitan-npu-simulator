# 独立 Block Diffusion 训练／推理参考验收

日期：2026-11-05。开发基线：`feat/npu-simulator-local` 的
`4f200e8abd318a0a813061d9f3946e85dcbc2e03`。

结论：独立 raw 参考的数值、autograd、状态、工作量和权重格式转换验收通过。
这不表示原生 TorchTitan/NPU 分布式训练、实际硬件时延或下游模拟器集成已通过。

## 范围与设计

本次只更新 `torchtitan_npu/simulator/raw_model/block_diff` 和相关说明。
原生 `models/block_diffusion`、converter、并行计划及已有 state-dict adapter
没有改动。raw 核心保留单 rank 计算语义；推理并行描述和训练算子捕获位于
独立文件。训练和推理共享同一组网络参数，不复制网络实现。

原始 raw Python 接口及默认规模已更新，旧调用方需迁移。旧参数布局通过
`legacy_state_dict.py` 显式转换，不自动猜测格式。详见
[接口与迁移说明](../../torchtitan_npu/simulator/raw_model/block_diff/README.md)。

## 独立测试

环境：Python 3.12.3、PyTorch 2.9.0+cpu、pytest 9.1.1，使用临时隔离解释器。
主 agent 在格式与类型修正后重新执行：

```bash
python -B -m pytest -q --import-mode=prepend \
  --confcutdir=torchtitan_npu/simulator/raw_model/block_diff/tests \
  torchtitan_npu/simulator/raw_model/block_diff/tests
```

结果：**120 passed in 3.71s**。

| 范围 | 测试数 | 关键证据 |
| --- | ---: | --- |
| 推理核心 | 34 | 独立 GQA 参考、缓存等价、逐请求输出位置、meta 图捕获、真实 MoE 求和 |
| 推理语义负载 | 32 | TP/EP 守恒、空 rank、KV dtype、采样归约、动态未确定位置和零接受轮次 |
| 训练数值与梯度 | 21 | 两种精确 mask、独立 logits/全部参数梯度参考、有限差分、零目标 loss、meta backward |
| 实际训练算子捕获 | 22 | CPU/meta AdamW、反向矩阵乘法、别名与原地写入、冻结及未激活参数、实际状态容量 |
| 旧权重格式转换 | 11 | 精确回环、严格加载、专家排列与 dtype、未知和不可表达格式拒绝 |

测试由独立 agent 编写，并由主 agent 复跑。验收中发现并修正了专家权重
归一化下溢、遗漏 KV cast、跨块非法提交、错误的静态未确定位置假设，以及
合成训练标签与 ignore index 冲突等问题。

## 可执行导出示例

```bash
python torchtitan_npu/simulator/raw_model/block_diff/example.py --device cpu --moe
python torchtitan_npu/simulator/raw_model/block_diff/example.py --device meta --moe
python torchtitan_npu/simulator/raw_model/block_diff/export_example.py --tp 2 --ep 4 --output /tmp/block-diff-inference.json
python torchtitan_npu/simulator/raw_model/block_diff/training_workload.py --device meta --optimizer adamw --output /tmp/block-diff-training.json
```

- 推理示例：6 个阶段、每阶段 8 个 rank，总接受 token 数 16。
- 默认训练示例：4 个有效标签；实际捕获 forward 236 个算子、backward 371 个、
  AdamW 317 个。参数与已生成梯度分别为 427264 字节，优化器状态为 854612 字节。
- 训练数值参考显式使用 PyTorch SDPA math backend，矩阵 FLOPs 仅对观察到的
  mm/bmm/addmm 计算，其他算子的成本保留为未知。以上数值不是 NPU 性能估计。

## Codecheck 与代码审查

- Ruff 0.14.14：修改目录检查和格式检查通过。
- Pyrefly 0.45.1：本次 6 个生产 Python 文件单独检查 **0 errors**。
- 修改文件的 pre-commit：空白、文件结尾、大文件、冲突标记、私钥、Ruff、
  codespell 和 OAT 检查通过。完整 project Pyrefly hook 仍失败。
- 执行了 `pre-commit run --all-files`，**未通过**：范围外已有格式／拼写问题，
  并且环境缺少 `/tmp/torchtitan` 及原生 NPU 训练依赖。该命令自动格式化的
  范围外文件已恢复，没有混入提交。
- 为区分基线与新增问题，对干净基线 worktree 使用相同环境执行 project
  Pyrefly：基线 1561 errors，本分支 1553 errors；本次修改目录没有类型错误。
  这些计数不代表全仓检查通过。

已按 `.agents/skills/torchtitan-npu-code-reviewer/SKILL.md` 独立审查：未发现
明确的 S1/S2 问题。覆盖设计与架构、调用点、checkpoint 迁移、训练 infra 和
NPU 成本声明。`git grep` 未发现原生路径导入被移除的 raw Python 类；现有
native adapter 的 stacked 专家排列与显式 legacy 导出一致。

## 尚未执行

- 原生 TorchTitan 全量测试及其分布式组合。
- 真实 NPU 数值与性能、C4 收敛及相同 checkpoint 的 loss/grad-norm 对齐。
- 独立 raw JSON 到原生模拟器 IR／调度图的端到端接入。

因此提交保留为 Draft PR，供上游环境补齐检查。训练捕获会保留引用来稳定
张量身份，不可据其寿命计算峰值显存；推理 JSON 仍是语义事件描述。两个
输出都不能直接视为已经校准的 HBM／互联流量或硬件时延结果。
