# Raw 契约版本 1 与 native 的人工对齐验收

日期：2026-11-05。工作分支：`feat/npu-simulator-local`。

## 职责与范围

Raw 目录仅维护规范 `README.md` 与独立 `model.py`；工具迁入
`scripts/block_diffusion`，测试迁入 `tests/unit_tests/models/block_diffusion_reference`。
Native 模型人工实现 raw 契约，不在运行时导入 raw 或 loader。

本轮 native 对齐范围是 **exact 训练**。Raw 保留训练和 KV 推理；native 的
KV 推理、多轮去噪尚未适配。原有 compute-proxy 模式继续可选，其结果不被
重新标记为 exact。没有创建新的 fix 开发分支。

## 环境与命令

- Python 3.11.16，官方 PyTorch 2.12.0+cpu，pytest 9.2.0。
- 真实 TorchTitan 源码基线：`ac13e536c84e7f6647b14fa9375c3c8a8a2b8578`。
- 不修改 TorchTitan 源码。CPU 对照测试仅绕开 NPU 插件包初始化；网络、
  normalization、RoPE、MoE、loss 及权重 adapter 使用真实源码。
- Converter 检查的 NPU 硬件入口由 CPU 数学实现代替；不声称真实 NPU 通过。

```bash
python -B -m pytest -q --import-mode=prepend \
  --confcutdir=tests/unit_tests/models/block_diffusion_reference \
  tests/unit_tests/models/block_diffusion_reference
```

主 agent 最终复跑：**163 passed, 14 warnings in 5.96s，无跳过**。
14 条 warning 来自上游 TorchScript 弃用提示。仅安装独立参考依赖时 native
测试可能跳过，不能把那种运行结果当成本页的 native 对齐证据。

## 验收矩阵

| 范围 | 结果 |
| --- | --- |
| 独立参考／工具／权重转换 | 121 项通过，包含本轮新增 raw router autocast 精度回归 |
| 新增 native、converter 与配置对齐 | 42 项通过 |
| Dense / MoE × 两种 exact mask | 同配置、同权重 logits、loss_sum、有效 token 归一化 loss 及全部参数梯度一致 |
| 独立全 mask GQA attention 参考 | 前向与 Q/K/V 梯度一致 |
| Native NPU attention wrapper | 矩形分段、causal/noncausal 参数及显式 backward 接口通过 CPU 仿真检查 |
| Native simulator exact ledger | 保留真实 Sq/Sk；每个前后向 matmul 的输入输出几何一致；GQA 梯度先按 query heads 计算再归约 |
| 原有 compute-proxy | 默认 causal 数值、有效 key 缩放及显式 RoPE override 保持 |
| 配置 | raw v1、RoPE、router 策略校验；exact + CP 明确拒绝；exact dataloader 使用窗口位置 |
| MFU | 每 token 单位、活跃专家与 tied output head 计算正确 |
| 共享 MoE gate helper | CPU/meta 混合 dtype 和梯度正确；CPU 外层 autocast 不降低 FP32 gate 精度 |

同权重整模型对照本轮使用 FP32。FP16/BF16 的整模型后端数值、真实 NPU、
训练收敛和分布式组合仍需各自的验收证据。

## 审查发现及修复

1. **RoPE 和路由语义分叉**：exact 使用 split-half cos/sin、无频率缩放、FP32
   router 和 selected-logits softmax；禁用 raw 未定义的专家 bias/负载均衡。
2. **每 token 统计单位错误**：原返回值额外乘了 sequence length，且把所有未
   激活专家计入。现在使用框架活跃参数约定，参数容量仍统计实际唯一参数。
3. **BF16 hidden 与 FP32 gate 不匹配**：共享 helper 依 gate 权重 dtype 计算，
   并防止 CPU 外层 autocast 覆盖该契约。相同 dtype 基线和梯度均有回归测试。
4. **Proxy 静默覆盖 RoPE**：保留用户显式 backend/scaling，不强制改回 flavor 默认值。
5. **Exact matmul 输出几何错误**：元素数相同但 BSH 不符合逐头矩阵乘法输出。
   改为逐头结果，再显式 view/transpose/reshape；GQA backward 记录归约。
6. **Raw 的 AMP 精度契约未落实**：参考 router 的 FP32 输入仍可能被外层
   autocast 降为 BF16。现显式关闭路由评分中的 autocast，修复既定 v1 精度
   规则的执行；不是新增训练范式或改变 contract version。

两个独立审查发现的 S2（第 4、5 项）均已复测关闭。已按仓库 reviewer skill
审查规范、模型/converter/并行规则、设计、调用点及训练基础设施。

## 检查与限制

本轮修改 Python 文件的 Ruff 检查和格式检查通过；修改生产模块的 scoped
Pyrefly 为 `0 errors (4 suppressed)`。修改文件的空白、文件结尾、大文件、
冲突、私钥、Ruff、codespell 与 OAT 检查通过。

已执行全仓 pre-commit，仍未通过范围外格式／拼写和 project 类型检查；其
自动格式化的 91 个范围外文件已恢复。相同原生 CPU 环境的 project Pyrefly
对照：干净基线为 578 errors，本分支为 561 errors。本轮 scoped 检查不替代
全仓通过，也不把 CPU 环境结果当成 NPU 验收。PR 因此保留 Draft。

Exact native 现有工厂覆盖 dense 与一个 shared expert 的 MoE，其他 raw 可选
结构不能仅凭契约版本号宣称已适配。Exact + CP 禁止；TP/EP/ETP/PP/FSDP 等
组合未由单 rank 对照证明。独立工具 JSON 到 native IR 的端到端导入、完整
trainer 运行和硬件性能校准也不属于本页的通过结论。
