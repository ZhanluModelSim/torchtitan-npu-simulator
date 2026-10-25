# Kimi K3 的原生组件复用和模拟融合边界

- Status: implemented；simulator 验收完成
- Created: 2026-10-24
- Last updated: 2026-10-25
- Issue/PR: 当前任务，尚未创建 PR

## Context

旧 Kimi K3 实现基于早期 TorchTitan 接口，模型含 KDA、Gated MLA、SiTU、LatentMoE 和 Attention Residual。
迁移目标为当前 v0.3.0 simulator，同时保留旧虚拟融合边界。
本次范围延续旧模型的文本 decoder，不扩展到官方多模态 encoder。

官方依据固定为 [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)
的 `config.json`、`configuration_kimi_k3.py`、`modeling_kimi_linear.py`。
TorchTitan 基线为 `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`；NPU 基线为 `7eb02313a7fb5ac764133f0c4d909a7e8ce44da1`。

## Proposal

模型定义归属 `torchtitan_npu/models/kimi_k3`，虚拟融合的显式激活归属 sidecar `compat/kimi.py`。
主包导入不激活模拟替换，真实设备调用虚拟算子明确失败。
虚拟算子使用 dispatcher schema、Meta kernel 和 autograd 注册，替代旧版手工记图；保存真实输入，不以占位输入替代依赖。

| 模块 | 对照结果 | 复用或保留方式 |
| --- | --- | --- |
| gate/up/down 专家投影 | 当前 V4 同样使用两次 gate/up GMM 和一次 down GMM | 原生 GroupedExperts 参数布局、`_grouped_mm` seam；保留三次投影边界 |
| SiTU | `beta*tanh(gate/beta)*sigmoid(gate)`；可选 `linear_beta*tanh(up/linear_beta)` | 保留 `situ_glu` 和 `situ_glu_backward`，不可替换成 SwiGLU/clamp |
| Router | sigmoid/top-k、choice bias、归一化、scale 可使用原生组件 | 使用 TokenChoiceTopKRouter；模拟显式均衡分配 |
| LatentMoE | Router 读取原隐藏态；压缩后执行专家；合并后归一化并升维；共享分支读取原隐藏态 | 原生 RoutedExperts 和 dispatcher；仅扩展 latent 阶段 |
| Router score | 旧 Kimi 和官方在 down 投影后加权；当前 NPU 公共工厂默认吸收到投影前 | Kimi 明确关闭 `absorb_router_scores`，不以代数近似改变 BF16 舍入和融合边界 |
| KDA | q/k/v 卷积、gate/beta 投影、输出 gate/norm 均在核心外 | 保留 chunk_kda/core backward，七个输入含 A_log/dt_bias；beta sigmoid 延续旧边界，位于核心外 |
| MLA | 官方 `mla_use_nope=True`，q_rope/k_rope 切片直接拼接 | 保留 attention core；投影、pad/trim、输出 sigmoid gate 均在外 |
| Attention Residual | 先完成块残差混合，再 attention/MLP 累积 | 保留原阶段顺序；首层 self-attention residual 的参数存在但路径恒不调用，保留 key 并冻结，不伪造梯度 |

### 验收条件

- 完整配置符合官方 93 层（69 KDA、24 MLA，末层 MLA）、896/top-16 专家及 latent 几何。
- 小模型经正常配置入口执行 F/B、loss、clip、optimizer；每个参与训练的参数具备正确梯度 shape/dtype/placement。
- SiTU 和 LatentMoE 对照独立官方公式和 token-by-token 专家实现，比较输出、输入、router 和所有参数梯度。
- 每个融合算子的输入身份、保存槽和 backward 依赖完整；图无环且引用闭合。
- 并行矩阵覆盖单卡、DP/FSDP/HSDP、TP、EP、CP 和有不同生产路径的交叉组合；不得用 DP 通过代替 TP/EP/CP。
- 原生 FSDP storage、通信完成依赖和梯度归约顺序正确；权重、优化器状态及活跃峰值有独立先验对照。
- 所有原 simulator 回归通过，并完成改动文件及全仓检查，记录已有失败。

## Consequences

投影、norm、卷积与 Attention Residual 已提升为可遍历配置树，默认启用公共 RMSNorm 和 NPU token dispatcher override。
并行声明复用原生 decoder 与 MoE sharding；TP 部分和在 latent norm 之前完成归约。
CP 使用连续输入区间，先完成 TP/SP 收集，再以原生 autograd collective 收集完整 CP 上下文；
在全部序列耦合计算之后、输出投影之前普通切片回本 rank 区间，避免重复传播其他 CP rank 的 loss 梯度。
已覆盖 TP、EP、CP、交叉组合、末 rank AC 和 FSDP 度数大于专家数的分片检查。
MXFP8 公共转换器保留 SiTU 子类，精确核对 routed/shared experts 的选择范围。
完整 93 层、world2048、TP4/CP2/EP64 的 rank0 BF16 none 与 rank2047 MXFP8 selective 均完成 F/B/O；
2,642 个训练参数梯度齐全，逐参数分片、权重/state 字节和独立峰值 sweep 均通过。

KDA 使用普通 FSDP 计算精度，未保留 A_log/dt_bias 的特殊 FP32 policy；原始参数仍显式 FP32。
Attention Residual 按官方 FP32 folded score-weight 顺序，不能替换成普通 BF16 RMSNorm 后再投影；
MLA 低秩 norm 的 epsilon 修正为官方默认 1e-6。
Linear/Conv1d/Embedding/expert 权重显式使用旧 normal std=0.02 初始化；不承诺跨实现 RNG 抽取顺序比特一致。

checkpoint 名称因原生专家与 dense 参数布局而变化，adapter 已迁移到原生接口；
测试覆盖 HF key 到真实模型 key、非方阵专家堆叠顺序，以及低秩 KDA output gate。
旧原生 checkpoint key 的兼容转换及硬件加载尚未验收。
新增虚拟 schema 的名称带 `kimi_sim` namespace，canonical op_type 延续旧融合名；raw 名称的变化需要在接入文档中说明。
虚拟算子只验证 Meta 依赖和旧边界，无法证明真实 KDA 内部 workspace、kernel 数值或硬件耗时。

完整回归 521 项通过，随后 14 项专项包含 4 个新增测试，最终共 525 个不同 testcase。
原生 FSDP AG 到投影权重、HSDP AR 的延后传播有专项测试；CP/EP 另有真实两 rank Gloo 对照。
详细支持范围、精度限制、checkpoint key 变化和复验数据见
[Kimi 接入说明](../../../../experiments/simulator/docs/kimi-k3-migration.md)。
主机全仓类型检查缺少训练依赖；配套容器中主包的基线和当前版本同为 357 个既有类型错误，新增 0。
模型专项与 sidecar 类型检查均为 0 errors；不将全仓 Lint 声称为通过。
主仓正常 pytest 入口的 Kimi CPU + 公共量化转换器 41 项通过。完整主仓回归收集 1,557 项，
但在未修改的 V4.1 metadata 测试进入真实 CANN CPU fallback 后停滞，诊断后停止；不计为全套通过。
