# 辅助损失框架（AuxLoss）

torchtitan-npu 通过 `torchtitan_npu/patches/torchtitan/models/common/aux_loss.py` backport 上游 torchtitan 的 `AuxLoss`（对应上游 PR [#3864](https://github.com/pytorch/torchtitan/pull/3864)）。辅助损失的梯度按 carrier 注入前向图，归一化分母取本步全局有效 token 数，与主 loss 同一分母，使各并行度下的贡献可比。DeepSeek-V3.2 的稀疏 indexer 损失是目前的使用者。

## 两条实现路径的指标口径

| 路径 | 指标名 | 累加时机 |
|---|---|---|
| 参考路径（`SparseIndexerLoss`） | `sparse_indexer_loss/mean` | 前向，`_accumulate_and_inject` |
| 融合路径（`AscSparseIndexerLoss`，由 `sparse_attn.asc` override 替换） | `asc_sparse_indexer_loss/mean` | 反向，`ctx.acc_buffer.add_(...)` |

源码位置：参考路径为 `torchtitan_npu/models/deepseek_v3_2/model.py`，融合路径为 `torchtitan_npu/override/deepseek_v3_2/sparse_attn/ascendc.py`。

融合算子在反向才产出求和后的 KL 值，两条路径的累加时机因此不同，激活检查点（AC）策略对日志值的影响也不同：

- `FullAC` 在反向重放所属 block 的前向，参考路径的累加随之重复执行，其指标高于融合路径；两条路径的指标不能直接比较。
- 本仓 backport 未引入保留累加区域的 `torch_remat`（上游用保留区域避免重放，当前 pinned TorchTitan 未提供该机制），重复累加只在 `FullAC` 下确认过，不能推广到 `SelectiveAC`、`RegionAC` 等其他 AC 配置。跨 AC 策略比较日志时应说明两侧策略，必要时按重放次数换算。
- 该差异只影响日志指标：`loss` 与 `grad_norm` 不受 AC 重放影响。

比较参考路径与融合路径时，以 `loss` 与 `grad_norm` 作为一致性判据；indexer loss 指标只在相同 AC 策略下比较。具体测量条件与数值由改动方记录在对应 PR 的验证说明中。

## 已知限制

**与 graph_trainer 图捕获的组合暂时不支持。** `AuxLoss` 的每步归一化分母当前保存在类级状态上，`minimal_fx_tracer` 捕获时会把它当作 trace 期常量折叠进图，后续步骤继续用首步分母缩放注入梯度。因此在该路径上启用辅助损失时，aux loss 的梯度与日志值都可能有精度差异，请勿在 graph_trainer 路径上依赖辅助损失结果；相关能力待分母改为运行时输入（或受管理 buffer 且能作为图输入重采样）后另行设计。graph_trainer 的接入方式见 [graph_trainer 说明](deepseek_v4_graph_trainer.md)。
