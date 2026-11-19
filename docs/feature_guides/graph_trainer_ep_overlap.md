# GraphTrainer EP overlap 适配

DeepSeek-V4（DSV4）在 TorchTitan v0.4.0 的 GraphTrainer 路径中按 token 将 MoE 拆成两个 chunk。调度器重排联合 FX 图中的节点，让一个 chunk 的计算与另一个 chunk 的 EP 通信重叠。

## 启用方式

在 Python recipe 中设置 `config.compile.ep_overlap`。DSV4 使用 `EpOverlapConfig(enabled=True, chunk_dim="seq", module_fqn="layers.*.feed_forward")`，要求参与切分的 MoE 输入 token 数为偶数。TorchTitan v0.4.0 的 SPMD 路径要求 EP overlap 使用单 microbatch，DSV4 示例将 `training.num_tokens_per_train_step` 设为 `-1`。

DSV4 A3 八卡 GraphTrainer 示例默认开启 EP overlap 并使用 NPU Auto Overlap 调度。启动方式和步数设置见 [DSV4 GraphTrainer 使用指南](deepseek_v4_graph_trainer.md#启动训练)。如果要单独使用上游固定调度，在 Python recipe 中保留 EP overlap 配置，将 `config.compile.pass_pipeline` 设为 `"mutation-functionalization"`，以保留 DSV4 的重计算 mutation 修复。

## 能力范围、依赖与验证边界

| 项 | 值 |
| --- | --- |
| 能力对象 | 使用 GraphTrainer、MoE 和真实 EP token exchange 的训练图，EP degree > 1 |
| 训练路径 | GraphTrainer SPMD 联合图 |
| TorchTitan 基线 | `0.4.0`（`requirements.txt` 固定版本） |
| 环境 | Ascend NPU、`torch-npu`、CANN 和 HCCL |
| 默认状态 | `EpOverlapConfig.enabled` 默认关闭，DSV4 GraphTrainer 示例显式开启 |

## 实现原理

### 切分与调度

1. DSV4 在 SimpleFSDP 之后、FX 捕获之前包装 `layers.*.feed_forward` 的 forward。包装器沿同一 token 边界等分激活、可选的 hash 路由 token IDs 和 padding mask，两个 chunk 共用原辅助损失分母，最后拼接输出。
2. `populate_eager_chunk_metadata_pass` 从捕获后的联合图读取 split、chunk body 和拼接标注，生成调度所需的 metadata。
3. 上游 `ep_overlap_schedule_pass` 消费该 metadata 并应用固定调度。NPU Auto Overlap 在同一位置先校验 EP 结构，再根据 NPU benchmark 调度，详见 [NPU Auto Overlap](graph_trainer_auto_overlap.md)。

## DSV4 整网实跑验证（2026-09-16）

以下记录对应 TorchTitan v0.3.0 的 FX 图切分路径。

在同一台机器上按“关闭 EP overlap -> 开启 EP overlap”的顺序各运行 10 step。两组均使用 8 张 Ascend 910、CANN 9.2.0、TorchTitan `v0.3.0`（`086bf6c166ec85c1298eb5596fa9bf95f6a2d840`）和本仓库 HEAD `ec98b86809e024e105958b50c7714666c8c65ff2`。模型为 flash 43 layers / 16 experts，EP=8、DP shard=8、TP/CP/PP=1，local/global batch size 为 1/64，sequence length 为 4096，seed 为 42。

| 指标 | EP overlap 关闭 | EP overlap 开启 |
| --- | ---: | ---: |
| 结果 | 10/10 step，退出码 0 | 10/10 step，退出码 0 |
| graph pass | 7 个，43.617 s | 13 个，228.628 s |
| EP 图变换 | 无 | seq chunking 86 个 region；scheduling 86 个 region |
| step 2-10 平均耗时 | 64.975 s | 62.923 s |
| step 2-10 平均 TPS | 505.000 | 521.222 |
| loss（step 1 -> 10） | 12.19569 -> 9.91401 | 12.19564 -> 9.94998 |
| 日志记录的峰值显存 | 52.22 GiB（85.23%） | 59.37 GiB（96.90%） |

该次顺序单跑中，开启组 step 2-10 的平均 TPS 比关闭组高约 3.2%。

开启组的代价也较明显：日志峰值显存增加 7.15 GiB。

## 相关文档

- DSV4 模型侧 GraphTrainer 适配：[`deepseek_v4_graph_trainer.md`](deepseek_v4_graph_trainer.md)
- Override 注册表：[`torchtitan_npu/override/README.md`](../../torchtitan_npu/override/README.md)
- 上游 GraphTrainer 与 EP overlap 说明：[`torchtitan/experiments/graph_trainer/README.md`](https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/graph_trainer/README.md)
