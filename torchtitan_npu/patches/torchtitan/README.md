<!--
待合入的上游 PR：
- https://github.com/pytorch/torchtitan/pull/3430
- https://github.com/pytorch/torchtitan/pull/3634
- https://github.com/pytorch/torchtitan/pull/3864
- https://github.com/pytorch/torchtitan/pull/4411
- https://github.com/pytorch/torchtitan/pull/4474
- https://github.com/pytorch/torchtitan/pull/3985
- https://github.com/pytorch/torchtitan/pull/4650
- https://github.com/pytorch/torchtitan/pull/4651
- https://github.com/pytorch/torchtitan/pull/4734
- https://github.com/pytorch/torchtitan/pull/4708
- https://github.com/pytorch/torchtitan/pull/4763
-->

# TorchTitan 临时补丁

本目录仅保存已向 TorchTitan 上游提交、但尚未合入当前依赖版本的临时补丁。

固定格式（每个补丁文件必须遵守）：

- 文件**第一行**必须是 PR 链接注释：

  ```python
  # Pending upstream PR: https://github.com/pytorch/torchtitan/pull/NNNN
  ```

- 模块 docstring 需说明补丁内容，并以 "Remove this module after the TorchTitan dependency includes the PR." 结尾。
- 对原模块/类做属性替换的 monkey patch 必须封装在 `def apply() -> None:` 中并在文件末尾调用（`apply()`），补丁逻辑内使用完整模块路径（`import torchtitan.models.common.moe` 后以 `torchtitan.models.common.moe.X = ...` 赋值），不使用短导入别名。纯定义被模型目录引用的 backport 类（如 `BatchedLinear`、`BaseMaskHandler`、`SingleComplexRoPE`、`AuxLoss`、`VarlenCPMetadata`）不需要 `apply()`。

校验（除 `__init__.py` 等包胶水文件外，每个补丁文件都应命中）：

```bash
grep -L "Pending upstream PR: https://github.com/pytorch/torchtitan/pull/" \
  torchtitan_npu/patches/torchtitan -r --include="*.py"
```

| PR | 说明 |
| --- | --- |
| [#3430](https://github.com/pytorch/torchtitan/pull/3430) | 为变长注意力补充 CP 和 Full DTensor 支持 |
| [#3634](https://github.com/pytorch/torchtitan/pull/3634) | 补充 DeepSeek-V4 所需的公共组件及训练接入 |
| [#3864](https://github.com/pytorch/torchtitan/pull/3864) | 为 torchtitan 补充 AuxLoss 辅助损失框架 |
| [#4474](https://github.com/pytorch/torchtitan/pull/4474) | 补齐部分初始化的 optimizer state，支持完整 checkpoint 恢复 |
| [#3985](https://github.com/pytorch/torchtitan/pull/3985) | 为 Trainer 补充 EMA 权重维护及 checkpoint 集成 |
| [#4411](https://github.com/pytorch/torchtitan/pull/4411) | 为 DistMuon 补充 HSDP replica compute 去重 |
| [#4529](https://github.com/pytorch/torchtitan/pull/4529) | 隔离 EP chunk 与 packed/varlen 动态符号，并修正 singleton batch、copied unbacked range 和具体化范围，对应 `ep_chunk_concretization` |
| [#4650](https://github.com/pytorch/torchtitan/pull/4650) | 修正 EP overlap 对 shape query 通信标注的校验，对应 `ep_overlap_shape_queries` |
| [#4516](https://github.com/pytorch/torchtitan/pull/4516) | 去重跨 chunk 共享的 EP ready node，对应 `ep_ready_nodes_dedup` |
| [#4651](https://github.com/pytorch/torchtitan/pull/4651) | 将同一 root 的 AllToAll dim-0 shape query 保留在 chunk 内，对应 `ep_shape_live_out` |
| [#4734](https://github.com/pytorch/torchtitan/pull/4734) | 在支持的一维 SimpleFSDP 场景下，将 GraphTrainer 每个 microbatch 的 chunk-loss head 梯度 ReduceScatter 合并为一次，对应 `experiments/graph_trainer/chunked_loss` |
| [#4534](https://github.com/pytorch/torchtitan/pull/4534) | 在 TorchFT 模型构造前应用配置 override 并重新校验配置，对应 `experiments/torchft/trainer.py` |
| [#4575](https://github.com/pytorch/torchtitan/pull/4575) | 为 TorchFT 的 chunked loss 绑定 `lm_head`，覆盖普通训练及 PP 最后阶段，对应 `experiments/torchft/trainer.py` |
| [#4578](https://github.com/pytorch/torchtitan/pull/4578) | 消除 TorchFT optimizer 递归分派，保留一次容器 hook 调用，对应 `experiments/torchft/optimizer.py` |
| [#4598](https://github.com/pytorch/torchtitan/pull/4598) | 导出副本恢复状态前刷新缓存中的学习率等元数据，对应 `experiments/torchft/optimizer.py` 和 `checkpoint.py` |
| [#4663](https://github.com/pytorch/torchtitan/pull/4663) | 按当前参与副本数平均 TorchFT 日志 loss，对应 `experiments/torchft/trainer.py` |
| [#4664](https://github.com/pytorch/torchtitan/pull/4664) | 为每个 FSDP 参数组注册 TorchFT 梯度平均 hook，对应 `experiments/torchft/manager.py` |
| [#4708](https://github.com/pytorch/torchtitan/pull/4708) | 修复 functionalize_recompute_mutations_pass 和 EpOverlap 的冲突 |
| [#4763](https://github.com/pytorch/torchtitan/pull/4763) | 向 GraphTrainer graph pass 提供运行时上下文，用于 auto-overlap 调度时整图 profiling |

上述 6 个 TorchTitan PR（#4534、#4575、#4578、#4598、#4663、#4664）已在上游合入，但 `v0.3.0` 尚未包含这些修复。

四个 EP overlap 补丁的作用、启用方式与排查手段见
[`docs/feature_guides/graph_trainer_ep_overlap.md`](../../../docs/feature_guides/graph_trainer_ep_overlap.md)。

对应 PR 合入且 TorchTitan 依赖更新后，应删除相关补丁及导入；全部补丁清理完成后，删除本目录。
