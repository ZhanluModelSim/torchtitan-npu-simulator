<!--
待合入的上游 PR：
- https://github.com/pytorch/torchtitan/pull/3634
- https://github.com/pytorch/torchtitan/pull/4411
- https://github.com/pytorch/torchtitan/pull/4474
- https://github.com/pytorch/torchtitan/pull/4650
- https://github.com/pytorch/torchtitan/pull/4734
- https://github.com/pytorch/torchtitan/pull/4708
- https://github.com/pytorch/torchtitan/pull/4763
- https://github.com/pytorch/pytorch/pull/201621
-->

# TorchTitan 临时补丁

本目录仅保存已向 TorchTitan 上游提交、但尚未合入当前依赖版本的临时补丁。

固定格式（每个补丁文件必须遵守）：

- 文件**第一行**必须是 PR 链接注释：

  ```python
  # Pending upstream PR: https://github.com/pytorch/torchtitan/pull/NNNN
  # Pending upstream PR: https://github.com/pytorch/pytorch/pull/NNNN
  ```

- 模块 docstring 需说明补丁内容，并以 "Remove this module after the TorchTitan dependency includes the PR." 结尾。
- 对原模块/类做属性替换的 monkey patch 必须封装在 `def apply() -> None:` 中并在文件末尾调用（`apply()`），补丁逻辑内使用完整模块路径（`import torchtitan.models.common.moe` 后以 `torchtitan.models.common.moe.X = ...` 赋值），不使用短导入别名。纯定义被模型目录引用的 backport 类（如 `BatchedLinear`、`BaseMaskHandler`、`SingleComplexRoPE`）不需要 `apply()`。

校验（除 `__init__.py` 等包胶水文件外，每个补丁文件都应命中）：

```bash
grep -L -E "Pending upstream PR: https://github.com/pytorch/(torchtitan|pytorch)/pull/" \
  torchtitan_npu/patches/torchtitan -r --include="*.py"
```

| PR | 说明 |
| --- | --- |
| [#3634](https://github.com/pytorch/torchtitan/pull/3634) | 补充 DeepSeek-V4 所需的公共组件及训练接入 |
| [#4474](https://github.com/pytorch/torchtitan/pull/4474) | 补齐部分初始化的 optimizer state，支持完整 checkpoint 恢复 |
| [#4411](https://github.com/pytorch/torchtitan/pull/4411) | 为 DistMuon 补充 HSDP replica compute 去重 |
| [#4650](https://github.com/pytorch/torchtitan/pull/4650) | 修正 EP overlap 对 shape query 通信标注的校验，对应 `ep_overlap_shape_queries` |
| [#4734](https://github.com/pytorch/torchtitan/pull/4734) | 在支持的一维 SimpleFSDP 场景下，将 GraphTrainer 每个 microbatch 的 chunk-loss head 梯度 ReduceScatter 合并为一次，对应 `experiments/graph_trainer/chunked_loss` |
| [#4708](https://github.com/pytorch/torchtitan/pull/4708) | 修复 functionalize_recompute_mutations_pass 和 EpOverlap 的冲突 |
| [#4763](https://github.com/pytorch/torchtitan/pull/4763) | 向 GraphTrainer graph pass 提供运行时上下文，用于 auto-overlap 调度时整图 profiling |
| [#201621](https://github.com/pytorch/pytorch/pull/201621) | 为本地 FileSystemReader 增加可选 mmap 加载，对应 `distributed/checkpoint/filesystem` |

当前依赖固定为 `requirements.txt` 中的 commit。已由该基线提供的 AuxLoss、EMA、DistMuon NPU storage 检查和 EP ready-node 去重不再保留本地 backport；随上游 EP chunk pass 删除的旧 chunk 补丁一并退役。

TorchFT 的配置 override、chunked-loss head 绑定、optimizer 分派、恢复状态刷新、loss 副本平均和 FSDP 参数组 hook 已由目标上游提供，对应 [#4534](https://github.com/pytorch/torchtitan/pull/4534)、[#4575](https://github.com/pytorch/torchtitan/pull/4575)、[#4578](https://github.com/pytorch/torchtitan/pull/4578)、[#4598](https://github.com/pytorch/torchtitan/pull/4598)、[#4663](https://github.com/pytorch/torchtitan/pull/4663) 和 [#4664](https://github.com/pytorch/torchtitan/pull/4664)，不再覆盖原生 TrainingEngine 实现。完整故障恢复运行仍需单独验证。

保留的 EP overlap shape-query 补丁的作用与排查手段见
[`docs/feature_guides/graph_trainer_ep_overlap.md`](../../../docs/feature_guides/graph_trainer_ep_overlap.md)。

对应 PR 合入且 TorchTitan 依赖更新后，应删除相关补丁及导入；全部补丁清理完成后，删除本目录。
