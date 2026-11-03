# 尚待定义的负载范围

本文只记录会改变负载含义、尚未确认的事项。当前明确拒绝或不在范围的功能见[能力矩阵](capability-matrix.md)，不得把旧计划写成已实现能力。

## attention metadata 的准备阶段

当前 `runner.simulate()` 在结构预热和正式观测器安装之前调用模型的 `build_attention_masks()`，生成 mask、CP plan 等 metadata。CANN metadata helper 已有普通 dispatcher 入口；必要的 Meta 分配兼容仍复用原厂 shape inference，并随作用域恢复。

因此这些 helper 可以在独立探针中被 dispatch 捕获，但标准 F/B/O 负载不包含这次输入准备。它们的输出和后续实际 consumer 仍按真实输入参与捕获。是否增加独立准备阶段图、是否每次迭代重建以及如何由平台计费，尚需先定义场景，不能直接把准备操作加到每个 F 模板。

调查入口：[runner.py](../torchtitan_simulator/runner.py)、[compat/cann_metadata.py](../torchtitan_simulator/compat/cann_metadata.py)、[metadata.py](../torchtitan_simulator/metadata.py)、[test_metadata.py](../tests/test_metadata.py)。该范围选择与厂商 Meta 能否执行是两件事，不应通过无条件替换健康 kernel 来解决。
