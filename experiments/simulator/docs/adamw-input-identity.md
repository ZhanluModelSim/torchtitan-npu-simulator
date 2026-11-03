# AdamW 输入身份

原生 `aten._fused_adamw_.default` 的输入身份随捕获结果交付，供算子 adaptor 区分参数、梯度、动量与 step 张量。该信息默认捕获，不增加用户开关，不改变 AdamW 计算、预热或卸载行为。

## 交付字段

`capture/optimizer.py` 按实际原生 schema 绑定参数，`capture/dispatch_capture.py` 将该算子的输入组织为 schema 顺序。`OpNode.attrs` 保存：

- `input_tensor_list_sizes`：六个列表的长度，依次对应 params、grads、exp_avgs、exp_avg_sqs、max_exp_avg_sqs、state_steps。原生第一个参数名为 `self`，对下游表达参数列表角色。AMSGrad 关闭时第五项明确为 0。
- `optional_tensor_roles`：实际非 None 的尾部张量角色，按 `grad_scale`、`found_inf` 的原生顺序排列；全部未提供时为显式空列表。

下游先按六个长度切分 inputs，再将剩余输入与可选角色逐一对应。不同列表即使 shape、dtype 相同也保持角色；同一张量重复出现时保留每个输入位置。只有顺序归一，真实 tensor/storage 身份、mutation 输出及后继依赖仍来自原生调用。

已有 `optimizer_updated_parameter_count`、`optimizer_declared_group_sizes`、`optimizer_logical_parameter_shapes` 和 `optimizer_work_scope` 继续作为独立来源注记。它们描述逻辑参数及更新范围，不能代替本地物理输入的列表边界。

## 范围与验证

当前只对标量 LR 的 `aten._fused_adamw_.default` 添加绑定；不扩展 `tensor_lr` overload、其他 optimizer 或其他算子的源字段。未知 AdamW Tensor 参数 schema 明确拒绝，不按 shape 猜测新角色。原生 tensor 值不读取，Meta 运行时也能获得参数列表长度与身份。

`tests/test_optimizer_contract.py` 覆盖真实 fused AdamW 混合 dtype 分桶、未更新参数、六列表边界、AMSGrad 空／非空列表、可选 overflow/scale、反序 keyword 调用、重复输入及后继 mutation 依赖。Meta 捕获只验证结构与逻辑 storage，不证明真实 NPU 数值或性能。

配合 `tests/test_warmup.py`、`tests/memory/test_optimizer_state.py` 与 `tests/test_memory_delivery.py` 检查预热后的状态身份、驻留量和卸载交付。新增输入身份不得改变静态显存摘要或优化器计算路径。
