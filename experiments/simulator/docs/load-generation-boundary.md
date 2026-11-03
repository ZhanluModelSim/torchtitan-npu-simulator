# 负载生成边界

本说明记录 2026-11-03 的职责收敛，替代旧文档中默认强制写 memory summary 和训练内部估算成本的描述。

## 内存交付与可选导出

`simulate()` 和 CLI 默认 `output_formats=[]`，返回 `WorkloadGraph`，不创建输出目录、不写文件、不打印导出预算。正式静态显存对象保存在 `workload.iteration.schedule.annotations["memory_plan"]`，其 `to_summary_dict()` 与同级 `memory_summary` 一致。平台可直接消费内存对象，无需文件中转。

显式选择 `output_formats=["mem"]` 或 CLI `--simulation.output-formats mem` 才导出 `memory/memory_summary.json` 及内存诊断明细，并打印预算。直接调用 `export_result()` 也遵循该选项，空列表无操作；不新增 summary-only 模式。旧目录文件不会自动清理。

## 移除的接口

- 删除 `torchtitan_simulator.cost` 和 `OpDispatchCapture` 的 `cost_model` 参数；捕获不再执行 FLOPs 或访存成本计算。
- `OpNode` 删除 `flops`、`peak_mem`、`param_mem`、`comm_bytes`，不提供伪兼容属性。
- `StepGraph` 删除 `total_flops`、`peak_active_mem`、`param_mem`、`comm_volume`；`WorkloadGraph` 删除 `total_cost_est`。
- 删除 `cost_unknown`、`cost_unknown_counts`、重复 `communication_input_bytes` 及阶段内重复的 `memory_summary` 注记。

以上字段的消费方应迁移到下游预测结果或正式 `MemoryPlan`，不能把删除后的成本字段补成零。训练仓内测试和验收脚本已迁移；平台本来就自行分类、预测和读取外层静态显存。

## 保留的捕获事实

保留图依赖、stream/event、FSDP 通信视图、张量 shape/dtype/storage/alias、FQN/recompute、通信组/rank、AdamW 输入角色及卸载传输节点。静态显存预算、预热状态和卸载驻留修正仍由训练仓负责。

`op_type` 是捕获名称归一，不是平台 cube/vec 等资源分类。`metadata_view` 影响流观察和别名语义，不能作为无用标记删除。未知算子的内存诊断仍保留，与预测是否缺模型无关。

通信事件与调度槽位的 `volume_bytes` 仍服务张量容量、FSDP 内存和调度数据流，含义是源张量容量，不是网络通信预测流量；下游应按所需算子语义自行建模。训练仓不添加性能预测或硬件资源类别。

## 验证范围

默认无文件和显式 mem 导出由 `tests/test_memory_delivery.py` 覆盖；IR、捕获、通信、卸载和静态显存回归验证移除成本不改变负载事实。真实 Meta 集成对比归平台验收记录，不表示真实 NPU 性能或数值精度验证。


最终验证：capture/IR/memory/step相关回归340 passed。训练全量首次为577 passed、14 failed；14项均为旧测试假设（12项precision默认文件断言、2项FSDP观察器未区分无捕获预热）。修正后完整test_precision.py为18 passed、test_fsdp_stream_capture.py为3 passed，未改生产语义来迁就测试。修改Python文件Ruff/format通过，pyrefly 0 errors。独立静态review无阻塞。干净目录构建的wheel从源码外site-packages运行真实Kimi两例：默认无目录，显式mem摘要/明细可读取并等于内存对象；zip检查确认无cost模块。全仓既有lint问题未纳入修改，未运行真实NPU数值/性能验收。
