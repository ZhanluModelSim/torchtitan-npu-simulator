# NPU smoke test 规则

`tests/smoke_tests/` 验证需要真实 NPU runtime 、但不需要完整多步训练的小范围行为。

- smoke test 不是 CPU UT，不得通过全量 mock NPU 语义来伪装设备验证。
- 测试必须明确设备数、runtime/算子依赖、dtype、shape 和预期失败条件。
- 只覆盖最小设备语义、注册或数值 sanity；分布式训练、checkpoint resume 和多 step 数值对齐放在 `tests/integration_tests/`。
- 结果必须记录实际执行环境；仅静态检查时不声称 smoke test 已通过。
