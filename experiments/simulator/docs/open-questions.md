# 迁移遗留问题

## CANN metadata 准备阶段的成本范围

四个 metadata 算子已恢复普通 `torch.ops` dispatcher 入口。输入从 host planner 转为 Meta；
设备分配能力探针仅在厂商 Meta 实现显式申请 NPU buffer 时注册局部修正，内部复用原厂 shape inference。
正常、异常退出均恢复 Python binding 与 Meta 注册；健康的厂商 Meta kernel 不覆盖。

这些 helper 当前仍在输入准备阶段构建 mask/CP plan，随后才开始捕获训练 F/B/O。
因此它们可被独立 dispatch 捕获，但不计入每步训练图或成本。是否加入单独的准备阶段图仍需定义统计范围。

本问题已收敛为成本范围选择；普通 dispatcher 的可行性已由当前实现和 metadata 测试确认。
改动、证据和边界见 [原生 dispatch 收敛记录](dispatch-first-remediation.md)。
