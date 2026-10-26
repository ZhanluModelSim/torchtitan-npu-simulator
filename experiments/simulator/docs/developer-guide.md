# Simulator 开发指南

本指南将 TorchTitan v0.3.0 迁移中已经验证的做法组织为两个开发领域，供后续开发者和 AI 接入模型、升级依赖或修改捕获基础设施时使用。
源码、配置和测试是行为真源；历史迁移报告用于解释取舍，不作为新版本的自动验收证明。

## 两个开发领域

| 领域 | 职责与入口 | 先读哪些内容 |
| --- | --- | --- |
| 捕获运行时与负载契约 | 环境基线、显式激活、Meta/autograd 兼容、tensor/storage 身份、通信控制依赖、L0–L3、内存与对外接口 | [捕获运行时开发](capture-runtime.md) |
| 模型适配与验收 | 官方结构对照、原生配置树、组件复用、融合边界、模型并行、量化、checkpoint、保存值归属与独立先验 | [模型接入开发](model-integration.md) |

修改 `compat/`、通用 capture/IR/memory 或公开接口时，首先按运行时领域检查所有模型调用点；模型接入不得用模型专属 patch 绕过这些通用契约。
修改模型、loss、AC、量化或并行时，首先按模型领域检查实际选中路径；遇到通用能力缺口再回到运行时领域修复。

## 按任务选择专题

| 任务 | 进一步阅读 |
| --- | --- |
| 平台同进程调用、CLI 参数和结果消费 | [上游接入](upstream-integration.md) |
| 六项预算、逐层 payload、offload 与 dtype | [单卡内存结果](steady-state-memory.md) |
| FSDP 等待、AG/RS/AR 和零耗时节点 | [流控制契约](stream-control-dependencies.md)、[依赖审计](fsdp-dependency-audit.md) |
| 保存槽、storage 所有权、AC 与状态 | [S3 生命周期审计](s3-lifetime-audit.md)、[生命周期优化](storage-lifetime-optimization.md) |
| CANN Meta 与 Python 包装接口差异 | [签名兼容](cann-metadata-signature-compat.md) |
| 原生 MXFP8 recipe、code/scale 和保存值 | [量化审计](s4-mxfp8-audit.md) |
| Kimi 的组件、数学和融合边界案例 | [Kimi 接入](kimi-k3-migration.md) |
| 当前支持范围与历史验收区别 | [能力矩阵](capability-matrix.md)、[分阶段验收](acceptance.md) |

## 通用开发顺序

1. 固定源码、软件栈、配置和输入，确认基线与候选实际加载的文件路径。保持工作树和安装源码隔离。
2. 沿真实入口找到问题，先做最小行为探针，再判断属于环境、配置、Meta、autograd、归属、图依赖还是估算问题。
3. 遵循原生 dispatch 优先；复用已有实现；必要兼容集中在显式、可恢复的作用域内。不复制旧 patch 后再逐个屏蔽错误。
4. 修改实现及所有调用点，补能区分错误路径的独立预期；兼顾正常、异常退出和连续捕获。
5. 从小模型扩到相关并行／AC／量化路径，最后执行代表性完整几何验收。只在存在不同生产路径时扩大组合，不用无意义的笛卡尔积替代分析。
6. 同步公开接口和能力边界，保留实际命令、源码哈希、模型几何、rank 坐标和原始证据。新增失败不能混入既有基线告警。

没有真实 NPU 时，CPU 数学对照、Meta 图、逻辑内存和 Gloo 契约分别报告；不能改写为 NPU 数值、HCCL 性能或 allocator 峰值通过。
完整测试次数只说明执行范围，不能替代独立字节先验、缺梯度检查和因果依赖测试。

## 本次迁移的证据入口

[迁移状态](migration-status.md) 保存固定版本、镜像和旧 patch 处置；
[dispatch 收敛](dispatch-first-remediation.md) 保存从手工补图转向原生机制的验证；
[精简记录](simulator-cleanup.md) 保存折叠、补边和多格式图导出删除的范围。
这些报告包含历史快照。新的源码、镜像、参数规模或依赖版本必须重新验证，不能沿用其中的计时和峰值作为通过结论。
