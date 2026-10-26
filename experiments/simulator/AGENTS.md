# Simulator sidecar

独立 distribution；主包导入不得自动激活本包。复用目标主线的配置、模型与算子，不编辑上游安装源码。

- 开发先读 [统一开发指南](docs/developer-guide.md)。运行时、兼容、capture/IR/通用内存修改按 [捕获运行时与负载契约](docs/capture-runtime.md)；模型结构、配置、组件、并行、量化及 checkpoint 修改按 [模型适配与验收](docs/model-integration.md)。同时涉及两个领域时，两套规则都适用。
- `compat/` 集中可撤销的运行时兼容，按接口能力与最小行为探针选择，不按版本号分支。
- `capture/`、`ir/`、`memory/`、`cost/` 延续旧 simulator 的语义与测试；移动不等于已完成新版验收。
- 新配置或并行组合须从真实入口验证到图、梯度及内存消费者。
- 未支持项在构图前拒绝，未知成本显式标注，禁止占位 tensor 冒充真实保存值。
- 仅 Meta 执行不能证明硬件数值正确；实际/模拟事实应明确区分。
- 接入新模型或修改 loss、AC、量化、并行、offload 时，按模型指南检查真实配置／数学／保存／使用／释放链路，不能仅凭图构建或回归数量声明完成。
- `unattributed` 的目标是没有未解释项；已知 head/norm/loss/梯度桥接不伪装成 decoder 层。默认逐层策略只卸载归属明确的层或显式 checkpoint 保存值，其余层外和未确认归属的值继续计入设备生命周期和峰值；禁止通过删除记录或只过滤报表得到零未知项。
- 保存槽不等于激活或卸载候选。区分 F action 内交错反向与后续 B/recompute，检查同一 storage 的全部保存及使用；AC 先确认层／checkpoint 绑定再清零，覆盖无保存槽、无绑定的非层重算引用。
