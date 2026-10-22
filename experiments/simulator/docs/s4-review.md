# S4 自审记录

审查范围为原生 MXFP8 配置接入、显式 CLI、配置别名与 Pro 20T 几何、量化 op 分类、已构建参数清单及 Meta SwiGLU arity 兼容；另包含类型化几何 CLI 的原生配置树重建。
适用 `experiments/AGENTS.md` 与独立 sidecar 规则；不修改生产 `torchtitan_npu/` 或上游安装源码。

设计合理：配置转换沿用目标 `TrainerEx` 的原生入口，量化数学与保存对象由 TorchAO-NPU 拥有；
兼容缺口集中在可撤销的 Meta runtime 作用域；复用 S3 的 storage/generation、FSDP、SAC 与 ownership 内存模型。

| 正向功能 | 独立证据 | 状态 |
|---|---|---|
| 原生 recipe 实际启用，旧 FQN CLI 消失 | 配置→模型→56 个实际 wrapper 参数；旧参数由 argparse 拒绝；六个真实规模配置构建 | 已覆盖 |
| MM/BMM/grouped MM 的真实量化保存值 | 8448/16896/17152-byte 手算，storage 与原生输出逐项对应，反向三次低精度 matmul | 已覆盖 |
| full/selective 的量化释放及重算 | 24576-byte 间隙先验、原始 codes 已释放、重算值保留至 backward 使用 | 已覆盖 |
| SwiGLU 元数据反向与真实保存对象 | 实际 x/score storage、native backward 与输入/score 梯度、scope 退出恢复 | 已覆盖 |
| 并行、AC、参数与 optimizer | 九个 native MXFP8 模型组合：全部梯度、量化来源、归约祖先、通信唯一归属、DAG 与峰值 storage 去重 | 已覆盖 |

SwiGLU 兼容新增一个固定 Function.backward arity 的局部类型忽略，与同类 S3 bridge 一致；独立包检查原有 13 项变为 14 项，
未增加全局忽略或 Any cast。其余类型按真实源码根检查，包括独立 TorchAO-NPU 的 editable 源码根。

未发现当前已验证范围内未解决的 S1/S2 问题。Pro 20T 依赖原生私有 builder，是后续同步时须由 geometry 测试保护的边界。
无 NPU，真实 kernel 数值/收敛与硬件峰值未验证；独立模拟包测试不作为生产 NPU ST 的覆盖证据。
未修改生产训练路径，不新增一次性 NPU CI 入口。大规模捕获、设备侧量化误差及所有下游消费按后续验收范围记录。

最终执行 `scripts/validate.sh`：410 passed，独立包 0 errors。Linux OAT 为 152 文件、0 问题；
全仓候选与基线同条件 64 个相同既有错误、新增 0。安装镜像的源码与图/保存值/峰值已独立复核。

几何复写使用原生 builder，空复写与四个原生 preset 的结构对照、三种 AC 的 CLI 量化 F/B/O 及
安装镜像的独立 Flash 几何用例均通过。同步更新 loss 的词表大小与 optimizer profile；
未知/未迁移旧字段明确拒绝，不整体复制旧 Model.Config，也不将 FQN 复写换名恢复。
