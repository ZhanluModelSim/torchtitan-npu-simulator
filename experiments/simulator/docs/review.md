# S0–S2 代码审查

2026-10-21。按 torchtitan-npu-code-reviewer 的专项规则、坏味道、设计、上游交互及训练 infra 维度审查。
本次不修改 `torchtitan_npu/`，未匹配该生产包下的专项规则；适用目标仓与 experiments 的分层要求。

## 已修复的问题

| 问题 | 根因与处置 | 验证 |
|---|---|---|
| Meta 前向有 grad_fn 但断梯度 | vendor RoPE/RMSNorm/token-permute 缺 Meta autograd；行为探针确认后用 native backward 桥接 | 输入/权重梯度、保存对象身份、完整模型梯度清单 |
| 真实 prefetch AG 从图中消失 | 归一化只保留完成 compute transition 的 AG；保留实际未消费 launch，采集 native finalize 释放 | 源事件唯一性、pending buffer 生命周期、全部通信字节 |
| 跨 capture mesh 身份污染 | DTensor 原生/Python cache、RoPE pool 和 SPMD TLS 持有旧 world | 同进程多配置矩阵、非零 rank、退出恢复 |
| 通信域和规模记录错误 | `(DeviceMesh, axis)` 被当作二元 rank 列表 | 独立四 rank oracle、实际 process group、混合轴模型用例 |
| 阶段峰值和 clip 依赖不可信 | chunked loss 内交错 backward；旧 phase 区间混算；optimizer 未包含所有真实内存输入依赖 | 手算时间轴、跨模板 DAG、归约→clip→optimizer 祖先检查 |
| fused optimizer 重复计算风险 | 一次参数子集调用扩成完整组，分桶时可能重复 | 完整声明组与实际更新集合分开；分桶/unused 测试；全局 shape 与 local 字节双 oracle |
| 设备和捕获坐标混用 | 宿主 `_IS_A5` 可能影响逻辑目标；worker 和 logical rank 共用字段 | 明确 target adapter、宿主 A5 干扰/恢复测试、worker 0 与 rank 15 分离 |
| 异步调用被当成同步控制依赖 | 旧 eager all-reduce 无视 async_op | 异步不添加同步完成边；同步保守控制边明确标注来源 |

## 设计判断

独立可选 distribution 合理：生产包不导入 simulator，不编辑上游安装文件；配置、模型、
loss、parallelize 和 optimizer 使用目标原生实现，成熟 IR/memory 代码沿用，兼容集中在局部 seam。
没有按版本号分支；原生功能满足需求时不安装旧 workaround。

私有 FSDP/DTensor 观测点和 `_IS_A5` 仍是升级耦合。选择保留少量 scoped adapter，
以 schema/capability 和模型级测试约束，而非复制新旧训练框架。后续升级必须重跑验收。

新增公共配置显式校验非法并行、目标和延后功能；IR 的数字 op ID 注解按真实类型修正，
ScheduleAction 的既有 int/string ID 均保留。没有修改 checkpoint key 或真实训练数值路径。

## 验证与限制

334 项 sidecar 测试通过，独立类型检查 0 errors，Ruff 通过，Linux OAT 无问题。
完整目标环境中的全仓类型检查与未改动 master 同为 64 个错误，本次新增 0；没有通过屏蔽整类错误伪造全仓成功。
当前审查范围内未留下明确的新 S1/S2 阻断项。

真实 NPU 数值、AC、MXFP8、完整规模、PP 和下游全量导出验收不在本阶段通过范围内。
没有以 Meta 梯度非 None 代替 NPU 数值精度，也没有以保存 inventory 累计字节代替内存峰值。
