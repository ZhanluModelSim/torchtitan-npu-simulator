# S5 代码与测试审查

审查范围为 S4 树 `cac9febbcbda6ad68bc5300242a27bc2746a383c` 之后的 sidecar 变更。
本次没有修改生产模型、算子、配置 override 或上游源码。全规模矩阵与独立导出复核均完成；逐项证据见能力矩阵与 S5 结果。

## 设计与架构

设计合理。图一致性检查放在独立的 `validation.py`，由原生执行入口在捕获结束后调用；
规模验收脚本只编排同一入口和消费其导出，不建立另一套训练或调度实现。
兼容层、模型几何、autograd bridge 和 FSDP 驻留建模保持原有职责，没有新增核心实验分支。
适用规则为 `.agents/AGENTS.md`、`experiments/AGENTS.md` 和 sidecar 的 AGENTS；未命中生产目录的专项规则。

S5 发现并修复五个具体问题：

1. `StepGraph.step_id` 随机生成，不能与模板字典 key 和 `template_ref` 直接关联。现统一为稳定模板 ID，
   在输出 manifest 的 schema version 2 和迁移文档中说明。旧随机 ID 消费者需改用模板 key。
2. 通信归属归一化逐模板刷新 `successors`，丢失跨模板出边。现依据保持不变的全局 predecessor DAG
   恢复出边，验证双向一致性。改变图索引表示，不改变模型执行、反向或通信调用顺序。
3. DOT 只绘制模板内部边，丢失跨模板依赖。现基于完整节点表绘图；别名 view 的收缩仍保留可达依赖。
4. 20T 的旧压缩比 `0` 不能通过新主线 metadata 布局。旧、新 attention 均以不建立压缩流表示窗口注意力，
   配置入口现使用原生 `1`，typed override 保留 `0` 作为显式兼容别名。未移除最后一层或缩减几何；
   BF16/MXFP8 的别名对照执行事件、梯度与内存峰值一致，完整 20T 六项复测和独立安装镜像执行均通过。

5. 计划层曾把 logical rank 写入 `capture_process_rank`，与真实 recorder worker 不同。现由捕获入口统一标注
   worker 和 logical rank，并在正常入口验证。此修正不改变 mesh、算子或 tensor DAG。

CSV rank 限制接入现有 exporter，默认展示一个 rank；不调整 logical world 或 mesh。
输出 manifest 明确其代表模板投影语义，未将复制展示行标为逐 rank 独立捕获。
SchedulePlan CSV 直接调用原有权威计划导出，不从 kernel CSV 重建调度。

## 训练 infra 与数值边界

- 梯度、clip、optimizer 更新集合及归约依赖在原生执行后检查；未替换算子实现或修改数值表达式。
- storage/保存槽检查只读取捕获元数据，不创建新张量，也不改变保存槽持有关系。
- 峰值通过生命周期区间独立求和，参数 shard dtype 与激活 offload 使用明确的投影口径。
- 大模型独立子进程顺序执行，CSV 限制避免输出按 world 无界复制；主机 RSS 和耗时单独记录。
- 未涉及权重 key、checkpoint 加载或保存格式；未声称跨版本真实 checkpoint 或数值精度已验收。
- CANN metadata 直调 Meta kernel 的专项问题已记录，按用户要求保持现有路径，未作为已解决项。

## 测试与质量

| 检查 | 结果与证据 |
| --- | --- |
| 完整 sidecar 测试 | 424 passed；`evidence/s5-tests-coordinates.log` |
| 独立包类型 | 0 errors，原有 14 suppressed；`evidence/s5-types-coordinates.log` |
| 全仓类型基线比较 | 原始主线 64、候选 64，消息／路径／行号相同，新增 0；`s5-pyrefly-comparison.json` |
| 完整 pre-commit | 已执行；常规、Ruff、architecture、OAT 通过；主机缺少 Pyrefly，故不标全绿 |
| 格式检查 | Ruff check/format 通过；原生执行入口和三个图缺陷均有回归断言 |
| NPU ST／数值 | 未执行：当前环境无真实 NPU；CPU/Meta 测试不计作 NPU ST |
| 安装包与源码 | 82 个 Python 文件 SHA256 一致；实际安装镜像 world=2048 用例通过 |
| 原生源码完整性 | TorchTitan/NPU 663 个、TorchAO-NPU 50 个 Python 文件未改变 |

正向测试通过真实 CLI/registry 执行到导出消费者；图检查的错误引用、环、峰值错误和保存张量身份错误
均通过对真实捕获的最小破坏验证其拒绝能力。MXFP8 × 三种 AC 的 offload/dtype 投影对照同时检查
原始执行事件、梯度、量化选择和 GA 元数据，不以仅构造配置替代执行。

当前无未修复的 S1/S2 代码问题。21 项全规模捕获及独立 CSV/内存复核通过；NPU 数值验收仍明确未执行。
