# [[PR_OR_CHANGE]] UT/ST 静态设计与审查

| 项目 | 内容 |
| --- | --- |
| 基线版本 / PR 版本 | [[BASE]] / [[CANDIDATE]] |
| 目标分支 | [[TARGET_BRANCH]] |
| TorchTitan 固定版本 | [[UPSTREAM_REVISION]] |
| 工作方式 | [[草拟/审查]] |
| 测试执行 | `未执行（仅静态设计/审查）` |

## 结论

**合入建议：**[[可以合入 / 补充测试后合入 / 暂停并澄清 / 不适用]]

**UT 格式：**[[符合 / 有修改建议 / 有隔离风险 / 未评价]]

**UT 正向功能：**[[已覆盖 / 有缺口 / 有无效检查 / 不适用]]
**ST 处理：**[[复用 / 调整 / 新增 / 无需 ST / 暂停 / 延后 / 混合]]

**阻塞问题：**

1. [[不超过三个；指向 UT 覆盖表或 ST 事实表。]]

**合入前置条件：**

[[先写拟修改或新增的准确路径、测试名称和一句动作；无则写“无”。]]

```python
[[必要时给不超过约 20 行的伪代码或短代码；无则删除代码块。]]
```

## 语义变换与独立 oracle

运行时代码变更时保留本节；文档、skill、规则或模板-only 变更按 `report-output.md` 的“不适用”规则删除本节及其余覆盖表。

| 语义变换 | 旧路径 | 新路径 | 等价条件 | 失效条件 | 独立 oracle | 状态 | 合入前置条件 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [[NONE_OR_TRANSFORMATION]] | [[OLD_PATH]] | [[NEW_PATH]] | [[CONDITIONS]] | [[INVALIDATORS]] | [[ORACLE]] | [[已覆盖/部分覆盖/未覆盖/检查无效/不适用]] | [[REQUIRED_ACTION]] |

语义变换存在但独立 oracle 未闭合时，合入建议只能是“补充测试后合入”。训练 loss 一致不能替代该 oracle。

## UT 正向功能覆盖

| 正向功能 | 生产代码和应观察结果 | PR 中的直接检查 | 状态 | 合入前置条件 |
| --- | --- | --- | --- | --- |
| [[POSITIVE_OUTCOME]] | [[PATH_AND_RESULT]] | [[SETUP_ACTION_ASSERT_OR_NONE]] | [[STATUS]] | [[REQUIRED_ACTION]] |

`部分覆盖` 行必须使用警示色高亮。

## ST 触发判断

| 生产代码变更 | 受影响训练场景 | 为什么需要 ST | 现有代表测试 | 判断 | 合入前置条件 |
| --- | --- | --- | --- | --- | --- |
| [[PRODUCTION_CHANGE]] | [[TRAINING_SCENARIO]] | [[WHY_CPU_UT_IS_NOT_ENOUGH]] | [[EXISTING_TEST]] | [[DECISION]] | [[REQUIRED_ACTION]] |

## 变更目标与生产模块

### 变更目标

[[正常支持条件 -> 生产路径 -> 用户或训练结果。]]

### 受影响模块

| 生产模块 | 使用位置 | 行为变化 | CPU 可观察结果 | 是否需要真实 NPU |
| --- | --- | --- | --- | --- |
| [[PATH_OR_MODULE]] | [[CALLSITE]] | [[IMPACT]] | [[OBSERVABLE]] | [[YES_NO_AND_WHY]] |

## UT 文件、命名和隔离

| 位置 | 类型 | 代码阅读结果 | 修改建议 |
| --- | --- | --- | --- |
| [[NODE_OR_LOCATION]] | [[TYPE]] | [[FACT]] | [[RECOMMENDED_CHANGE]] |

### 需要展开的 UT 缺口

[[只解释部分覆盖、未覆盖和检查无效的行。先给路径、名称和短代码，再解释原因。]]

## 4 张 NPU ST 事实表

| 测试 | 模型/配置 | 并行数值 | 替换实现/融合算子 | 编译模式 | NPU 数 | 启用与完成检查 | golden | 执行阶段 |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| [[EXISTING_OR_PROPOSED_CASE]] | [[MODEL_CONFIG]] | [[EXACT_DEGREES]] | [[EXACT_VARIANT_OPERATOR]] | [[EAGER_OR_BACKEND]] | [[LE_4]] | [[STATIC_DESIGN]] | [[GOLDEN_PATH_SOURCE_TOLERANCE]] | [[PR_MAIN_NIGHTLY]] |

### 按模型投影

| DeepSeek-V4 并行方式（不超过 4 张 NPU） | 参考实现/eager | 目标替换实现/eager | 参考实现/编译 | 目标替换实现/编译 |
| --- | --- | --- | --- | --- |
| [[BASELINE_EXACT_DEGREES]] | [[CASE_GAP_OR_UNSUPPORTED]] | [[CASE_GAP_OR_UNSUPPORTED]] | [[CASE_GAP_OR_UNSUPPORTED]] | [[CASE_GAP_OR_UNSUPPORTED]] |
| [[CHANGED_EXACT_DEGREES]] | [[CASE_GAP_OR_UNSUPPORTED]] | [[CASE_GAP_OR_UNSUPPORTED]] | [[CASE_GAP_OR_UNSUPPORTED]] | [[CASE_GAP_OR_UNSUPPORTED]] |

[[模型专属 PR 增加一个同结构表；否则不增加。]]

## ST 不能证明的内容

| 声明 | 负责的测试 | 基础校验要求 |
| --- | --- | --- |
| [[NUMERICAL_LIFECYCLE_PERFORMANCE_OR_NONE]] | [[COMPONENT_NUMERICAL_LIFECYCLE_BENCHMARK]] | [[MINIMAL_DESIGN]] |

## 附录

- **生产代码改动：**[[PATHS]]
- **PR 测试/ST 定义：**[[PATHS_OR_NONE]]
- **已有本仓/上游测试：**[[REVISION_AND_PATHS]]
- **测试入口静态包含情况：**[[REACHABLE_EXCLUDED_OR_UNKNOWN]]
- **排除的文档：**[[PATHS_OR_NONE]]
- **不支持的组合：**[[COMBINATIONS_AND_CODE_BASIS_OR_NONE]]
- **测试执行：**`未执行（仅静态设计/审查）`
