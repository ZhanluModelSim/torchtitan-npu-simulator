# NPU ST Review

ST 审查只回答一个问题：现有 `tests/integration_tests` 是否覆盖了 PR 改变的真实 NPU 训练路径。ST 不读取 CPU UT 的覆盖结论，也不把 CPU UT、CPU oracle 或单元测试计入 ST 覆盖。

## Workflow

### 1. 确定 PR 需要进入的训练路径

从生产代码改动确定受影响的模型、训练配置、`override.imports`、并行方式、编译模式、训练入口和应当完成的步骤。只记录 PR 实际改变的组合，不因为测试名称或 PR 描述中的笼统表述扩大支持范围。

### 2. 建立现有 integration testcase 清单

读取目标分支的 [`tests/integration_tests/README.md`](../../../../tests/integration_tests/README.md)，再读取同目录的 `run_tests.py`、相关模型 case 定义和 `tests/assets/losses`。最后读取 `.ci/smoke_test.sh`，确认 suite 注册、case 选择、NPU 数量和 CI 入口。README 只提供测试意图和矩阵说明，实际覆盖仍以代码和入口为准；README 中没有在代码和入口中落地的描述不能单独作为覆盖证据。

`.ci/smoke_test.sh` 是既有测试框架入口。审查或更新某个 PR 时不得为它追加一次性模型命令、kernel pytest、selector 或其它默认流程；需要新模型时，应沿用现有 integration runner，把模型 case 接入 `run_tests.py` 的 `build_models_test_list`（并按需登记独立 suite），由 `OverrideDefinitions` 提供模型、配置、override、并行和资产参数。

### 3. 对照改动判断复用、调整或新增

逐项将 PR 训练路径与现有 case 的实际参数和命令比较。已有 case 经过相同模型、配置、override、并行和编译路径，并且检查目标路径启用和训练完成时，结论为“复用”。已有 case 只缺少一个 PR 改变的条件时，结论为“调整”。没有任何 case 能进入目标路径时，必须在现有 `tests/integration_tests` 框架中新增最小 testcase。

### 4. 输出 ST 结论

报告先给出每项生产改动对应的现有 case、覆盖状态和缺口，再给出需要调整或新增的准确文件、case 名称、配置组合和完成条件。只要目标路径没有被现有 case 实际执行，就不能写成“已覆盖”。

## Rules

### 1. Testcase 来源

ST testcase 只能来自 `tests/integration_tests` 的注册和执行链路。默认从 `.ci/smoke_test.sh` 追踪到 `tests.integration_tests.run_tests`，再追踪到 `build_*_test_list` 和具体 `OverrideDefinitions`；被 `disabled`、suite 未注册或入口不会选择的 case 不算覆盖。

### 2. 覆盖判定

一个 case 只有在真实命令中使用了目标模型和配置、目标 `override.imports` 或融合实现、目标并行数值、目标编译选项，并进入真实训练入口时，才覆盖对应路径。仅在 README、case 名称或配置字符串中声明目标条件，不算覆盖。

### 3. 完成证据

每个被计入的 case 都必须有可观察的训练完成条件，例如指定训练步正常退出、初始化、前向、反向和优化器步骤完成。`check_loss=True` 的 case 还要读取对应 golden 并比较 loss；`check_loss=False` 的 case 可以在框架明确不支持确定性比较时只做完成性检查，但报告必须记录原因，不能把它写成数值等价证明。

### 4. 复用优先

如果现有 case 已经进入 PR 的目标路径，优先复用；如果只缺 PR 新增的模型、override、并行或编译条件，调整该 case 的定义和入口参数。不得为了填满模型或并行矩阵复制没有新路径的 case。

### 5. 新增条件

只有在现有 case 无法进入 PR 的新生产路径时才新增 testcase。新增 case 必须使用现有 `run_tests.py`、`OverrideDefinitions`、suite 注册、启动脚本和结果检查方式；不得另起测试框架或只写一个不会被 suite 选择的孤立脚本。

新增模型必须进入 `tests/integration_tests/<model>.py` 的 `build_*_test_list`，并在 `run_tests.py` 的 `build_models_test_list` 中加入默认模型 suite（若资产或资源需要条件门禁，应在 case 定义或 runner 的既有选择机制中表达）。不得通过修改 `.ci/smoke_test.sh` 来绕过模型列表或为单个 PR 建立专用入口。

### 6. 最小组合

新增或调整的 case 只保留能触发 PR 改动的最小模型、并行和编译组合，单个 PR 阶段不超过 4 张 NPU。第二个及以上 case 必须说明第一个 case 无法进入的不同生产路径。

### 7. 输出格式

ST 覆盖判断使用以下表格：

| 生产代码改动 | 目标训练路径 | 现有 integration testcase | 状态 | 需要的调整或新增 |
| --- | --- | --- | --- | --- |

状态只使用“已覆盖”“部分覆盖”“未覆盖”“调整”“新增”“无需 ST”。“部分覆盖”表示 case 能启动但缺少目标 override、并行、编译或完成检查；“未覆盖”表示没有 case 进入目标路径。
