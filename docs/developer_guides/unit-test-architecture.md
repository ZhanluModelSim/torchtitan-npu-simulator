# Unit Test 架构与迁移方案

本文记录 CPU unit test 的目录、执行和生产代码归属。它是开发约定，不替代 `test-review` skill 中针对某个 PR 的审查报告。

## 目标

测试目录要能回答「被保护的对象是谁」，测试入口要能回答「这条测试是否会被执行」。目录归属和执行归属是两个维度，不能因为一个测试不计入产品 UT，就让它从 CI 中静默消失。

## 目录与统计边界

`tests/unit_tests/` 是 CPU 测试的统一执行根目录。生产语义测试按 `torchtitan_npu/` 的职责镜像放置：

- `tests/unit_tests/compile/`、`config/`、`models/`、`ops/` 和 `override/` 分别保护对应的生产模块；
- `tests/unit_tests/patches/torchtitan/` 只镜像有上游依据的临时 patch；`tests/unit_tests/patches/torch_npu/` 和 `tests/unit_tests/patches/workaround/` 镜像当前有效的 NPU runtime compatibility patch；
- `tests/unit_tests/tooling/` 放仓库脚本和 skill 的自测，便于沿用现有 unit-test 入口执行，但这类测试不计入产品行为覆盖；
- 多进程 worker 与启动它的测试放在同一个语义目录；
- `tests/integration_tests/` 只放进入真实 `torchtitan.train` 的 NPU 模型 ST，不把 CPU UT、kernel pytest 或 runner 自测计入 ST。

本次目录调整将以下文件归入统一执行根，但保留其 tooling 属性：

```text
tests/unit_tests/tooling/profiler/test_profiler_tools.py
tests/unit_tests/tooling/training_log_visualization/test_training_log_visualization.py
```

它们不再散落在 `scripts/tests/` 或 `.agents/skills/.../tests/`，也不因此变成产品 UT。产品 UT 统计和报告必须按语义归属排除 `tooling/`。

## `patches/` 的职责和边界

`torchtitan_npu/patches/torchtitan/` 只保存当前固定上游版本尚未包含、且已有上游 PR 或 commit 依据的临时 patch。上游版本吸收对应改动后，应删除 patch、导入入口和专用测试。

`torchtitan_npu/patches/torch_npu/` 与 `torchtitan_npu/patches/workaround/` 的现有文件不是「上游 torchtitan 临时 patch」：它们改变 PyTorch、Inductor 或 `torch_npu` 的运行时行为，例如 `determinism.py` 替换 `torch.use_deterministic_algorithms`，`device_copy.py` 修改 Inductor 的 `DeviceCopy` 和 `prims.device_put`。这两个目录是当前有效的 runtime compatibility 归属，可以继续维护；本次测试目录整理不移动生产代码，也不要求新增 `runtime/compat` 层。若未来要迁移，必须另开包含导入时机、兼容别名和真实激活测试的原子架构变更。

## 执行归属

只要测试放在 `tests/unit_tests/`，现有 `.ci/unit_test.sh` 的 `pytest tests/unit_tests` 就能发现它；报告仍须单独标注 `tooling`，避免把 tooling 通过数写成产品覆盖。后续若保留其他目录外的测试，必须在对应 CI suite 显式列出执行路径，不能依赖开发者手工运行。
