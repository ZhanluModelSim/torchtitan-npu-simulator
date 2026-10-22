# Simulator 迁移基线

目标主线的生产依赖继续由 `requirements.txt` 和 `.ci/setup_torchtitan.sh` 管理。
本表记录 simulator 独立迁移分支实际使用的上游基线，不改变生产依赖或发布兼容性声明。

| torchtitan_npu_branch | torchtitan_repo_url | torchtitan_branch | torchtitan_commit | updated_at | notes |
| --- | --- | --- | --- | --- | --- |
| codex/simulator-titan030 | https://github.com/pytorch/torchtitan.git | v0.3.0 | 086bf6c166ec85c1298eb5596fa9bf95f6a2d840 | 2026-10-21 | NPU master snapshot 7eb02313a7fb5ac764133f0c4d909a7e8ce44da1; simulator S0-S5: seven presets x three AC modes, native-size BF16-MXFP8/MTP1/non-PP Meta F/B/O acceptance; real NPU numerical validation not performed; no upstream checkout edits. |

旧 simulator 源码基线为 `da283eb674af2962695e120cad1d193a138d51c7`，旧上游运行时
来自 `ac13e536c84e7f6647b14fa9375c3c8a8a2b8578`。详细验证范围、镜像和遗留项见
[sidecar 迁移状态](../../experiments/simulator/docs/migration-status.md)。

升级时固定新的 NPU/TorchTitan SHA，运行组件梯度、真实模型/并行、身份、通信依赖及驻留测试，
再更新本表。不能只改包版本号或将未执行的能力标为通过。
