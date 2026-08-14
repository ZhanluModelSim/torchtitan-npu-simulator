# 版本策略

`torchtitan-npu` 采用“分支 + commit 基线”的方式与上游 `torchtitan` 保持对齐。
本政策用于定义发布兼容性约束，以及分支同步信息的唯一事实来源（source of truth）。

## 发布兼容性表

下表列出了已发布 `torchtitan-npu` 版本对应的上游 `torchtitan` 基线及关键依赖兼容范围。

| torchtitan-npu | torchtitan | Python | Stable CANN | PyTorch/torch_npu | triton-ascend |
| --- | --- | --- | --- | --- | --- |
| master | main (ac13e536c84e7f6647b14fa9375c3c8a8a2b8578) | 3.11.x | 9.0.0 | 2.12.0 / 2.12.0.dev20260808 | 3.2.1 |
| v0.2.2-dev | v0.2.2 (73a0e6979dd10b6b1904098eb3c8f62c18ab87ce) | 3.11.x | 9.0.0 | 2.10.0 / 2.10.0 |3.2.1 |

对于活跃开发分支，请始终以 `分支同步表` 为准。

## 分支同步表

本表是 `torchtitan-sync` skill 的唯一事实来源。
在执行同步前，必须先读取与当前 `torchtitan-npu` 分支对应的数据行。

| torchtitan_npu_branch | torchtitan_repo_url | torchtitan_branch | torchtitan_commit | updated_at | notes |
| --- | --- | --- | --- | --- | --- |
| master | https://github.com/pytorch/torchtitan.git | main | ac13e536c84e7f6647b14fa9375c3c8a8a2b8578 | 2026-05-11 | f7d0133 breaking refactor: adapt to upstream dataclass ConfigManager/config_registry, remove TOML/custom_config flow for migrated models, flatten models layout. |

格式规则：
- `torchtitan_commit` 必须仅包含纯 commit hash。
- 不得在 `torchtitan_commit` 中附加说明性文本（例如 `, v0.2.2 tag`）。

## 分支策略

- `master`：长期集成分支，应跟踪稳定上游基线。
- `releases/vX.Y.Z`（如使用）：发布维护分支，使用固定上游基线。
- `feature/*` 或 `dev_*`：开发分支，仍必须维护同步元数据的准确性与时效性。

常规流程应先将变更合入 `master`，再按需回移（backport）到维护分支。

## 同步更新策略

每次完成上游同步后，必须同步更新本文件：
1. 将 `torchtitan_commit` 更新为本次同步目标 commit。
2. 更新 `updated_at`。
3. 在 `notes` 中记录 commit 区间与主要适配点。
4. 若兼容性发生变化，同步更新 `发布兼容性表`。

若当前分支在表中无映射行，必须先补充该行后再执行同步。
