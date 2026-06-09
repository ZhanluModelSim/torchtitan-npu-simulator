# torchtitan-npu skill 集合

该目录包含 `torchtitan-npu` 的 skill 定义。基于昇腾NPU的训练调试经验总结，支持Agent快速定位和解决问题。

## skill列表

| Skill | 说明 |
| --- | --- |
| `accuracy-debug` | 有基线对照的训练精度异常定位（loss 偏离、NaN/Inf），基于代码审查 + detect_anomaly + msprobe dump/compare 流程 |
| `premerge-accuracy-check` | PR 合入前的数值精度一致性验证，对比两个分支/commit 的 loss 和 grad_norm，并生成包含复现步骤的 PDF 报告 |
| `oom-analysis` | NPU 训练 OOM 问题诊断，按日志分类 → 静态内存估算 → Memory Snapshot 深度分析 → 优化建议的流程定位和解决问题 |
| `torchtitan-sync` | 上游 torchtitan 分支同步与适配，读取 versioning_policy.md 分支同步表，生成变更分析并完成代码适配 |
| `training-log-visualization` | 训练日志指标提取与 loss/grad_norm 曲线绘制，支持双日志对比 |
| `torchtitan-npu-code-reviewer` | 本仓代码审查与 codecheck 失败定位 |
| `default-skills` | 安装 GitCode PR 和 pipeline 等远程默认 skills |
| `torchtitan-npu-env-setup` | 当开发者需要在 Ascend NPU 上按 torchtitan-npu 官方安装文档完成环境搭建，或排查安装步骤不一致导致的环境问题时使用。 |
