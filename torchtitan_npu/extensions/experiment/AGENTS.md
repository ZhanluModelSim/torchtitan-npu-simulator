# extensions/experiment 规则

`extensions/experiment/` 放置与 `TrainerEx` 或训练调度紧密集成、尚未承诺稳定 API 的试验性能力。

- 实验能力必须显式配置或显式选择，默认不影响核心 TrainerEx 路径。
- 不在核心 patch、model、override 或 ops 中增加对实验功能的反向依赖或条件分支。
- 当前跨 `patches`、`models`、`override` 和 `ops` 的直接依赖是待迁移债务，不作为新模块的先例。
- 功能转正前先通过 Agent Note 确认 owner、公开接口、默认状态和退出条件，再迁入正式 extension 子层。
