# Simulator 历史修复与当前边界

> 现状核对：2026-10-07，`feat/npu-simulator`，实现基线 `ecde2f0`。依据源码与 Git 历史静态核对，未执行测试或仿真；后续行为以当前源码为准。

返回[使用入口](../user-guides/simulator.md)。本表记录问题、根因和已合入处理，提交缩写可用 `git show <sha>` 查询。

以下从当前代码、设计分析和已合入提交重建；不是本次重新运行的结果。

| 问题 | 根因 | 当前处理/代表提交 |
|---|---|---|
| Fake PG 仍报 c10d 无 Meta kernel | Fake backend 只替代通信进程组，不给 c10d 算子补 meta 实现 | 拦截 dist/funcol/autograd/P2P，构造 shape 正确输出并记录事件 |
| 无 NPU 时初始化仍触发硬件 | `.npu()`、硬编码 device、torch.npu stream/probe、pinned host state、AMP 等绕过统一 device_type | meta device accessor/stub 与专门 patch，禁 swap_optimizer，meta AMP 分支 |
| MoE split `.cpu().tolist()` 崩溃 | token counts 依赖数值，meta 无数据 | 均衡路由 + 静态 splits，保留通信桥 |
| EP AllToAll 丢失，连 backward 也不见 | fake 分支在调用拦截器之前短路通信，autograd 节点也没有建立 | dispatch/combine 自定义 autograd bridge；7 月捕获修复，`883bf7f` 等补依赖 |
| RoPE/RMSNorm/MoE/GMM 的 backward 或保存激活不完整 | meta forward 能走，但缺显式 autograd/backward bridge；shape placeholder 丢失真实生产者身份 | 完整 backward bridges；`e7c712d`、`6687204`、`f177362`，GMM 保存真实前向中间值 |
| fused AdamW 被展开成 foreach 或少参数 | 早期为了 meta 改成 foreach；只用代表参数无法表达完整工作 | 保留 fused 路径 synthetic `npu_apply_adam_w`，公开 param_groups 捕获，区分逻辑 shape/本地内存 |
| 把输出大小求和当 peak | 单 op cost 与生命周期指标混淆 | active set/lifetime estimator，phase peaks 和 pre-optimizer peak |
| parameter/DTensor view 重复计费、leaf/gradient shape 丢失 | nn.Parameter/DTensor 边界及 materialization alias 语义不同 | 参数身份保留、本地 shard/placement 校验，`20ab1b0`、`632858f`、`833b90c` |
| full/selective AC 生命周期错误 | 原始 forward、重算、缓存命中混淆，缓存 tensor 被提前释放；一次性 context 不能多次重入 | execution_kind、可重入 context、synthetic output cache；`5446b46` |
| fake A2A 的 SAC 选项含糊 | 通用 comm 保存和 synthetic A2A 目标混淆 | `0ca254b` 接入 cache，`4516f3a` 要求显式 all-to-all |
| MoE permute/slice_backward 被当成 alias | 旧规则按 `permute/slice` 子串匹配，误伤分配型 op | `ecde2f0` 精确 namespace/operator 匹配，多输出独立核算 |
| PP phase/stage/MB 和 P2P peer 错归属 | stale 上下文、延迟 P2POp、物理/逻辑 rank 混用、只跟踪首个 model part | semantic contexts、P2POp 标记、SimulationRankContext、多 root ModulePathTracker |
| 后续 MB 的 FSDP 通信消失或重复 | 只复用首个计算模板，未处理通信变化和执行 ownership | 不可变 comm variants、group/transition ID、ownership normalization |
| FSDP prefetch 成环或全 stage 被错误门控 | 粗暴补控制边，忽略参数组和真实 source invocation | 按参数组区域及 source readiness 重建；`8bd4736`、`23e3e71` |
| FSDP RS/AR 被串行或污染 CP RS | 观察顺序冒充 data dependency、按 primitive 名称归类 | 真实梯度依赖、HSDP RS→AR 保留、CP/TP 隔离；9 月系列修复 |
| replicated first layer 被当 FSDP shard、HSDP 轴被弄错 | replicate 没有 shard 语义；2D mesh 的 shard/replicate 轴不同 | `bbe4282` replica-aware capture；`627f15a` HSDPMeshInfo 显式轴 |
| DeepSeek PP replay 不稳 | 多 virtual stage 路径、cross-action prefetch invocation、结构校验不一致 | `b980b74`：多 root tracker、源 FQN/末次 invocation、通用 readiness 校验 |
| 大 world 启动进程过多或 CLI 不生效 | 使用 logical world 启动 worker，或在最终 override 之前读取 degree | 最终配置先解析，仅 PP workers；`e3177b2`、`83c9b27`、`e171992` |
| FSDP init 创建海量空 meta shards/mesh | 每参数 eagerly 扩展 world-size chunk list，重复构建 shard/SPMD mesh | LazyMetaChunks、mesh identity cache；`84902a4`、`204d534`、`2a1f9d5` |
| 6 万卡量级 DeviceMesh 坐标初始化慢 | root mesh 反复扫描、累计代价大 | `00c5f59` 用 arange mesh 算术解坐标，任意 submesh 回退；历史 commit 记录 300s 超时降到秒级，本次未复测 |
| FSDP prefetch 重复重建拓扑慢 | 重复 graph topology/path 查询 | `70016c6` cache，按 invocation/区域索引复用 |
| 图/CSV 过大、无工作 view 被当 kernel | 全 logical ranks 展开、模板重复、metadata view 未折叠 | 模板首现捕获、metadata contraction、csv_max_ranks、按需格式、JSON 快速路径 |

DeviceMesh 优化的实际代码仍先构造/比较 arange mesh，因此坐标推导是常数级并不意味着整个 helper 或整个 mesh initialization 严格 O(1)。不能只拿提交标题作为当前复杂度证明。

## 当前边界

- Kimi PP、DeepSeek ETP、DeepSeek MTP+PP、memory_budget AC、reentrant AC 均有明确限制，详见[矩阵](../user-guides/simulator/model-matrix.md)。
- PP 内存回放暂不重放精确 autograd save slots，使用 use-def/checkpoint 路径；不能声称与非 PP 的精确保存记录具有相同覆盖度。
- saved activation offload 与 FP8 通信是建模选项，尚未提供完整 offload executor、传输时延和硬件校准。
- `target_npu_device_type` 仅检索到配置定义与默认值测试，未发现生产消费点。
- generic ExpertParallel 与 NpuExpertParallel 的 FP8 dispatch / synthetic AllToAll SAC 接入不同，公共开关存在不等于所有模型均有效。
- 未知算子成本、非代表 rank 的动态路由和负载波动仍有覆盖边界。

上述是静态核对的实现边界，未新增失败复现或测试结果。原始分析和方案保留在[文档导航](../user-guides/simulator/documentation-map.md)中，不能把其中的待办状态直接视作当前缺陷。
