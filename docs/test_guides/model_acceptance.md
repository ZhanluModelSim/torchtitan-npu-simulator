# 新模型最终验收规范（Meta 模拟器）

本文定义什么时候可以判定一个新接入模型“可用”。它是模型开发完成后的最终验收，不是各开发步骤的局部验收。接入步骤见[新模型接入开发流程](../feature_guides/new_model_onboarding.md)。

## 1. 可用的定义

只有同时满足以下条件，才可把模型标记为“模拟器建模可用”：

1. 声明支持的模拟器特性和并行组合全部通过必选验证。
2. 参数切分、核心算子、通信依赖和显存均有独立预期值，并与实测证据一致。
3. meta 路径验证了结构、shape、autograd 连通性、依赖和资源估计。
4. 不支持的模式被明确记录并 fail fast，不能静默降级或产生看似成功的错误结果。
5. 验收证据记录 commit、基线、配置、world size、软件版本和输出路径，可重复执行。

“进程退出码为 0”或“模拟器能生成报告”都只是必要的局部信号，不能单独证明模型建模正确。meta 不产生可用于精度判断的 loss/grad norm 数值；真实 NPU 数值训练和硬件 profiler 是独立目标，不是本文的默认门槛。

## 2. 验收前准备

验收前冻结以下输入：

- 被测 commit 和它基于的训练主分支 commit。
- 模型配置、tokenizer/test asset、用于固定 MoE 路由的输入或策略，以及 state-dict 样本 schema。
- 模拟环境，包括 Docker 镜像或 digest、PyTorch、torch_npu、CANN、模拟 world size 和 capture rank。
- 模型支持矩阵，以及明确排除的功能。
- 独立计算的参数量、逐层权重 shape、并行后 local shape。
- 核心算子清单和理论调用次数。
- 每种并行预期产生的通信类型、消息 shape/字节数和先后关系。

建议先用 `debug` 规格定位，再用 `reduced` 规格完成 meta 核心组合测试，最后用正式规格进行模拟器的容量与数量级验证。不能只用缩小后改变了 layer 类型、MoE 分布或 attention 规律的配置代替正式结构。

## 3. 必测配置矩阵

先声明模型支持矩阵，再从下表选择必测项。FSDP（含 eFSDP）、TP、EP、CP 是 MoE 模型接入的核心门槛；ETP 在模型声明支持时同样为必测。PP 最后验证。

| 场景 | 并行配置 | 目的 |
| --- | --- | --- |
| 单卡基线 | 所有并行度为 1 | 参数、算子和显存的参考基线 |
| FSDP | `dp_shard=2`，分别测试 FSDP/eFSDP | 参数/梯度切分和 unshard/reshard |
| TP | `tp=2` | attention、FFN、Norm 边界和 TP 通信 |
| EP | `ep=2` | expert placement、token dispatch/combine |
| ETP | `etp=2`，并与 EP/TP 的目标组合测试 | 专家内部切分及组合布局 |
| CP | `cp=2` | sequence 切分、mask/RoPE offset 和 CP 通信 |
| 核心组合 | FSDP/eFSDP + TP + EP + CP，按需加入 ETP | mesh 正交性、local shape 和通信组合 |
| PP | `pp=2`，最后执行 | stage 边界、microbatch 和 send/recv |
| 最终组合 | 加入 PP 的声明支持组合 | 全功能最终门槛 |

不需要机械覆盖所有并行度的笛卡尔积，但必须覆盖：

- 每个并行维度单独开启一次。
- 每对会共同作用于同一参数、激活或通信域的并行至少组合一次。
- 至少一个不含 PP 的核心组合。
- 声明支持 PP 时，至少一个含 PP 的最终组合。
- 所有验证分别覆盖 activation checkpoint 关闭和开启；开启至少包含实际支持的 `full` 或 `selective`。

若支持激活 offload、compile 或其他会改变图和显存的功能，应在核心组合上增加开启/关闭对照。

## 4. 验收维度与判定标准

### 4.1 配置、结构和权重

检查：

- 默认配置不依赖隐藏命令行覆盖即可解析和构造。
- 实际 layer 类型、数量和分布与配置一致。
- 全局参数总量、逐层参数量和 tied weight 去重规则与独立公式一致。
- state-dict 名称、shape、dtype 和合并/拆分顺序正确。
- DCP/HF 转换路径按声明支持，保存/重新加载后 key、shape、dtype 和权重布局保持一致。

静态参数字节数按实际 storage dtype 计算：

```text
parameter_bytes = numel × bytes_per_element
```

不能把计算 dtype 当成参数 storage dtype，也不能因 tied weight 重复计数。

### 4.2 并行切分后的 local shape 和静态显存

对每个参数和核心激活记录：

```text
FQN, global_shape, local_shape, dtype, mesh, placement, replicated/sharded
```

判定要求：

1. local shape 由 tensor 语义维和 placement 推导，与实际 DTensor/local tensor 一致。
2. replicated 参数不除并行度；sharded 参数只除实际作用在该维度的 mesh。
3. MoE routed expert 只出现在所属 EP rank；shared expert 的复制/切分符合设计。
4. TP、ETP、FSDP 同时开启时，不重复切分同一维，也不漏掉应切分的维度。
5. padding、对齐和空 expert rank 必须有显式解释。
6. 在一组小的并行对照矩阵中，静态参数字节数的变化与实际 placement 语义一致；不要求为所有模型建立通用的自动精确公式。

建议至少对照单卡、FSDP/eFSDP、TP、EP、ETP、CP 和一个核心组合：FSDP 应降低可分片常驻权重；EP 主要影响 routed experts；ETP 只影响专家内部语义维；CP 不应改变静态权重。组合场景重点检查没有重复切分或漏切分，而非把所有参数简单除以 world size。

FSDP/eFSDP 还需同时验证：

- 常驻 shard 的大小。
- 计算前 all-gather 后 full parameter 的 shape 和临时驻留。
- 反向梯度 reduce-scatter 后的 local shape。
- 直接通信依赖：all-gather 在首个消费者前，reduce-scatter 在最后一个梯度生产者后。

`fsdp_state`、residency transition 和 L2 prefetch 标注属于公共模拟器调度能力，不是模型专用语义。非 PP 基线若所有模型都未产出这些标注，应以 L0 collective 的 group、shape 和 producer/consumer 边验收模型；公共标注缺失应单列为 simulator 任务，不能在新模型中引入特例补丁。

### 4.3 核心算子的 shape、数量和融合

先建立算子账本，至少包括：

| 字段 | 含义 |
| --- | --- |
| raw op/FQN | 实际捕获到的算子名和所属模块 |
| layer 类型 | attention、dense FFN、shared/routed expert 等 |
| execution kind | original forward、recompute、backward、optimizer |
| global/local shape | 各输入输出和关键权重 shape |
| 理论次数 | 按 layer 数、激活专家数、microbatch、AC 模式计算 |
| 实际次数 | 模拟器统计值 |
| 融合要求 | 允许的实现及禁止出现的小算子分解 |

重点核对：

- attention 核心算子以及 Q/K/V、score、output 的 MM/BMM shape。
- GMM 专家输入 token 数、expert offset、权重 shape 和 routed/shared expert 次数。
- RMSNorm、门控 RMSNorm、RoPE、激活函数等融合算子。
- 模型特有的 Triton/KDA/其他算子。
- 前向、反向的 MM/BMM/GMM 数量和比例。

算子次数必须分别按 execution kind 统计。若某段被 full AC 包裹，通常会出现一次 original forward 和一次 recompute；未重计算的段不能被统一乘二。验收报告要写出理论公式，而不是只给出一个总数。

MoE 的 GMM shape 和执行量受实际路由 token 数影响。应固定输入与 router 结果，或保存每个 expert 的 token histogram，再由该 histogram 推导预期值；不能简单用 `num_experts × layer 数` 代替实际调用量。空 expert、capacity/padding 和 shared expert 需要分别计数。

对于模拟器中的模型特有算子，允许使用 shape-only 实现，但必须：

1. 以真实融合 op 名称被捕获。
2. 前向、反向 shape 和梯度输入数正确。
3. 参与 hook、依赖图和显存生命周期。
4. 不退化为大量逐元素或小 MM 算子。

### 4.4 通信行为和依赖回放

每个通信事件至少记录：

```text
rank, process_group, collective/p2p type, tensor shape, bytes, peer/root, phase
```

数量和字节数由 local tensor shape 推导，并按 rank-local 语义统计；不能把一个 rank 捕获到的 collective 人工扩成全 world 的事件。

按并行类型验证以下不变量：

- TP：局部 GEMM 的生产者先于 all-reduce/reduce-scatter，后续消费者等待通信结果。
- CP：KV/attention 交换与对应计算块匹配，mask、RoPE offset 和 sequence chunk 不错位。
- EP：router/dispatch → all-to-all → expert 计算 → all-to-all → combine 链路完整；backward 按正确的逆向数据依赖执行。
- FSDP/eFSDP：参数 all-gather/unshard 在首个参数消费者前完成；梯度 reduce-scatter 在最后一个梯度生产者之后开始。
- PP：send/recv 成对，tensor shape、microbatch、stage 和 schedule slot 匹配。

依赖图必须满足：

1. 无未解析依赖、悬空节点和环。
2. 每个通信有唯一归属，不重复插入。
3. producer/consumer 边跨原始前向、重计算和反向时仍指向正确 execution kind。
4. 通信前后的 compute 节点可追踪，回放顺序与训练语义一致。
5. 必需的 P2P transfer 成对，collective 的 group/rank 成员一致。

设计约束可参考[通信归属契约](../design/communication-ownership-contract.md)和[调度计划依赖重建契约](../design/schedule-plan-dependency-reconstruction-contract.md)。

### 4.5 端到端显存估计

显存报告至少分解为：

- 常驻参数 shard/replica。
- FSDP full-parameter 临时驻留。
- 梯度和 optimizer state。
- saved activations。
- recompute 临时激活。
- attention、GMM 和融合算子的 workspace/temporary。
- collective/P2P buffer。
- offload/prefetch 相关驻留。

必须对 AC 关闭与开启分别验证；支持 offload 时再做开关对照。除了 peak 数值，还需核对 peak 发生在哪个执行阶段、由哪些 tensor 构成。

以下趋势应符合逻辑，否则必须解释：

- FSDP 后常驻参数通常随 shard 增大而下降，但计算阶段仍可能出现 full-parameter 峰值。
- full/selective AC 应减少 saved activation，增加相应区域的 recompute 算子。
- offload 应降低设备驻留，但不能让逻辑 tensor、prefetch 或传输成本凭空消失。
- sequence length、microbatch 或 expert token 数增加时，相关激活/workspace 应按理论阶数增长。
- TP/CP/EP 改变的是对应语义维的 local tensor，而不是把所有显存简单除以 world size。

验收应给出各对照配置之间的变化原因和 peak 构成。对于模型特有 workspace 或动态 token 路由，重点验证量级、单调性和 tensor 生命周期逻辑；不要求使用真实 NPU allocator 结果作为基线。

显存输出字段见[模拟器使用指南](../user-guides/simulator.md)，建模原则见[模拟器显存模型设计](../design/simulator-memory-model-design.md)。

### 4.6 插件、Hook 和训练特性兼容性

在单卡和核心组合上检查：

- memory estimator、snapshot 和逐 module/activation hook。
- 算子计数、raw op 捕获和 CSV/JSON 导出。
- AC、activation offload、compile 等声明支持的图变换。
- fused RMSNorm、GMM、attention 和模型特有算子是否仍被 hook。
- optimizer、梯度裁剪，以及 checkpoint/state-dict schema 的 save/load 闭环。
- converter 后模块 FQN、参数身份和 state-dict 路径是否稳定。

任何 fused/custom op 若在统计、显存或依赖图中不可见，视为未兼容；不能以真实 kernel 尚未实现为由跳过 shape-only hook。

### 4.7 回归和可维护性

最后执行仓库现有单测、模型单测和 smoke test，并确认：

- 没有修改其他模型的注册、默认配置和 converter 行为。
- 没有恢复目标训练主分支已删除的兼容层或旧接口。
- 非法或未支持配置有清晰错误信息。
- 新增测试使用当前 API、配置名和算子名。

通用测试命令见[测试指南](test_guide.md)。

## 5. 推荐执行顺序

建议按以下顺序收敛问题：

1. 单卡：参数、shape、核心算子账本。
2. FSDP/eFSDP、TP、EP/ETP、CP 分别开启。
3. 不含 PP 的核心组合。
4. 在单卡和核心组合上分别执行 AC 关闭/开启。
5. 插件、hook、显存和依赖回放。
6. state-dict/checkpoint schema 的 meta 闭环。
7. 最后执行 PP 和含 PP 的最终组合。
8. 正式规格模拟器容量验证及证据归档。

模拟器示例：

```bash
python3 scripts/run_simulator_spawn.py \
  --config <model>_simulate \
  --simulation.output_formats mem \
  --activation-checkpoint.mode none

python3 scripts/run_simulator_spawn.py \
  --config <model>_simulate \
  --simulation.output_formats mem \
  --activation-checkpoint.mode full
```

## 6. 验收证据模板

每个必测配置保留一行汇总，并链接原始报告：

| 字段 | 内容 |
| --- | --- |
| Commit/base | 被测 commit、训练主分支基线 |
| Environment | Docker 镜像/digest、torch、torch_npu、CANN、模拟 world size、capture rank |
| Model/config | 模型 flavor、layer、seq、microbatch、dtype |
| Parallel | DP/FSDP/eFSDP/TP/EP/ETP/CP/PP |
| Features | AC、offload、compile、converter |
| Parameters | 全局数量、每 rank 字节数及并行对照下的变化解释 |
| Operators | 预期/实际核心 op shape 和 F/R/B 次数 |
| Communication | 类型、次数、字节数、依赖校验结果 |
| Memory | 模拟 peak、阶段、构成与 AC/并行对照 |
| Lifecycle | state-dict schema save/load、hook/plugin 结果 |
| Evidence | trace、CSV/JSON、memory report、日志路径 |
| Result | PASS/FAIL，以及差异说明 |

最终汇总应明确给出以下三种结论之一：

- **Ready**：所有声明支持的模拟器范围通过，证据完整，无静默降级。
- **Conditionally ready**：明确缩小了模拟器支持范围；未支持项有文档并 fail fast，不能按完整支持宣传。
- **Not ready**：任一核心并行、shape/数量、通信依赖或显存逻辑未通过。

以下情况必须判定为 **Not ready**：

- 只验证了单卡或只验证进程能退出。
- 只看总显存，没有参数/激活/临时量分解。
- 只统计前向，没有重计算和反向。
- 只看通信节点存在，没有验证 shape、字节和依赖。
- 自定义融合算子被展开成小算子或未进入 hook。
