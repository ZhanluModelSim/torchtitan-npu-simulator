# 新模型接入开发流程

本文用于固化一个模型从“代码可以构造”到“具备完整 meta 模拟器建模能力”的接入流程。流程适用于 Dense、MoE 以及包含模型特有融合算子的架构；并行重点覆盖 FSDP（含 eFSDP）、TP、EP、ETP、CP，PP 作为最后阶段接入。

本文默认验收目标是 simulator 的结构、shape、通信依赖和显存建模，不包含真实 NPU 数值训练或硬件 profiler。若项目另行声明需要真实训练可用性，应另立数值/性能验收，不把它混入模型建模接入的结论。

最终是否可以对外声明模型的 simulator 建模可用，不以某个单测或一次单步执行成功为准，而应执行[新模型最终验收规范](../test_guides/model_acceptance.md)。

## 1. 先定义接入范围和模型契约

编码前先形成一份模型契约，避免后续用运行现象反推架构。至少记录：

| 类别 | 必须明确的内容 |
| --- | --- |
| 来源 | 参考实现、配置、权重格式、tokenizer、基线 commit |
| 网络结构 | layer 数、hidden size、head 数、KV head 数、head dim、FFN/intermediate size |
| Attention | MHA/GQA/MLA/KDA 等类型，RoPE、门控、Q/K norm、滑窗或全局 attention 规则 |
| MoE | routed/shared expert 数、每 token 激活专家数、路由方式、专家权重布局 |
| Norm/激活 | RMSNorm 变体、门控方式、激活函数及其融合边界 |
| 权重 | 每类参数的全局 shape、参数量公式、初始化和 tied weight 关系 |
| 状态字典 | 原始权重名与框架权重名的双向映射，是否有合并/拆分权重 |
| 支持范围 | FSDP/eFSDP、TP、EP、ETP、CP、PP、AC、compile、offload 的目标状态 |
| 验证基线 | 算子清单、local shape、通信行为和显存趋势 |

同时记录当前功能分支真正对应的开发基线。模型分支应 rebase 到它所属的训练主分支，而不是默认选择 `master`。解决冲突时优先保留目标基线中较新的通用实现，再把模型差异重新适配进去，避免恢复已经废弃的 shim、配置项或通信建模逻辑。

建议为每类 layer 写出参数量公式。全模型参数量至少能拆成：

```text
embedding + dense layers + moe layers + final norm + output
```

MoE 参数还要区分 shared experts 与 routed experts。这份计算结果会成为后续权重切分、静态显存和算子 shape 验收的独立基线。

## 2. 完成单卡模型骨架

先实现不含并行语义的纯模型，通常包括：

- `model.py`：模型、decoder layer、初始化和 loss 接口。
- `attention.py`：模型特有 attention 及其 mask/cache-free 训练路径。
- `feed_forward.py`：Dense FFN、MoE router、shared/routed experts。
- `config_registry.py`：至少提供 `debug`、`reduced` 和正式规格。

这一阶段应满足：

1. 配置字段能完整描述架构，不能依赖不可见的命令行覆盖才能启动。
2. 单卡前向、反向和 loss 可执行，输入输出 shape 与模型契约一致。
3. 参数总量和逐层参数量与独立公式一致。
4. Dense/MoE layer 的选择规则、特殊首尾层和共享权重关系正确。
5. 对非法配置尽早报错，例如 head、序列长度、专家数不能被目标并行度整除。

不要在基础模型里提前混入 DTensor、DeviceMesh 或模拟器特例。并行布局、NPU 融合和模拟器 shape-only 行为应分别由并行计划、converter 和模拟器适配承担。

## 3. 接入注册、配置和权重转换

一个可被框架发现的模型至少需要完成：

1. 在模型包中提供 `ModelSpec`，注册 model、parallelize、pipeline、loss、optimizer 后处理和 state-dict adapter。
2. 在 `torchtitan_npu` 的模型注入/注册入口中声明模型，使 `--module` 和配置解析能命中它。
3. 提供可直接运行的训练配置与模拟器配置，并使用仓库内有效的 tokenizer/test asset。
4. 实现 checkpoint、DCP 或 HF 权重所需的 state-dict 双向转换。

权重转换要与运行时模块布局保持同一份约定。例如 MoE 的 `w1/w3` 合并成 `w13` 后，converter、state-dict adapter、并行切分和算子建模都必须使用同一个维度顺序。不能只让随机初始化路径可用，而让加载或保存后的权重悄悄错位。

推荐至少验证以下回路：

```text
外部权重 -> 框架 state_dict -> 构造模型 -> 保存 -> 重新加载
```

若存在合并/拆分权重，回路后逐 tensor 校验名称、shape、dtype 和拆分/合并顺序。meta 建模只要求 state-dict schema 闭环；数值逐元素比较属于另行声明的真实权重兼容性范围。

更多 converter 和 state-dict 扩展方式见[模型定制](model_custom.md)。

## 4. 接入 NPU 融合算子和模型特有算子

优先复用通用 converter，例如 RMSNorm、RoPE、GMM；只有语义或权重布局确实不同才增加模型专用 converter。接入时需明确：

- converter 的执行顺序以及它依赖的前置转换。
- 替换前后的模块 FQN 是否稳定，已有 hook、参数和 state dict 是否仍可追踪。
- 训练前向和反向分别对应哪些原始算子。
- 算子是否真正保持融合边界，还是退化成大量小算子。
- 自定义 autograd 的梯度个数、shape 和 dtype 是否与输入一一对应。

模型特有的 Triton、KDA、门控 RMSNorm 等算子，在模拟器中不要求实现真实数值 kernel，但必须提供准确的 shape-only 前向/反向，并以真实 raw op 名称被 hook。不能为了“能跑”把它展开成标量、小 MM 或逐元素算子，否则算子数量、依赖、耗时和显存估计都会失真。

自定义 autograd 与 DTensor 组合时需要特别处理：

1. 本地 shape-only 算子只接收 local tensor。
2. 输出需要按输入的 mesh 和 placement 重新封装。
3. backward 返回的本地梯度也要满足原 placement。
4. hook 应在模型构造和并行化完成后的正确阶段绑定，避免替换模块时丢失并行参数或 hook。

融合算子的通用接入方式见[NPU 融合算子](npu_fused_ops.md)。

## 5. 按统一顺序接入并行

并行实现要显式表达参数、激活和通信布局，不应只以“多卡不报错”作为目标。推荐按以下顺序开发：

1. 非 MoE 部分 TP。
2. MoE 的 EP/ETP，以及它与 TP 的组合。
3. CP。
4. activation checkpoint。
5. FSDP 或 eFSDP。
6. DP replicate 或其他外层数据并行。
7. 最后接入 PP。

具体代码顺序应与当前训练主分支的通用模型保持一致；顺序变化必须说明其布局依据。每种并行需写清下面的契约：

| 并行 | 参数布局 | 关键激活布局 | 典型通信 | 必查约束 |
| --- | --- | --- | --- | --- |
| TP | attention/FFN 指定维度 shard | hidden/head/intermediate 维的 shard 或 replicate | all-reduce、reduce-scatter、all-gather | head、KV head、intermediate size 可切分 |
| EP | routed experts 按 expert 维分布 | token 按目的专家重排 | all-to-all/dispatch/combine | expert 数、top-k、group 与 rank 映射 |
| ETP | 单个专家内部矩阵切分 | expert token 与 intermediate shard | expert 内 collective | 与 EP mesh 正交且 shape 不重复除 |
| CP | sequence/context 维切分 | local sequence、KV/attention 中间量 | ring/all-to-all 等 CP 通信 | sequence length 与 mask/RoPE offset |
| FSDP | 参数/梯度按 DP shard | 计算前临时 unshard | all-gather、reduce-scatter | replicated 参数、padding、reshard 时机 |
| eFSDP | composable FSDP 布局 | 与模块级 fully_shard 边界一致 | all-gather、reduce-scatter | shard group、模块边界和 DTensor placement |
| PP | layer/stage 切分 | microbatch 激活跨 stage | send/recv | stage 边界、loss stage、schedule |

组合并行时，逐个维度计算 local shape，禁止简单使用 `global/world_size`。例如：

- replicated 参数不应除以任何并行度；
- routed expert 参数只按 EP 和可能的 ETP/FSDP 维度切分；
- TP 与 ETP 作用于不同语义维度时分别计算；
- EP/ETP 通常不作为 world-size 公式之外的额外乘数，而是已有 mesh 维度的解释；
- FSDP 计算瞬间存在 full-parameter residency，静态 shard 大小不能代替峰值。

并行入口还应在执行前校验 mesh 和整除条件，使不支持的组合明确失败，而不是静默回退成 replicate。

## 6. 补齐 AC、Hook、插件和训练生命周期

模型接入不仅包括主前向，还要检查框架在模型生命周期上的所有扩展点：

- activation checkpoint：`none`、`full`、`selective`。
- memory estimator、memory snapshot、激活值 hook、module hook。
- 算子统计、原始 op 捕获和导出。
- optimizer 构造后处理、梯度裁剪、checkpoint save/load。
- compile、激活 offload 等声明支持的特性。

重计算场景必须区分 `original_forward`、`recompute` 和 `backward`。如果把两次 forward 合并统计，核心算子数量、依赖关系和激活生命周期都会错误。

模块替换或 converter 执行后，检查 FQN 是否仍可被插件找到。模型专用 fused op 也必须经过同一套 hook 和统计链路，不能成为显存、算子统计或依赖捕获的盲区。

## 7. 接入模拟器建模

模拟器配置应尽量复用真实模型配置，仅覆盖运行规模、硬件和输出项。接入内容包括：

1. 模型能在 meta/fake 路径构造并执行。
2. 所有模型特有算子都有 shape-only 前向和反向。
3. raw op 名称、FQN、dtype、global/local shape 可被记录。
4. 通信事件按 rank-local 语义捕获，并保留 group、peer 和字节数。
5. 前向、重计算、反向依赖能够生成可回放 DAG。
6. 参数、梯度、optimizer、saved activation、临时 buffer 和通信 buffer 能进入显存模型。

不要用全局 monkeypatch 恢复已经删除的模拟器 shim；应使用当前主分支提供的模块级 bridge 或 converter 扩展点。模拟器只替代数值计算，不应绕开真实模型结构、并行计划和 autograd 拓扑。

运行和输出文件说明见[模拟器使用指南](../user-guides/simulator.md)。通信归属、依赖重建和显存模型分别见：

- [通信归属契约](../design/communication-ownership-contract.md)
- [调度计划依赖重建契约](../design/schedule-plan-dependency-reconstruction-contract.md)
- [模拟器显存模型设计](../design/simulator-memory-model-design.md)

## 8. 建立测试分层并进入最终验收

开发阶段至少建立以下测试：

- 配置和 registry 单测。
- 单卡模型构造、参数量、前向/反向 shape 单测。
- state-dict 转换和 round-trip 单测。
- 每种并行的布局计划与非法配置单测。
- 模型特有算子的 forward/backward shape、DTensor 和 hook 单测。
- 模拟器单步、通信捕获和显存输出 smoke test。
- reduced 规格的 meta 单步、AC 开关和核心并行组合测试。

这些测试用于尽早发现局部问题，但不能替代最终验收。实现完成后，按照[新模型最终验收规范](../test_guides/model_acceptance.md)生成完整证据包并给出结论。

## 常见失效模式

| 现象 | 常见根因 | 应对方式 |
| --- | --- | --- |
| 单卡可跑，多卡 shape 错 | local/global shape 混用，重复除并行度 | 为每个 tensor 记录语义维和 placement |
| TP 后 hook 调用 `to_local` 失败 | 自定义算子丢失 DTensor 包装 | local 执行后按原 mesh/placement 重封装 |
| 算子总数很大且都是小算子 | fused/Triton op 被分解或未被 hook | 使用真实 raw op 的 shape-only bridge |
| AC 后算子数翻倍但统计无法解释 | 原始前向和 recompute 未区分 | 按 execution kind 分桶统计 |
| FSDP 通信存在但回放错误 | unshard/reshard/grad RS 的依赖边错误 | 校验通信前后最后生产者和首个消费者 |
| EP 前向正常、反向挂起或次序错 | dispatch/combine 的逆向依赖缺失 | 独立核对 backward all-to-all 链路 |
| 显存远小于真实值 | 只统计静态 shard，漏 full-param 或临时量 | 分 persistent、transient、saved activation 建模 |
| 默认配置无法启动 | tokenizer、配置名或注册依赖隐藏覆盖 | 使用仓库资产执行零额外参数 smoke test |
| rebase 后旧 bug 回归 | 冲突时恢复了模型分支旧通用实现 | 以训练主分支当前通用实现为基线重适配 |
| 测试命令失效 | 沿用过期配置名、算子名或接口 | 以目标基线现有 API 编写测试 |
