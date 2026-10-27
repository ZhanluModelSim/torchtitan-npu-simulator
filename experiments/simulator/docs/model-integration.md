# 模型适配与验收

本文用于向 simulator 接入新模型，涵盖结构对照、配置、组件、并行、量化、checkpoint、保存值归属和完整验收。
通用运行时缺口按 [捕获运行时开发](capture-runtime.md) 处理。结构正确、没有缺梯度或全套回归通过，都不能单独证明新模型的内存口径正确。

## 固定模型基线并列出差异

固定官方 config、模型实现及 revision，保留旧模型源码和目标 TorchTitan/NPU 版本。先列出结构和行为对照，不从名称相同推断公式相同：

- 完整几何：层数、层型顺序、head/rank/dim、专家数与 top-k、共享专家、词表、MTP，以及位置／mask／压缩配置。
- 每个模块的投影、激活、normalization epsilon、gate、router score 位置、dtype 转换、初始化和冻结参数。
- 旧虚拟融合的前后向名称、输入／输出、保存值和 attrs；明确哪些边界必须保留。
- 官方 checkpoint key、tensor 方向、分片方式；旧原生 checkpoint 是否另需转换。

小配置用于快速验收，但必须包含新模型的关键不同路径。完整配置保留真实层数和几何；build-only 不等于完整训练捕获。旧版峰值若基于不同几何，不能作通过阈值。

## 代码归属和原生配置树

| 内容 | 承载位置 |
| --- | --- |
| 模型定义、配置工厂、sharding、parallelize、state adapter | 主包 `torchtitan_npu/models/<family>/` |
| 已有公共 NPU 组件选择 | 原生配置级 override／扩展注册入口 |
| 仅模拟所需的 Meta/autograd／虚拟融合激活 | sidecar `compat/`，由 runtime 显式配置 |
| 模型矩阵、原始图／梯度／内存消费者 | sidecar `tests/`；CPU 数学与状态映射按主仓测试目录规则 |

新模型层只依赖标准库、上游和模型层内部模块，不直接导入 `extensions`、`override` 或 `ops`，详见 [模型层规则](../../../torchtitan_npu/models/AGENTS.md)。
需要 kernel 时通过配置或既有 override 注入。不能为了 simulator 构建，把真实 kernel 依赖和实验分支硬编码到通用模型 forward。

原生 `Trainer.Config`／`ModelSpec` 必须包含实际 model config、parallelize、loss、optimizer 和需要的 adapter／hook。
sidecar 当前调用 `factory(num_mtp_layers=...)`；不支持的 MTP 显式拒绝。新增模型还需接入 `prepare_config()` 的 family/preset 校验、默认 override 和 runtime 激活，不能仅添加模型目录或接受任意私有对象。

Linear、Conv、Embedding、norm、dense／routed/shared experts 和子模块应形成可遍历的原生配置树。constructor 内临时创建的模块可能绕过 override、量化和 sharding；用实际选中对象验证整条路径。
修改几何时重建完整配置树，不只赋值顶层 `dim`。共享嵌套 dataclass／list 不因浅拷贝而隔离；避免一次 prepare 修改另一个 preset 或下一次运行。

复用原生 Module 的构建、状态初始化和分布式生命周期；显式保留原模型初始化策略。native Linear/Conv/expert 的默认初始化可能不同于旧实现，不能只因加载接口正常就忽略。
参数分发／重建后重新检查 `requires_grad`。确实不参与路径的参数应说明并冻结；不能通过合成梯度让清单看起来完整。
模型构建必须使用配置声明的 dtype 上下文并恢复进程原状态，不能受宿主默认 dtype 干扰或将全局设置泄漏到后续捕获。

## 组件、虚拟融合与量化

优先扫描已有模型和公共组件。完全等价的 router、dispatcher、RoutedExperts、GroupedExperts 和投影复用原生实现；只保留新模型不同的数学与阶段。
例如当前 V4 和 Kimi 均为独立 gate/up GMM 再 down GMM，不能据旧印象擅自合成；router score 在 W2 前后相乘会改变 BF16 舍入，必须按基线验证。

能走原生 CANN dispatch 的组件不额外手工记图。原生 Meta/autograd 健康时不安装 bridge；缺口验证和兼容安装见运行时指南。
保留虚拟融合时，注册 schema、Meta 输出和真实输入的 autograd 保存；backward 消费实际 dy，输出梯度的 shape/dtype/身份关系可追溯。
Meta factory 可以创建算子输出元数据，但不能用同 shape tensor 伪造本应保存的真实前向输入。作用域退出恢复调用点，真实设备无 provider 时明确失败，不暗中使用数值不等价的替代实现。

模型 preset／工厂只定义结构，不增加 `_bf16`／`_mxfp8` 工厂变体或根据模型名改写量化配置。调用者通过 `enable_quantized_training` 显式开启量化，默认关闭；recipe、基础 dtype、FSDP 计算／归约 dtype 各自独立。新增模型验收同一工厂的量化开／关两条路径，核对实际参数包装与配置不被静默覆盖。

量化走公共原生 recipe／转换器，核对实际包装参数集合与 code/scale。配置被遍历不等于 custom expert/activation 子类被保留；需要检查转换后的实际 SiTU 等路径，并排除错误替换成公共 SwiGLU。
不复制旧 FQN 名单、不默认启用其他模型专属的 sparse/LI/KV 量化选项；新模型缺少扩展字段时按公共配置机制组合，不能静默忽略量化开关。

## 并行、精度和 checkpoint

沿模型的原生 config-sharding 与 parallelize 顺序构建参数分布，再使用目标版本的 AC／FSDP 接口；参考 [Kimi parallelize](../../../torchtitan_npu/models/kimi_k3/parallelize.py) 与 [sharding](../../../torchtitan_npu/models/kimi_k3/sharding.py)。

- 使用实际 dense/expert mesh 和 process group，分别验证 TP/EP/FSDP 的 local shape、placement 和非零 rank。原生轴选择可能随 degree 和专家数改变，不硬编码「所有参数沿第 0 维切」。
- `dp_mesh_dims` 等参数必须与已构建的原生 SPMD 参数状态匹配，不能直接把旧 plain-Tensor FSDP 调用加上新参数就视为迁移。
- TP partial 必须在后续非线性前完成归约，例如 latent MoE 的 RMSNorm；不能为了图连通把归约挪到 norm/up 投影后。
- CP 需保留 attention、因果卷积、递归状态和序列位置的完整语义。KDA 不能直接切开序列后各自执行；保守 full-context gather／local slice 也要验证输出和全部梯度，不把两个 gather 的反向叠加成重复 loss。
- 参数总量和每 rank shard 大小用独立几何／分片公式核查；输入 tensor 的逻辑 shape 不自动等于物理 local allocation。

区分原始参数 dtype、FSDP 计算 dtype、reduce dtype、量化 dtype 和常驻存储投影。
当前 DSV4／Kimi base BF16 与显式 FP32 参数可能违反原生 FSDP 的统一原始 dtype 约束；使用 FP32 原始参数与 BF16 `mixed_precision_param` 不是全 FP32 计算。
KDA 的 FP32 原始 A_log/dt_bias 不等于需要强制 FP32 FSDP 计算；没有 kernel 契约证据不要新增特殊 policy。build-only、真实混合精度捕获和内存 dtype 投影分别验收。

state adapter 必须对照实际模型 `state_dict`，覆盖 dense/router/latent/shared、低秩 gate、norm key、专家堆叠顺序和非方阵方向。
不能只比较 key 数量或做同源 roundtrip。旧原生 key 因 w1/w2/w3、router 和专家布局变化而失效时，需要明确转换路径；HF adapter 通过不等于旧 native checkpoint 自动兼容。
CPU 数学对照固定同一权重与输入；真实训练数值按仓库精度指南执行，不能用相同 seed、有限 stdout 精度或 Meta 梯度存在证明比特一致。

## 接入顺序与捕获边界

沿用「原生 dispatch → 补齐必要的 Meta/autograd 能力 → 无法表达的部分才显式建模」的顺序。
先检查模型实际选中的配置、override 和算子，再决定是否需要兼容代码。模拟替换必须在显式作用域内激活并恢复，不修改上游安装源码。

确认新模型的调用边界，包括 decoder、MTP、最终 norm、head、loss、梯度桥接、clip 和 optimizer。
这些边界不一定都对应 `nn.Module`。函数式 loss 的 `module_path` 为空，不自动表示 module tracker 失效；反之，decoder 内大量保存值失去层归属，也不能以「函数式调用」为由跳过排查。

与既有模型复用组件前，核对公式、dtype 转换、router score 的乘法位置、保存输入和前反向融合边界。例如 SiTU 不等于 SwiGLU，FP32 folded score-weight 归约也不等于 BF16 affine RMSNorm 后再投影。CPU 数学对照和 Meta 图验收分别记录，不相互替代。

## 保存值不等于卸载候选

`autograd_saved_tensors` 记录 autograd 实际保存的对象，可能包含激活、参数视图、索引、归一化统计量、量化 code/scale 或已计算的梯度。不能把所有保存槽都归类为 decoder 激活，也不能将它们全部投影为 offload。

| 实际来源 | 默认处理 | 必须核查的证据 |
| --- | --- | --- |
| 明确层或显式 checkpoint 边界保存、供后续反向／重算使用的激活 | 可作为 offload 候选 | 保存模块、实际 consumer、边界和本地 storage |
| chunked loss 内保存并立即反向消费的 logits/log-prob 等 | 保留设备驻留，不计 offload | pack/unpack、全部使用与语义 F action |
| 最终 norm、head、输出 Attention Residual 等层外保存值 | 保留设备驻留，不计 offload | 已知模块路径、调用链及生命周期 |
| head 累积梯度交给 decoder 的桥接张量 | 保留设备驻留，不当作层内激活卸载 | 梯度累积、桥接 Function 的保存与返回、实际 consumer |
| 参数、FSDP full parameter、通信 buffer、optimizer state | 按对应驻留模型处理 | 参数所有权、通信／FSDP marker 和状态 storage |
| 无法解释的保存值 | 保守保留设备驻留，记录并继续定位 | 生产、保存、使用、释放的完整证据 |

原生 `ChunkedLossWrapper` 会逐 chunk 执行 head、loss 和 backward，最后通过梯度桥接进入 decoder backward。其保存值可能很大，但生命周期很短。
`phase=backward` 或 `execution_kind=backward` 不意味着已经进入整个训练步的 B action：head 的交错反向仍可能属于 F action。需要同时核对 `comp_type`、stage、microbatch，以及 storage 的全部实际使用。

只有全部保存消费和使用都在同一已知 F action 内完成，才可判定为 chunk-local。若同一 storage 还由后续 B/recompute 读取，即使没有第二个保存槽，也不能误判为仅在 F 内使用。缺失锚点不能当作「已经使用完毕」。

## `unattributed` 的处理规则

传输列表的目标是没有未解释项，标准路径应为 0 个 `<unattributed>` 传输条目。但「列表为 0」不是消除归属问题的证据，不能通过过滤记录、清空模块路径或强行归入最后一层达到这个目标。

1. 优先检查保存槽自身的 `module_path`，再检查实际保存／消费／生产事件及显式 checkpoint 绑定。明确属于 head/norm/root 的保存，不借用滞后的 backward 层路径归入 decoder。
2. 已知的层外保存值不是未知类型。保留其实际来源和设备生命周期，不计入逐层卸载量。
3. 真正无法解释的值保持设备驻留，并记录 `storage_key`、shape/dtype/bytes、生产算子、保存模块、全部 consumer、释放和 stage/microbatch。不能只给一个比例就结束分析。
4. decoder 层内存在未解释的大对象、新增类型或不合理字节量时，必须定位或明确限制，不能据全套回归通过声明该模型已完成内存验收。不设一个百分比来代替来源分析。
5. 不能只隐藏 summary 中的 `<unattributed>`。若改变了 offload 选择，必须同步检查 `modeled_num_bytes`、`residency_policy`、checkpoint 记录和 alloc/free 时间线。

逐层候选归属见 `memory/activation_offload.py`，AC 预投影见 `memory/activation_checkpoint.py`，简化汇总见 `memory/records.py`。
尤其检查 AC 的先后顺序：先确认层或显式 checkpoint 绑定，再将驻留投影为 0。非层 wrapper 中即使有 recompute consumer，缺少边界且没有该 storage 的保存槽时，也不能绕过归属检查提前清零。

## 字节、生命周期与峰值

- 计量使用实际本地 storage 身份和原始字节；同层重复保存、view、alias 和不同 checkpoint role 不重复计量。区分 allocation generation，不能只依赖可复用的 Python 对象 ID。
- 传输量不能取卸载后为 0 的设备驻留字节。`mixed_precision_param` 通过实际保存 dtype 影响大小；权重存储投影不能改变激活传输量。FP32、BF16、整数和量化 code/scale 分别按实际 storage 计量。
- 不卸载的值必须继续参加设备峰值；卸载传输列表与显存预算不能两边都漏掉它。传输字段表示单向 payload，H2D/D2H 各按此值，不翻倍。
- 不把各 chunk 的累计保存量直接加到峰值。按真实分配／释放区间求同时活跃字节；多个短生命周期对象可以顺序复用显存。
- 最后一次读取、一次 unpack、Python 对象销毁和设备 storage 释放不是同一件事。使用实际所有权、保存槽释放及 FSDP 显式释放证据，不以「最后 consumer」替代全部生命周期语义。
- 暂时不将释放时间固化到某个 `OpNode`。当前捕获序号是原执行顺序的依据，不自动成为任意 DES 重调度下的释放时间。

权重、梯度、optimizer state、FSDP full/staging 和注册 buffer 应分别核查。图中的零耗时 `stream_wait` 仍是依赖约束，AG 未完成不能消费 full weight，归约完成前不能推进相应 clip／更新。遵循原生 wait/event 的实际约束，不将捕获序号直接补成全串行依赖。
真实设备 allocator、碎片和虚拟 kernel 内部不可观察的 workspace 不在当前逻辑 storage 预算中，必须保持边界说明。

## 新模型必须提供的验收证据

从真实 preset/config/override 入口捕获小模型 F/B/clip/optimizer，不仅检查构建成功。
至少覆盖以下可区分的行为：

- AC none/full/selective、offload 开／关；有 MTP 或 chunked loss 时进入真实路径。验证每类保存值的来源，而非只比较条目数量或未知比例。
- 层内候选正常计传输，loss/head/norm/梯度桥接驻留；检查真实非零 alloc/free 记录，而非只断言报表中没有 `<unattributed>`。
- 共享 storage 的多个保存槽、view、跨 F/B 使用和 B/recompute 普通引用；覆盖无保存槽、无 checkpoint 绑定的非层引用。
- 独立字节和峰值预期。例如共享 64-byte allocation 加独立 32 bytes 只形成 96-byte 单向 payload；128-byte 不卸载保存值必须形成实际 128-byte 驻留峰值，不能被 offload 过滤。
- 参数／优化器的本地分片先验、梯度 shape/dtype/placement、图引用闭合与 DAG；FSDP AG 到实际权重使用、RS/AR 到后续操作的完成依赖。扩大到实际改变执行路径的 TP/EP/CP/AC 组合和非零 rank。
- 量化 code/scale 与其实际 consumer、混合精度边界，以及文件导出和图内 summary 一致。汇总只维持六项核心预算加一个逐层 payload 列表，不恢复多套重复统计。

相关回归不是新模型的替代证明。需要增加进入新模型真实路径的薄测试，并沿用独立 oracle。当前参考入口：

| 验收对象 | 参考测试 |
| --- | --- |
| chunk-local 驻留、层外保存、共享与非层重算 | [test_interleaved_loss_offload.py](../tests/memory/test_interleaved_loss_offload.py) |
| storage/view/保存角色去重与文件输出 | [test_layer_offload_summary.py](../tests/memory/test_layer_offload_summary.py) |
| DSV4/MTP、Kimi、MXFP8 与 offload 投影隔离 | [test_memory_delivery.py](../tests/test_memory_delivery.py) |
| FSDP 的流控制与延后传播 | [test_fsdp_stream_capture.py](../tests/test_fsdp_stream_capture.py) |
| Kimi 并行分片、非线性前归约与独立字节 | [test_kimi_parallel.py](../tests/test_kimi_parallel.py) |

在已配置的 sidecar 开发环境中，从 `experiments/simulator` 运行针对这类问题的快速回归：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/memory/test_interleaved_loss_offload.py \
  tests/memory/test_layer_offload_summary.py
```

完整测试和类型检查入口见 [README](../README.md#验证)。验收报告记录源码／依赖版本、配置、实际执行命令、原始保存与生命周期证据、独立预期和未验证边界，不仅记录「通过多少项」。
历史 DSV4/Kimi 案例与修正结果见 [单卡内存结果](steady-state-memory.md#chunked-loss-的设备驻留修正)，对外字段见 [同进程接入指南](upstream-integration.md)。
