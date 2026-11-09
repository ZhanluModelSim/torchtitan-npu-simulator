# 模型接入常见问题与排查

本页保存从现有模型获得的可复用经验。先按[接入工作流](model-integration.md)确定变更和验收范围；此处用于解释常见错误的原因，不把某次历史测试结果当作新模型通过证据。

## 新 family 接入时运行期才暴露的约定

| 症状 | 根因 | 正确处理 |
| --- | --- | --- |
| 捕获完成后 `AttributeError: 'Config' object has no attribute 'n_layers'` | runner 汇总读取 `model_spec.model.n_layers`，上游/自建 Config 未必有该字段 | 在 Config 上提供 property 从 `layers` 派生（层切片 preset 不会漂移） |
| 自建 `build_attention_masks` 对 load_balancer_type 立即 raise | torchtitan `ParallelismConfig.context_parallel_load_balancer` 默认非 None，runner 原样透传 | factory 显式 `ParallelismConfig(context_parallel_load_balancer=None)`（Kimi/qwen3 均如此） |
| MoE 模型直接换 dense 模型后 `install_balanced_dispatch` 报"requires verified balanced routing" | 旧版对无 router 模型直接 raise | 现已对 dense 返回 `not-applicable-dense`；新 family 应在捕获测试中断言该能力标记 |
| 数值参考对照 attention 偏差巨大 | 上游 `FusedQKVLinear` 的融合权重**按 KV 组交织**（`r_dim = heads_per_kv + 2`，每组 [Q×hpk, K, V]），不是 [Wq;Wk;Wv] 拼接；state dict 经 hook 转回 wq/wk/wv | 参考实现按 `view(B, S, n_kv, r_dim, hd)` 切片，见 [test_qwen3_numerics.py](../tests/test_qwen3_numerics.py) |
| varlen 元数据传给 NPU 算子后行为异常 | Ascend `actual_seq_qlen/kvlen` 是**不含前导 0** 的累加和；`VarlenMetadata.cu_seq_q` 以 0 开头 | 传 `cu_seq_q[1:]`；约定来源是 torch_npu binding docstring，不确定时先读 docstring 再探测 |

## 构建与配置正常，实际路径却不对

| 症状 | 常见原因／应检查的位置 | 正确处理 |
| --- | --- | --- |
| 新 factory 能 import，但 simulator 拒绝 | `prepare_config()` 的 family 校验和默认 override 表仍只支持已有模型 | 完整接线后逐关验收，不仅移除校验 |
| 修改 dim 后出现 shape 冲突 | 只改顶层字段，嵌套投影、head、loss 词表仍为旧值 | 使用 builder 重建完整配置树；校验子配置及实际模块 |
| 设置量化或 override，却没有目标算子 | constructor 绕过配置构建，或新子类没被原生转换器保留 | 对照配置遍历 FQN 与实际构建模块／wrapper 清单 |
| 第二次 simulate 与第一次不同 | 共享配置、RoPE／mesh cache、临时 patch 或 FSDP policy 泄漏 | 同进程换配置／rank 连续运行和异常退出测试，恢复全局状态 |
| build-only 成功但 FSDP 失败 | 原始参数 dtype 不统一；build-only 未经历训练／分片 | 分开检查原始参数、计算、归约、量化和预算投影 dtype |
| 没有 memory JSON 就认为失败 | 默认已经改为内存对象交付 | 检查返回对象；只有显式 `mem` 才要求文件 |

当前 DSV4/Kimi 以 FP32 原始参数满足统一 dtype 约束，FSDP 可按 BF16 计算；这与「全 FP32 计算」不同。`memory_parameter_storage_dtype` 只影响常驻权重预算，不应改变激活或传输 dtype。不要无依据地给 KDA A_log/dt_bias 加一套特殊 FP32 计算 policy。

## 复用相似组件时改了模型数学

- SiTU 与 SwiGLU 不等价；量化转换后也必须保留实际激活路径。
- router score 在 W2 前吸收或 W2 后相乘会改变舍入和融合边界，不能只比较最终 shape。
- Kimi Attention Residual 使用 FP32 归一化及 folded score-weight 顺序，不能替换成普通 BF16 affine RMSNorm 后再投影。
- dense／shared／routed experts 的输入来源可能不同；latent MoE 的 router/shared branch 可消费原始 hidden state，而专家消费降维后的张量。
- 相同的 `Module.Config` 名称不证明相同初始化、epsilon、冻结参数或 checkpoint 方向。

对差异组件用同权重和同输入的独立 CPU 参考验证数学；Meta 图只验证执行结构和梯度元数据。checkpoint 对照应覆盖非方阵、专家顺序与真实 key，不仅做同一实现的导出／导入 roundtrip。HF adapter 通过不表示旧 native checkpoint 可直接恢复。

## 并行图可连通，但依赖或分片错误

**FSDP。** dense 与 expert mesh 不同，`dp_mesh_dims` 必须匹配实际 SPMD 参数。单卡／rank 0 通过不能证明非均匀 shard 或尾 rank 正确；某些 singleton 参数在部分 rank 是空 shard。权重预算使用本地 storage，不把逻辑全局 shape 当作本地分配。

**完成依赖。** AG 发起不等于权重可用，RS/AR 发起不等于归约完成。保留 stream wait/event 及 producer 路径，用延后通信的测试证明消费者随之延后；不要按捕获序号补全串行边。FSDP 通信视图隐藏准备算子是既定边界，不应为省事切换到完整源图做下游性能统计。

**TP。** partial sum 不能直接进入非线性 norm/gate。Kimi latent MoE 的专家归约必须在 norm 和 up projection 前完成；把归约挪到后面即使 shape 一致也错误。

**CP。** KDA 因果卷积与递归计算依赖完整前缀，不能将序列切段后独立执行。若采用 gather 全上下文再 local slice，必须计入通信，并用独立整序列参考校验输出、输入梯度及累计参数梯度；不要因额外 gather 的 backward 重复放大 loss。

**EP。** count 交换、payload 交换、专家计算、combine 的组和依赖要一致；保持公共均衡 dispatcher 路径，不为新模型复制专用 A2A。均衡路由是当前建模假设，不能称为真实数据相关负载均衡。

## 自定义融合只保存了 shape，或反向缺依赖

Meta 输出可以新建；autograd 应保存的前向输入不能用同 shape 新 tensor 替代。检查真实保存对象／storage、实际 dy 和每个输出梯度，并沿 producer 祖先追溯。缺参数梯度时先判断数学上是否参与路径；不要合成梯度或冻结所有失败参数来通过检查。

新自定义边界必须有 [`tensor_semantics`](custom-tensor-semantics.md)。例如 Kimi attention 为 B/H/S/D，KDA 为 B/S/H/D，不能通用地把第二维当作序列。MLA 的 V 已 padding，边界 shape 是填充后的宽度。SiTU 前轴未知就用 `None`，不能凭 rank 猜 batch/token。可选输入缺席与 shape 为 0 的真实 tensor 不同，后者仍占索引；同一 tensor 多次出现也不能按身份去重输入。

原生 CANN 可用时不要手工再插一个融合节点；兼容层只补已确认的 Meta/autograd 缺口，并保持原始 raw overload。真实设备 provider 与纯 Meta 虚拟实现分开说明，不能因捕获通过宣称 kernel 数值可用。

## 首轮与稳态、chunked loss 和阶段边界

`simulate()` 已默认结构预热。首个 optimizer 更新常有惰性 state 初始化；稳定 O 图不再出现这些创建算子，但状态仍驻留，不能少算优化器显存或卸载量。每轮计数 reset、fill 等仍有意义，禁止按名称删除。底层 `run_simulation_step()` 不自动执行预热，测试不能混淆两种入口。

Chunked loss 在 F action 内执行多次 head/loss 及其反向，再通过梯度桥接进入主干 B。因此 `phase="backward"` 不能单独决定归属 B；同时看 `comp_type`、真实生产／使用及层边界。F/B/O 输出应沿既有 step boundary 机制组织，不另按 raw 名称分桶。

曾出现 head 的临时 reshard 策略未恢复，导致后续迭代的通信模板改变。当前作用域恢复逻辑见[chunked loss policy](chunked-loss-policy.md)；新 wrapper 或 loss 变更仍须验证策略在正常和异常路径均恢复。不能将实现泄漏直接当作 GA 本来就需要不同通信模板。

## 显存与卸载：不要靠删报表消除异常

autograd 保存槽不等于激活卸载候选。先追踪保存、全部使用和实际释放，再决定层归属：

| 来源 | 应有处理 |
| --- | --- |
| 明确层／checkpoint 保存，供 B 或重算读取的激活 | 按完整保存与 consumer 事实选为候选 |
| head／norm／Attention Residual 等已知层外保存 | 保留实际来源与设备驻留，不借用上一层 FQN |
| chunked loss 当场消费的 logits／log-prob | 按真实短生命周期驻留，不累计成整步峰值 |
| head→decoder 梯度桥接值 | 保留到实际使用，不因 shape 类似 hidden state 就当普通层激活卸载 |
| parameter、FSDP full/staging、optimizer state | 按各自所有权与驻留机制处理，不混入激活清单 |
| 无法解释的保存值 | 保守保留，并调查生产／保存／消费／释放证据；不能隐藏记录或强塞最后一层 |

只有 storage 的全部使用都在已知 F action 内完成，才能判为 chunk-local。后续 B/recompute 的普通引用也算使用，即使没有第二个保存槽。AC 投影先确认层或 checkpoint 绑定，再决定驻留；缺少绑定的非层值不能仅因发生重算就清零。

内存计量须同时守住：

- view、重复保存和多个 wrapper 按真实 storage／allocation generation 去重；共享 64-byte allocation 加独立 32 bytes 是 96 bytes，不是重复相加。
- 一次 unpack、最后算子读取、Python 对象销毁与 storage 释放不是同一时刻。保留所有权、保存槽释放及 FSDP 显式释放证据。
- 峰值是同时活跃 storage 的和，不是保存量累计；短生命周期 chunk 不能全部堆进峰值。
- 激活 payload 使用卸载前的实际本地字节，不能取投影后为 0 的驻留量。H2D/D2H 各使用单向 payload，不先翻倍。
- 优化器卸载独立修正 F/B/O 驻留，保留原始状态总量；与激活卸载一起测试，不能互相覆盖修正。
- 设备预算排除 CPU 控制 metadata；allocator 碎片、reserved/cache 和不可观察的融合 workspace 仍不在当前静态模型内。

出现异常时记录 storage key、shape/dtype/bytes、producer、保存模块、全部 consumer、释放位置与阶段。decoder 内新增的大块未解释对象必须定位或明确限制，不能设一个「未知比例阈值」代替验收。原始完整事件保留用于审计，正式峰值和逐层 payload 口径见[稳态内存](steady-state-memory.md)。
