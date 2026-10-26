# Kimi K3 接入与验证

Kimi K3 文本 decoder 已迁移到 TorchTitan v0.3.0 的配置、Module、MoE 和并行接口。
入口仍为 `simulate(SimulationConfig(...))`，返回完整 `WorkloadGraph`，成功后默认导出内存汇总，`mem` 选项增加明细。
本页的验证范围是 Meta 图、CPU 数学契约和逻辑活跃内存；不包含真实 NPU 训练数值与 kernel workspace。

## 配置与调用

运行前准备 [接入指南](upstream-integration.md#1-接入方式与运行前提) 所列环境，使用本次源码中的主包和 sidecar。
仅升级 sidecar、仍加载旧 `torchtitan_npu` 安装包不能取得新模型。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m torchtitan_simulator \
  --config kimi_k3_smoketest \
  --training.seq-len 16 \
  --training.mixed-precision-param bfloat16 \
  --simulation.output-dir /tmp/kimi-k3-smoke
```

成功仿真默认导出 `memory/memory_summary.json`；添加 `--simulation.output-formats mem` 可额外导出内存明细。

| Preset | 几何与用途 |
| --- | --- |
| `kimi_k3_smoketest` | 4 层、dim256、8 专家；3 KDA + 1 MLA，包含 dense、routed/shared experts、Attention Residual |
| `kimi_k3_16layer_reduced` | 16 层缩小配置，用于较长模块链路检查 |
| `kimi_k3_full` | 原始 93 层、dim7168、896/top-16 专家，69 KDA + 24 MLA；2,779,484,393,568 个参数 |

Kimi 没有 MTP 模块，`num_mtp_modules` 必须为 0。当前 flat `model_overrides` 仍限定 DeepSeek-V4；
自定义 Kimi 几何使用 `module.factory` 返回原生配置树。
PP、compile、非 `spmd_types` 后端和非连续 CP load balancer 不在当前支持范围。

默认计算采用 `mixed_precision_param=bfloat16`，可选择 `float32`；梯度归约为 `float32`。
`training_dtype=float32` 保留原始参数，FSDP 按计算精度转换全参数，包括 KDA 的 `A_log`、`dt_bias`，没有特殊 FP32 计算策略。
base `bfloat16` 与显式 FP32 的 KDA 原始参数混合，无法满足当前原生 FSDP 的统一原始 dtype 约束，因此完整捕获提前拒绝；`--build-only` 可检查构建。
`memory_parameter_storage_dtype` 仅投影常驻权重字节，不改变计算与通信精度。

MXFP8 使用公共转换器，额外传入：

```bash
--extension.quantization.enable-quantized-training \
--extension.quantization.recipe all_mxfp8
```

Kimi 的选择范围延续旧版：routed experts 与 shared experts 的三个投影；attention 和首层 dense 不量化。
转换器保留 SiTU 子类，未将其替换成 SwiGLU。DeepSeek 专属的 LI、KV norm 和 sparse attention 量化选项不适用。

## 组件与融合边界

| 部分 | 当前实现与检查 |
| --- | --- |
| Linear、Conv1d、Embedding、普通 RMSNorm | 原生可遍历配置；普通 RMSNorm 使用公共 NPU override；初始化显式 normal std=0.02，norm 权重为 1 |
| MoE 路由与通信 | 公共 TokenChoiceTopKRouter、RoutedExperts、NPU dispatcher；模拟均衡路由；router 读取原始隐藏态 |
| LatentMoE | 压缩后进入专家；加权合并后 latent norm/up；共享专家读取原始隐藏态；TP 专家部分和在非线性 norm 前完成归约 |
| 专家投影与激活 | 与 V4 一样，gate/up 为两次 GMM，down 为第三次；保留不同的 SiTU 数学与融合边界；关闭 pre-W2 router score 吸收 |
| KDA | q/k/v 因果卷积、gate/beta 投影和输出 gate/norm 在核心外；beta sigmoid 延续旧边界，位于核心外 |
| MLA | 官方 no-RoPE 路径；两个低秩 norm epsilon=1e-6；保留 V pad/trim 和输出 sigmoid gate |
| Attention Residual | 按官方 FP32 归一化和 folded score-weight 顺序；普通 BF16 affine RMSNorm 后再投影不等价，因此保留专用组件 |

官方结构依据固定为 [moonshotai/Kimi-K3 f831ab6](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06)。
首层 self-attention residual 路径恒不调用，参数 key 保留并冻结；权重预算包括它们，优化器状态不包括它们。

KDA、MLA attention core、SiTU 使用显式作用域内的 dispatcher custom op、Meta kernel 和 autograd。
保存真实输入并由原生 dispatch 记录前反向，不另行手工插入算子节点。退出运行作用域恢复调用点；主包导入不激活 simulator。

| 原融合名 | 当前 raw forward / backward | canonical `op_type` |
| --- | --- | --- |
| KDA | `kimi_sim.chunk_kda.default` / `kimi_sim.chunk_kda_grad.default` | `chunk_kda` |
| MLA | `kimi_sim.fusion_attention.default` / `kimi_sim.fusion_attention_grad.default` | `fusion_attention` |
| SiTU | `kimi_sim.situ_glu.default` / `kimi_sim.situ_glu_backward.default` | `situ_glu` |

依赖 raw 名称的上游匹配需更新；前反向通过原始名称和 phase 区分。
KDA 成本保持 unknown，其他已有成本公式也不能解释为实测设备耗时。
模型层不导入 Triton 或 NPU 扩展；非模拟调用需显式向 `KimiDeltaAttention.Config.kernel` 注入兼容 provider，默认缺少 provider 时明确报错。
本次没有交付或验证真实 KDA 硬件 provider。

HF checkpoint adapter 已适配原生 dense/expert/router 命名、专家堆叠与低秩 output gate。
旧原生 checkpoint 名称发生变化，尚无旧 native key 的自动兼容转换，不能直接承诺恢复；HF 映射有 CPU roundtrip 检查，真实设备加载未验证。

## 并行、依赖与内存验收

CP 使用连续序列区间，先完成 TP/SP 收集，再经原生 autograd collective 收集 CP 完整上下文。
完整序列用于 KDA 因果卷积与注意力，完成序列耦合后普通切片回本 rank 区间，再进入输出投影。
两 rank Gloo 的 MLA 输出、输入梯度和累加参数梯度与整序列独立参考比较；EP 另有真实 Gloo all-to-all 数学对照。
这不是 HCCL 硬件验证。

| 检查 | 证据与范围 |
| --- | --- |
| 图与梯度 | `test_kimi_capture.py`：六组 DP/FSDP/HSDP/AC，110 个训练参数梯度；三个虚拟核心的保存输入身份、实际 dy、producer 祖先依赖 |
| 并行分片 | `test_kimi_parallel.py`：TP、EP、CP、交叉组合、首末 rank、full/selective AC、FSDP 度数大于专家数；逐参数独立 shape 与权重/state 字节断言 |
| 量化 | `test_kimi_quantization.py`：四组 MXFP8/EP/TP/CP/AC；精确的 18 个量化参数、SiTU 前反向、code/scale 与完整梯度 |
| FSDP 依赖 | `test_fsdp_stream_capture.py`：AG 对应的 full-weight storage 到达投影；延后 AG 会延后计算；HSDP AR 延后通过 wait/event 传播到后续计算 |
| 数学与状态 | `tests/unit_tests/models/kimi_k3/`：SiTU、LatentMoE、Attention Residual、MLA 输出和梯度、CP/EP Gloo、HF state 映射、KDA provider 参数 |

mem 汇总保留 [六项核心预算](steady-state-memory.md)，其中四个峰值均包含稳态优化器状态；另有逐层 offload payload 列表，未开启 offload 时为 0。
完整 93 层、B=1/S=512、world2048、TP4/CP2/EP64 的当前源码复验如下，单位为 GiB：

| 单 rank 指标 | rank0，BF16，AC none | rank2047，MXFP8，AC selective |
| --- | ---: | ---: |
| 稳态总峰值 | 36.759 | 26.610 |
| 前向峰值（含状态） | 34.580 | 20.941 |
| 反向峰值（含状态） | 36.759 | 26.610 |
| 优化器阶段峰值（含状态） | 20.387 | 20.340 |
| 权重大小 | 5.097 | 5.081 |
| 优化器状态大小 | 10.193 | 10.162 |

两组均有 2,642 个训练参数梯度；总参数公式为 2,779,484,393,568，逐参数本地 shape、权重/state 字节和独立生命周期 sweep 均吻合。
捕获加校验分别约 42.2 秒、58.2 秒；该时间来自当前无 NPU 的容器，不是设备训练时间。
两列同时改变 rank、精度和 AC，不能据此单独归因量化或 AC 的收益。
精确字节、图验证和源码 SHA256 见 [本轮结果](kimi-k3-results.json)。

最终全套回归 521 项通过；随后 14 项专项包含 4 个新增测试，合计 525 个不同 testcase，收集检查通过。
通过主仓正常 pytest 入口执行 Kimi CPU 测试与公共量化转换器测试，41 项通过；这 41 项与前面的计数重叠，不累计为新增覆盖。
模型专项和 sidecar 类型检查为 0 errors。全仓 pre-commit 的常规检查通过，整体因主机缺少训练依赖而未通过；
配套容器的主包基线与当前版本同为 357 个既有类型错误，错误消息与文件多重集合对比新增为 0，不能将全仓 Lint 写成通过。
上面的轻量 CLI 已按原命令执行并产生六项 mem 汇总。

扩大执行主仓 `pytest tests/ -x -q` 可收集 1,557 项，但运行在未修改的 V4.1 launcher-contract 测试处停滞；
诊断栈落在真实 CANN `sparse_flash_mla_metadata` 的 CPU fallback，已停止该次运行。
对应三条测试单独运行通过；主仓全套未完成，不计为通过，也未据此修改 V4.1 或 CANN 安装源码。

逻辑内存包含可观察 storage、保存值、FSDP 全参数/staging 和原生 AdamW 状态；
不包含 allocator/cache/碎片及虚拟融合 kernel 内部不可观察 workspace、内部保存状态。
当前只捕获单 microbatch；完整多 microbatch 累积和真实 NPU 数值不在本次验收范围。
