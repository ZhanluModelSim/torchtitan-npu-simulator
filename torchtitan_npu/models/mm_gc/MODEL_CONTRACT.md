# mm_gc 模型契约（SLA2 + Multi-Head MoE）

本文记录 mm_gc 的模型契约与当前接入状态，遵循
`docs/feature_guides/new_model_onboarding.md` 的模型契约要求。验收目标与判定标准见
`docs/test_guides/model_acceptance.md`。

## 1. 来源

| 项 | 内容 |
| --- | --- |
| 架构 sketch | `torchtitan_npu/simulator/raw_model/mm_gc/multi-head_moe.py`（放大版规格，以此为准，不对齐 MAGI-2 真实 114B 规格） |
| SLA2 attention | 用户提供的 Sparse Linear Attention 参考实现（块级 router + 稀疏/线性双分支 + per-block alpha），落地于 `core.py` |
| MoE 部署/通信语义 | MAGI-2-preview reference（`raw_model/mm_gc/reference/MAGI-2-preview`）：仅参考 CoreMultiHeadMoE 的路由语义（sigmoid + expert_bias + route_norm + route_scale）、CSR sort + grouped expert 执行、head-dim EP dispatch；其余（mHC、attention sinks、多模态 adapter、真实 114B 规格）明确排除 |
| 外部权重 | 暂无公开权重；native state-dict schema 为唯一 checkpoint 事实来源 |
| tokenizer/测试资产 | `tests/assets/tokenizer/deepseekv3_tokenizer` + `c4_test`（占位，仅用于训练链路连通） |

## 2. 网络结构（full / 放大版规格）

| 项 | 值 |
| --- | --- |
| layers | 96；前 2 层与后 2 层 dense FFN，中间 92 层 Multi-Head MoE |
| dim / n_heads / head_dim | 12288 / 96 / 128（MHA 同宽 QKV，Q/K 各过 per-head RMSNorm 后 RoPE） |
| vocab_size | 102400（embedding 与 output 不 tied） |
| attention | SLA2：双向（无因果 mask）、块粒度 top-k 稀疏 softmax 分支 + 全局 softmax feature map 线性分支，按 query block 可学 alpha 融合 |
| SLA2 超参 | blkq=blkk=64；topk 比例按 seq 分档（≤1M：0.5%，>1M：0.1%），绝对块数 floor=1；训练 stage=1（router 可学、soft top-k），stage=2（frozen router，从 `router_data_path/block<i>/block_<i>_2.pt` 加载，缺失即 fail fast）；compute dtype bf16 |
| MoE | S=32 heads × hs=384，Es=1024 experts/head，每 head top-6；expert FFN 384→1536→384 SwiGLU；shared expert 12288→6144→12288；`proj_in` dim→S×hs，`proj_out` S×hs→dim |
| 路由 | 每 head 独立 router `[S, hs, Es]`；sigmoid 打分；`expert_bias`（非持久 buffer）只作用于 top-k 选择；probs 从原始 score gather 后 L1 `route_norm`，再乘 `route_scale` |
| Norm/激活 | pre-RMSNorm 残差；RMSNorm eps=1e-6；专家/FFN 均 SwiGLU |
| RoPE | theta=10000，freqs_cis 每次 forward 现算（非常驻参数；`torch.polar` 已验证 meta 可用） |

## 3. 参数量公式（独立基线）

```text
embedding + output        = 2 × V × D
每层公共                  = 4×D×D (qkv+out) + 2×Dh (qk norm) + 2×(Dh²+Dh) (sla router) + ceil(L/blkq) (alpha) + 2×D (layer norms)
dense FFN 层              = 3 × D × I_dense
MoE 层                    = 2×D×(S×hs) (proj_in/out) + S×hs×Es (router) + 3×(S×Es)×hs×(hs×mult) (routed) + 3×D×I_shared (shared)
```

| flavor | 参数量 | 备注 |
| --- | --- | --- |
| debug | 11,165,708 | 6 层，与 `sum(p.numel())` 逐字节一致（已验证） |
| reduced | 42.69B | 16 层 / Es=32 |
| full | 5.447T | 96 层 / Es=1024（放大版 sketch 规格） |

`expert_bias` 是 non-persistent buffer，不计入参数量；不 tied weight，embedding/output 分别计数。

## 4. 权重布局与 state-dict 约定

- routed experts 采用 torchtitan 统一 `[E, out, in]` 布局：`experts.w1/w2/w3`
  形状 `[S×Es, ·, ·]`，全局 expert id = `head × Es + local_e`（head-major flatten）。
  converter/state-dict/并行切分/算子建模必须共用该布局。
- SLA router：`core.proj_q/proj_k`（fp32 Linear，含 bias）、`core.alpha [ceil(L/64), 1]`。
- `expert_bias` / `expert_bias_ema` 类负载统计不进 state dict（persistent=False）。
- state-dict 回路（save → reload）已验证一致；HF 转换待有权重映射时补充。

## 5. 支持范围

| 特性 | 状态 |
| --- | --- |
| 单卡 eager（CPU/NPU） | 已验证（debug：前向/反向/loss/grad/state-dict 回路） |
| meta 构造与前向 | 已验证（debug/reduced/full 全部可跑；alpha 改用 `torch.empty`+`fill_` 构造以规避模拟器 `torch.full` 包装与 device context 的冲突） |
| FSDP（dp_shard）+ AC（none/selective/full） | 已接：embedding、每 block、[norm, output]、根模型各自成组（与 llama3 布局一致，根组必须有） |
| DP replicate | 已接 |
| TP | 已接：attention 按 head 维切分（QKV colwise / out rowwise + all-reduce）；SLA2 全部算子按 head 独立，router/alpha 复制；MoE 头切分见 EP 行 |
| EP（head parallel） | 已接：`proj_in` colwise / router+experts 按 head-major 行切分 / `proj_out` rowwise + all-reduce；每层通信 = 1 次 all-reduce ∝ N·D（head-parallel 契约：与 top-k 无关）；`moe_num_heads % ep == 0` 校验 |
| EP+TP（ETP=TP） | 已接：MoE 头切分走 ["ep","etp"] 组合 mesh（扁平 head 切分），每层 2 级 all-reduce（层级化，对应 Head Parallel 的跨节点/节点内两级通信）；attention 走 tp mesh |
| CP | 已接：attention 输入按 seq 维 all-gather（保守全序列策略，SLA2 全局块路由与线性分支可见全序列），输出切回本地 chunk；`seq_len % cp == 0` 校验 |
| PP | **未实现，fail fast**（onboarding 顺序最后接入） |
| ETP（独立专家内切分） | **未声明，fail fast**（仅作为 EP+TP 组合中 MoE 头切分的 TP 轴） |
| torch.compile | 未支持，fail fast |
| MXFP8 量化 | 已接（`MMGcMXFP8Converter`，`torchtitan_npu/converters/kernels/mm_gc_mxfp8.py`）：经 module-path FQN 控制覆盖**全部 MM**（attention QKV/输出投影、dense FFN、MoE proj_in/proj_out、shared expert）与 **routed experts 的 grouped matmul**（3D `[E,out,in]` 权重 → `npu_grouped_matmul` MX 路径）；SLA2 块路由（`attention.core.proj_*`）与 MoE 头路由（`moe.gate.router`）**强制 fp32 排除**，FQN 命中即 fail fast；FQN 未命中任何模块 fail fast；硬件门槛：A5（模拟器 meta 自动放行） |
| npu_rms_norm / npu_rope converter | 配置中未启用（SLA 的 RoPE 是自定义路径，接入需单独评估） |
| 模拟器 shape-only 融合算子 | 已接：`triton_ascend_kernels.sla2_block_route_topk`（stage1/2 按输出个数区分）、`sla2_sparse_attn`（stage1/2 按 selection dtype 区分）、`sla2_linear_attn`、`mh_moe_route_topk`（含 `*_grad` 反向记录）、`aten._grouped_mm`（专家 GMM，原生捕获）；alpha 融合保持 eager（生产同为小逐元素算子）。规划中的生产 kernel 名已登记于本契约，下游 cost-model registry 接入前为约定名。**算子清单/输入契约/实现详见 `OP_REFERENCE.md`** |

## 6. 已知限制与风险

1. `alpha` 尺寸在构造期由 `seq_len` 固定；运行时 seq 与构造 seq 不一致会显式报错。
   变序列训练（含 CP）需要先扩展该契约。
2. SLA topk 绝对块数 = `max(1, int(Nb × ratio))`（floor 1），短序列下 ratio 语义退化，
   算子账本须按该公式推导。
3. stage-1 soft top-k 的温度平移 t 由二分搜索得到（前向数据依赖、不参与梯度），
   训练/推理路径行为不同；模拟器建模时按 execution kind 记录。
4. routed expert 的 NPU 路径使用 `torch._grouped_mm`（CPU 为逐专家 loop），
   真机行为待 NPU 验证；GMM converter 接入时对齐。
5. `precompute_freqs_cis` 每次 forward 在 CPU 现算后搬运，正式规格需改为按
   `buffer_device` 注册的 buffer（模拟器显存建模项）。
6. 无 SLA/MoE 模型专用融合算子的真实 ACLNN kernel；模拟器算子名属约定生产名，
   需在算子账本中声明。

## 7. 已完成的验证（截至本契约）

- debug 单卡 CPU：参数量与公式一致；前向/反向/loss 有限值；关键参数梯度齐全；
  state-dict 保存/加载回路一致；层类型分布（4 dense / 2 MoE）符合规则。
- debug/reduced/full meta 构造与前向 shape 正确（logits `[B, S, V]`）。
- 模拟器配置矩阵（全部单步跑通，`--simulation.output_formats mem`）：

| 配置 | world | 并行 | 通信事件（memory_events.csv） |
| --- | --- | --- | --- |
| `mm_gc_smoketest` | 1 | - | 无 |
| `mm_gc_smoketest_fsdp2` | 2 | dp_shard=2 | allgather + reduce_scatter（FSDP） |
| `mm_gc_smoketest_tp2` | 2 | tp=2 | allreduce（attention/MoE head 轴） |
| `mm_gc_smoketest_ep2` | 2 | dp_shard=2, ep=2 | allreduce（EP）+ FSDP allgather/RS |
| `mm_gc_smoketest_cp2` | 2 | dp_shard=2, cp=2 | allgather（CP+FSDP）+ RS |
| `mm_gc_smoketest_tp2ep2` | 4 | tp=2, ep=2, etp=2 | attention allreduce + MoE 两级 allreduce + FSDP allgather/RS |
| `mm_gc_smoketest_fsdp2ep2` | 4 | dp_shard=4, ep=2 | FSDP + EP 组合 |

- 融合算子捕获核对（`mm_gc_smoketest`，selective AC，6 层 debug）：
  `sla2_block_route_topk`/`sla2_sparse_attn`/`sla2_linear_attn` 各 12 次
  （6 层 × original+recompute），`*_grad` 各 6 次；`mh_moe_route_topk` 4 次
  （2 MoE 层 × 2），grad 2 次；`aten._grouped_mm` 24 次（2 层 × 3 GMM × 4）。
- MXFP8 量化核对（`mm_gc_smoketest_mxfp8`）：`npu_dynamic_mx_quant` 404、
  `npu_quant_matmul` 184（覆盖全部受控 MM）、`npu_grouped_matmul` 24 +
  `npu_grouped_dynamic_mx_quant` 12（专家 grouped MM）；路由器权重保持原生
  fp32 tensor（converter 内建守卫）；`mm_gc_smoketest_mxfp8` + CLI
  EP=2 组合验证量化算子与 EP all-reduce 共存。配置入口：
  `mm_gc_smoketest_mxfp8` / `mm_gc_reduced_mxfp8` / `mm_gc_baseline_mxfp8`
  （模拟器同名）。
