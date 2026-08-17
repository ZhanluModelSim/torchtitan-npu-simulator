# Kimi K3 simulator fused-op README

本文说明 Kimi K3 simulator 中三个核心融合算子的 IR 语义。每个前向算子
都有一个对应的反向算子，因此一次训练步在 L0 中会出现六类节点：

| 语义 | 前向 raw op name | 反向 raw op name | simulator 实现 |
| --- | --- | --- | --- |
| KDA core | `triton_ascend_kernels.chunk_kda` | `triton_ascend_kernels.chunk_kda_grad` | `kda_shim.py` |
| Gated MLA attention core | `fusion_attention` | `fusion_attention_grad` | `kimi_k3_fusion_shim.py` |
| SiTU-GLU activation | `situ_glu` | `situ_glu_backward` | `kimi_k3_fusion_shim.py` |

这些不是实际执行的 NPU/Triton kernel，也不是通过 `torch.library` 注册的
custom op。它们是 simulator 在 meta 执行时用 `torch.autograd.Function` 建立
autograd 连通性、再调用 `capture.record_synthetic_op()` 写入的 synthetic L0
OpNode。KDA 保留生产 Triton 算子名，便于与真实 profiler 对齐；MLA 和
SiTU-GLU 使用模型无关的语义名。

## 记号与 roofline 口径

以下记号用于全部估算：

| 记号 | 含义 |
| --- | --- |
| `B` | micro-batch 内 batch size |
| `S` | 当前 rank 的 sequence length |
| `H` | attention heads |
| `D` | 每 head 的 attention/KDA hidden dimension |
| `I` | MLP intermediate dimension |
| `T` | SiTU-GLU 的 token 数；共享专家通常为 `B*S`，路由专家为本 rank 的 routed-token 数 |
| `b` | 一个元素的字节数，例如 BF16 为 2、FP32 为 4 |
| `N` | `B*H*S*D` |

文中的访存量是理想化的 external-tensor 下界：每个输入只读一次、每个输出
只写一次，不计 cache miss、重复加载、临时 buffer、参数分片通信、重计算或
硬件布局转换。它适合用于设备无关的 arithmetic-intensity / roofline 比较，
不能替代真实硬件带宽测量。

## 1. KDA core

### Tensor contract

`KimiDeltaAttention` 在 q/k/v 投影、ShortConv、gate/beta 投影之后调用 KDA。
因此这些投影和卷积不属于本融合节点。

| Tensor | Shape | 维度语义 |
| --- | --- | --- |
| `q` | `[B, S, H, D]` | batch、token 位置、head、每-head query feature |
| `k` | `[B, S, H, D]` | batch、token 位置、head、每-head key feature |
| `v` | `[B, S, H, D]` | batch、token 位置、head、每-head value feature |
| `g` | `[B, S, H, D]` | batch、token、head、per-channel decay/gate feature |
| `beta` | `[B, S, H]` | batch、token、head 的 delta update rate |
| `A_log` | `[H]` | 每 head 的稳定化/decay 参数 |
| `dt_bias` | `[H*D]` | 展平的 head-channel bias；逻辑形状为 `[H, D]` |
| `o` | `[B, S, H, D]` | KDA 输出，和 `v` 同形状 |

反向节点输入为 `q/k/v/g/beta` 和 `dO`，其中 `dO` 为 `[B,S,H,D]`；它产生
`dQ/dK/dV/dG`（均 `[B,S,H,D]`）、`dBeta`（`[B,S,H]`）、`dA_log`（`[H]`）和
`dDt_bias`（`[H*D]`）。当前 capture 的反向 raw 输入列表只显式记录动态输入和
`dO`，但 autograd bridge 仍向 `A_log`、`dt_bias` 返回梯度。

### 融合边界与伪代码

KDA 是有状态、因果的 delta-attention recurrence。真实 Triton 实现使用
chunked 算法和内部状态分块；下列伪代码只描述融合边界，而非其精确数值实现：

```text
# Outside KDA: q/k/v projections, ShortConv, f_a/f_b gate projections,
#              beta projection and sigmoid.
state = initial_state
for t in causal_order(S):
    decay = decay_from(g[:, t], A_log, dt_bias)
    state = gated_delta_update(state, k[:, t], v[:, t], beta[:, t], decay)
    o[:, t] = query_state(q[:, t], state)
return o

# Backward: one fused reverse recurrence produces gradients for all inputs.
```

被融合的是因果 state update、decay/beta 处理、query-state 读取以及它们的
反向 recurrence。`q/k/v` 的线性投影、ShortConv、`g`/`beta` 投影、RMSNorm、
residual 和后续 MLP/MoE 都保持独立 OpNode。

### 粗粒度 roofline

KDA 的主要工作是每 token/head 对 `D x D` state 的读取、更新和查询。忽略
chunk 边界优化后，可使用下列设备无关近似：

| Pass | FLOPs | external tensor traffic lower bound |
| --- | --- | --- |
| forward | approximately `4*B*S*H*D^2 + O(B*S*H*D)` | `(5*N + B*S*H + H + H*D) * b` |
| backward | approximately `8*B*S*H*D^2 + O(B*S*H*D)` | `(9*N + 2*B*S*H + 2*H + 2*H*D) * b` |

这里 `4` 和 `8` 分别表示 state update/query 与其反向的数量级系数；真实
chunk-KDA 会因 chunk size、门控实现和状态重算而变化。这个近似的 forward
arithmetic intensity 为约 `4*B*S*H*D^2 / ((5*N+B*S*H+H+H*D)*b)`，大序列时
约为 `4D/(5b)`。

## 2. Gated MLA attention core

### Tensor contract

这个 synthetic op 只替换 `KimiGatedMLA._attention_core()` 中的 causal SDPA。
名称中的 `gated` 是架构名称；输出 gate 本身仍是融合节点外的
`sigmoid(g_proj(hidden_states))` 与逐元素乘法。

| Tensor | Shape | 维度语义 |
| --- | --- | --- |
| `Q` | `[B, H, S, Dq]` | batch、head、query token、query/key feature |
| `K` | `[B, H, S, Dq]` | batch、head、key token、query/key feature |
| `V` | `[B, H, S, Dq]` | batch、head、value token、padded value feature |
| `O` | `[B, H, S, Dq]` | batch、head、attention output token、padded value feature |

`fusion_attention` 和 `fusion_attention_grad` 均带有两个非 tensor kwargs，它们
以 `OpNode.attrs` 写入工作负载图：`num_heads=H` 和固定的
`layout="BNSD"`。其中 `N` 表示 head 维，因此该 layout 对应下表的
`[B,H,S,Dq]` 物理 tensor 排列。

`Dq = qk_nope_head_dim + qk_rope_head_dim`。如果 `v_head_dim < Dq`，模块会先
对 value pad 到 `Dq`，attention 后再 trim 回 `v_head_dim`；pad、trim、transpose
和 reshape 都不属于 `fusion_attention`。反向输入为 `Q/K/V/dO`，输出为
`dQ/dK/dV`，三者均为 `[B,H,S,Dq]`。

### 融合边界与伪代码

```text
# Outside fused op:
q = q_b_proj(rms_norm(q_a_proj(hidden_states)))
kv = kv_b_proj(rms_norm(kv_a_proj_with_mqa(hidden_states)))
Q, K, V = split_and_layout(q, kv)
V = pad_to_qk_head_dim(V)

# fusion_attention(num_heads=H, layout="BNSD"):
scores = causal_mask((Q @ transpose(K)) / sqrt(Dq))
probabilities = softmax(scores)
O = probabilities @ V

# Outside fused op:
O = trim_and_layout(O)
O = O * sigmoid(g_proj(hidden_states))
return o_proj(O)
```

因此 q_a/q_b、kv_a/kv_b 的 LoRA/MQA 投影，两个 RMSNorm，布局处理，输出 gate
和 `o_proj` 都会作为普通 MM/RMSNorm/elementwise OpNode 被捕获。融合节点只
代表 FlashAttention/MLA attention core，避免把整个 attention module 误建模为
单个大算子。

### 粗粒度 roofline

令 `Nq = B*H*S*Dq`。在自注意力 `Sq=Sk=S` 且不物化 score/probability 矩阵时：

| Pass | FLOPs | external tensor traffic lower bound |
| --- | --- | --- |
| forward | `4*B*H*S^2*Dq + O(B*H*S^2)` | `4*Nq*b` |
| backward | `8*B*H*S^2*Dq + O(B*H*S^2)` | `6*Nq*b` |

forward 的两个 GEMM 分别为 `QK^T` 和 `P@V`；backward 包含 dV、dP、dQ、dK
四个同数量级 GEMM。softmax 的标量工作以 `O(B*H*S^2)` 记入余项。对应 forward
arithmetic intensity 下界约为 `S/b`，说明长序列下该算子天然偏计算密集；实际
FlashAttention 的 tiling、causal 下三角和重算策略会改变常数，但不改变该量级。

## 3. SiTU-GLU activation

### Tensor contract

SiTU-GLU 的两个输入由两个独立投影/GMM 产生，投影本身和后续 down projection/GMM
不属于本融合节点。

| Tensor | Shape | 维度语义 |
| --- | --- | --- |
| `gate` | `[..., I]` | 任意 token 前缀维度、intermediate channel；展平后为 `[T,I]` |
| `up` | `[..., I]` | 与 `gate` 相同的 token/channel 布局 |
| `y` | `[..., I]` | 激活后的 intermediate tensor |
| `dY` | `[..., I]` | 来自 down projection/GMM 的梯度 |
| `dGate`, `dUp` | `[..., I]` | 返回给两条输入投影/GMM 的梯度 |

共享专家通常使用 `[B,S,I]`，路由专家通常使用 `[T_routed,I]`；两者共享完全相同
的逐元素语义。

### 融合边界与伪代码

```text
# Outside fused op:
gate = gate_proj(x)                 # or first grouped_mm
up   = up_proj(x)                   # or second grouped_mm

# situ_glu, beta=4 and linear_beta=25 by default:
activated_gate = beta * tanh(gate / beta) * sigmoid(gate)
bounded_up = linear_beta * tanh(up / linear_beta)
y = activated_gate * bounded_up

# Outside fused op:
out = down_proj(y)                  # or third grouped_mm
```

当 `linear_beta` 为 `None` 时，`bounded_up = up`。反向节点融合上述表达式的导数，
直接生成 `dGate` 和 `dUp`，不会在 L0 中拆成 `tanh`、`sigmoid` 和多个小乘法节点。

### 粗粒度 roofline

令 `M = T*I`。把 `tanh`、`sigmoid` 视为常数代价的逐元素函数：

| Pass | FLOPs / element | external tensor traffic lower bound |
| --- | --- | --- |
| forward | about 6 scalar arithmetic + 3 transcendental evaluations | `3*M*b` |
| backward | about 12 scalar arithmetic + 3 transcendental evaluations | `5*M*b` |

因此其 forward arithmetic intensity 约为 `(6 + 3*c_trans)/ (3*b)`，其中
`c_trans` 是将一次 transcendental 换算为普通 FLOP 的约定值。无论如何，它的
复杂度为 `O(T*I)`，通常比 attention 和 GMM 更偏带宽/指令开销受限；将其合成一个
节点的目的正是避免把这种单一逐元素 kernel 错误放大为大量设备小算子。

## Capture 与后端使用建议

- 后端读取 `self.workload_graph` 时可将上述六个 raw op 视为真实设备工作节点。
- `metadata_view` 已在 WorkloadGraph 构建前收缩，不会把 weight transpose/view
  等纯元数据操作传给后端。
- 反向节点保留与前向相同的 module path，并通过 autograd bridge 连到对应输入；
  不能将 `_backward` 当成独立、无依赖的 kernel。
- 本文 roofline 不包含重计算。启用 AC 后，应按实际 execution kind 将重算前向节点
  额外计入总 FLOPs/traffic。
