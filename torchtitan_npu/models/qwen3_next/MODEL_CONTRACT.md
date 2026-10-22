# Qwen3-Next 模型契约（MODEL_CONTRACT）

> 依据《新模型接入开发流程》§1 编写。来源、结构、参数量公式（独立于代码手工
> 推导）、支持矩阵、已知风险与验证记录。参数量公式由
> `tests/unit_tests/models/test_qwen3_next.py::test_2layer_param_count_matches_independent_formula`
> 以**精确相等**断言守护。

## 1. 来源

| 项 | 内容 |
| --- | --- |
| 参考实现 | MindSpeed-LLM fsdp2 分支 `mindspeed_llm/fsdp2/models/qwen3_next/`（实测对齐基线树 commit `6082e0de`） |
| 结构来源 | HF Qwen3-Next-80B-A3B（hybrid：3/4 线性注意力 GatedDeltaNet + 1/4 门控 GQA） |
| 部署参考 | MindSpeed fsdp2 实测口径：`pretrain_qwen3_next_16k_fsdp2_A3.yaml`（FSDP=8, EP=2, eFSDP=4, recompute off, reduce bf16） |
| 算子流口径 | 与实测 fsdp2 运行逐算子对齐（含 6 轮管理方反馈闭环），见 `docs/user-guides/simulator/qwen3-next-op-stream-alignment.md` |
| 权重格式 | v1 仅框架原生 FQN（见 §4 state-dict）；HF 权重转换显式不支持 |

## 2. 结构与参数量公式

### 结构（80B-A3B）

```text
dim=2048, vocab=151936, n_layers=48
hybrid 布局: layer i 为 full_attention 当且仅当 i % 4 == 3（36 GDN + 12 GQA）
GQA: n_heads=16, n_kv_heads=2, head_dim=256; q 与 gate 融合投影 (out=2*n_heads*head_dim)
GDN: num_k_heads=16, num_v_heads=32, head_k_dim=head_v_dim=128, conv_kernel=4
MoE: 全部层为 MoE (mlp_only_layers=[]), 512 experts, top-10, norm_topk,
     moe_inter=512, shared_expert_inter=512, shared_expert_gate=sigmoid
```

### 参数量公式（独立推导，逐层）

```text
embed   = vocab × dim
output  = dim × vocab                       (lm_head 无 bias)
norm    = dim
attn层  = dim·(2·n_heads·head_dim)          # 融合 q|gate
        + 2·dim·(n_kv_heads·head_dim)       # k, v
        + (n_heads·head_dim)·dim            # o
        + 2·head_dim                        # q/k norm
gdn层   = dim·(2·key_dim + 2·value_dim)     # in_proj_qkvz (key=16·128, value=32·128)
        + dim·(2·num_v_heads)               # in_proj_ba
        + (2·key_dim+value_dim)·4           # depthwise conv1d (groups=conv_dim, 无 bias)
        + 2·num_v_heads                     # dt_bias + A_log
        + head_v_dim                        # 门控 RMSNorm
        + value_dim·dim                     # out_proj
moe层   = dim·num_experts                   # router gate (fp32 语义, 不量化)
        + experts·(dim·2·moe_inter + moe_inter·dim)   # 融合 gate_up + down
        + 3·dim·moe_inter                   # shared expert gate/up/down
        + dim                               # shared_expert_gate
每层    = mixer + 2·norm + moe

总参数  = embed + output + norm + Σ层
```

| flavor | 公式值 (params) | 状态 |
| --- | --- | --- |
| 80B-A3B-2L（1 GQA + 1 GDN） | 3,912,940,224 | 单测精确相等 ✓ |
| 80B-A3B（48 层） | 79,674,391,296 | meta 构造实测 ✓ |
| debugmodel（4 层：3 GDN + 1 GQA） | 小规模 | 单测结构断言 ✓ |

## 3. 支持矩阵

| 维度 | 状态 | 说明 |
| --- | --- | --- |
| FSDP / eFSDP | ✅ | 镜像 MindSpeed `fsdp_modules` 计划；expert 单元 eFSDP（ep>1），param bf16 / reduce fp32 |
| EP | ✅ | MindSpeed eager dispatcher 镜像（count AG + a2a dispatch/combine + 逐专家 matmul） |
| TP + 序列并行 | ✅ | llama3 风格计划；fused qkvz / q+gate 按 head 连续块均匀 colwise（要求 `num_k_heads % tp == 0`）；norms SequenceParallel；conv1d/dt_bias/A_log 手工切；见 §5 偏差 1 |
| TP+EP 组合 | ✅ | experts 跨 TP 复制、按 EP 切 |
| DP replicate | ✅ | 上游 `apply_replicate` |
| AC full / selective | ✅ | `apply_moe_ac`；AC none/full 双覆盖已验证 |
| MXFP8 量化 | ✅ | `MXFP8Converter`（mxfp8_rceil）；FQN 控制见 §6 |
| ETP | ❌ fail fast | v1 支持矩阵外（experts 跨 TP 复制） |
| CP | ❌ fail fast | 未实现 |
| PP | ❌ fail fast | 未实现 |
| compile | ⚠️ 未验证 | 模拟器配置强制关闭并告警 |
| offload | ❌ | 未声明 |

非法配置 fail fast：head 数与 TP 不整除（ValueError）、ETP/CP/PP（NotImplementedError
并写明原因）、mxfp8 FQN 命中 router（ValueError）。

## 4. state-dict

v1 检查点 schema = 框架原生 FQN（`Qwen3NextStateDictAdapter` 恒等直通，DCP
save/load 闭环）；HF 导入/导出显式 `NotImplementedError`（无静默错位风险）。
`w1/w3→gate_up` 类合并权重不存在（experts 为融合 `gate_up_proj` 原生参数，
与 MindSpeed 布局一致：`(num_experts, dim, 2·inter)` + `(num_experts, inter, dim)`）。

## 5. 已知偏差与风险

1. **TP 下 lm_head 复制**：MindSpeed 实测为 tp=1，无 vocab-parallel 头的
   tp>1 证据；chunked loss 需全词表本地 logits。v1 在 lm_head 入口对
   SP 分片流做一次 all-gather（`full_tensor`），头权重跨 TP 复制。tp>1 时
   激活显存口径按全词表 logits 计。待实测 tp>1 数据后可切换 vocab-parallel。
2. **EP/eFSDP 显存记账**：`persistent_param_bytes` 在 fsdp2ep2 与 fsdp2 相同
   （7.83 GB），未随 EP 切分再减——疑为显存模型对 `.data.chunk` EP 切分的
   记账口径（公共模拟器行为，非模型特例）；TP/FSDP 维度的下降（tp2ep2
   4.23 GB → tp2ep2fsdp2 2.12 GB）符合预期。单列为 simulator 观察项。
3. **debug 分支 load-balance**：`debug.moe_force_load_balance` 仅用于解析
   a2a 拆分的值强制；权重仍流经真实 softmax→topk→div→cast 反向图（管理方
   反馈轮已修）。
4. **divinplace / catoutout / 显式 zero_**：钉住的 core 参考版
   （/tmp/opencode/MindSpeed）中不存在；需服务器 mindspeed/ops/triton 源码或
   实测算子行落位（管理方第六轮遗留项）。
5. **仓内既有单测失败 5 项**（kimi overrides ×4、multimodal 资产 ×1），
   与 qwen3_next 无关，见验证记录。

## 6. MXFP8 量化范围

- 量化（`mxfp8_rceil`）：token_mixer 全部投影（q/k/v/o、in_proj_qkvz、
  in_proj_ba、out_proj）+ shared_expert（gate/up/down）。
- **强制排除（fp32 语义）**：`moe.gate`（router）、`moe.shared_expert_gate`。
  配置 FQN 子串命中即 fail fast（`_guard_mxfp8_fqns`）+ 单测守护
  （converter 按 `target_fqn in cur_fqn` 子串匹配，故 FQN 必须精确到
  `moe.shared_expert.gate_proj` 级别以避开 `shared_expert_gate`）。
- **范围外**：routed experts（裸参数 + `torch._grouped_mm` 路径，torchao
  nn.Linear 包装不适用）——bf16，契约声明；`npu_gmm` 无需前置（本模型
  EP dispatcher 不走 `npu_gmm` converter）。

## 7. 验证记录

### 7.1 模拟器矩阵（全部 exit=0，2 层 80B 维度、seq 16384、AC=none 除注明）

| 配置 | NGPU | AC | comm (fwd+bwd 合计) |
| --- | --- | --- | --- |
| qwen3_next_2l_tp2 | 2 | none | AG6 RS4 AR3 |
| qwen3_next_2l_ep2 | 2 | none | A2A12 AG14 AR2 RS5 |
| qwen3_next_2l_fsdp2 | 2 | none | AG10 AR2 RS5 |
| qwen3_next_2l_tp2ep2 | 4 | none | A2A12 AG26 AR8 RS13 |
| qwen3_next_2l_fsdp2ep2 | 4 | none | A2A8 AG16 AR3 RS7 |
| qwen3_next_2l_tp2ep2fsdp2 | 8 | none | A2A12 AG26 AR8 RS13 |
| qwen3_next_2l_mxfp8 | 1 | none | 量化算子入账（§7.3） |
| qwen3_next_2l_fsdp2 / _ep2 / _tp2ep2 / _tp2ep2fsdp2 | — | **full** | 全部 exit=0 |
| qwen3_next_smoketest | 2 | none | 回归 ✓（ep=2 实测口径） |
| qwen3_next_80b_a3b_2layer | 2 | none | 回归 ✓ |

### 7.2 显存对账（memory_summary.json，2 层 80B 维度）

| 配置 | persistent_param_bytes | active peak |
| --- | --- | --- |
| fsdp2 | 7.83 GB | 23.5 GB |
| ep2 | 7.83 GB | 23.5 GB |
| tp2ep2 | 4.23 GB | 15.9 GB |
| tp2ep2fsdp2 | 2.12 GB | 8.0 GB |

TP/FSDP 维度切分单调下降 ✓；EP 维度的记账观察见 §5.2。

### 7.3 量化算子入账（qwen3_next_2l_mxfp8）

```text
npu.npu_dynamic_mx_quant.default  forward 26 | backward 52
npu.npu_quant_matmul.default      forward 13 | backward 26
```

### 7.4 单测与回归

- `tests/unit_tests/models/test_qwen3_next.py`：29 passed（配置注册、
  mxfp8 守卫、层布局、参数量公式精确断言、并行 fail-fast、TP 整除守卫）。
- 仓内既有失败 5 项与 qwen3_next 无关（`test_kimi_k3.py::test_model_overrides_validate_before_run`
  ×4、`test_multimodal_helpers.py` ×1），在本分支改动前已失败。

### 7.5 证据路径

- 算子流：`simulator_output/<config>/kernel_summary/rank_0.csv`
- 显存：`simulator_output/<config>/memory/memory_summary.json`
- 对齐过程：`docs/user-guides/simulator/qwen3-next-op-stream-alignment.md`
- 环境容器 `titan-llf`（模拟器走 meta；branch qwen3.5）
