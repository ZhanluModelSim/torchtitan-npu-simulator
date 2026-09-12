# ar_llm（DeepSeekV4-Sparse）模型契约

来源：`torchtitan_npu/simulator/raw_model/ar_llm/`（参考实现：`config.py`、`model.py`、`ARCHITECTURE.md`、`DIMENSION_SCALING.md`）。
本文件是按 `docs/feature_guides/new_model_onboarding.md` 第 1 节要求形成的模型契约，记录参考实现与本仓框架化之间的所有约定与偏差。接入实现以本契约为准；与 raw 参考实现冲突时，以本契约记录的决策为准并注明原因。

## 1. 来源

- 参考实现：raw_model 自带纯 PyTorch 实现（self-contained，无框架依赖）。
- 无外部权重、无 tokenizer、无基线 commit；本仓先以随机初始化 + state-dict schema 闭环为验收目标（meta 建模），真实权重兼容性另行声明。
- 训练/模拟 tokenizer 资产复用 `./tests/assets/tokenizer/deepseekv3_tokenizer`（仅 c4_test smoke 数据通道，与模型词表不要求一致，同 kimi_k3 做法）。

## 2. 网络结构

| 项 | 值 |
| --- | --- |
| hidden_size | 16384（2^14） |
| num_layers | 60 = 10 × 6 层单元 |
| 层单元 | `[CSA, HCA, KDA, KDA, KDA, KDA]`（CSA:HCA:KDA = 1:1:4） |
| num_attention_heads | 128 |
| head_dim | 256 = nope 192 + rope 64 |
| vocab_size | 524288（2^19），embedding 与 lm_head **tied** |
| 每层 | attention + MoE 同时存在（1:1），无 dense FFN 层 |

层类型规则：`pos = layer_idx % unit_size`；`pos==0 → CSA`，`pos==1 → HCA`，`pos>=2 → KDA`。

## 3. Attention（三种类型，按层分发）

### 3.1 CSA（Compressed Sparse Attention，每单元 1 层）
- MLA 式低秩投影：`q: d→q_lora→(nope|rope)`，`kv: d→(kv_lora|rope)→(k_nope|v)`，`o: 分组低秩 o_groups×(o_lora)→d`。
- 核心：滑窗 local attention（window_size）+ stride 压缩 global attention（compress_ratio），两路 softmax 输出相加。
- RoPE 仅作用于 rope 维（64），YaRN 外推（yarn_factor / yarn_original_max）。

### 3.2 HCA（Heavily Compressed Attention，每单元 1 层）
- 同一套 MLA 低秩投影；核心：KV 按 compress_ratio（256/512）压缩，indexer（独立 64×128 投影头 + RMSNorm）对压缩 KV 打分取 top-4096，再在选中 KV 上做 attention，indexer 分数作为 attention logit 偏置。
- raw 参考实现中的 `HadamardRotation` 在 forward 中未被使用（死代码），本仓不建模；生产化如需再按模型定制算子补齐。

### 3.3 KDA（Kimi Delta Attention，每单元 4 层）
- delta-rule 线性注意力：状态 `S ∈ [b, H, d_state, d_state]`（非参数 buffer）。
- 门：`alpha`（读门，[H, d_k]）、`beta_erase/beta_write`（双写门，各 [H, d_state]）。
- raw 实现的 `da_proj`（d_a=3）在 delta-rule 计算中未被使用（死代码），**本仓删除该参数**；参数量公式相应扣减。
- 维度约束：**要求 `kda_d_k == kda_d_v`**（否则 alpha 门与输出维度错位，raw 语义不成立），`d_state ≥ d_k` 时按 raw 的零填充嵌入语义（等价于 [d_v, d_k] 有效状态）。
- 融合边界：生产算子为外部 kernel（Kimi KDA 同族 `chunk_kda`，本仓 raw op 名 `triton_ascend_kernels.chunk_kda[_grad]`）；框架路径提供顺序参考实现（逐 token，仅 debug 规格可负担），模拟器以 shape-only shim 记录融合算子。

### 3.4 语义修正（相对 raw 参考）
- raw `SparseBlock.forward` 中 Engram 分支引用未定义变量（bug），契约定义为：**Engram 作用于 block 输入**，`h = hidden_states + engram(hidden_states, input_ids)`。
- raw `SparseModel.forward` 中 sampling loss 悬空表达式（bug），契约定义为：`total_loss = ce + z_loss_alpha * Σaux + sampling_loss_alpha * Σsampling`；框架化后 aux loss 通过 autograd scaler 注入，等价生效。

## 4. MoR（Mixture of Recursions）训练态语义

- raw 的 `RecursiveDynamicCache`（cycle KV 共享，base_depth=12 × 5 递归）**仅存在于推理路径**；raw 训练 forward 未使用 cache。
- 契约：**训练/模拟器建模不实现跨层 KV 复用**。MoR 在训练态只体现为 `mor_expert_ratio`（5% 层，仅 CSA/HCA）使用 expert-choice 路由。KV 压缩收益属于推理服务建模，不在本仓 meta 验收范围。
- PP 与 MoR 的跨 stage cache 复用未定义，PP 接入前必须先补充该契约（当前 PP fail fast）。

## 5. LatentMoE

- 每 token：top-16 路由（token-choice，`sqrtsoftplus` 打分 × route_scale=2.5，renormalize）+ 2 个 shared expert（全部 token）。
- 单 expert（routed 与 shared 同构）5 矩阵：`gate_down/up_down: d→latent(7168)`，`latent_to_inter: latent→inter(4096)`（gate/up 共享），SwiGLU(clamp=10)，`inter_to_latent: inter→latent`，`latent_to_out: latent→d`。
- grouped 权重布局 `[E, out, in]`：`w1=gate_down [E,latent,d]`、`w3=up_down [E,latent,d]`、`w4=latent_to_inter [E,inter,latent]`、`w5=inter_to_latent [E,latent,inter]`、`w2=latent_to_out [E,d,latent]`；融合边界 = 5 组权重、6 次 grouped_mm（latent_to_inter 对 gate/up 各一次）+ 1 次 swiglu。
- 路由：
  - token-choice（57 层）：top-K + load-balance aux loss（z_loss_alpha=1e-3）。
  - expert-choice（3 层，均匀分布于深度、仅 CSA/HCA）：每 expert 选 capacity=⌈factor·T·topk/E⌉ 个 token；未选中 token 以 weight=0 经 expert 0 通路（保持 `num_tokens_per_expert ≡ capacity` 静态）；aux loss = 选中计数方差（sampling loss）。
  - `debug_force_load_balance=True`（模拟器强制）：所有层退化为 round-robin top-K，路由结果与输入值无关。
- expert 内部并行：TP/ETP 切 **inter 维**（colwise→swiglu→rowwise+all-reduce，单次 all-reduce）；latent 维切分需两次 all-reduce，不采用。

## 6. mHC / Engram / 其他

- mHC：每层 2 个 HyperConnectionBlock（expand d→hc_mult·d → Sinkhorn(20) 混合 [hc,hc] → contract → 残差+post norm）；TP 下权重 Replicate、按 sequence-shard 本地计算；模拟器以融合 kernel 名记录。
- Engram：10/20 个指定层；每层 = 多头多项式 rolling hash（n-gram orders [2,3]）→ `nn.Embedding(num_hash_heads × capacity, per_head_dim)` 查表 → context gate → memory proj + 深度可分离 Conv1d。hash 乘法在 int64 下允许回绕（仅影响 hash 分布）。Engram hash 表参与训练（可训练参数），必须进入 FSDP 分片与显存模型。
- **Engram 部署（训练，按 Engram paper §2.5）**：hash 表按**桶维连续分片**（`Shard(0)`，owner = `bucket // E_local`）分布到 EP mesh（EP 开启时，与专家同待遇、计算期保持分片）或 TP mesh（无 EP 时）；前向 = 本地 hash → **All-to-All gather**（发 keys / 回 embedding 行，两次 a2a，autograd 自动完成反向 **All-to-All dispatch** 梯度）；模拟器强制负载均衡下 owner/splits 为形状纯函数（round-robin，meta 不读值）。TP/CP 下不再 all-gather hidden：hash 在全局 ids 上求值后按本地序列窗口切片（CP 下仅 all-gather 微量 int64 ids）；门控/卷积在本地 hidden 上计算；**短卷积的跨 rank halo 以零填充近似**（paper 消融显示 conv 贡献边际，随计算流对齐后置）。纯 DP/FSDP（无 EP/TP）时表退化为本地查表。
- **计算流与 paper 的偏差（后置对齐，不影响部署方式）**：hash 函数（multiplicative-XOR vs 多项式 rolling）、mHC 分支特异 W_K^(m)/共享 W_V 门控、conv 的 SiLU 包裹结构。
- KDA 双写门在框架模型中拆分为 `erase_gate`/`write_gate` 两个 head-major Linear（融合的 `[2, H, ds]` 输出在 TP colwise 下会切错维度）；参数量不变。
- TP 下的 HCA indexer 保持 Replicate（输入本就是全序列，避免 indexer 分数 partial-sum 通信）；O 分组投影按 group 切分后以 all-gather 恢复 sequence shard。
- 稳定性项：attn_sink（每 head 可学习 bias，仅 CSA/HCA）、attn softmax clamp=50、swiglu clamp=10。
- loss：`cross_entropy（sum reduction，框架统一）`；aux loss（load-balance / sampling）经 autograd scaler 注入梯度，不计入打印 loss 数值口径。

## 7. 参数量公式（独立基线）

框架模型 output 头与 embedding **不 tied**（FSDP/DTensor 下共享参数易引入边界问题），两者各计 `V·d`；HF 权重若为 tied 形式由 state-dict adapter 展开为两个 key。

```
embedding          = V·d
output(lm_head)    = V·d
每 CSA/HCA 层 attention（MLA 低秩）:
  q:  d·q_lora + q_lora·nh·nope + q_lora·nh·rope
  kv: d·(kv_lora+rope) + kv_lora·nh·nope + kv_lora·nh·hd
  o:  G·(nh/G)·hd·o_lora + G·o_lora·(d/G)
  indexer（仅 HCA）: 2·d·(idx_heads·idx_head_dim) + 2·(idx_heads·idx_head_dim)
  norm/sink: q_lora + nope + (kv_lora+rope) + kv_lora + nh·use_attn_sink
每 KDA 层:
  q/k/v: d·nh·dk + d·nh·dk + d·nh·dv
  alpha: d·nh·dk ; beta(双门): d·2·nh·ds ; o: nh·dv·d
每层 MoE:
  routed: E·(3·d·latent + 2·latent·inter)
  shared: n_shared·(同上单 expert)
  router: d·E
每层 mHC: 2·(hc·d·d + hc·hc + hc·d·d)
每层 norm: 4·d（block 级）+ 4·d（mHC pre/post）+ 最终 norm·d
Engram 每层: n_orders·nh·cap·(mem_dim/n_orders/nh) + 2·d·mem_dim
             + 2·mem_dim + 2·d + 4·d(conv) + 1(gate_bias)
```

注意：raw 仓库 `estimate_params_from_config` 将 tied embedding 计了 2 次，且 Engram 数值与 ARCHITECTURE.md 不一致；本仓以模型实参 `sum(p.numel())` 与上述公式对账为准（单测覆盖 debug/reduced 两个规格）。

## 8. 支持范围（首版声明）

| 特性 | 状态 |
| --- | --- |
| 单卡构造/前向/反向 | 支持（debug/reduced 规格可真实执行） |
| FSDP / eFSDP | 支持 |
| TP（非 MoE + 专家内部 TP） | 支持（o 分组输出 all-gather 回 sequence shard；专家切 inter 维） |
| EP（routed experts expert 维分布） | 支持 |
| CP（all-gather 全序列，kimi_k3 保守方案） | 支持；**暂不支持 CP+TP 组合**（fail fast）；RoPE 全局位置由 attention CP pre-hook 重切 buffer 保证 |
| ETP（EP+ETP 组合） | 首版不支持，fail fast |
| PP | 首版不支持（MoR 跨 stage cache 契约未定），fail fast |
| AC none/full/selective | 支持（复用 `apply_moe_ac`） |
| MXFP8 | 支持（`ar_llm_debug_mxfp8` / `ar_llm_reduced_mxfp8`；`--mxfp8-fqns` 逗号分隔自定义覆盖，fqn 子串匹配） |
| compile / offload / MTP | 首版不支持，fail fast 或显式关闭 |

## 9. 模拟器建模契约（raw op 名）

**全部复用 DSV4（smla/mhc shim）与 kimi_k3（kda shim）已建模的真算子名与输入签名**，不引入自造名字：

| 模型点 | 捕获 raw op | 签名约定（与 DSV4 对齐） |
| --- | --- | --- |
| CSA 核心 | `aclnn.npu_sparse_attn_sharedkv[_grad]` + `..._metadata` | 5 输入 `[query, ori_kv, sinks, metadata, cmp_kv]`（压缩 KV、无 topk 索引），输出 `[result, softmax_lse]`；**ori_kv/cmp_kv 为 4-D `[B, T, 1, kv_dim]`**（DSV4 的 head-collapsed shared-KV 约定，dim1=kv_seq_len；ar_llm 的 per-head K/V 展平进 kv_dim = nh×(k_dim+v_dim)，诚实反映每 token KV 带宽） |
| HCA 核心 | 同上 | 6 输入 `[..., cmp_kv, cmp_sparse_indices]`（topk 索引变体）；**按输入 tensor 个数区分 CSA/HCA**，indices `[B,S,1,K]` |
| HCA indexer | `aclnn.npu_lightning_indexer` | `[query_idx [B,S,1,D_idx], key_idx [B,S2,1,D_idx], weights [B,S,1]]` → `[sparse_indices(int32), sparse_values]`；**无反向算子**（DSV4 非 A5 行为）；ar_llm 的 indexer 为单 dot-product 头（N_idx=1，D_idx=idx_dim） |
| indexer 梯度 | `aclnn.npu_sparse_lightning_indexer_grad_kl_loss` | 反向记录，6 输入 `[query [B,S1,N,D], key [B,S2,1,D], query_idx [B,S1,1,D_idx], key_idx [B,S2,1,D_idx], weights [B,S1,1], sparse_indices [B,S1,1,topK]]` → `[d_query_idx, d_key_idx, d_weights, loss]`；softmax_max/sum 不记（解析器 len<7 默认分支） |
| KDA 核心 | `triton_ascend_kernels.chunk_kda` / `_grad` | kimi_k3 同族；ar_llm 双门变体传 `alpha/beta_erase/beta_write` |
| mHC | `triton._triton_hc_sinkhorn_comb_fwd/bwd_kernel` + `triton.hc_pre_bmm_forward/backward` | Sinkhorn comb 与通道混合 bmm 分别复用 DSV4 核名；rms_norm/matmul 步骤走 SimRMSNorm converter 与真实 meta matmul |
| LatentMoE | `aten._grouped_mm.default` ×6/层 + swiglu | 真实 meta kernel 执行即被捕获 |
| Engram / router / hash | aten 原生算子 | 生产形态即 embedding gather + 小算子，不合成融合名 |

shape-only shim 在模型构造与并行化完成后绑定（`apply_ar_llm_shims`），保持 FQN、DTensor placement 与既有 hook 不变。

### 9.1 MXFP8 量化建模

- 机制：`MXFP8Converter`（torchao）按 **module fqn 子串匹配**包装目标模块下的 `nn.Linear` 权重与 3D `nn.Parameter` 为 `MXFP8TrainingWeightWrapperTensor`；wrapper 只拦截 `linear/mm/matmul/addmm/_grouped_mm`，故命中模块的矩阵乘真实派发为 `npu.npu_dynamic_mx_quant ×2 + npu.npu_quant_matmul`（Linear 的 F/dx/dw）或 `npu.npu_grouped_dynamic_mx_quant + npu.npu_grouped_matmul`（routed experts 的 `_grouped_mm`），meta kernel 直接执行并进入算子/内存/依赖账本。
- 默认范围（`DEFAULT_MXFP8_FQNS`）：`moe.experts`、`moe.shared_experts`、`attention.q_proj`、`attention.kv_proj`、`attention.core`（indexer）、`attention.kda`。CSA/HCA 的 fused core 本身仍走 shape-only shim，不受影响。
- **einsum 覆盖**：wrapper 原生不拦截 einsum。`patches/torchao_npu/mxfp8_wrapper_einsum.py` 将「单个 3D `[n,out,in]` 包装权重、逐 expert 线性形（`A @ W[i].t()`）」的 einsum 降级为 `n` 次 `NpuMXFP8MM` 后 stack——`moe.shared_experts` 的 6 个投影由此全部走 FP8；`attention.o_proj`（GroupedOProjection，einsum 含交叉维度重排）不匹配该模式，保持 BF16 回退（显式排除于默认清单）。
- 内存口径：wrapper 保留 fp32 主权重（FP8 副本为瞬态），`persistent_param_bytes` 不变；激活侧新增量化 scale/FP8 输入的瞬态占用。
- CLI：`--mxfp8-fqns "moe.experts,moe.shared_experts"`（逗号分隔）整体替换默认清单；要求 config 恰含一个 MXFP8 converter，否则报错。

## 10. 首版实现状态

- 已落地：`torchtitan_npu/models/ar_llm/`（model/attention/feed_forward/parallelize/state_dict_adapter/config_overrides/config_registry），`--module ar_llm` 注册，flavors：`debug`（1 个 6 层单元，全路径覆盖）/`reduced`（2 个单元）/`50t`/`100t`（正式规格，仅 meta）。
- 训练配置工厂：`ar_llm_debug / ar_llm_reduced / ar_llm_50t / ar_llm_100t`；模拟器配置：`torchtitan_npu/simulator/config_registry.py` 同名包装。
- 模拟器 shim：`torchtitan_npu/simulator/hardware_shims/ar_llm_shim.py`（KDA/CSA/HCA/mHC 融合算子 shape-only 记录）；LatentMoE GMM 走真实 `aten._grouped_mm` meta kernel 捕获。
- 已知首版限制（验收口径为 Conditionally ready 的声明范围）：
  - CP/PP/ETP/DeepEP/compile fail fast（错误信息指向本契约）；MXFP8 已支持（§9.1）。
  - KDA 真实融合 kernel 未接入（顺序参考实现仅 debug 规格可负担）；模拟器/大规模验证依赖 shim。
  - MoR 训练态不建模 KV 复用（见第 4 节）。
  - `ExpertParallel` 分区假设上游通过 `self._partition_fn` 分发，`parallelize` 内有 DTensor 校验 fail-fast 兜底。
