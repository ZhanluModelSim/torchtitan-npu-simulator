# glm5_next（GLM-5.3-Flash / unified_mm）模型契约

来源：`torchtitan_npu/simulator/raw_model/unified_mm/`（`config.json` + `model.py`，HF transformers 风格参考实现，`model_type: glm5_next`，`architectures: Glm5NextForConditionalGeneration`）。`reference/MAGI-2-preview` 为第三方推理仓旁证，不作为结构依据。
本文件按 `docs/feature_guides/new_model_onboarding.md` 第 1 节形成；与 raw 参考实现冲突时，以本契约记录的决策为准并注明原因。复用约定：mHC 参照 `models/deepseek_v4`（HcPre/HcPost/Sinkhorn 同构），KDA 参照 `models/kimi_k3`（`chunk_kda` kernel seam + 模拟器 shape-only shim）。

## 1. 来源与范围

- 参考实现自带 `config.json` 与 `model.py`（纯 HF 风格，依赖 transformers 内部模块），无独立权重文件、无 tokenizer；以随机初始化 + state-dict schema 闭环为 meta 验收目标。
- 训练/模拟 tokenizer 复用 `./tests/assets/tokenizer/deepseekv3_tokenizer`（c4_test 通道）；多模态数据通道复用 `cc12m-test`（NLD collator：`pixel_values [N, L, D_patch]`、`grid_thw [N, L, 3]`）。
- `quantization_config`（fp8 e4m3, block 128×128, activation dynamic）是 **checkpoint 存储格式**，不参与显存/参数建模（MXFP8 口径：计算侧量化不改变参数 storage dtype）。**计算侧量化**：`MXFP8Converter`（mxfp8_rceil）作用于 attention/MoE 矩阵乘（FQN 见下），与 DSv4/K3 同机制；router gate、前 4 层 dense MLP、embedding/lm_head、vision tower 保持高精度。能力门（`has_mx_capability` 要求 A5）在模拟器下由 `meta_env` 的 meta-safe 补丁放行（非 A5 宿主机可跑 mxfp8 meta 模拟），真机训练仍由原门把关 Ascend950；capture 中量化路径记录 `npu.npu_dynamic_mx_quant`、`npu.npu_quant_matmul`、`npu.npu_grouped_matmul`（MoE grouped 替换 `aten._grouped_mm`）。

## 2. 网络结构（名义 96 层 → 框架 41 个唯一 block）

| 项 | 值 |
| --- | --- |
| hidden_size | 24576 |
| 名义层数 | 96 = 16（pre，0–15）+ 56（loop 区，16–71）+ 24（post，72–95） |
| 唯一 block 数 | **41** = 16 pre + **1 共享 loop block** + 24 post（`share_loop_weights=true`） |
| attention 比例 | 每 4 层 3×KDA + 1×DSA（`layer_types`）；DSA 层 index ≡ 3 (mod 4) |
| MLP 分布 | 前 4 层 dense（inter 73728），其余 sparse MoE |
| num_attention_heads | 192；vocab 154880；embedding 与 lm_head **不 tied**（config 声明 false，框架保持不 tied） |
| MTP | `num_nextn_predict_layers=2`，**v1 不建模**（fail fast） |
| 位置编码 | 全模型 **NoPE**（`qk_rope_head_dim=0`，`position_embeddings=None`）；max_position 4M 仅约束配置 |

## 3. Looped layers 建模（契约决策）

config `loop_config`：`looped_layer_start=16, end=71, num_looped_layers=56, share_loop_weights=true, train_min/max_steps=1/4, adaptive_halting per_sequence threshold 0.98, infer 1–32, hard cap 64`。raw `model.py` **未实现** loop/halting（纯 config 声明）。契约决策：

1. **共享权重**：loop 区建模为 1 个共享 block（KDA + sparse MoE），执行 `loop_train_steps` 次（config 字段，默认 4 = `train_max_steps`，校验范围 1–4）。参数量按 1 个 block 计。
2. **固定步数，无 halting**：模拟器/训练侧不做 halting 判断（shape 静态、算子账本确定、DAG 静态可回放）。`adaptive_halting` / per_sequence 退出 / infer steps > train 声明为不支持，fail fast。
3. **结构偏差声明**：loop 区名义上含 DSA 槽位（每 4 层第 4 个），但权重共享下 block 必须同构，故共享 block 取区域多数类型 **KDA**；loop 区 DSA 槽位不作为独立参数/执行存在。`share_loop_weights=false`（56 个独立 block）v1 不支持，fail fast。若真实训练系统语义与此不符（如"1 步 = 1 个 4 层单元"），以真实系统为准修订本节并重算参数/算子账本。
4. `kv_mirror` / `loop_kv_mirror_ratio`（KV 迭代间镜像复用）属推理/服务建模，v1 不实现（同 ar_llm 对 MoR 的处理）。

## 4. Attention（按层分发）

### 4.1 KDA 线性注意力（72/96 名义层；框架：pre 12 + loop 1 + post 18 = 31 个唯一 block）

- 投影：`q/k/v_proj: d→192·128`；**fused qkv 短卷积**：单条 depthwise `Conv1d(3·qkv_dim, kernel=4, groups=3·qkv_dim)` + silu（kimi_k3 为 3 条独立 ShortConv，本模型为 1 条 fused，契约按 1 条）。
- Forget gate（低秩 + lower bound）：`f_a_proj d→128` → `f_b_proj 128→qkv_dim`；`g = lower_bound·sigmoid(exp(A_log)·(f_b(f_a(x)) + dt_bias))`，`lower_bound=-5.0`，`A_log/dt_bias` fp32。
- Beta：`b_proj d→192`，sigmoid。输出门（低秩）：`g_a d→128` → `g_b 128→qkv_dim`；输出 `RMSNormGated(head_dim=128)`（sigmoid 门控）→ `o_proj qkv_dim→d`。
- 核心 kernel：`chunk_kda`（chunk 64，`use_qk_l2norm_in_kernel=true`，g 为**预计算值**传入 kernel，即 `use_gate_in_kernel=false` 语义；A_log/dt_bias/lower_bound 在 gate 模块内完成）。生产路径按 kimi_k3 同一 seam 接入：`_chunk_kda` 懒加载 `triton_ascend_kernels.attention.fla.kda.chunk.chunk_kda` 并以预计算 g 调用（`use_gate_in_kernel=false, safe_gate=false`）；包缺失时（CPU/debug 环境）回退到逐 token 顺序参考实现并告警一次（数学精确，非性能等价）。模拟器由 `apply_glm5_next_shims` 将同一 seam 替换为 shape-only shim 记录 `triton_ascend_kernels.chunk_kda[_grad]`（shim 单测 + 端到端 capture 验证）。
- 状态语义：训练态无 cache（与 kimi_k3 相同）；`initial_state=None, output_final_state=False`。

### 4.2 DSA 稀疏全注意力（24/96 名义层；框架：pre 4 + post 6 = 10 个唯一 block）

MLA（**纯 NoPE**，无 rope 维）：
- `q_a_proj d→6144` + `q_a_layernorm(RMSNorm 6144)` + `q_b_proj 6144→192·256`；
- `kv_a_proj_with_mqa d→2048`（无 rope 段）+ `kv_a_layernorm(RMSNorm 2048)` + `kv_b_proj 2048→192·(256+256)`；
- `o_proj 192·256→d`；scale = qk_head_dim^-0.5 = 256^-0.5。

Indexer（k-pool 压缩版 DSA indexer；`indexer_types` 全为 `"full"`，跨层 topk 共享 `"shared"` v1 fail fast）：
- `wq_b 6144→64·128`、`wk d→128` + `k_norm(LayerNorm 128)`、`weights_proj d→64`；
- k-pool 压缩：pool=8，`gate [128,d]` 打分 + `ape [8,128]` 加权 softmax 池化出 pool key；
- 选择：`select_k = min(index_topk//kpool, P)` = 1024 pools → 展开 8·1024 = 8192 token 索引 + 不足整池 tail（≤7）→ 输出宽度 **8199**；
- **语义偏差声明**：raw indexer forward 带 `@torch.no_grad()` 且 topk 仅作为 mask（无 additive bias），indexer 梯度路径为零。框架 v1 忠实建模为**冻结 indexer**：`requires_grad=False` + no_grad 前向，参数进入 FSDP 分片与显存模型但不参与优化器/梯度。真实训练若需训练 indexer（独立 aux loss 或 additive-bias 语义），列为后续契约项。
- 训练态 KV 为本层同序列（无 cache）；attention 实现为 **gather-based 精确稀疏**：按 topk 索引 gather k/v（`[B,S,K=8199,H,D]`）+ 无效槽位 additive -inf → softmax。shape 全静态。声明：生产融合 kernel（smla/sparse-attn 族）接入时在同一 seam 替换；SDPA+additive-mask 路径与 gather 路径数学等价（topk 并集内 attention）。

## 5. MoE

- 框架结构约定：text backbone 平铺在顶层（`tok_embeddings` / `layers` / `norm` / `output`，loop block 占据 `layers.{pre_layers}` 槽位），使上游 AC（`model.layers`）与 PP 约定直接生效；vision tower 挂在 `self.visual`。
- 前置 4 层 dense：`gate/up/down d→73728`，clamp-swiglu（gate 上限 +10，up ±10）。
- sparse（其余 92 名义层；框架 12 个唯一 block）：**2048 routed experts / top-16 / 1 shared**（moe_inter 3072）。
- 权重布局（torchtitan 惯例 `[E, out, in]`）：`w1 [E,inter,d]`、`w3 [E,inter,d]`、`w2 [E,d,inter]`；HF `gate_up_proj [2·inter, d]` 前半 gate 后半 up，converter/state-dict 层负责拆分/合并（与 npu_gmm 的 w13 融合共用同一约定）。
- 路由：sigmoid 打分（fp32）+ `e_score_correction_bias`（noaux_tc）；`n_group=1, topk_group=1` → 组过滤退化为全专家 top-16；`norm_topk_prob=true`（权重 renormalize）× `routed_scaling_factor=2.5`。v1 中 bias 固定零初始化：不参与梯度（topk 不可导）也不参与 LB hook 更新（`moe.expert_bias` 未暴露），路由直方图 `tokens_per_expert` 仍供 hook/算子账本使用；模拟器叠加 round-robin 强制均衡保证 GMM 形状静态。
- shared expert：dense MLP（inter 3072），全 token 通路。
- aux loss：load-balance（Switch 式 density·proxy），经 autograd scaler 注入，不计入打印 loss 口径；`debug_force_load_balance=true`（模拟器强制）退化为 round-robin top-K。
- 专家并行：EP 切 expert 维（ExpertParallel，Shard(0)）；无 EP 时专家内部按 inter 维 TP（colwise→swiglu→rowwise + 单次 all-reduce）。ETP 组合 v1 fail fast。

## 6. mHC 超连接

- 每层 attn/ffn 两处 + 末层 head。残差流为 **[B, S, 4, D]** 四流贯穿主干（hc_mult=4），activation 显存约 4×。
- 结构（与 deepseek_v4 的 HcPre/HcSplitSinkhorn/HcPost **同构**，参数全部 fp32）：
  - 每处 1 组参数：`fn [(2+H)·H, H·d]` = [24, 4d]、`base [24]`、`scale [3]`；输入无权重 RMSNorm（eps=rms_norm_eps，fp32）。
  - pre = sigmoid(w·scale₀+base₀)+ε（流塌缩权重）；post = 2·sigmoid(w·scale₁+base₁)；comb = H×H 经 softmax + **Sinkhorn 20 次迭代**（行/列交替归一，+ε）。
  - forward：`collapsed = Σ_h pre·stream` → 子层输入；子层输出后 `new_streams = post ⊗ out + combᵀ ⊛ streams`。
  - **hc_head 与 DSv4 的差异**：GLM 末层为**无参数 mean**（DSv4 为可学习 head）；框架实现 `HcHeadMean`（纯归约，无 converter 需求）。
- 复用：直接 import `torchtitan_npu.models.deepseek_v4.model` 的 `HcPre/HcPost`（npu_mhc_pre/npu_mhc_post converter 与模拟器 mHC shim 按 isinstance 识别，自动生效）；hc 参数挂在 block 上（`hc_attn_fn/base/scale`、`hc_ffn_fn/base/scale`，fp32，与 DSv4 相同）。
- TP：mHC 按序列分片本地计算，参数 Replicate（与 DSv4 相同）。

## 7. Vision tower（纳入范围，v1 约束）

- 结构：patchify（`Conv3d/Conv2d`，patch 14、temporal v1=1）→ hidden 2048，**32 blocks**（pre-norm attention + clamp-swiglu MLP，q/k per-head RMSNorm，axial 2D RoPE 由 `grid_thw` 逐 patch 坐标张量运算得到）→ post RMSNorm → **downsample `Conv2d(2048→24576, k=merge=2)`** → `PatchMerger`（proj+LayerNorm+GELU+clamp-swiglu 49152→24576）→ 输出 [N_img, L/4, 24576]。
- 融合：`image_token_id`（154854）占位 mask + `scatter_visual_embeddings` 早期融合进 text embedding；video token（同 id 经 start/end span 区分）v1 **fail fast**（仅声明 image 通路）。
- **v1 形状约束**：批内**均匀网格**（每图 patch 数相同，`h·w == L`，`h/w` 由 config `vision_image_size` 推出），否则 fail fast；非均匀网格需要 per-image bucketing（数据侧能力），列为后续项。meta 下禁止读取 grid 数值（RoPE 坐标、valid mask 全部走张量运算；h/w 来自 config）。
- 并行：v1 vision tower 跟随外层 FSDP 分片，**不支持 vision TP/CP**（fail fast）；vision 激活随微批次进入显存模型。

## 8. 参数量公式（独立基线，41 个唯一 block）

```
d=24576, nh=192, kda_heads=192, kda_hd=128, kv_lora=2048, q_lora=6144, nope=256, v_hd=256
E=2048, topk=16, moe_inter=3072, dense_inter=73728, V=154880, H=4
D = d

embedding   = V·d ;  lm_head = V·d（不 tied）

每 KDA block:
  q/k/v   = 3·d·(kda_heads·kda_hd)          # 各 d·24576
  conv1d  = 3·kda_heads·kda_hd·4            # fused qkv depthwise, kernel 4
  f_gate  = d·kda_hd + kda_hd·(kda_heads·kda_hd) + qkv(kda_heads·kda_hd) + kda_heads   # f_a,f_b,dt_bias,A_log
  b_proj  = d·kda_heads
  o_gate  = d·kda_hd + kda_hd·(kda_heads·kda_hd)
  o_norm  = kda_hd ; o_proj = (kda_heads·kda_hd)·d

每 DSA block:
  q: d·q_lora + q_lora + q_lora·nh·nope
  kv: d·kv_lora + kv_lora + kv_lora·nh·(nope+v_hd)
  o: (nh·v_hd)·d
  indexer: q_lora·(64·128) + d·128 + 128(LN) + d·64 + 64·128(ape) + 128·d(gate)

每 MoE block:
  routed  = E·(2·moe_inter·d + d·moe_inter)      # w1,w3,w2
  router  = d·E + E（e_score_correction_bias）
  shared  = 2·moe_inter·d + d·moe_inter
每 dense block = 2·dense_inter·d + d·dense_inter

每 block mHC = 2·[(2H+H²)·H·d + (2H+H²) + 3] = 2·[24·4d + 24 + 3]
每 block norm = 2·d（input/post_attention RMSNorm）+ KDA 额外 o_norm·kda_hd
vision（config: depth 32, v_h=2048, heads 32×64, v_inter 8192, out 24576, proj 49152）:
  per block = qkv 3·v_h² + proj v_h² + q/k norm 2·64 + mlp 3·v_h·v_inter + 2 norm·v_h
  stem = Conv(3·1·14²·v_h) ；downsample = v_h·24576·4 ；merger = v_h² + v_h(LN) + 2·v_h·proj + proj·v_h
```

debug/reduced 规格的单测以 `estimate_glm5_next_params(config)` 与 `sum(p.numel())` 对账（不 tied，无共享重复计数；loop block 只计 1 次）。

## 9. 算子账本（每 step，microbatch m，序列 S，loop 步数 T=loop_train_steps）

| 模型点 | 每 block 理论调用（fwd） | 执行 block 数 |
| --- | --- | --- |
| KDA conv1d / chunk_kda | 1 / 1 | (12+18)·1 + 1·T = 30+T |
| DSA MLA 线性（q_a/q_b/kv_a/kv_b/o） | 5 MM | 10 |
| DSA indexer（wq_b/wk/weights_proj + pool/topk） | 3 MM + pool/softmax/topk/gather | 10 |
| DSA 核心 attention | gather k/v + 2 BMM + softmax | 10 |
| MoE grouped GMM | 3（w1/w3/w2） | 12（dense block 0 个） |
| dense MLP MM | 3 | 4（pre 0–3） |
| mHC pre/post | 线性 1 + sinkhorn 逐元素 | (30+T)·2 |
| RMSNorm（含 gated） | 见算子清单 | 常规 |
| vision block（SDPA + MLP） | 3 MM + SDPA + 3 MM | 32 |
| vision stem/downsample/merger | conv2d×2 + 4 MM | 1 |

backward 次数按 autograd 规则对称；AC full 时 recompute 段整体翻倍（按 execution kind 分桶统计）。MoE GMM 受 `debug_force_load_balance` round-robin 保证 `num_tokens_per_expert` 静态（每 expert `m·S·topk/E`）。

## 10. 支持范围（首版声明）

| 特性 | 状态 |
| --- | --- |
| 单卡构造/前向/反向（debug/reduced 真实执行，full 仅 meta） | 支持 |
| FSDP / eFSDP | 支持 |
| TP（attention head 维 + 专家 inter 维 + vision 不切） | 支持；KDA/DSA head 维须被 TP 整除 |
| EP（routed experts Shard(0)） | 支持（2048 % EP == 0） |
| 模拟器单步 smoke | 已验证：debug/reduced 规格、FSDP（NGPU=2）、TP=2、EP=2、CP=2、TP2+EP2 组合、AC full 开关；融合 op 捕获数量与第 9 节账本一致 |
| ETP（EP+ETP 组合） | v1 fail fast |
| CP | 支持（kimi_k3/ar_llm 保守 all-gather 方案，仅无 TP 组合） |
| PP | v1 fail fast（loop 区 + hc_head 跨 stage 契约未定义） |
| AC none/full/selective | 支持（复用 apply_moe_ac；loop block 执行 T 次，AC 包裹每次迭代） |
| MXFP8（attention/MoE 矩阵乘计算侧量化） | 支持（`glm5_next_debug_mxfp8` / `glm5_next_reduced_mxfp8` flavor；CLI `--mxfp8-fqns` 覆盖） |
| compile / offload / MTP / video 通路 / 非均匀 vision 网格 | v1 fail fast 或显式关闭 |
| halting / kv_mirror | 不支持（见 §3），fail fast |

## 11. 模拟器建模契约（raw op 名）

| 模型点 | 捕获 raw op | 说明 |
| --- | --- | --- |
| KDA conv1d | `aten.convolution.default` / `aten.convolution_backward.default` | **不 shim**：与 kimi_k3 的 `ShortConvolution` 一致，fused qkv 单条 depthwise conv 走真实 aten 算子（`causal_conv1d` 不在已建模算子集合内，参考文档 §0.1 禁止自造名） |
| KDA 核心 | `triton_ascend_kernels.chunk_kda`（5 进 1 出）/ `chunk_kda_grad`（**6 进 `[q,k,v,g,beta,do]` 5 出同形梯度**） | 与 kimi_k3 同族；接口对齐 `hardware_shims/OP_INTERFACE_REFERENCE.md` §1 |
| DSA indexer 选择 | `aclnn.npu_lightning_indexer`（**3 进 `[query_idx 4D, key_idx 4D, weights 3D]` → 2 出 `[sparse_indices 4D int32, sparse_values]`**；无反向算子，indexer 冻结） | pool 打分 + topk 汇总；K=有效 topk（pool 级 `min(topk//kpool, cl)`）；接口对齐 §3 |
| DSA 核心 | `aclnn.npu_sparse_attn_sharedkv_metadata`（先行）+ **6 输入主 op** `[query, ori_kv, sinks, metadata, cmp_kv, cmp_sparse_indices]` → `[result, softmax_lse]`；grad **7 进 → 4 出 `[d_query, d_ori_kv, d_sinks, d_cmp_kv]`** | 接口对齐 §2；ori_kv 头折叠 per-head KV 约定 `nh×(k_dim+v_dim)`；sinks 记 `zeros[N]` f32 占位；ratio=1 时 cmp_kv 镜像 ori_kv 以携带 topk 索引 |
| mHC pre/post | DSv4 `npu_mhc_pre/post` shim（isinstance 复用） | 直接生效 |
| MoE 路由/GMM | aten 原生（router/topk/`_grouped_mm`） | 不合成融合名 |
| clamp-swiglu / RMSNormGated | aten 原生（v1） | 生产融合 converter 列后续项 |

shim 在模型构造 + 并行化完成后绑定（`apply_glm5_next_shims`），保持 FQN/DTensor/hook 不变。

## 12. MXFP8 FQN 清单（config_registry `_GLM5_NEXT_MXFP8_FQNS`，子串匹配）

```
attention.q_proj / k_proj / v_proj / o_proj            # KDA
attention.q_a_proj / q_b_proj / kv_a_proj_with_mqa / kv_b_proj / o_proj   # DSA（NoPE MLA）
attention.indexer.wq_b / weights_proj                  # DSA indexer
moe.experts                                            # routed experts（3D w1/w2/w3）
moe.shared_experts                                     # shared expert MLP
```

排除项（高精度）：`moe.gate`（router）、`layers.*.mlp`（前 4 层 dense）、`tok_embeddings`/`output`、`visual.*`、KDA 门控（f_a/f_b/g_a/g_b/b_proj）与 conv1d。vision tower 的 nn.Linear 已统一为 torchtitan `Linear` 以满足 `verify_module_protocol`。

## 13. 已知首版限制（Conditionally ready 声明范围）

- loop 语义按 §3 固定步数共享块建模；halting/动态深度/kv_mirror 不建模。
- DSA 走 gather 精确稀疏路径；生产 fused kernel 未接入（模拟器经 shim 记录融合名）。
- indexer 冻结（忠实 raw no_grad 语义）；indexer aux loss 未注入。
- video 通路、非均匀网格、MTP、fp8 存储、CP+TP、ETP、PP fail fast。
