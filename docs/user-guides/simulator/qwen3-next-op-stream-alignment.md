# Qwen3-Next 仿真算子口径对齐说明

与 NPU 实测负载对比后发现的算子流偏差及修正记录。参考准绳：MindSpeed-LLM fsdp2 版
`mindspeed_llm/fsdp2/models/qwen3_next/`（commit `6082e0de`）。修正范围 commit
`4f1dee9` → `d8b9536`。

## 修正清单

| Commit | 修正 | 对照实测问题 |
|---|---|---|
| `4f1dee9` | GatedAttention 抽 `_attention_core` 缝，仿真态记录 `fusion_attention` 融合虚拟算子 | flash attention 未融合 |
| `fa1ed3d` | MoE 严格 mirror MindSpeed `Qwen3NextMoeExperts`（见下） | 缺 index_copy、gather/index_add 偏少、scatter 漏捕获 |
| `1f4cb6c` | `RMSNormGated` 绑定融合 `npu_rms_norm` 边界（`run_meta_rms_norm`） | 搬运/copy 类算子差距大 |
| `eefe481` | router topk 归一化去掉 `+1e-20`（参考无 epsilon） | 严格逐算子对齐 |
| `d8b9536` | GDN 核心解除合成算子屏蔽，eager 分解开放捕获（实测走 MindSpeed 文件内 `torch_chunk_gated_delta_rule` torch eager 路径） | elementwise 个数明显少于实测 |

### MoE 生产算子流（`fa1ed3d`）

- 专家权重融合：`gate_up_proj` `[E*h, 2i]` + `down_proj` `[E*i, h]`，每层 GMM 3→2 次
  （EP 分组下扁平权重 Shard(0) 切分后，本地专家数按 `gate_up.shape[0] // hidden_size` 推导）
- token 分发改为生产调用序：`npu_moe_token_permute` → GMM → `npu_swiglu` → GMM →
  `npu_moe_token_unpermute`，前反向 + grad 全套虚拟算子（`qwen3_next_converter.py`）
- router：去掉 mm 前 float 强转，`F.softmax(dtype=torch.float32)`，topk 归一化无 epsilon
- `op_mapping.py` 补 index/scatter 族（index/gather/index_add_/index_put/scatter*/
  index_copy/sort/histc/cumsum/topk）——此前 eager scatter 算子已捕获但被归一化成
  `unknown`，呈现为"漏捕获"

### GDN triton kernel 序列（`d8b9536` 已被 `6552d68` 取代）

GDN 核心的生产绑定是 `use_triton_gdn=True` →
`mindspeed_llm/tasks/models/transformer/chunk_gated_delta_rule.py`（triton-ascend
实现，**每阶段一个 kernel**）。文件内的 `torch_chunk_gated_delta_rule` 只是 fla/triton
不可用时的 fallback。曾先后试过"整个核心 1 个合成算子"（elementwise 过少）和
"eager 分解全展开"（elementwise 虚高 6 倍），均与实测不符；最终口径为**逐 kernel 记录**
生产调用序：

- 前向（8 kernel）：`l2norm_fwd`×2（q,k）→ `chunk_local_cumsum` →
  `chunk_scaled_dot_kkt_fwd` → `solve_tril` → `recompute_w_u_fwd` →
  `chunk_gated_delta_rule_fwd_h` → `chunk_fwd_o`
- 反向（9 kernel + eager）：`recompute_w_u_fwd` → `chunk_gated_delta_rule_fwd_h` →
  `chunk_bwd_dv_local` → `chunk_gated_delta_rule_bwd_dhu` → `chunk_bwd_dqkwg` →
  `prepare_wy_repr_bwd` → eager `add_`×2 → `chunk_local_cumsum`(reverse) →
  `l2norm_bwd`×2 → 输出 cast×5

效果（80B，rank_0）：总节点 17,031；elementwise 3,152（eager 展开时 155,072）；
triton kernel 记录 900 个（12 种 × 36 层 × 前向/重算/反向倍数）。
实现：`torchtitan_npu/simulator/hardware_shims/gdn_shim.py`。

## 效果（80B a3b，world-size 8，rank_0）

| 指标 | 对齐前 | 对齐后 |
|---|---|---|
| 总节点数 | ~25,000 | **321,879** |
| elementwise | ≈0（被合成算子隐藏） | **155,072**（GDN 内 152,844：mul 54K / add 53K / bmm 46K / exp 14K） |
| copy 类 | 4,084 | 46,708（GDN chunk 循环 in-place 写 / contiguous / float cast） |

MoE 区（逐算子 = MindSpeed 生产口径）：`moe_token_permute` 94+grad47、`npu_swiglu`
94+47、`moe_token_unpermute` 94+47、`grouped_mm` 376、`fusion_attention` 24；手写
index/index_add/gather/sort 全部消失。

## loss / softmax 对齐状态

- loss：两侧均为 `F.cross_entropy` 分解（`_log_softmax`×1 + `nll_loss`×1 + 反向各 1 +
  logits float cast），逐算子一致；唯一差异是 reduction 属性（`sum` vs HF 默认 `mean`），
  不影响算子个数/类型
- router softmax：`F.softmax(dim=1, dtype=float32)` / topk / 归一化 / `.to(dtype)` 与
  MindSpeed 逐行一致（`eefe481` 后）
- attention softmax：融合在 `fusion_attention` 内，与实测 `npu_fusion_attention` 同口径

## 给成本模型的口径建议

1. `view` / `_unsafe_view` 是零搬运元数据操作，不应计入搬运类（全局 1151 次）
2. `_to_copy`（dtype cast）与 `clone`/`copy_` 应分开，按 tensor 字节数走带宽模型，
   而非统一 "copy" 单价
3. 融合算子（`moe_token_permute`/`moe_token_unpermute`/`npu_swiglu`/`fusion_attention`/
   `npu_rms_norm`）必须有条目，否则对应区块贡献零耗时
4. GDN eager 分解算子按常规 eager 算子计价即可（shape 元数据齐全）

## FSDP 单元划分与 all-gather 对齐（`5be2808`）

MindSpeed fsdp2 默认计划：`fsdp_modules=['model.layers.{*}', 'model.embed_tokens',
'lm_head']`（80B `tie_word_embeddings=false`，lm_head 独立单元），专家在 EP>1 时走
EFSDP（`ep_fsdp_modules`，edp mesh 上 Shard(1)），`recompute_modules` 仅 layers。
对应结构（EP=2、4 层、3 MoE）：embed 1 + [norm,output] 1 + 4 层×2（前向+重算）+
3 experts×2 = **16 个 all-gather，与实测一致**。实现：`parallelize.py` 自带
`_apply_fsdp`/`_apply_expert_fsdp`（首/末单元 `reshard_after_forward=False` → 仅一次
前向 AG；layers/experts True → 前向+重算 AG；EP=1 不做 EFSDP，与 MindSpeed 门控一致）。

## 融合 norm 的精确算子序列（`b198a6d`）

MindSpeed（`use_fused_rmsnorm=True`）的 `Qwen3NextRMSNorm` =
`npu_rms_norm(x.float(), 1.0 + weight.float(), eps)[0].type_as(x)`——float cast、
`1.0+weight` 偏置加法、输出 cast 都是实测流中的真实 eager 算子（80B 每步 ~1.2k 个
小算子，裸融合边界会藏掉）。已按 qwen3_next 门控绑定 `NPURMSNorm` 实例到该序列；
GDN 门控 norm 升级为 float32 全序列。同时配置对齐官方示例：`reduce_dtype=bfloat16`、
debug preset 改 full 重算（`recompute_modules` 语义）。

## 验证

smoketest EP=1/2/8、`kimi_k3_smoketest` 回归、80B 全量捕获均通过；与实测的对表方法：
按 `module_path` 分解 `kernel_summary/rank_0.csv` 算子计数，逐模块比对。

## EP eager dispatcher 与融合 permute 的分界（`eefd16a`）

实测 ep=2 且 `moe_grouped_gemm=True` 走 **eager dispatcher**（`expert_parallelize_modules`
整专家 Shard(0) + `_ep_dispatcher` 前向替换：bincount→计数 AG→argsort/index_select→
all_to_all_single→argsort/index_select→逐专家 matmul 循环→silu*up→gmm→unpermute
（zeros+index_copy_）→a2a→probs 加权 unpermute）。实现要点：

- 专家权重 `gate_up_proj.data.chunk(ep_size,0)[ep_rank]` 后 `Shard(0)` 切到 ep mesh
  （连续行=整专家）；eFSDP 在 edp mesh 上 `Shard(1)` + `reshard_after_forward=False`
  （实测每单元仅 1 次前向 AG）。
- `aten.bincount` 无 Meta kernel——在 `meta_env` 注册 fake（返回 `[minlength]`）；
  `repeat_interleave` 需显式 `output_size=`；meta 上 `.to('cpu')` 抛错，D2H 切分用
  合成 `aten._to_copy.default` 记录；a2a 必须 autograd Function 包装。
- **ep=1 保留融合 `npu_moe_token_permute/unpermute`**（实测 ep=1 同为融合）。
  实验佐证融合算子吸收算术：同 preset 下 ep=1 全流 add/mul=236 个，ep=2 eager=283 个，
  +47 个即融合 permute/unpermute 内部吸收的 probs 加权 mul 与 unpermute 求和。

## 分块 loss（ChunkLoss，实测 `chunk_loss_size: 1024`）

实测语言模型 loss 不在 `lm_head.forward` 里：`_build_chunk_loss` 令 model 返回
hidden（lm_head 模块仍被调用，[norm,output] FSDP 单元照常 AG，`reshard=False` 保
权重全量），`ChunkLoss.apply` 在**前向**逐 chunk（1024 token）做
`torch.func.grad_and_value(calculate_lm_loss)`：`F.linear`→`.float()`→
`F.cross_entropy(sum, ignore_index)`→`/alpha`（alpha=有效 token 数），梯度经
`copy_`/`add_` 回填预分配缓冲（grad_inputs chunk copy_、grad_weight add_、loss add_），
backward 仅按 trainer 预除的 grad_output 缩放。16k 序列 → 16 chunk → 16 个
log_softmax/nll_loss、16 次 `[1024,151936]` 的 lm_head matmul。

sim 实现（`models/qwen3_next/loss.py`）：`build_loss_fn` 工厂在 post-build shim 中把
`model.output` 绑定为透传并注册权重；`_SimChunkLoss` 用 `enable_grad` +
`torch.autograd.grad` 在 Function forward 内求 chunk 梯度（不引 functorch，meta 捕获
风险）；backward 返回保存梯度×grad_output。验证：80B（ep2/16k）log_softmax /
`_log_softmax_backward` / nll fwd+bwd 各 16、shape `[1024,151936]`，smoketest 1 chunk，
EP=1/2 comm 数（AG10/16、A2A0/12、RS6/9）无回归。

## add/mul 对账（管理员检查 #5）

大桶来源与归属（80B，47 MoE+1 dense）：

- 残差 add `[1,16384,2048]`：attn 2/层×48 + moe 47
- norm 权重 add `[2048]×48 / [256]×12`：`1.0+weight.float()`（实测流中的真实 eager
  算子，非融合内部）
- 反向双分支重组 add：gate/up、q/k/o in_proj、GDN token_mixer 的
  `d_a@Waᵀ + d_b@Wbᵀ`（silu_bwd→mm→mm→add）
- eager dispatcher 逐专家循环 add `[256,2048,1024]` 等 ×11985（47 层×255 swap）
- probs 加权 mul `[16384,10,2048]×[16384,10,1]` fwd ×47 + mul_ bwd ×47
- loss：add_ scalar ×16、add_ `[151936,2048]`×16（grad_weight）、bwd 缩放
  mul `[]×[151936,2048]` 与 `[]×[1,16384,2048]` 各 1

待复核（需实测 trace）：管理方若在 rms_norm 处观察到 add/mul 差异，最可能是实测侧
把 `1.0+w` 吸进融合边界（实测前向无 `[2048]` add）——当前序列是按实测 CSV 推导的，
以实测 grep `aten.add` shape `[2048]` 为准。

## 实测对齐配置与 all-gather 全清单（`62ccfa3` 后）

管理方实测口径修正：**完整 80B 维度、仅层数改为 2（1 层 full-attention + 1 层
linear-attention，均 MoE）**，ep=2。对应 preset `qwen3_next_80b_a3b_2layer`。
关键语义（读实测树确认）：

- `reshard_after_forward`：fsdp_modules 三组（layers/embed_tokens/lm_head）共用
  `parallel_args.reshard_after_forward`，**默认 True**——每个单元反向各 1 次
  pre-bwd AG；eFSDP `fully_shard(experts)` 不传该参 → torch 默认 True → 专家权重
  反向也有 AG。根 wrap `fully_shard(model, {mesh, ignored_params, mp_policy})`
  不传 → torch 默认 None → **根单元 False**（final norm 归根单元：前向 1 次 AG、
  反向无）。final norm 不在 fsdp_modules 里，由根单元兜底捕获（[2048]/8=[256]）。
- eFSDP 精度策略在 MindSpeed 中硬编码 `MixedPrecisionPolicy(param=bf16,
  reduce=fp32)`，与主 FSDP 的 `reduce_dtype` 无关（专家梯度 RS 为 fp32）。
- **LMHead 前向被无条件调用两次**（`qwen3_next.py`：先 `self.lm_head(h)` 算全量
  logits——loss_ctx 存在时结果被覆盖（死分支，但 mm/AG 都进流），再
  `self.lm_head(h, loss_ctx=loss_ctx)` 在模块内跑分块 loss）→ lm_head 共 3 次 AG
  （前向 2 + 反向 1）。模型返回预计算 loss，trainer 侧 loss_fn 为 passthrough。

2 层配置的 16 个 AG（与实测一致）：

| # | phase | 单元 | shape（元素） | 说明 |
|---|-------|------|--------------|------|
| 1 | fwd | root(norm) | [256]→[2048] | final norm 归根单元，仅前向 |
| 2 | fwd | embed | [38895616]→[311164928] | 151936×2048 全表 |
| 3 | fwd | layer0 | [3934784]→[31478272] | full-attn 层非专家参数 |
| 4 | fwd | layer0.experts | [201326592]→[805306368] | eFSDP 权重（Shard(1)/4）|
| 5 | fwd | layer0.experts | [512]→[1024] | dispatcher 计数 AG |
| 6 | fwd | layer1 | [4741656]→[37933248] | linear-attn 层 |
| 7/8 | fwd | layer1.experts | 同 4/5 | |
| 9 | fwd | output | [38895616]→[311164928] | 死 logits 调用 unshard |
| 10 | fwd | output | 同上 | loss_ctx 调用 unshard |
| 11 | bwd | output | 同上 | pre-bwd |
| 12-16 | bwd | layer1 / layer1.experts / layer0 / layer0.experts / embed | 同各自前向 | reshard=True 的反向重 unshard |

校验数：`log_softmax`/nll 各 16（16k/1024 chunk）、全量死 logits mm 1 个 +
chunk mm 16 个；A2A=8（2 层×4）、RS=7（root1+embed1+layers2+experts2+head1）。
