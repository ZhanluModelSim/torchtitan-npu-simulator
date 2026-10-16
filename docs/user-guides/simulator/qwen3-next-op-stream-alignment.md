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

## 分组 GEMM 为融合 npu_grouped_matmul（管理方第二轮反馈修正）

管理方确认：实测 MoE 段 elementwise 每层仅 ~20 次、shape 为 `[16384,10,2048]` /
`[81920,512]` 等全 token 维——**逐专家 python 循环（zeros+cumsum+切片写，每层
255×4 全张量 op）在实测流中不存在**。`ep_dispatcher: eager` 指 dispatcher 路径
（permute/a2a/unpermute 走 eager index 算子），分组 GEMM 本身是融合
`npu_grouped_matmul`（`_GroupedMatmul` 自定义 autograd Function）。

sim 实现：`_SimGroupedMatmul`——前向 1 个 `npu.npu_grouped_matmul.default` 合成算子
（[M,K]×[E,K,N]→[M,N]），反向 2 个（输入梯度用转置权重；权重梯度
`split_item=3/group_type=2` 输出 [E,K,N]，实测还有 1 次 stack）。2 层配置验证：
gmm=12（2 层×[前向2+反向4]）、add/mul 全流 168（此前 ~24.6k）、experts 桶每层仅
残差 add + probs mul(×4) + silu/silu_bwd/mul [81920,512]，comm 数不变
（AG16/A2A8/RS7/AR3）。

## 无 dense 层修正（管理方确认）

Qwen3-Next 全系**没有 dense 层**：transformers `Qwen3NextConfig` 默认
`mlp_only_layers=[]`、`decoder_sparse_step=1`（实测树 modeling_qwen3_next.py:1075
的判定：不在 mlp_only_layers 且 num_experts>0 且命中 sparse_step → MoE）。
此前 flavor 里 `mlp_only_layers=[0]` 是未经核实的假设，已改为 `[]`。

修正后计数（48 层、ep=2、seq16k）：AG=246（root1 + embed2 + layers 48×2 +
experts 48×3 + head3）、A2A=192、RS=99、gmm=288、log_softmax=16；smoketest
4 层全 MoE：AG26/A2A16/gmm24。

## router 反向链修复（`cff5694`，管理方第三轮反馈）

管理方指出：shape 缺口（[16384,32]/[16384,8]/[16384,12288] 等）大部分属于优化器部分，已被融合算子覆盖，可忽略；真正差异是**反向 MoE gate/梯度逐元素乘，实测 aclnnMul/InplaceMul 共 8 次**（2 层 = 4/层）。

根因：debug 负载均衡分支用 `torch.full` 常量构造 routing_weights，**梯度路径被切断**——整个 router 反向链缺失：
- unpermute probs 加权（`u *= probs`）反向的 d(probs) mul：`[16384,10,2048]`
- norm_topk_prob 除法反向 mul：`[16384,10]`
- softmax 反向（`_softmax_backward_data`）

修复：仅强制 selected_experts 的值（解析式均匀 split 仍成立）；routing_weights 走真实
softmax → topk → div → cast 计算图。topk 的 values 保持图连通（meta 下值无关紧要），
indices 单独覆盖，因此**前向与实测逐算子一致，无需额外 gather**。

修复后 gate 路径反向 Mul/InplaceMul = 3/层（`[16384,10,2048]`×2 + `[16384,10]`×1）
+ shared_expert_gate 反向 1/层 = **4/层 × 2 = 8 次**，与实测精确吻合。
其余反向 mul（silu [81920,512]、down_proj、GDN norm 等）不变。

## 非 EP 分支融合 permute 猴补丁移除（管理方第四轮反馈）

管理方指出 FSDP 搬运缺 transdata/tensormove/transpose/splitV，归因于"仿真被
MoETokenPermute 融合算子代表，实测是 eager index copy/select"。核实结果：

- **2 层实测口径 preset（ep=2）一直是干净的 eager**（CSV 无任何融合 permute，
  index_select/index_copy_ 与实测 `fsdp2/distributed/expert_parallel/utils.py`
  的 eager_permute/eager_unpermute 逐算子一致），管理方归因对该流不成立。
- **问题确实存在，但在 smoketest/debugmodel（ep=1，非 EP 分支）**：
  `apply_qwen3_next_fusion_shims` 曾把 `_ff._token_permute/_token_unpermute`
  重绑为虚拟融合算子（`npu.npu_moe_token_permute.default` ×4 等），导致 ep=1
  流被单个融合 op 代表。已移除该猴补丁，非 EP 分支走本就存在的 eager 实现
  （argsort+index_select / zeros+index_add_）。修复后 smoketest 无融合算子，
  2 层 preset 计数不变。

为什么此前全量检查未发现：eefd16a 轮的修复与验证都基于 2 层实测口径 preset
（ep=2 → EP 分支，本就 eager）；猴补丁只影响非 EP 分支，且管理员检查 #1-#5
均在 2 层配置上验证，smoketest 流从未与实测 eager 语义对照——修复覆盖了一条
路径，另一条遗留旧表示，属验证范围缺口。

遗留（需管理方提供实测计数）：transdata/tensormove/transpose/splitV 是 NPU
aclnn 层对 view 类算子的物化内核（GDN qkvz/ba `torch.split`→splitV、attention
`chunk(2)`→splitV、`transpose.contiguous()`→transpose+tensormove、非连续布局
→transdata），我方 meta 采集只记录产生计算的 aten 算子（view 类零记录），
结构上无法自动出现；如需对齐须按实测计数在对应源点加合成记录。

# 适配全程回顾（初始实现 → 最终修改）

按时间线记录 qwen3_next 适配中遇到的每个问题、所做修改、解决了什么、以及为什么
一开始有问题/没发现——供下一次模型适配对照。

## 时间线与提交链

| 阶段 | 提交 | 内容 |
|---|---|---|
| 初始实现 | `5517f8c` | 模型包（GDN 线性注意力 + gated GQA MoE）+ simulator 六处接线 |
| | `4f1dee9` | 融合 attention/GLU 虚拟算子绑定 |
| | `fa1ed3d` | MoE 算子流镜像 Qwen3NextMoeExperts（融合 permute 时代） |
| | `1f4cb6c` | RMSNormGated → npu_rms_norm 融合核 |
| GDN 口径 | `d8b9536`→`6552d68` | eager torch 分解 → triton kernel 序列（12 种 kernel 逐个记录） |
| FSDP 对齐 | `5be2808`/`b198a6d` | fsdp_modules 三组计划 + 专家 eFSDP；融合 norm 精确序列 |
| 实测口径 | `eefd16a` | eager EP dispatcher 镜像 + 实测 fsdp2 config（**遗留非 EP 分支猴补丁**） |
| 检查 #4 loss | `4bc1460` | chunked loss（ChunkLoss，chunk=1024，前向求梯度） |
| 检查 #5 add/mul | `62ccfa3` | MoE 权重梯度累加修复（select → chunk+squeeze 视图） |
| 检查 #1 AG | `3647bfb` | reshard 语义 + LMHead 双调用 + 2 层实测口径 preset |
| 检查 #5 补充 | `1ab6906` | eager 循环 gmm → 融合 npu_grouped_matmul |
| 检查 #2 补充 | `e9f7e40` | 无 dense 层（mlp_only_layers=[]） |
| 检查 #5 再补充 | `cff5694` | router 反向链接回真实 softmax/topk 图（8 次 gate mul） |
| 检查 #2 搬运 | `ace2013` | 移除非 EP 分支融合 permute 猴补丁 |

## 问题清单：现象 → 修改 → 根因（为什么一开始有问题/没发现）

### 1. 初始表示以融合算子为中心
- **现象**：早期流把 permute/unpermute/swiglu/attention 各表示为 1 个融合 op。
- **修改**：逐步按实测改为逐算子 eager（permute/unpermute/swiglu），attention 保留融合。
- **为什么一开始有问题**：初始研读的是 mindspeed.ops 库（有融合核），误把"库有此核"
  等同于"实测走此核"。实测配置（ep_dispatcher=eager、无相关开关）才是唯一判据。

### 2. GDN 三种口径试错
- **现象**：整核 1 合成算子（elementwise 少 6 倍）→ eager torch fallback 全展开
  （elementwise 虚高）→ 逐 triton kernel 记录才对齐。
- **修改**：`gdn_shim.py` 按生产调用序记录 12 种 triton kernel（前向 8 + 反向 9 + eager 小算子）。
- **为什么没一开始就做对**：没先确认参考的**符号绑定**（`use_triton_gdn` 决定
  `self.chunk_gated_delta_rule` 指向 triton 实现还是 torch fallback）。

### 3. reshard_after_forward 根单元语义（检查 #1，缺 7 个 AG）
- **现象**：我方全流 AG 9 个，实测 16 个（2 层口径）。
- **修改**：fsdp_modules 三组（layers/embed/lm_head）共享 `reshard_after_forward=True`
  → 每单元反向各 1 次 pre-bwd AG；eFSDP(experts) 不传参 → torch 默认 True；根 wrap
  不传 → torch 默认 None → **根单元 False**（final norm 被根兜底）。AG 清单 =
  root1+embed2+layers4+experts6+head3=16。
- **为什么没发现**：假设了统一的 reshard 策略；没读 mindspeed `fully_shard_parallel.py`
  根 wrap 的实参——"不传参→None→False"是 torch 语义细节，代码里一行都没写。

### 4. LMHead 双调用死分支（同检查 #1）
- **现象**：lm_head 只有 2 次 AG，实测 3 次。
- **修改**：镜像实测 `qwen3_next.py:90`——无条件先 `self.lm_head(h)` 算全量 logits
  （loss_ctx 存在时被覆盖=死分支，但 mm/AG 进流），再带 loss_ctx 调一次；模型返回
  预计算 loss，trainer 侧 loss_fn 透传。
- **为什么没发现**：我们实现了"干净"的单调用版本；死分支只有读实测 forward 才看得到。

### 5. MoE 权重梯度累加（检查 #5）
- **现象**：eager gmm 反向按 `weights[i]` select 逐专家产生独立梯度 → add/mul 数千个。
- **修改**：chunk+squeeze 视图链，autograd 坍缩为单 cat（实测同构）。
- **为什么一开始有问题**：视图语义下 select 与 chunk+squeeze 数值等价（meta 无值），
  但**梯度累积行为完全不同**——meta 仿真不能只对前向数值等价，要对反向图等价。

### 6. 分组 GEMM 融合 vs eager 循环（检查 #5 补充）
- **现象**：逐专家 python 循环产生全张量 add/select_backward/slice_backward。
- **修改**：`_SimGroupedMatmul`（npu_grouped_matmul：前向 1 op，反向 2 op）。
- **为什么没发现**：`ep_dispatcher: eager` 字面误导——它只指 dispatcher（permute/a2a/
  unpermute），不指 GMM 实现；实测 GMM 是融合核。

### 7. 无 dense 层假设（检查 #2 补充）
- **现象**：48 层口径 layer0 无 experts 算子，与管理方反馈矛盾。
- **修改**：`mlp_only_layers=[]`（HF/实测默认，decoder_sparse_step=1 → 全部层 MoE）。
- **为什么一开始有问题**：`mlp_only_layers=[0]` 是**未核实的假设**——没查 config
  默认值，也没问管理方。

### 8. router 反向链被切断（检查 #5 再补充，8 次 gate mul）
- **现象**：实测反向 gate 路径 4 mul/层 ×2 层=8，我方 0。
- **修改**：debug 负载均衡只强制 selected_experts 的**值**（indices）；routing_weights
  走真实 softmax→topk→div→cast 图（topk values 保图连通，无需 gather）。
- **为什么一开始有问题**：debug 分支用 `torch.full` 常量构造权重——**值强制顺带切断
  了梯度路径**，整个 router 反向链（d(probs) mul、div/softmax 反向）从未入流。此前
  检查都聚焦前向计数与 comm，没人对反向 mul 计数。

### 9. 非 EP 分支融合 permute 猴补丁（检查 #2 搬运）
- **现象**：管理方称"仿真被 MoETokenPermute 融合算子代表，实测是 eager index"。
- **核实**：2 层 preset（ep=2）一直是纯 eager（归因对该流不成立）；**但 smoketest
  （ep=1，非 EP 分支）确实有此问题**——`apply_qwen3_next_fusion_shims` 把
  `_token_permute/_token_unpermute` 猴补丁成融合虚拟算子。
- **修改**：移除猴补丁，非 EP 分支走 eager helpers。
- **为什么没发现**：`eefd16a` 的修复与验证都基于 2 层 preset（EP 分支，本就 eager）；
  猴补丁只影响非 EP 分支；管理员检查 #1–#5 全在 2 层配置上验证，smoketest 流从未
  与实测对照。**修复覆盖一条路径、另一条遗留旧表示，属验证范围缺口。**

### 10. 遗留未解（需管理方输入）
- transdata/tensormove/transpose/splitV：NPU 对 view 类算子的 aclnn 物化内核，meta
  采集只记录产生计算的 aten 算子（view 类零记录），结构上无法自动出现；对齐需管理方
  提供实测计数后在源点加合成记录。
- 实测 profiler 中的优化器算子（aclnnMul 等）多数被融合优化器覆盖——对表前先请管理方
  区分反向/前向与优化器来源，避免追错方向。

## 元经验（为什么问题总是这几类）

1. **"库有此核"≠"实测走此核"**：融合/eager 边界逐算子以实测配置判定。
2. **参考代码的怪癖必须镜像**（死分支双调用、硬编码 reduce fp32）：别实现"更干净"的版本。
3. **未核实的假设是最大错误源**（mlp_only_layers、GDN 绑定、根单元 reshard）：
   每个 config 默认值、每个实参缺省都要查源码或问管理方。
4. **值强制不能切断梯度**：meta 仿真要前向数值与反向图**双重等价**。
5. **多路径模型每条路径都要验证**：修一处跑全部 config 档位（smoketest + 实测口径
   preset），猴补丁/模块级重绑会静默切换另一条路径的表示。
6. **验证范围随管理员反馈扩大**：早期只对前向+comm 计数，后期反向 mul、搬运语义
   都要进清单——每轮反馈后把新口径补进本文档。

## 布局算子节点缺失（管理方第五轮反馈：图捕获层甄别）

管理方指出"图捕获层没生成这些布局算子的节点"——**甄别结论：说法正确**。机制：

1. 捕获层（`OpDispatchCapture.__torch_dispatch__`）其实**记录了**所有 view 类 op，
   并标注 `metadata_view=True`（dispatch_capture.py:835）；
2. 但 `fold_metadata_views`（trainer.py:373）在导出前把它们折叠掉——设计注释称
   "not launched device kernels"，该假设对 CPU/meta 成立，**对 NPU 不成立**：
   split/chunk 在 NPU 上物化为 aclnnSplitV 内核，转置后 contiguous 物化为
   TransData/TensorMove；
3. 因此我方 CSV 中 split/transpose/chunk/view 全为 0，实测侧却有——层级假设错误，
   不是 MoETokenPermute 融合算子归因（那只在旧流/ep=1 成立）。

### 修复：源点合成记录（新提交）

新增 `models/qwen3_next/layout_ops.py` 的 `record_split_v`：在真实 view op 之后
用 `run_synthetic_op` 记录 `npu.aclnn_split_v.default`（canonical `split_v`，
成本模型按 `_data_move` 计费），输入/输出用真实张量 → 依赖边与 liveness 一致。

| 源点 | 记录 | 2 层口径频次 |
|---|---|---|
| GDN `torch.split(qkvz/ba)`（gated_deltanet.py） | split_v ×2 | [16,768]→4 段 + [16,4]→2 段 |
| gated attention `chunk(q_and_gate,2)`（attention.py） | split_v ×1 | [16,512]→2×[16,256] |
| swiglu `chunk(2)`（feed_forward.py，EP eager 路径） | split_v ×1/层 | [81920,1024]→2×[81920,512] |
| chunked loss 3 个 `torch.split`（loss.py） | split_v ×3 | [1,16384,·]→16 chunks |

2 层口径合计 **split_v ×8**；smoketest ×10。comm/gmm/elementwise 计数不变。

### transdata/tensormove/transpose 的映射说明

这些内核对应的物化点在我方流中**已存在**，只是以 aten 级名字记录
（clone/contiguous 不在折叠名单里）：

- GDN split 输出的非连续 reshape 物化 → `aten.clone` ×9（token_mixer 桶，
  如 [1,16384,16,256]/[1,16384,16,2]）
- attention 出口 `transpose(1,2).reshape().contiguous()` 物化 → aten.clone
- NPU 侧这些显示为 TransData/TensorMove/Transpose 内核——按 shape 对表即可。

### 未覆盖（如管理方实测有更多再补）

- 非 EP 路径 swiglu 走融合 `npu_swiglu`（无 chunk）→ 无 split_v；
- FSDP 运行时参数搬运（mindspeed fsdp2 unshard/unpad）：AG 输入通常已连续
  （contiguous 为空操作），暂无内核；如实测 profiler 在 AG 前有 TensorMove 再加。
