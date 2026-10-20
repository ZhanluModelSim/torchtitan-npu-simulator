# Qwen3-Next 算子账本（OP_REFERENCE）

> 依据《新模型接入开发流程》§4 与《最终验收规范》§4.3 编写。账本数据来自
> `simulator_output/qwen3_next_80b_a3b_2layer`（80B 维度 2 层：1 GQA + 1 GDN，
> 均 MoE，ep=2 实测口径，AC=none，seq 16384，world 2）的
> `kernel_summary/rank_0.csv`。模型特有算子保持融合边界（验收 4.3 红线），
> 反向次数按 execution kind 分列。

## 1. 模型特有算子（MindSpeed triton GDN 序列，真名 + shape-only）

每 GDN 层一次前向的捕获序列（1 GDN 层的实测值 = 下表；bwd 列为该层反向）：

| raw op | fwd | bwd | 模块 FQN | 备注 |
| --- | --- | --- | --- | --- |
| mindspeed.ops.triton.l2norm_fwd | 2 | 2 | layers.1.token_mixer | q/k 归一化（fwd 2 = q,k；bwd 2 = 对应 grad 侧） |
| mindspeed.ops.triton.l2norm_bwd | — | 2 | 同上 | dq/dk 梯度归一化 |
| mindspeed.ops.triton.chunk_local_cumsum | 1 | 2 | 同上 | β/g 门控前缀和（bwd 2 = g,k 路径） |
| mindspeed.ops.triton.chunk_scaled_dot_kkt_fwd | 1 | 1 | 同上 | K Kᵀ（A 矩阵，f32） |
| mindspeed.ops.triton.solve_tril | 1 | 1 | 同上 | 下三角逆（zeros_like bf16 填充入流） |
| mindspeed.ops.triton.recompute_w_u_fwd | 1 | 2 | 同上 | W/U 重计算 |
| mindspeed.ops.triton.chunk_gated_delta_rule_fwd_h | 1 | 2 | 同上 | 分块状态传递（训练无 final_state） |
| mindspeed.ops.triton.chunk_fwd_o | 1 | 1 | 同上 | 输出块计算（cat padding 分支 seq%64==0 不触发） |
| mindspeed.ops.triton.chunk_bwd_dv_local | — | 1 | 同上 | dv 局部 |
| mindspeed.ops.triton.chunk_gated_delta_rule_bwd_dhu | — | 1 | 同上 | dh/dv 状态梯度 |
| mindspeed.ops.triton.chunk_bwd_dqkwg | — | 1 | 同上 | dq/dk/dw/dg（含 sum(0) 后处理） |
| mindspeed.ops.triton.prepare_wy_repr_bwd | — | 1 | 同上 | Wᵧ 表示梯度 |

包装层 eager 算子（同 FQN，2 层口径 GDN 桶）：`aten.empty_like` ×28、
`aten.empty` ×8、`aten.new_empty` ×3、`aten.zeros_like` ×1、`aten.clone` ×37、
`aten.sum` +1、cast（`aten._to_copy`）fwd +1 / bwd +3 每层（f32 内核输出 →
包装层 `.to()` 下转，`aten._to_copy.default` 即 `.to(dtype)` 的真名）。
逐阶段清单见 `docs/user-guides/simulator/qwen3-next-op-stream-alignment.md`
"GDN triton 包装层 eager 算子镜像"节。

## 2. 注意力与通用 NPU 融合算子

| raw op | fwd | bwd | FQN | 次数推导 |
| --- | --- | --- | --- | --- |
| fusion_attention | 1 | 1 | layers.0.token_mixer | 每 GQA 层 1（`torch_npu.npu_fusion_attention`，BNSD） |
| fusion_attention_grad | — | 1 | 同上 | 反向虚拟对 |
| npu.npu_rotary_mul.default | 2 | 2 | 同上 | 每 GQA 层 1（q,k）+ 反向 2 |
| npu.npu_rotary_mul_backward.default | — | 2 | 同上 | |
| npu.npu_rms_norm.default | 8 | 7 | layers.*.{attention_norm,ffn_norm,token_mixer.q_norm/k_norm,token_mixer.norm} | 2 层 = 层 norm×4 + q/k norm×2 + GDN 门控 norm×1 + 最终 norm |
| npu.npu_rms_norm_backward.default | — | 8 | 同上 | |
| npu.npu_grouped_matmul.default | 4 | 12 | layers.*.moe.experts | ep=2 eager dispatcher 口径；routed 专家 GMM，次数由固定路由的 token 分布决定（debug load-balance 解析分布，权重走真实反向图） |

## 3. MoE EP dispatcher 流（ep>1，MindSpeed eager 镜像）

每 MoE 层：count all-gather → a2a dispatch（`comm.all_to_all`）→ 逐专家
matmul → a2a combine。2 层 ep=2：`comm.all_to_all` fwd 4 / bwd 8，
count AG fwd 2 / bwd 2。permute/unpermute 为 eager index 算子
（`index_select`/`copy_`，实测树口径，无融合虚拟算子）。

非 EP 路径（ep=1）：`npu.npu_moe_token_permute/unpermute` + `npu.npu_swiglu`
融合虚拟算子对（smoketest ep=1 历史口径）；swiglu 前反向在两路径均已核实
入流（EP eager：silu+mul+silu_backward；非 EP：npu_swiglu 融合对）。

## 4. Chunked LM loss（MindSpeed ChunkLoss 镜像）

每 chunk（CHUNK_LOSS_SIZE=1024，seq 16384 → 16 chunks）：`F.linear` +
`.float()` + `cross_entropy(sum)/alpha`，前向内 `torch.autograd.grad`
回填 `copy_`/`add_` 梯度缓冲；反向仅 rescale。LM head 调用 2 次/前向
（首次全 logits 死值 + loss_ctx 调用），FSDP 单元 3 次 all-gather
（fwd×2 + bwd×1）——与实测一致。

## 5. MXFP8 量化算子（qwen3_next_2l_mxfp8 配置）

| raw op | fwd | bwd | FQN 范围 |
| --- | --- | --- | --- |
| npu.npu_dynamic_mx_quant.default | 26 | 52 | §契约量化范围（mixer 投影 + shared expert；routers 排除） |
| npu.npu_quant_matmul.default | 13 | 26 | 同上 |

## 6. 通信事件（rank-local 语义，2 层 ep=2）

| 类型 | fwd | bwd | opt | 归属 |
| --- | --- | --- | --- | --- |
| comm.allgather | 8 | 6 | — | FSDP/eFSDP unshard + LM head ×3 + EP count AG |
| comm.all_to_all | 4 | 8 | — | EP dispatch/combine（每层 fwd 1 dispatch + 1 combine） |
| comm.reduce_scatter | — | 5 | — | FSDP 梯度 RS + eFSDP |
| comm.allreduce | 1 | — | 1 | router 相关 + optimizer 阶段 |

字节数按 rank-local tensor shape 推导并随 CSV 输出；依赖回放（AG 在首个
消费者前、RS 在最后梯度生产者后）由模拟器公共调度能力提供。

## 7. 湛卢对接提示（黄区工作，此处仅登记）

- 全部模型特有算子使用**已建模真名**（`mindspeed.ops.triton.*`、
  `fusion_attention`、`npu.npu_*`），无自造名。
- 未命中闭环（`ZHANLU_DISABLE_PRUNE_OP=1`）属黄区流程，见
  `torchtitan_training_modeling_guide.md` §8。
