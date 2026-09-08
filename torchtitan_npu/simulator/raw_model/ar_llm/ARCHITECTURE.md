# DeepSeekV4-Sparse: 50T & 100T Model Architecture

> DeepSeekV4 Pro backbone extended with:
> - **CSA / HCA / KDA** tri-attention (1:1:4 per 6-layer unit)
> - **LatentMoE** with `moe_latent_dim=7168` bottleneck
> - **MoR (Mixture of Recursions)** with cycle KV cache sharing
> - **Engram** conditional memory (~10% of params)
> - **Expert-choice MoR** routing on 5% of CSA/HCA layers
> - Low-rank Q/KV/O (MLA-style), mHC hyper-connections
> All key dims aligned to powers of 2 (2^N)

---

## 1. Layer Structure (6-layer unit, repeated 10× for 60 layers)

```
Unit layout: [CSA, HCA, KDA, KDA, KDA, KDA]
CSA : HCA : KDA = 1 : 1 : 4
Attention : MoE = 1 : 1 (every layer has both)
```

| Position | Layer | Attention | MoR routing |
|----------|-------|-----------|-------------|
| 0 | CSA | Compressed Sparse | token-choice (95%) / expert-choice (5%) |
| 1 | HCA | Heavily Compressed | token-choice (95%) / expert-choice (5%) |
| 2-5 | KDA | Kimi Delta (linear) | token-choice |

**Total**: 60 layers = 10 units × 6 layers/unit
- CSA layers: 10
- HCA layers: 10
- KDA layers: 40

---

## 2. MoR (Mixture of Recursions) — Cycle KV Sharing

```
num_layers = 60 = base_depth × num_recursion = 12 × 5

Recursion 1: L0-L11   (writes KV cache slots 0-11)
Recursion 2: L12-L23  (reuses slots 0-11)
Recursion 3: L24-L35  (reuses slots 0-11)
Recursion 4: L36-L47  (reuses slots 0-11)
Recursion 5: L48-L59  (reuses slots 0-11)

→ KV cache only stores 12 layers' KV (not 60)
→ 5× KV compression from MoR alone
```

- `base_depth = 12` = 2 complete 6-layer units
- `num_recursion = 5` (within Recursion paper's 2-5 range)
- `sharing = "cycle"`: `layer_idx % base_depth` maps to cache slot

**Expert-choice layers** (`mor_update_cache=True`): only scatter-update selected tokens' KV into cache, further compressing cache writes.

---

## 3. Expert-Choice vs Token-Choice Routing

| Aspect | Token-choice (95%) | Expert-choice (5%) |
|--------|-------------------|-------------------|
| Direction | Token → selects experts | Expert → selects tokens |
| Capacity | Unlimited (top-K) | Capacity-constrained |
| Aux loss | Load-balance loss | Sampling loss (uniformity) |
| KV update | All tokens write | Only selected tokens write |
| # layers | 57 | 3 |

5% of 60 layers = 3 expert-choice layers, distributed across depth (shallow / mid / deep). Only CSA/HCA layers can be expert-choice; KDA always token-choice.

---

## 4. Shared Backbone (50T and 100T)

| Parameter | Value | Power of 2 |
|-----------|-------|------------|
| hidden_size | 16384 | 2^14 |
| num_layers | 60 | — |
| num_attention_heads | 128 | 2^7 |
| head_dim | 256 | 2^8 (noPE 192 + RoPE 64) |
| vocab_size | 524288 | 2^19 (4× expansion) |

| Low-rank | Value | Power of 2 |
|----------|-------|------------|
| q_lora_rank | 4096 | 2^12 |
| qk_rope_head_dim | 64 | 2^6 |
| qk_nope_head_dim | 192 | derived |
| kv_lora_rank | 1024 | 2^10 |
| o_groups | 32 | 2^5 |
| o_lora_rank | 4096 | 2^12 |

| LatentMoE | Value | Power of 2 |
|-----------|-------|------------|
| num_shared_experts | 2 | 2^1 |
| moe_intermediate_size | 4096 | 2^12 |
| moe_latent_dim | 7168 | — |
| num_experts_per_token | 16 | 2^4 |

| MoR | Value |
|-----|-------|
| mor_sharing | cycle |
| num_recursion | 5 |
| base_depth | 12 |
| mor_expert_ratio | 5% |
| mor_expert_capacity | 1.0 |

---

## 5. KDA (Kimi Delta Attention)

Linear attention with **delta-rule** state update:
```
State S ∈ [H, d_state, d_state]  (fixed buffer, not learnable)

Per token t:
  q, k, v = W_q x, W_k x, W_v x
  α = σ(W_α x)        # read gate [H, d_k]
  β_erase, β_write = σ(W_β x)  # double gate [H, d_state]

  Delta rule:
    S_t = S_{t-1} × (1 - β_erase) + β_write × (v - S_{t-1} k) k^T

  Read:
    o = α ⊙ (S_t q)

KV cache: only fixed state S, no token-wise KV → constant memory w.r.t. seq_len.
```

| KDA Param | 50T | 100T |
|-----------|-----|------|
| d_state | 256 (2^8) | 512 (2^9) |
| d_k (= d_v) | 128 (2^7) | 256 (2^8) |
| d_a | 3 | 3 |
| α gate | [128, 128] | [128, 256] |
| β gate | 2 × [128, 256] | 2 × [128, 320] |

---

## 6. LatentMoE Expert Structure

```
Input:  d → moe_latent_dim           (gate_down, up_down: 2 matrices)
Latent: moe_latent_dim → intermediate (latent_to_inter: 1 matrix, shared for gate+up)
SwiGLU: intermediate → intermediate
Compress: intermediate → moe_latent_dim  (inter_to_latent)
Output: moe_latent_dim → d          (latent_to_out)

Params per expert = 3 × d × moe_latent_dim + 2 × moe_latent_dim × intermediate
                  = 3 × 16384 × 7168 + 2 × 7168 × 4096
                  = 352M + 59M = 411M
```

vs original MoE (3 × d × intermediate = 201M), LatentMoE is +104% per expert.

---

## 7. Engram (~10% of total params)

| Parameter | 50T | 100T |
|-----------|-----|------|
| engram_layers | 10 | 20 |
| engram_ngram_orders | [2, 3] | [2, 3] |
| engram_num_hash_heads | 8 (2^3) | 8 |
| engram_table_capacity | 8,388,608 (2^23) | 8,388,608 |
| engram_memory_dim | 4096 (2^12) | 4096 |
| Per-layer params | 536B | 536B |
| Total Engram | 5.36T (~10%) | 10.72T (~10%) |

---

## 8. KV Cache Compression (≥2000× vs MHA)

**MHA baseline (G1)**: `2 × H × head_dim × num_layers = 2 × 128 × 256 × 60 = 3,932,160` elements/token

**Our design**:
| Component | Calculation | Per-token KV |
|-----------|-------------|--------------|
| CSA (10 layers, MLA, /16 stride, /5 MoR) | 10 × 2112 / 16 / 5 | 264 |
| HCA (10 layers, MLA, /256 stride, /5 MoR) | 10 × 2112 / 256 / 5 | 16.5 |
| KDA (40 layers, fixed state, /seq_len) | 40 × 128 × d_state² / 16M | 21 |
| **Total** | | **~302** |

**Compression ratio = 3,932,160 / 302 ≈ 13,000×** ✓ (target: 2000×)

---

## 9. System Resource Summary

| Metric | 50T | 100T |
|--------|-----|------|
| Total params | ~55T | ~109T |
| MoE params (% total) | 50.2T (91%) | 100.6T (92%) |
| Engram params (% total) | 5.36T (10%) | 10.72T (10%) |
| Active params / token | ~490B | ~960B |
| Max context | 16M | 32M |
| KV cache compression vs MHA | ~13,000× | ~13,000× |
| Model weights (FP8) | ~55 TB | ~109 TB |

---

## 10. File Structure

```
Sparse-Attn/
├── config.py        # SparseConfig + preset_50T() / preset_100T()
├── model.py         # Full model implementation
├── ARCHITECTURE.md  # This file
├── DIMENSION_SCALING.md
└── params_comparison.csv
```

`config.py`: `SparseConfig` dataclass with layer-type helpers (`is_csa_layer`, `is_hca_layer`, `is_kda_layer`, `mor_type_for_layer`, `mor_cache_layer_idx`).

`model.py`: All model classes (RMSNorm, RoPE, low-rank Q/KV/O, CSA, HCA, KDA, LatentMoE, MoR cache + routers, Engram, mHC, Block, SparseModel, smoke_test).
