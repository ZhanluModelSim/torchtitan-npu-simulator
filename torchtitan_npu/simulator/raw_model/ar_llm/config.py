"""
DeepSeekV4-Sparse Configuration
================================
50T / 100T parameter language model configuration.

Architecture:
  - CSA / HCA / KDA layer types in ratio 1:1:4 per 6-layer unit
  - LatentMoE with moe_latent_dim bottleneck (7168)
  - MoR (Mixture of Recursions) with cycle KV sharing
  - Engram conditional memory (~10% of total params)
  - 5% of CSA/HCA layers use expert-choice MoR routing
  - 95% of CSA/HCA layers use token-choice MoR routing
  - Low-rank Q/KV/O projections (MLA-style)
  - mHC (Manifold-Constrained Hyper-Connections)
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SparseConfig:
    """Unified configuration for DeepSeekV4-Sparse 50T / 100T models.

    Layer layout (per 6-layer unit, repeated 10 times for 60 layers):
      [CSA, HCA, KDA, KDA, KDA, KDA]
    CSA:HCA:KDA = 1:1:4
    Attention:MoE = 1:1 (every layer has both attention + MoE)

    MoR cycle sharing:
      base_depth=12 (2 units) × num_recursion=5 = num_layers=60
      KV cache only stored for base_depth layers, reused across recursions
    """

    # ====================================================================
    # Core dimensions
    # ====================================================================
    hidden_size: int = 16384              # 2^14
    num_layers: int = 60                  # 6 × 10 units
    num_attention_heads: int = 128        # 2^7
    head_dim: int = 256                   # 2^8 (noPE 192 + RoPE 64)
    vocab_size: int = 524288              # 2^19 (4× expansion)
    max_seq_len: int = 16_777_216         # 2^24 (16M)

    # ====================================================================
    # Low-rank projections (MLA-style)
    # ====================================================================
    q_lora_rank: int = 4096               # 2^12
    qk_rope_head_dim: int = 64            # 2^6
    qk_nope_head_dim: int = 192           # = 256 - 64, derived
    kv_lora_rank: int = 1024              # 2^10
    o_groups: int = 32                    # 2^5 (128 / 32 = 4 heads/group)
    o_lora_rank: int = 4096               # 2^12

    # ====================================================================
    # MoR (Mixture of Recursions) — cycle KV sharing
    # ====================================================================
    mor_enable: bool = True
    mor_sharing: str = "cycle"            # "cycle" | "middle_cycle" | "sequence"
    num_recursion: int = 5
    base_depth: int = 12                  # = num_layers / num_recursion, = 2 × 6-layer unit
    mor_update_cache: bool = True         # expert-choice layers only update selected token KV

    # ====================================================================
    # Layer type ratios (per 6-layer unit)
    # ====================================================================
    # CSA : HCA : KDA = 1 : 1 : 4
    # Unit layout: [CSA, HCA, KDA, KDA, KDA, KDA]
    unit_size: int = 6
    csa_per_unit: int = 1
    hca_per_unit: int = 1
    kda_per_unit: int = 4

    # ====================================================================
    # MoR routing
    # ====================================================================
    # 5% of total layers use expert-choice MoR (with capacity + sampling_loss)
    # Remaining CSA/HCA layers use token-choice MoR (standard top-K + balancing_loss)
    # KDA layers use token-choice MoR
    mor_expert_ratio: float = 0.05        # 5% of layers
    mor_expert_capacity: float = 1.0      # capacity factor for expert-choice
    mor_cap_warmup_steps: int = 1000      # warmup for capacity factor

    # ====================================================================
    # LatentMoE
    # ====================================================================
    num_routed_experts: int = 2048        # 2^11 (50T), 2^12 (100T)
    num_shared_experts: int = 2           # 2^1
    moe_intermediate_size: int = 4096     # 2^12
    moe_latent_dim: int = 7168            # LatentMoE bottleneck dim
    num_experts_per_token: int = 16       # 2^4
    router_score_function: str = "sqrtsoftplus"
    route_scale: float = 2.5

    # ====================================================================
    # CSA (Compressed Sparse Attention)
    # ====================================================================
    csa_compress_ratio: int = 16          # 2^4
    csa_window_size: int = 1024           # 2^10

    # ====================================================================
    # HCA (Heavily Compressed Attention) + Indexer
    # ====================================================================
    hca_compress_ratio: int = 256         # 2^8 → comp_len = 16M/256 = 65536
    indexer_n_heads: int = 64             # 2^6
    indexer_head_dim: int = 128           # 2^7 → idx_dim = 8192
    indexer_topk: int = 4096              # 2^12

    # ====================================================================
    # KDA (Kimi Delta Attention) — linear attention with delta rule
    # ====================================================================
    kda_d_state: int = 256                # 2^8 (50T), 2^9 (100T)
    kda_d_k: int = 128                    # 2^7 (50T), 2^8 (100T)
    kda_d_v: int = 128                    # = d_k
    kda_d_a: int = 3                      # delta attention extra dim
    # α gate: [H, d_k], β gate (double gate): 2 × [H, d_state]

    # ====================================================================
    # mHC (Manifold-Constrained Hyper-Connections)
    # ====================================================================
    hc_mult: int = 4                      # 2^2
    sinkhorn_iters: int = 20
    use_attn_sink: bool = True

    # ====================================================================
    # Engram (Conditional Memory, ~10% of total params)
    # ====================================================================
    engram_layers: List[int] = field(default_factory=list)
    engram_ngram_orders: List[int] = field(default_factory=lambda: [2, 3])
    engram_num_hash_heads: int = 8        # 2^3
    engram_table_capacity: int = 8_388_608  # 2^23 = 8M
    engram_memory_dim: int = 4096         # 2^12

    # ====================================================================
    # Position encoding
    # ====================================================================
    rope_theta: float = 16384.0           # 2^14
    yarn_factor: float = 32.0             # 2^5 (50T), 2^6 (100T)
    yarn_original_max: int = 65536        # 2^16

    # ====================================================================
    # Training stability
    # ====================================================================
    swiglu_clamp: float = 10.0
    attn_softmax_clamp: float = 50.0
    z_loss_alpha: float = 0.001
    sampling_loss_alpha: float = 0.001    # expert-choice sampling loss weight

    # ====================================================================
    # Dropout
    # ====================================================================
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # ====================================================================
    # Initialization
    # ====================================================================
    init_method_std: float = 0.02
    embedding_init_method_std: float = 0.02

    # ====================================================================
    # Helpers
    # ====================================================================
    def is_csa_layer(self, layer_idx: int) -> bool:
        """CSA on position 0 of each 6-layer unit."""
        return (layer_idx % self.unit_size) == 0

    def is_hca_layer(self, layer_idx: int) -> bool:
        """HCA on position 1 of each 6-layer unit."""
        return (layer_idx % self.unit_size) == 1

    def is_kda_layer(self, layer_idx: int) -> bool:
        """KDA on positions 2-5 of each 6-layer unit."""
        return (layer_idx % self.unit_size) >= 2

    def is_engram_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.engram_layers

    def mor_type_for_layer(self, layer_idx: int) -> str:
        """Return 'expert' or 'token' for MoR routing type.

        5% of total layers use expert-choice (distributed evenly across depth).
        Only CSA/HCA layers can be expert-choice; KDA always token-choice.
        """
        if not self.mor_enable:
            return "token"
        # Only CSA/HCA layers can be expert-choice
        if self.is_kda_layer(layer_idx):
            return "token"
        # 5% of total layers → 60 × 0.05 = 3 layers
        num_expert_layers = max(1, int(self.num_layers * self.mor_expert_ratio))
        # Distribute evenly: pick CSA/HCA layers at shallow/mid/deep
        # Positions: ~1/6, ~1/2, ~5/6 of num_layers
        candidate_positions = []
        for i in range(self.num_layers):
            if not self.is_kda_layer(i):
                candidate_positions.append(i)
        if num_expert_layers >= len(candidate_positions):
            return "expert"
        # Even spacing across CSA/HCA candidates
        step = len(candidate_positions) // num_expert_layers
        expert_indices = set(candidate_positions[i * step]
                             for i in range(num_expert_layers))
        return "expert" if layer_idx in expert_indices else "token"

    def mor_cache_layer_idx(self, layer_idx: int) -> int:
        """Map physical layer idx to KV cache slot idx (cycle sharing).

        cycle: layer_idx % base_depth
        """
        if self.mor_sharing == "cycle":
            return layer_idx % self.base_depth
        elif self.mor_sharing == "sequence":
            return layer_idx // self.num_recursion
        else:
            return layer_idx  # fallback: no sharing

    # ====================================================================
    # Presets
    # ====================================================================
    @classmethod
    def preset_50T(cls) -> "SparseConfig":
        """50 Trillion parameter model.

        60 layers (10 units of CSA+HCA+KDA×4), 2048 routed experts,
        LatentMoE with moe_latent_dim=7168, MoR 5-cycle sharing,
        Engram 10 layers (~10%), seq=16M, vocab=524288.
        """
        return cls(
            hidden_size=16384,
            num_layers=60,
            num_attention_heads=128,
            head_dim=256,
            vocab_size=524288,
            max_seq_len=16_777_216,

            q_lora_rank=4096,
            qk_rope_head_dim=64,
            qk_nope_head_dim=192,
            kv_lora_rank=1024,
            o_groups=32,
            o_lora_rank=4096,

            mor_enable=True,
            mor_sharing="cycle",
            num_recursion=5,
            base_depth=12,
            mor_update_cache=True,

            unit_size=6,
            csa_per_unit=1,
            hca_per_unit=1,
            kda_per_unit=4,

            mor_expert_ratio=0.05,
            mor_expert_capacity=1.0,
            mor_cap_warmup_steps=1000,

            num_routed_experts=2048,
            num_shared_experts=2,
            moe_intermediate_size=4096,
            moe_latent_dim=7168,
            num_experts_per_token=16,

            csa_compress_ratio=16,
            csa_window_size=1024,

            hca_compress_ratio=256,
            indexer_n_heads=64,
            indexer_head_dim=128,
            indexer_topk=4096,

            kda_d_state=256,
            kda_d_k=128,
            kda_d_v=128,
            kda_d_a=3,

            hc_mult=4,
            sinkhorn_iters=20,
            use_attn_sink=True,

            # 10 engram layers, evenly distributed
            engram_layers=[3, 9, 15, 21, 27, 33, 39, 45, 51, 57],
            engram_ngram_orders=[2, 3],
            engram_num_hash_heads=8,
            engram_table_capacity=8_388_608,
            engram_memory_dim=4096,

            rope_theta=16384.0,
            yarn_factor=32.0,
            yarn_original_max=65536,

            swiglu_clamp=10.0,
            attn_softmax_clamp=50.0,
            z_loss_alpha=0.001,
            sampling_loss_alpha=0.001,
        )

    @classmethod
    def preset_100T(cls) -> "SparseConfig":
        """100 Trillion parameter model (~109T total).

        Same 60 layers as 50T, doubled routed_experts (4096),
        larger KDA state (512/256), seq=32M, CSA=24x, HCA=512x,
        yarn_factor=64, 20 engram layers.
        """
        return cls(
            hidden_size=16384,
            num_layers=60,
            num_attention_heads=128,
            head_dim=256,
            vocab_size=524288,
            max_seq_len=33_554_432,         # 2^25 (32M)

            q_lora_rank=4096,
            qk_rope_head_dim=64,
            qk_nope_head_dim=192,
            kv_lora_rank=1024,
            o_groups=32,
            o_lora_rank=4096,

            mor_enable=True,
            mor_sharing="cycle",
            num_recursion=5,
            base_depth=12,
            mor_update_cache=True,

            unit_size=6,
            csa_per_unit=1,
            hca_per_unit=1,
            kda_per_unit=4,

            mor_expert_ratio=0.05,
            mor_expert_capacity=1.0,
            mor_cap_warmup_steps=1000,

            num_routed_experts=4096,        # 2^12 (doubled)
            num_shared_experts=2,
            moe_intermediate_size=4096,
            moe_latent_dim=7168,
            num_experts_per_token=16,

            csa_compress_ratio=24,          # 24x
            csa_window_size=2048,           # 2^11

            hca_compress_ratio=512,         # 2^9 → comp_len=65536
            indexer_n_heads=64,
            indexer_head_dim=128,
            indexer_topk=4096,

            kda_d_state=512,                # 2^9 (doubled)
            kda_d_k=256,                    # 2^8 (doubled)
            kda_d_v=256,
            kda_d_a=3,

            hc_mult=4,
            sinkhorn_iters=20,
            use_attn_sink=True,

            # 20 engram layers
            engram_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29,
                           32, 35, 38, 41, 44, 47, 50, 53, 56, 59],
            engram_ngram_orders=[2, 3],
            engram_num_hash_heads=8,
            engram_table_capacity=8_388_608,
            engram_memory_dim=4096,

            rope_theta=16384.0,
            yarn_factor=64.0,               # 2^6 (doubled for 32M)
            yarn_original_max=65536,

            swiglu_clamp=10.0,
            attn_softmax_clamp=50.0,
            z_loss_alpha=0.001,
            sampling_loss_alpha=0.001,
        )
