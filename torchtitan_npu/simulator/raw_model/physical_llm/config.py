"""
DeepSeekV4.1-Flash 40T Configuration
====================================
40T compute-equivalent (20T stored parameters × Loop 2) language model.
Autoregressive multimodal understanding: text / image / audio / embodied
(3D point cloud, haptic-somatosensory stream) inputs.

Architecture:
  - Looped Transformer backbone: 100 physical layers × Loop 2 = 200
    logical depth. Weights are stored once per physical layer, so
    实际参数量 ≈ 20.7T (标称 20T, 其中 mHC ≈ 26.8B、Engram ≈ 2.06T
    占 10%); compute follows the full logical depth, so 计算等效
    ≈ 41.5T (标称 40T).
  - Layer layout per 4-layer unit [ISA, GDN2, GDN2, GDN2] × 25 units:
      25 ISA (Indexed Sparse Attention — CSA 压缩 2× + indexer 逐 token
      稀疏选择, 全秩 Q/K/V/O 投影, 无 MLA 低秩)
      75 GDN2 (delta-rule linear attention)
    ISA : GDN2 = 1 : 3; every layer has both attention + LatentMoE.
  - Flash-style KV sharing (DeepSeek v4.1 Flash): the 25 ISA layers share
    only 5 KV slots. Per group of 5 ISA layers: 1 layer produces KV,
    1 layer reindexes (re-runs its own indexer against the shared KV),
    3 layers reuse the KV layer's top-K selection.
  - LatentMoE with moe_latent_dim=4096 bottleneck, 4096 routed experts
    (top-25 per token) + 1 shared expert.
  - mHC (Manifold-Constrained Hyper-Connections): Sinkhorn-constrained
    hyper-connection blocks before attention and MoE (2 blocks/layer).
  - Engram conditional memory (~10% of params, DeepSeek V4.1 Flash
    口径): 10 inject layers (每 10 层一个) carry 3-order n-gram hash
    tables × 8 heads × 2^25 rows × 256 dim; inject layers are memory
    layers without MoE, keeping the total parameter count unchanged.
  - Multimodal encoders: image (ViT-style patch) / audio (1D-conv stem +
    transformer) / embodied point3d (PointNet-style set encoder) /
    haptic stream (multi-channel 1D-conv + transformer). Each encoder
    projects into hidden_size and adds a learned modality-type embedding;
    the AR sequence interleaves modality tokens with text tokens and the
    loss is computed on text spans only.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class FlashConfig:
    """Unified configuration for DeepSeekV4.1-Flash 40T.

    Layer layout (per 4-layer unit, repeated 25 times for 100 physical
    layers): [ISA, GDN2, GDN2, GDN2]
    ISA:GDN2 = 1:3
    Attention:MoE = 1:1 (every layer has both attention + MoE)

    Loop (Looped Transformer):
      base_depth=100 (the 100-layer physical stack) × num_recursion=2
      = num_layers=200 (logical depth)
      weights stored once per physical layer (实际参数量 ≈ 20.7T,
      标称 20T, 含 mHC ≈ 26.8B),
      compute follows the full logical depth (计算等效 ≈ 41.5T,
      标称 40T).

    Flash KV sharing (25 ISA layers → 5 KV slots):
      per group of 5 consecutive ISA layers:
        position 0 → "kv"      (produces KV + top-K via its own indexer)
        position 2 → "reindex" (own indexer re-selects against shared KV)
        positions 1/3/4 → "reuse" (reuses the KV layer's top-K selection)

    mHC (Manifold-Constrained Hyper-Connections):
      per layer 2 hyper-connection blocks (before attention and before
      MoE): expand → Sinkhorn-constrained doubly-stochastic mixing
      matrix → contract, with residual connection. 轻量口径:
      hc_mult=4 branches × hc_branch_dim=1024 (d/16), 控制 mHC 的
      参数量与稠密激活开销。

    Engram (conditional memory, ~10% of total params):
      10 inject layers (engram_layers) carry the memory module and have
      NO MoE (memory layers) — this keeps the total parameter count
      unchanged. Per module (DeepSeek V4.1 Flash 口径): 3 n-gram orders
      [2,3,4] × 8 hash heads × 2^25 rows × 256 dim lookup tables +
      key/value/query projections + static rmsnorm_ssr gate.
    """

    # ====================================================================
    # Core dimensions
    # ====================================================================
    hidden_size: int = 16384              # 2^14
    num_layers: int = 200                 # 逻辑深度 = 100 物理层 × Loop 2
    num_attention_heads: int = 128        # 2^7 (MLA heads, ISA 用)
    head_dim: int = 256                   # 2^8 (noPE 192 + RoPE 64)
    vocab_size: int = 524288              # 2^19
    max_seq_len: int = 8_388_608          # 2^23 (8M)

    # ====================================================================
    # Attention heads (ISA: 全秩 Q/K/V/O 投影, 无 MLA 低秩)
    # ====================================================================
    # head_dim 拆分: noPE 部分 + RoPE 部分;noPE 维度 =
    # head_dim - qk_rope_head_dim (代码内推导)。
    qk_rope_head_dim: int = 64            # 2^6

    # ====================================================================
    # Looped Transformer — 100 物理层 × Loop 2 = 200 逻辑深度
    # ====================================================================
    # 权重按物理层只存一份(实际参数量 ≈ 20T);前向把同一份权重跑两遍
    # (计算等效 ≈ 40T)。反向时 autograd 按链式法则把两遍使用点的梯度
    # 自动累加到同一 Parameter,优化器只做一次更新。
    loop_enable: bool = True
    loop_sharing: str = "cycle"           # "cycle": 第 r 遍即同一物理层
    num_recursion: int = 2
    base_depth: int = 100                 # 物理层 = 完整 100 层堆栈

    # ====================================================================
    # Layer type ratios (per 4-layer unit)
    # ====================================================================
    # ISA : GDN2 = 1 : 3
    # Unit layout: [ISA, GDN2, GDN2, GDN2] × 25 units = 100 physical layers
    unit_size: int = 4
    isa_per_unit: int = 1
    gdn2_per_unit: int = 3

    # ====================================================================
    # Flash KV sharing (DeepSeek v4.1 Flash) — 25 ISA layers → 5 KV slots
    # ====================================================================
    # 每组 5 个 ISA 层共享 1 个 KV 槽位:
    #   1 层产 KV (position 0), 1 层 reindex (position 2, 用自己的 indexer
    #   对共享 KV 重新选 top-K), 3 层 reuse (position 1/3/4, 复用产 KV 层
    #   的 top-K 选择, 无 KV 投影、无 indexer 计算)
    kv_share_enable: bool = True
    kv_groups: int = 5
    kv_reindex_per_group: int = 1
    kv_reuse_per_group: int = 3

    # ====================================================================
    # ISA (Indexed Sparse Attention) — CSA 压缩 + 稀疏 (DeepSeek v4.1
    # Flash 口径)
    # ====================================================================
    # KV 序列按 2× 压缩存储并参与检索(压缩网格上 indexer 逐 token 选
    # top-K),Q/K/V/O 为全秩投影(无 MLA 低秩)。8M 上下文 → 压缩网格
    # 4M 位置,top-2048 稀疏注意。
    isa_compress_ratio: int = 2           # Flash CSA2: 压缩倍数 2
    indexer_n_heads: int = 64             # 2^6
    indexer_head_dim: int = 128           # 2^7 → idx_dim = 8192
    indexer_topk: int = 2048              # 2^11 (token-level top-K)

    # ====================================================================
    # GDN2 (delta-rule linear attention)
    # ====================================================================
    # 状态 S ∈ [H, d_state, d_state] 为固定缓冲(非可学习参数),
    # 每层 O(1) 内存、无 token 级 KV cache。
    gdn2_num_heads: int = 64              # 2^6
    gdn2_d_state: int = 64                # 2^6
    gdn2_d_k: int = 64                    # 2^6
    gdn2_d_v: int = 64                    # = d_k
    gdn2_d_a: int = 3                     # 逐头读出增益附加维
    # α gate: [H, d_k] read gate; β gate (double): 2 × [H, d_state]

    # ====================================================================
    # mHC (Manifold-Constrained Hyper-Connections)
    # ====================================================================
    # 每层 2 个超连接块(注意力前/FFN 前):expand → Sinkhorn 约束的
    # 分支混合矩阵 → contract,残差连接;混合矩阵经 Sinkhorn 迭代保持
    # 双随机(流形约束)。轻量口径:分支宽 hc_branch_dim = d/16,
    # 控制 mHC 的参数量与稠密激活开销(全宽 = 参考实现口径,会令
    # 单遍激活 ~3×)。
    hc_mult: int = 4                      # 2^2 (并行分支数)
    hc_branch_dim: int = 1024             # 2^10 (分支宽 = d/16)
    sinkhorn_iters: int = 20

    # ====================================================================
    # LatentMoE
    # ====================================================================
    num_routed_experts: int = 4096        # 2^12
    num_shared_experts: int = 1
    moe_intermediate_size: int = 4096     # 2^12
    moe_latent_dim: int = 4096            # LatentMoE bottleneck dim
    num_experts_per_token: int = 25       # top-25
    router_score_function: str = "sqrtsoftplus"
    route_scale: float = 2.5

    # ====================================================================
    # Engram (Conditional Memory, ~10% of total params)
    # ====================================================================
    # 参考 DeepSeek V4.1 Flash 官方 Engram(196.6B @ 748.6B ≈ 26%):
    # 多阶 n-gram 哈希表 + 多头查找 + key/value 投影 + rmsnorm_ssr
    # 静态门控(RMSNorm 点积 + signed-square-root + 静态 token mask)。
    # 本模型口径: 10 个注入层(每 10 层一个、浅层起,仿 Flash 层 1/14),
    # 每注入层 3 阶 × 8 头 × 2^25 行 × 256 维 = 206.2B,合计 ≈ 2.06T
    # ≈ 10% 总参数。注入层为记忆层(不含 MoE),以保持总参数量不变;
    # MoE 只分布在其余 90 层。
    engram_enable: bool = True
    engram_layers: List[int] = field(
        default_factory=lambda: list(range(1, 100, 10)))  # 层 1/11/.../91
    engram_ngram_orders: List[int] = field(
        default_factory=lambda: [2, 3, 4])                # Flash 官方口径
    engram_num_hash_heads: int = 8        # engram_n_heads
    engram_head_dim: int = 256            # engram_head_dim
    engram_rows_per_head: int = 33_554_432  # 2^25 行/(阶, 头)
    # 检索拼接向量 = len(orders) × heads × head_dim = 6144(代码内推导)
    engram_static_gate: bool = True       # rmsnorm_ssr 静态门控(Flash 口径)
    # 注入层为记忆层:替代 MoE(无 MoE 也无稠密 FFN);字段供静态导入器解析 MoE 层数
    engram_replaces_moe: bool = True

    # ====================================================================
    # Multimodal encoders (AR 多模态理解: 文本/图像/语音/具身)
    # ====================================================================
    # 文本直接走词嵌入;其余模态经独立 encoder → 投影到 hidden_size →
    # 加模态类型嵌入后与文本 token 交错成单一 AR 序列。
    multimodal_enable: bool = True
    modalities: List[str] = field(
        default_factory=lambda: ["text", "image", "audio", "point3d", "haptic"])

    # image: ViT-style patch encoder (双向, patch token 输出)
    image_patch_size: int = 16
    image_enc_hidden: int = 1024
    image_enc_layers: int = 12
    image_enc_heads: int = 16
    image_max_patches: int = 4096         # 1024×1024 @ patch 16

    # audio: 原始波形 1D-conv 前端 (≈40× 下采样, log-mel 风格简化) + Transformer
    audio_enc_hidden: int = 768
    audio_enc_layers: int = 8
    audio_enc_heads: int = 12
    audio_conv_kernel: int = 400          # 16kHz 下 25ms 帧
    audio_conv_stride: int = 160          # 10ms 帧移 → 40× 下采样
    audio_max_frames: int = 4096

    # point3d: 具身 3D 输入 (PointNet 式集合编码: 逐点共享 MLP + 全局池化
    # 汇总 token + 逐点 token)
    point3d_enc_hidden: int = 512
    point3d_enc_layers: int = 6
    point3d_enc_heads: int = 8
    point3d_num_points: int = 1024

    # haptic: 具身体感/触觉流 (IMU/触觉阵列/力觉等多通道时序 → 1D-conv + Transformer)
    haptic_enc_hidden: int = 512
    haptic_enc_layers: int = 6
    haptic_enc_heads: int = 8
    haptic_num_channels: int = 12         # 6-axis IMU + 触觉阵列 + 力觉
    haptic_conv_kernel: int = 8
    haptic_conv_stride: int = 4
    haptic_max_frames: int = 4096

    # ====================================================================
    # Position encoding
    # ====================================================================
    rope_theta: float = 16384.0           # 2^14
    yarn_factor: float = 32.0             # 2^5
    yarn_original_max: int = 65536        # 2^16

    # ====================================================================
    # Training stability
    # ====================================================================
    swiglu_clamp: float = 10.0
    attn_softmax_clamp: float = 50.0
    z_loss_alpha: float = 0.001

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
    # Consistency checks
    # ====================================================================
    def __post_init__(self):
        """配置一致性校验 (模型构造依赖这些关系成立)."""
        if self.qk_rope_head_dim >= self.head_dim:
            raise ValueError(
                f"qk_rope_head_dim({self.qk_rope_head_dim}) 必须小于 "
                f"head_dim({self.head_dim})")
        if self.isa_per_unit + self.gdn2_per_unit != self.unit_size:
            raise ValueError(
                f"unit_size({self.unit_size}) != isa({self.isa_per_unit}) "
                f"+ gdn2({self.gdn2_per_unit})")
        if self.loop_enable:
            if self.loop_sharing != "cycle":
                raise ValueError(
                    f"Loop sharing 仅实现 'cycle'(当前为 {self.loop_sharing!r})")
            if self.base_depth * self.num_recursion != self.num_layers:
                raise ValueError(
                    f"Loop 配置不一致: base_depth({self.base_depth}) × "
                    f"num_recursion({self.num_recursion}) = "
                    f"{self.base_depth * self.num_recursion} != "
                    f"num_layers({self.num_layers})")
            if self.base_depth % self.unit_size != 0:
                raise ValueError(
                    f"Loop cycle 共享要求 base_depth({self.base_depth}) 是 "
                    f"unit_size({self.unit_size}) 的整数倍, 保证各递归内 "
                    f"ISA/GDN2 层型模式对齐")
        if self.kv_share_enable:
            nl_phys = self.base_depth if self.loop_enable else self.num_layers
            n_isa = (nl_phys // self.unit_size) * self.isa_per_unit
            if n_isa % self.kv_groups != 0:
                raise ValueError(
                    f"KV 共享要求 ISA 层数({n_isa}) 是 kv_groups"
                    f"({self.kv_groups}) 的整数倍")
            gs = n_isa // self.kv_groups
            if 1 + self.kv_reindex_per_group + self.kv_reuse_per_group != gs:
                raise ValueError(
                    f"每组 KV 共享: 1 产 KV + reindex({self.kv_reindex_per_group}) "
                    f"+ reuse({self.kv_reuse_per_group}) = "
                    f"{1 + self.kv_reindex_per_group + self.kv_reuse_per_group} "
                    f"!= 组内 ISA 层数({gs})")
        if self.multimodal_enable:
            for name, h, nh in (
                ("image", self.image_enc_hidden, self.image_enc_heads),
                ("audio", self.audio_enc_hidden, self.audio_enc_heads),
                ("point3d", self.point3d_enc_hidden, self.point3d_enc_heads),
                ("haptic", self.haptic_enc_hidden, self.haptic_enc_heads),
            ):
                if h % nh != 0:
                    raise ValueError(
                        f"{name} encoder: hidden({h}) 必须整除 heads({nh})")

    # ====================================================================
    # Helpers
    # ====================================================================
    def is_isa_layer(self, layer_idx: int) -> bool:
        """ISA on position 0 of each 4-layer unit."""
        return (layer_idx % self.unit_size) < self.isa_per_unit

    def is_engram_layer(self, layer_idx: int) -> bool:
        """Engram inject layer (memory layer, 无 MoE)."""
        return layer_idx in self.engram_layers

    def is_moe_layer(self, layer_idx: int) -> bool:
        """MoE layer (non-Engram layers)."""
        return not self.is_engram_layer(layer_idx)

    def is_gdn2_layer(self, layer_idx: int) -> bool:
        """GDN2 on positions 1-3 of each 4-layer unit."""
        return not self.is_isa_layer(layer_idx)

    def layer_type(self, layer_idx: int) -> str:
        return "ISA" if self.is_isa_layer(layer_idx) else "GDN2"

    def isa_group_size(self) -> int:
        """Number of ISA layers per KV-sharing group (25/5 = 5)."""
        return (self.base_depth // self.unit_size) // self.kv_groups

    def kv_group_of(self, layer_idx: int) -> int:
        """Physical ISA layer idx → KV slot idx (5 slots total)."""
        return (layer_idx // self.unit_size) // self.isa_group_size()

    def isa_role(self, layer_idx: int) -> str:
        """KV-sharing role of the ISA layer at physical index.

        Per group of 5 ISA layers: position 0 → "kv" (produces KV + top-K),
        position 2 → "reindex" (own indexer against shared KV),
        positions 1/3/4 → "reuse" (reuses the KV layer's top-K).
        """
        if not self.kv_share_enable:
            return "kv"
        isa_pos = layer_idx // self.unit_size
        p = isa_pos % self.isa_group_size()
        if p == 0:
            return "kv"
        if self.kv_reindex_per_group > 0 and p == 2:
            return "reindex"
        return "reuse"

    def isa_layout_summary(self) -> str:
        """Human-readable ISA/KV-sharing layout summary."""
        roles = [self.isa_role(u * self.unit_size)
                 for u in range(self.base_depth // self.unit_size)]
        return (f"{len(roles)} ISA layers / {self.kv_groups} KV slots: "
                f"{roles.count('kv')} kv, {roles.count('reindex')} reindex, "
                f"{roles.count('reuse')} reuse")

    # ====================================================================
    # Presets
    # ====================================================================
    @classmethod
    def preset_40T(cls) -> "FlashConfig":
        """40T compute-equivalent (~20.7T stored params × Loop 2).

        100 physical layers (25 units of ISA + GDN2×3) × Loop 2 = 200
        logical depth; 4096 routed experts × top-25 + 1 shared expert,
        LatentMoE bottleneck 4096, MoE on 90 layers; Flash KV sharing:
        25 ISA layers → 5 KV slots (1 produce + 1 reindex + 3 reuse per
        group); mHC hyper-connections (hc_mult=4 × 分支宽 1024, 2
        Sinkhorn-constrained blocks/layer); Engram conditional memory
        ≈ 2.06T (10%, DeepSeek V4.1 Flash 口径) on 10 inject layers
        (无 MoE 记忆层); seq=8M, vocab=524288; multimodal encoders for
        image/audio/point3d/haptic.
        实际参数量 ≈ 20.7T (标称 20T, Engram 10.0%), 计算等效
        ≈ 41.5T (标称 40T), 单遍激活 ≈ 224B (标称 200B, Engram
        表格按查找口径计).
        """
        return cls(
            hidden_size=16384,
            num_layers=200,
            num_attention_heads=128,
            head_dim=256,
            vocab_size=524288,
            max_seq_len=8_388_608,

            qk_rope_head_dim=64,

            loop_enable=True,
            loop_sharing="cycle",
            num_recursion=2,
            base_depth=100,

            unit_size=4,
            isa_per_unit=1,
            gdn2_per_unit=3,

            kv_share_enable=True,
            kv_groups=5,
            kv_reindex_per_group=1,
            kv_reuse_per_group=3,

            isa_compress_ratio=2,
            indexer_n_heads=64,
            indexer_head_dim=128,
            indexer_topk=2048,

            gdn2_num_heads=64,
            gdn2_d_state=64,
            gdn2_d_k=64,
            gdn2_d_v=64,
            gdn2_d_a=3,

            hc_mult=4,
            hc_branch_dim=1024,
            sinkhorn_iters=20,

            num_routed_experts=4096,
            num_shared_experts=1,
            moe_intermediate_size=4096,
            moe_latent_dim=4096,
            num_experts_per_token=25,

            engram_enable=True,
            engram_layers=[1, 11, 21, 31, 41, 51, 61, 71, 81, 91],
            engram_ngram_orders=[2, 3, 4],
            engram_num_hash_heads=8,
            engram_head_dim=256,
            engram_rows_per_head=33_554_432,
            engram_static_gate=True,
            engram_replaces_moe=True,

            multimodal_enable=True,
            modalities=["text", "image", "audio", "point3d", "haptic"],
            image_patch_size=16,
            image_enc_hidden=1024,
            image_enc_layers=12,
            image_enc_heads=16,
            image_max_patches=4096,
            audio_enc_hidden=768,
            audio_enc_layers=8,
            audio_enc_heads=12,
            audio_conv_kernel=400,
            audio_conv_stride=160,
            audio_max_frames=4096,
            point3d_enc_hidden=512,
            point3d_enc_layers=6,
            point3d_enc_heads=8,
            point3d_num_points=1024,
            haptic_enc_hidden=512,
            haptic_enc_layers=6,
            haptic_enc_heads=8,
            haptic_num_channels=12,
            haptic_conv_kernel=8,
            haptic_conv_stride=4,
            haptic_max_frames=4096,

            rope_theta=16384.0,
            yarn_factor=32.0,
            yarn_original_max=65536,

            swiglu_clamp=10.0,
            attn_softmax_clamp=50.0,
            z_loss_alpha=0.001,
        )
