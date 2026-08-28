# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview tensor parallelism tests (parallelize._apply_tensor_parallel).

v1 scope under test: TP with the sequence REPLICATED (no sequence
parallel) — head-wise column splits for the attention projections and
pair-preserving column splits for the MLPs, conjugate row splits whose
partial outputs are all-reduced at the sublayer boundary, head-sharded
sinks, and the routed MoE core reusing the head-parallel sharding of
expert_parallel over the tp mesh.

CI-safe single-process coverage:
- pure algebra of the grouped-linear TP slicing helpers (per-expert
  head/pair out-dim slices, matmul equivalence, swiglu7 pairing under
  split, row-split partial sums);
- TP=2-emulated vs TP=1 fwd/bwd equivalence for attention, the dense MLP
  and the MoE layer, running the real partition functions on two virtual
  ranks without a process group (the boundary all-reduces are applied
  manually as tensor sums; with the real wiring the module hooks do them);
- parallelize guards (TP+CP / TP+EP / TP+ETP raise, divisibility errors);
- single-rank gloo wiring (degree 1): state-dict keys/placements,
  fwd/bwd parity, TP+FSDP composition and full-checkpoint loading.

Checkpoint note: every weight whose TP slice a single DTensor placement
expresses becomes a Shard DTensor (full-checkpoint loading distributes
through DTensor); the multi-expert column splits whose per-expert slices
have no honest placement (linear_qkv always; linear_g/up_gate_proj with
num_modality > 1; modality_specific_shared_expert_fc1) stay plain local
slices — state-dict keys unchanged, but full-state-dict loading of those
keys at TP > 1 needs loader-side slicing (documented follow-up, mirroring
cp_ulysses' head-sharded sinks note). The equivalence tests pin the local
contents directly against the full weights.

Nightly-gated real-collective coverage (RUN_MODEL_PARALLEL_MULTI_RANK,
following tests/smoke_tests/model_parallel/_multi_rank.py conventions):

    torchrun --nproc_per_node=2 -m pytest \
        tests/unit_tests/models/test_magi2_tp.py -m nightly -k TwoRank
"""

import copy
from types import SimpleNamespace

import pytest
import torch

TP = 2  # virtual TP degree used throughout the single-process tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_mesh(rank: int, degree: int = TP):
    """Duck-typed stand-in for a 1D DeviceMesh (size/local_rank/group API)."""
    return SimpleNamespace(
        ndim=1,
        size=lambda: degree,
        get_local_rank=lambda: rank,
        get_group=lambda: None,
    )


def _fake_parallel_dims(**overrides):
    """Minimal ParallelDims stand-in exercising the parallelize guards."""
    base = dict(
        pp_enabled=False,
        tp_enabled=True,
        cp_enabled=False,
        ep=1,
        etp=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _init_module(module):
    """Fill a standalone module like Magi2PreviewModel.init_weights."""
    from torchtitan_npu.models.magi2_preview.feed_forward import (
        CoreMultiHeadMoE,
    )
    from torchtitan_npu.models.magi2_preview.grouped_linear import (
        GroupedLinear,
    )
    from torchtitan_npu.models.magi2_preview.norms import MultiModalityRMSNorm

    with torch.no_grad():
        for submodule in module.modules():
            if isinstance(submodule, MultiModalityRMSNorm):
                torch.nn.init.zeros_(submodule.weight)
            elif isinstance(submodule, GroupedLinear):
                torch.nn.init.normal_(submodule.weight, mean=0.0, std=0.02)
            elif isinstance(submodule, CoreMultiHeadMoE):
                torch.nn.init.normal_(submodule.gate, mean=0.0, std=0.02)
                torch.nn.init.normal_(submodule.W_gate, mean=0.0, std=0.02)
                torch.nn.init.normal_(submodule.W_up, mean=0.0, std=0.02)
                torch.nn.init.normal_(submodule.W_down, mean=0.0, std=0.02)
                submodule.router.expert_bias.normal_(0.0, 0.05)
    return module


def _make_attention(backend: str = "sdpa", hidden_size: int = 128, seed: int = 3):
    from torchtitan_npu.models.magi2_preview.attention import Magi2Attention

    torch.manual_seed(seed)
    attn = Magi2Attention(
        Magi2Attention.Config(
            hidden_size=hidden_size,
            head_dim=32,
            num_modality=3,
            sink_token_num=1,
            attn_backend=backend,
        )
    )
    attn.sinks.data.normal_(mean=0.0, std=0.02)
    return _init_module(attn)


def _attention_original_inputs(seq_len: int = 12, hidden_size: int = 128):
    """Full-sequence inputs in ORIGINAL token order."""
    generator = torch.Generator().manual_seed(11)
    modality = torch.arange(seq_len) % 3
    x = torch.randn(seq_len, hidden_size, generator=generator)
    # rotary_dim 24 (< head_dim 32) exercises the RoPE pass-through dims.
    rope = torch.randn(seq_len, 24, generator=generator)
    return x, modality, rope


def _sorted_args(modality):
    """Modality sort bookkeeping mirroring Magi2PreviewModel.forward."""
    mod = modality.clone()
    mod[mod == 3] = 2  # TIME -> TEXT, same remap as the model entry
    sort_idx = torch.argsort(mod)
    inv_sort_idx = torch.argsort(sort_idx)
    m_splits = [int(v) for v in torch.bincount(mod, minlength=3).tolist()]
    return sort_idx, inv_sort_idx, m_splits


def _make_dense_mlp(num_modality: int = 3, seed: int = 4):
    from torchtitan_npu.models.magi2_preview.feed_forward import Magi2MLP

    torch.manual_seed(seed)
    return _init_module(
        Magi2MLP(
            Magi2MLP.Config(
                hidden_size=64,
                intermediate_size=32,
                num_modality=num_modality,
            )
        )
    )


def _make_moe_layer(seed: int = 6):
    from torchtitan_npu.models.magi2_preview.feed_forward import (
        MultiHeadMoELayer,
    )

    torch.manual_seed(seed)
    return _init_module(
        MultiHeadMoELayer(
            MultiHeadMoELayer.Config(
                hidden_size=64,
                num_modality=3,
                moe_num_heads=4,
                num_experts=6,
                moe_top_k=3,
                expert_intermediate_size=32,
                shared_expert_intermediate_size=32,
            )
        )
    )


def _small_model_config(**overrides):
    """Tiny MAGI-2-preview config; every TP-split dim is divisible by 2."""
    from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

    kwargs = dict(
        num_layers=2,
        hidden_size=128,
        head_dim=32,
        num_stream=2,
        video_in_channels=48,
        audio_in_channels=64,
        text_in_channels=64,
        time_channel_dim=64,
        dense_intermediate_size=64,
        mm_layers=[0],
        moe_layers=[1],
        moe_num_heads=4,
        num_experts=4,
        moe_top_k=2,
        expert_intermediate_size=32,
        shared_expert_intermediate_size=32,
        route_scale=4.9,
        sink_token_num=1,
    )
    kwargs.update(overrides)
    return Magi2PreviewModel.Config(**kwargs)


def _model_inputs(seq_len: int = 16, seed: int = 5):
    """Full packed batch: every modality (incl. TIME), multi-segment."""
    from torchtitan_npu.models.magi2_preview.model import Modality

    generator = torch.Generator().manual_seed(seed)
    pattern = [
        Modality.VIDEO,
        Modality.AUDIO,
        Modality.TEXT,
        Modality.TIME,
    ]
    modality = torch.tensor(
        [pattern[i % len(pattern)].value for i in range(seq_len)],
        dtype=torch.int32,
    )
    x = torch.randn(seq_len, 64, generator=generator)
    coords = torch.randn(seq_len, 9, generator=generator)
    time_embedding = torch.randn(seq_len, 64, generator=generator)
    cu_seqlens = torch.tensor([0, 6, seq_len], dtype=torch.int32)
    labels = torch.zeros(seq_len, 64)
    video_rows = modality == Modality.VIDEO
    audio_rows = modality == Modality.AUDIO
    labels[video_rows, :48] = torch.randn(
        int(video_rows.sum()), 48, generator=generator
    )
    labels[audio_rows, :64] = torch.randn(
        int(audio_rows.sum()), 64, generator=generator
    )
    inputs = dict(
        coords_mapping=coords,
        modality_mapping=modality,
        time_embedding=time_embedding,
        cu_seqlens=cu_seqlens,
    )
    return x, inputs, labels


def _head_range(rank: int, num_heads: int, degree: int = TP):
    per_rank = num_heads // degree
    return rank * per_rank, (rank + 1) * per_rank


def _expected_local_grad(name, ref_grad, rank, degree, config):
    """Rank-local expectation for a parameter's reference gradient.

    Encodes the partition rules of ``_apply_tensor_parallel``; replicated
    parameters pass through unchanged (their rank-local partial gradients
    are summed by the TP grad hooks, so the wired value equals the
    reference gradient directly). Also used for full-weight slicing in the
    checkpoint-distribution check (values and gradients share the layout).
    """
    from torchtitan_npu.models.magi2_preview.grouped_linear import (
        slice_grouped_linear_by_heads,
        slice_grouped_linear_by_pairs,
    )

    if not name.startswith("block.layers."):
        return ref_grad  # pre_adapter / post_adapter replicated
    layer_id = int(name.split(".")[2])
    attn_heads = config.hidden_size // config.head_dim
    head_dim = config.head_dim
    hs_a, he_a = _head_range(rank, attn_heads, degree)
    num_modality = 3 if layer_id in config.mm_layers else 1

    if name.endswith("attention.sinks"):
        return ref_grad[:, hs_a:he_a]
    if name.endswith("attention.linear_proj.weight"):
        return ref_grad[:, hs_a * head_dim : he_a * head_dim]
    if name.endswith("attention.linear_g.weight"):
        return slice_grouped_linear_by_heads(
            ref_grad, num_modality, attn_heads, 1, 1, (hs_a, he_a)
        )
    if name.endswith("attention.linear_qkv.weight"):
        return slice_grouped_linear_by_heads(
            ref_grad, num_modality, attn_heads, head_dim, 3, (hs_a, he_a)
        )
    if name.endswith(("pre_norm.weight", "q_norm.weight", "k_norm.weight")):
        return ref_grad  # replicated inside the TP-sharded sublayers

    if layer_id not in config.moe_layers:
        # Dense MLP: pair-range column split + conjugate row split.
        pairs = config.dense_intermediate_size
        p0, p1 = _head_range(rank, pairs, degree)
        if name.endswith("mlp.up_gate_proj.weight"):
            return slice_grouped_linear_by_pairs(
                ref_grad, num_modality, pairs, (p0, p1)
            )
        if name.endswith("mlp.down_proj.weight"):
            return ref_grad[:, p0:p1]
        return ref_grad

    # MoE layer.
    hidden_cols = config.hidden_size // degree
    c0, c1 = rank * hidden_cols, (rank + 1) * hidden_cols
    if name.endswith("mlp.split_linear.weight"):
        return ref_grad[c0:c1]
    if name.endswith("mlp.merge_linear.weight"):
        return ref_grad[:, c0:c1]
    if ".moe_mlp." in name:
        hs_m, he_m = _head_range(rank, config.moe_num_heads, degree)
        return ref_grad[hs_m * config.num_experts : he_m * config.num_experts]
    pairs = config.shared_expert_intermediate_size
    p0, p1 = _head_range(rank, pairs, degree)
    if name.endswith("modality_specific_shared_expert_fc1.weight"):
        return slice_grouped_linear_by_pairs(ref_grad, 3, pairs, (p0, p1))
    if name.endswith("shared_expert_fc1.weight"):
        return slice_grouped_linear_by_pairs(ref_grad, 1, pairs, (p0, p1))
    if name.endswith("shared_expert_fc2.weight") or name.endswith(
        "modality_specific_shared_expert_fc2.weight"
    ):
        return ref_grad[:, p0:p1]
    return ref_grad  # mlp.pre_norm and MHC params: replicated


def _local_value(param):
    from torch.distributed.tensor import DTensor

    value = param
    if isinstance(value, DTensor):
        value = value.to_local()
    return value


# ---------------------------------------------------------------------------
# Grouped-linear TP slicing algebra (single process)
# ---------------------------------------------------------------------------


class TestGroupedLinearTpSlicing:
    def test_by_heads_slices_per_expert_and_section(self):
        from torchtitan_npu.models.magi2_preview.grouped_linear import (
            slice_grouped_linear_by_heads,
        )

        E, S, H, Dh, Cin = 3, 3, 4, 2, 5
        weight = torch.arange(
            E * S * H * Dh * Cin, dtype=torch.float32
        ).reshape(E * S * H * Dh, Cin)
        per_rank = H // TP
        for rank in range(TP):
            hs, he = _head_range(rank, H)
            sliced = slice_grouped_linear_by_heads(
                weight, E, H, Dh, S, (hs, he)
            )
            assert sliced.shape == (E * S * per_rank * Dh, Cin)
            w = weight.view(E, S, H, Dh, Cin)
            expected = w[:, :, hs:he].reshape(E * S * per_rank * Dh, Cin)
            assert torch.equal(sliced, expected)

    def test_by_heads_rejects_bad_inputs(self):
        from torchtitan_npu.models.magi2_preview.grouped_linear import (
            slice_grouped_linear_by_heads,
        )

        weight = torch.randn(6, 4)
        with pytest.raises(ValueError, match="head_range"):
            slice_grouped_linear_by_heads(weight, 1, 4, 1, 1, (2, 2))
        with pytest.raises(ValueError, match="leading dim"):
            slice_grouped_linear_by_heads(weight, 1, 3, 1, 1, (0, 1))

    def test_by_pairs_keeps_gate_up_together(self):
        from torchtitan_npu.models.magi2_preview.grouped_linear import (
            slice_grouped_linear_by_pairs,
        )

        E, P, Cin = 2, 4, 3
        weight = torch.arange(E * 2 * P * Cin, dtype=torch.float32).reshape(
            E * 2 * P, Cin
        )
        per_rank = P // TP
        for rank in range(TP):
            p0, p1 = _head_range(rank, P)
            sliced = slice_grouped_linear_by_pairs(weight, E, P, (p0, p1))
            assert sliced.shape == (E * 2 * per_rank, Cin)
            w = weight.view(E, P, 2, Cin)
            expected = w[:, p0:p1].reshape(E * 2 * per_rank, Cin)
            assert torch.equal(sliced, expected)

    def test_by_pairs_rejects_bad_inputs(self):
        from torchtitan_npu.models.magi2_preview.grouped_linear import (
            slice_grouped_linear_by_pairs,
        )

        weight = torch.randn(8, 4)
        with pytest.raises(ValueError, match="pair_range"):
            slice_grouped_linear_by_pairs(weight, 1, 4, (0, 5))
        with pytest.raises(ValueError, match="leading dim"):
            slice_grouped_linear_by_pairs(weight, 1, 5, (0, 1))

    def test_head_column_split_matmul_equivalence(self):
        """Sliced-weight grouped matmul == head slice of the full output."""
        from torchtitan_npu.models.magi2_preview.grouped_linear import (
            GroupedLinear,
            slice_grouped_linear_by_heads,
        )

        E, H, Dh, Cin = 3, 4, 2, 5
        torch.manual_seed(0)
        x = torch.randn(6, Cin)
        m_splits = [1, 2, 3]
        for sections in (1, 3):
            out_full = sections * H * Dh
            weight = torch.randn(E * out_full, Cin)
            full = GroupedLinear(Cin, out_full, num_experts=E)
            with torch.no_grad():
                full.weight.copy_(weight)
            y_full = full(x, m_splits).view(6, sections, H, Dh)
            for rank in range(TP):
                hs, he = _head_range(rank, H)
                local = GroupedLinear(
                    Cin, sections * (he - hs) * Dh, num_experts=E
                )
                with torch.no_grad():
                    local.weight.copy_(
                        slice_grouped_linear_by_heads(
                            weight, E, H, Dh, sections, (hs, he)
                        )
                    )
                y_local = local(x, m_splits)
                expected = y_full[:, :, hs:he].reshape(
                    6, sections * (he - hs) * Dh
                )
                assert torch.equal(y_local, expected)

    def test_pair_column_split_preserves_swiglu7(self):
        """Pair-aligned slicing keeps the swiglu7 gate/up pairing intact."""
        from torchtitan_npu.models.magi2_preview.feed_forward import swiglu7
        from torchtitan_npu.models.magi2_preview.grouped_linear import (
            GroupedLinear,
            slice_grouped_linear_by_pairs,
        )

        E, P, Cin = 3, 4, 5
        torch.manual_seed(1)
        x = torch.randn(6, Cin)
        m_splits = [1, 2, 3]
        weight = torch.randn(E * 2 * P, Cin)
        full = GroupedLinear(Cin, 2 * P, num_experts=E)
        with torch.no_grad():
            full.weight.copy_(weight)
        raw_full = full(x, m_splits)
        y_full = swiglu7(raw_full)
        for rank in range(TP):
            p0, p1 = _head_range(rank, P)
            local = GroupedLinear(Cin, 2 * (p1 - p0), num_experts=E)
            with torch.no_grad():
                local.weight.copy_(
                    slice_grouped_linear_by_pairs(weight, E, P, (p0, p1))
                )
            raw_local = local(x, m_splits)
            # The raw projection is an exact row subset of the full one ...
            assert torch.equal(
                raw_local,
                raw_full.view(6, P, 2)[:, p0:p1].reshape(6, 2 * (p1 - p0)),
            )
            # ... and swiglu7 of the local tensor pairs exactly the local
            # pairs (CPU SIMD may shift the last fp32 bit vs the wide op).
            assert torch.allclose(
                swiglu7(raw_local), y_full[:, p0:p1], atol=1e-6, rtol=1e-6
            )

    def test_row_split_partials_sum_to_full(self):
        from torchtitan_npu.models.magi2_preview.grouped_linear import (
            GroupedLinear,
        )

        E, out, Cin = 3, 7, 8
        torch.manual_seed(2)
        x = torch.randn(6, Cin)
        m_splits = [1, 2, 3]
        weight = torch.randn(E * out, Cin)
        full = GroupedLinear(Cin, out, num_experts=E)
        with torch.no_grad():
            full.weight.copy_(weight)
        y_full = full(x, m_splits)

        width = Cin // TP
        partial = 0
        for rank in range(TP):
            local = GroupedLinear(width, out, num_experts=E)
            with torch.no_grad():
                local.weight.copy_(
                    weight[:, rank * width : (rank + 1) * width].contiguous()
                )
            partial = partial + local(x[:, rank * width : (rank + 1) * width], m_splits)
        assert torch.allclose(partial, y_full)


# ---------------------------------------------------------------------------
# TP=2-emulated vs TP=1 equivalence (real partition, virtual collectives)
# ---------------------------------------------------------------------------


class TestAttentionTpEmulatedEquivalence:
    @pytest.mark.parametrize("backend", ("sdpa", "flex"))
    @pytest.mark.parametrize(
        "cu_seqlens",
        [None, torch.tensor([0, 5, 12], dtype=torch.int32)],
        ids=["single-segment", "multi-segment"],
    )
    def test_tp2_matches_tp1_fwd_bwd(self, backend, cu_seqlens):
        from torchtitan_npu.models.magi2_preview.grouped_linear import (
            slice_grouped_linear_by_heads,
        )
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _shard_attention_tp,
        )

        T, hidden = 12, 128
        x_orig, modality, rope = _attention_original_inputs(T, hidden)
        sort_idx, inv_sort_idx, m_splits = _sorted_args(modality)

        ref = _make_attention(backend)
        x_ref = x_orig.clone().requires_grad_(True)
        out_ref = ref(
            x_ref[sort_idx], rope, m_splits, sort_idx, inv_sort_idx, cu_seqlens
        )
        out_ref.sum().backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

        num_heads = ref.num_heads
        head_dim = ref.head_dim
        num_modality = ref.num_modality

        rank_attns, x_ranks, outs = [], [], []
        for rank in range(TP):
            attn = _make_attention(backend)
            _shard_attention_tp(attn, TP, rank)
            hs, he = _head_range(rank, num_heads)
            # rank-local bookkeeping of the partition
            assert attn.num_heads == he - hs
            assert attn.linear_g.out_features == he - hs
            assert attn.linear_qkv.out_features == 3 * (he - hs) * head_dim
            assert attn.linear_proj.in_features == (he - hs) * head_dim
            assert attn.sinks.shape == (1, he - hs)
            rank_attns.append(attn)
            x_r = x_orig.clone().requires_grad_(True)
            x_ranks.append(x_r)
            outs.append(
                attn(
                    x_r[sort_idx],
                    rope,
                    m_splits,
                    sort_idx,
                    inv_sort_idx,
                    cu_seqlens,
                )
            )

        # The module-boundary output all-reduce is the sum of the partials.
        out_tp = sum(outs)
        assert torch.allclose(out_tp, out_ref, atol=1e-4, rtol=1e-4)
        out_tp.sum().backward()

        for rank, attn in enumerate(rank_attns):
            hs, he = _head_range(rank, num_heads)
            grads = {n: p.grad for n, p in attn.named_parameters()}
            assert grads["linear_g.weight"] is not None
            expected = slice_grouped_linear_by_heads(
                ref_grads["linear_g.weight"],
                num_modality,
                num_heads,
                1,
                1,
                (hs, he),
            )
            assert torch.allclose(
                grads["linear_g.weight"], expected, atol=1e-4, rtol=1e-4
            )
            expected = slice_grouped_linear_by_heads(
                ref_grads["linear_qkv.weight"],
                num_modality,
                num_heads,
                head_dim,
                3,
                (hs, he),
            )
            assert torch.allclose(
                grads["linear_qkv.weight"], expected, atol=1e-4, rtol=1e-4
            )
            expected = ref_grads["linear_proj.weight"][
                :, hs * head_dim : he * head_dim
            ]
            assert torch.allclose(
                grads["linear_proj.weight"], expected, atol=1e-4, rtol=1e-4
            )
            assert torch.allclose(
                grads["sinks"], ref_grads["sinks"][:, hs:he], atol=1e-4, rtol=1e-4
            )

        # Replicated norms: the rank-local grads are partial; the wiring's
        # grad hooks all-reduce them, so their SUM must match the reference.
        for name in ("pre_norm.weight", "q_norm.weight", "k_norm.weight"):
            total = sum(
                dict(a.named_parameters())[name].grad for a in rank_attns
            )
            assert torch.allclose(total, ref_grads[name], atol=1e-4, rtol=1e-4), name

        # Input gradient: the pre-hook all-reduce sums the rank partials.
        x_grad_sum = sum(x.grad for x in x_ranks)
        assert torch.allclose(x_grad_sum, x_ref.grad, atol=1e-4, rtol=1e-4)


class TestDenseMlpTpEmulatedEquivalence:
    @pytest.mark.parametrize("num_modality", (1, 3))
    def test_tp2_matches_tp1_fwd_bwd(self, num_modality):
        from torchtitan_npu.models.magi2_preview.grouped_linear import (
            slice_grouped_linear_by_pairs,
        )
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _shard_dense_mlp_tp,
        )

        torch.manual_seed(4)
        ref = _make_dense_mlp(num_modality)
        inter = ref.up_gate_proj.out_features // 2
        x = torch.randn(6, 64)
        m_splits = [6] if num_modality == 1 else [1, 2, 3]

        x_ref = x.clone().requires_grad_(True)
        ref(x_ref, m_splits).sum().backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

        rank_mlps, x_ranks, outs = [], [], []
        for rank in range(TP):
            mlp = _make_dense_mlp(num_modality)
            _shard_dense_mlp_tp(mlp, TP, rank)
            p0, p1 = _head_range(rank, inter)
            assert mlp.up_gate_proj.out_features == 2 * (p1 - p0)
            assert mlp.down_proj.in_features == p1 - p0
            rank_mlps.append(mlp)
            x_r = x.clone().requires_grad_(True)
            x_ranks.append(x_r)
            outs.append(mlp(x_r, m_splits))

        assert torch.allclose(sum(outs), ref(x, m_splits), atol=1e-4, rtol=1e-4)
        sum(outs).sum().backward()

        for rank, mlp in enumerate(rank_mlps):
            p0, p1 = _head_range(rank, inter)
            grads = {n: p.grad for n, p in mlp.named_parameters()}
            expected = slice_grouped_linear_by_pairs(
                ref_grads["up_gate_proj.weight"], num_modality, inter, (p0, p1)
            )
            assert torch.allclose(
                grads["up_gate_proj.weight"], expected, atol=1e-4, rtol=1e-4
            )
            expected = ref_grads["down_proj.weight"][:, p0:p1]
            assert torch.allclose(
                grads["down_proj.weight"], expected, atol=1e-4, rtol=1e-4
            )

        total = sum(dict(m.named_parameters())["pre_norm.weight"].grad for m in rank_mlps)
        assert torch.allclose(total, ref_grads["pre_norm.weight"], atol=1e-4, rtol=1e-4)
        x_grad_sum = sum(x_r.grad for x_r in x_ranks)
        assert torch.allclose(x_grad_sum, x_ref.grad, atol=1e-4, rtol=1e-4)


class TestMoeLayerTpEmulatedEquivalence:
    def test_tp2_matches_tp1_fwd_bwd(self):
        from torchtitan_npu.models.magi2_preview.grouped_linear import (
            slice_grouped_linear_by_pairs,
        )
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _shard_moe_layer_tp,
        )

        ref = _make_moe_layer()
        num_experts = ref.moe_mlp.num_experts
        inter = ref._shared_intermediate_size
        x = torch.randn(6, 64)
        m_splits = [1, 2, 3]

        x_ref = x.clone().requires_grad_(True)
        ref(x_ref, m_splits).sum().backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

        rank_mlps, x_ranks, outs = [], [], []
        for rank in range(TP):
            mlp = _make_moe_layer()
            _shard_moe_layer_tp(mlp, TP, rank)
            hs, he = _head_range(rank, ref.moe_mlp.num_heads)
            assert mlp.moe_mlp.head_range == (hs, he)
            assert mlp.moe_mlp.sharded_input is True
            assert mlp._shared_intermediate_size == inter // TP
            assert mlp.split_linear.out_features == 64 // TP
            assert mlp.merge_linear.in_features == 64 // TP
            rank_mlps.append(mlp)
            x_r = x.clone().requires_grad_(True)
            x_ranks.append(x_r)
            outs.append(mlp(x_r, m_splits))

        assert torch.allclose(sum(outs), ref(x, m_splits), atol=1e-4, rtol=1e-4)
        sum(outs).sum().backward()

        for rank, mlp in enumerate(rank_mlps):
            hs, he = _head_range(rank, ref.moe_mlp.num_heads)
            p0, p1 = _head_range(rank, inter)
            c0, c1 = rank * 32, (rank + 1) * 32
            grads = {n: p.grad for n, p in mlp.named_parameters()}
            assert torch.allclose(
                grads["split_linear.weight"],
                ref_grads["split_linear.weight"][c0:c1],
                atol=1e-4,
                rtol=1e-4,
            )
            assert torch.allclose(
                grads["merge_linear.weight"],
                ref_grads["merge_linear.weight"][:, c0:c1],
                atol=1e-4,
                rtol=1e-4,
            )
            rows = slice(hs * num_experts, he * num_experts)
            for name in ("moe_mlp.gate", "moe_mlp.W_gate", "moe_mlp.W_up", "moe_mlp.W_down"):
                assert torch.allclose(
                    grads[name], ref_grads[name][rows], atol=1e-4, rtol=1e-4
                ), name
            expected = slice_grouped_linear_by_pairs(
                ref_grads["shared_expert_fc1.weight"], 1, inter, (p0, p1)
            )
            assert torch.allclose(
                grads["shared_expert_fc1.weight"], expected, atol=1e-4, rtol=1e-4
            )
            expected = slice_grouped_linear_by_pairs(
                ref_grads["modality_specific_shared_expert_fc1.weight"],
                3,
                inter,
                (p0, p1),
            )
            assert torch.allclose(
                grads["modality_specific_shared_expert_fc1.weight"],
                expected,
                atol=1e-4,
                rtol=1e-4,
            )
            for name in ("shared_expert_fc2", "modality_specific_shared_expert_fc2"):
                assert torch.allclose(
                    grads[f"{name}.weight"],
                    ref_grads[f"{name}.weight"][:, p0:p1],
                    atol=1e-4,
                    rtol=1e-4,
                ), name
            # Router buffers carry no gradient in either form.
            assert mlp.moe_mlp.router.expert_bias.grad is None

        total = sum(dict(m.named_parameters())["pre_norm.weight"].grad for m in rank_mlps)
        assert torch.allclose(total, ref_grads["pre_norm.weight"], atol=1e-4, rtol=1e-4)
        x_grad_sum = sum(x_r.grad for x_r in x_ranks)
        assert torch.allclose(x_grad_sum, x_ref.grad, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# parallelize guards and divisibility validation (no process group)
# ---------------------------------------------------------------------------


class TestParallelizeTpGuards:
    def test_tp_with_cp_raises(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        model = Magi2PreviewModel(_small_model_config())
        with pytest.raises(NotImplementedError, match="TP with CP"):
            parallelize_magi2_preview(
                model,
                parallel_dims=_fake_parallel_dims(cp_enabled=True),
                training=None,
                model_converters=None,
                parallelism=None,
                compile_config=None,
                ac_config=None,
                dump_folder="",
            )

    def test_tp_with_ep_raises(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        model = Magi2PreviewModel(_small_model_config())
        with pytest.raises(NotImplementedError, match="TP with EP/ETP"):
            parallelize_magi2_preview(
                model,
                parallel_dims=_fake_parallel_dims(ep=2),
                training=None,
                model_converters=None,
                parallelism=None,
                compile_config=None,
                ac_config=None,
                dump_folder="",
            )

    def test_tp_with_etp_raises(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        model = Magi2PreviewModel(_small_model_config())
        with pytest.raises(NotImplementedError, match="TP with EP/ETP"):
            parallelize_magi2_preview(
                model,
                parallel_dims=_fake_parallel_dims(etp=2),
                training=None,
                model_converters=None,
                parallelism=None,
                compile_config=None,
                ac_config=None,
                dump_folder="",
            )

    def test_requires_attention_head_divisibility(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_tensor_parallel,
        )

        # hidden 160 / head_dim 32 = 5 heads, not divisible by TP=2
        model = Magi2PreviewModel(_small_model_config(hidden_size=160))
        with pytest.raises(ValueError, match="num attention heads"):
            _apply_tensor_parallel(model, tp_mesh=_fake_mesh(0))

    def test_requires_moe_head_divisibility(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_tensor_parallel,
        )

        # hidden 192: 6 attention heads (divisible by TP=2) but
        # moe_num_heads=3 is not.
        model = Magi2PreviewModel(
            _small_model_config(hidden_size=192, moe_num_heads=3)
        )
        with pytest.raises(ValueError, match="moe_num_heads"):
            _apply_tensor_parallel(model, tp_mesh=_fake_mesh(0))

    def test_requires_dense_intermediate_divisibility(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_tensor_parallel,
        )

        model = Magi2PreviewModel(
            _small_model_config(
                dense_intermediate_size=65, mm_layers=[], moe_layers=[]
            )
        )
        with pytest.raises(ValueError, match="dense intermediate size"):
            _apply_tensor_parallel(model, tp_mesh=_fake_mesh(0))

    def test_requires_shared_intermediate_divisibility(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_tensor_parallel,
        )

        model = Magi2PreviewModel(
            _small_model_config(shared_expert_intermediate_size=33)
        )
        with pytest.raises(ValueError, match="shared expert intermediate size"):
            _apply_tensor_parallel(model, tp_mesh=_fake_mesh(0))

    def test_rejects_multi_dim_mesh(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_tensor_parallel,
        )

        model = Magi2PreviewModel(_small_model_config())
        with pytest.raises(ValueError, match="1D mesh"):
            _apply_tensor_parallel(model, tp_mesh=SimpleNamespace(ndim=2))


# ---------------------------------------------------------------------------
# Wiring on a single-rank gloo mesh (CI-safe)
# ---------------------------------------------------------------------------


@pytest.fixture
def single_rank_process_group():
    """Shared single-rank gloo group (mirrors tests/conftest.py, redefined
    here so the file also runs standalone in integration harnesses)."""
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://localhost:12358",
            world_size=1,
            rank=0,
        )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


class TestApplyTensorParallelSingleRank:
    def _wire_small_model(self):
        from torch.distributed.device_mesh import DeviceMesh

        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_tensor_parallel,
        )

        config = _small_model_config()
        torch.manual_seed(7)
        model = Magi2PreviewModel(config)
        model.init_weights()
        model.train()
        reference = copy.deepcopy(model)
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("tp",))
        _apply_tensor_parallel(model, tp_mesh=mesh)
        return config, model, reference, mesh

    def test_state_dict_keys_and_placements(self, single_rank_process_group):
        from torch.distributed.tensor import DTensor, Shard

        config, model, reference, _ = self._wire_small_model()
        assert set(model.state_dict().keys()) == set(
            reference.state_dict().keys()
        )

        layers = model.block.layers
        # Layer 0 is mm (num_modality=3): grouped column splits stay plain
        # local slices; row splits and sinks are honest Shard DTensors.
        attn0 = layers["0"].attention
        assert not isinstance(attn0.linear_g.weight, DTensor)
        assert not isinstance(attn0.linear_qkv.weight, DTensor)
        assert attn0.linear_proj.weight.placements == (Shard(1),)
        assert attn0.sinks.placements == (Shard(1),)
        assert attn0.num_heads == config.hidden_size // config.head_dim
        # Layer 1 attention has num_modality=1: linear_g is Shard(0).
        attn1 = layers["1"].attention
        assert attn1.linear_g.weight.placements == (Shard(0),)
        assert not isinstance(attn1.linear_qkv.weight, DTensor)
        # Dense MLP on the mm layer: E=3 up_gate stays a plain slice.
        mlp0 = layers["0"].mlp
        assert not isinstance(mlp0.up_gate_proj.weight, DTensor)
        assert mlp0.down_proj.weight.placements == (Shard(1),)
        # MoE layer placements.
        mlp1 = layers["1"].mlp
        assert mlp1.split_linear.weight.placements == (Shard(0),)
        assert mlp1.merge_linear.weight.placements == (Shard(1),)
        assert mlp1.shared_expert_fc1.weight.placements == (Shard(0),)
        assert mlp1.shared_expert_fc2.weight.placements == (Shard(1),)
        assert not isinstance(
            mlp1.modality_specific_shared_expert_fc1.weight, DTensor
        )
        assert mlp1.modality_specific_shared_expert_fc2.weight.placements == (
            Shard(1),
        )
        moe = mlp1.moe_mlp
        assert moe.head_range == (0, config.moe_num_heads)
        assert moe.sharded_input is True
        for name in ("gate", "W_gate", "W_up", "W_down"):
            assert getattr(moe, name).placements == (Shard(0),)
        for name in ("expert_bias", "expert_bias_ema"):
            assert getattr(moe.router, name).placements == (Shard(0),)

    def test_forward_backward_match_nontp(self, single_rank_process_group):
        config, model, reference, _ = self._wire_small_model()
        x, inputs, labels = _model_inputs()

        x_tp = x.clone().requires_grad_(True)
        x_ref = x.clone().requires_grad_(True)
        pred = model(x_tp, **inputs)
        pred_ref = reference(x_ref, **inputs)
        assert torch.allclose(pred, pred_ref, atol=1e-5, rtol=1e-5)
        torch.nn.functional.mse_loss(pred, labels).backward()
        torch.nn.functional.mse_loss(pred_ref, labels).backward()
        assert torch.allclose(x_tp.grad, x_ref.grad, atol=1e-5, rtol=1e-5)

        ref_params = dict(reference.named_parameters())
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            grad = _local_value(param.grad)
            expected = _expected_local_grad(
                name, ref_params[name].grad, 0, 1, config
            )
            assert torch.allclose(grad, expected, atol=1e-5, rtol=1e-5), name

    def test_tp_and_fsdp_compose(self, single_rank_process_group):
        from torch.distributed.device_mesh import DeviceMesh

        from torchtitan_npu.models.magi2_preview.parallelize import _apply_fsdp

        config, model, reference, _ = self._wire_small_model()
        training = SimpleNamespace(
            mixed_precision_param="float32",
            mixed_precision_reduce="float32",
            enable_cpu_offload=False,
        )
        parallelism = SimpleNamespace(fsdp_reshard_after_forward="default")
        dp_mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))
        _apply_fsdp(
            model,
            dp_mesh,
            training=training,
            parallelism=parallelism,
            pp_enabled=False,
        )

        x, inputs, labels = _model_inputs()
        x_tp = x.clone().requires_grad_(True)
        x_ref = x.clone().requires_grad_(True)
        pred = model(x_tp, **inputs)
        pred_ref = reference(x_ref, **inputs)
        assert torch.allclose(pred, pred_ref, atol=1e-5, rtol=1e-5)
        torch.nn.functional.mse_loss(pred, labels).backward()
        torch.nn.functional.mse_loss(pred_ref, labels).backward()
        assert torch.allclose(x_tp.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
        ref_params = dict(reference.named_parameters())
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            grad = _local_value(param.grad)
            expected = _expected_local_grad(
                name, ref_params[name].grad, 0, 1, config
            )
            assert torch.allclose(grad, expected, atol=1e-5, rtol=1e-5), name

    def test_tp_and_ac_compose(self, single_rank_process_group):
        """TP is applied before activation checkpointing (kimi's ordering)."""
        from torchtitan.config import ActivationCheckpointConfig

        from torchtitan_npu.models.common.activation_checkpoint import (
            apply_moe_ac,
        )

        config, model, reference, _ = self._wire_small_model()
        apply_moe_ac(
            model.block,
            ActivationCheckpointConfig(mode="full"),
            model_compile_enabled=False,
            base_folder="/tmp/magi2_tp_ac_test",
        )

        x, inputs, labels = _model_inputs()
        x_tp = x.clone().requires_grad_(True)
        x_ref = x.clone().requires_grad_(True)
        pred = model(x_tp, **inputs)
        pred_ref = reference(x_ref, **inputs)
        assert torch.allclose(pred, pred_ref, atol=1e-5, rtol=1e-5)
        torch.nn.functional.mse_loss(pred, labels).backward()
        torch.nn.functional.mse_loss(pred_ref, labels).backward()
        assert torch.allclose(x_tp.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
        ref_params = dict(reference.named_parameters())
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            # AC wrapping inserts a _checkpoint_wrapped_module path segment.
            clean = name.replace("_checkpoint_wrapped_module.", "")
            grad = _local_value(param.grad)
            expected = _expected_local_grad(
                clean, ref_params[clean].grad, 0, 1, config
            )
            assert torch.allclose(grad, expected, atol=1e-5, rtol=1e-5), name

    def test_full_checkpoint_loads_into_tp_model(self, single_rank_process_group):
        """Full-weight loading via the DTensor distribute path.

        Degree 1 keeps every slice full-size, so the whole state dict (incl.
        the plain-sliced parameter classes) loads through the standard
        ``set_model_state_dict(full_state_dict=True)`` mechanism; the TP=2
        nightly test verifies the rank-local contents against the same
        distribute_tensor expectations.
        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )
        from torch.distributed.device_mesh import DeviceMesh

        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_tensor_parallel,
        )
        from torchtitan_npu.models.magi2_preview.state_dict_adapter import (
            Magi2PreviewStateDictAdapter,
        )

        config = _small_model_config()
        torch.manual_seed(7)
        model = Magi2PreviewModel(config)
        model.init_weights()
        model.train()
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("tp",))
        _apply_tensor_parallel(model, tp_mesh=mesh)

        adapter = Magi2PreviewStateDictAdapter(model_config=config)
        torch.manual_seed(5)
        hf_dict = {
            key: torch.randn_like(value)
            for key, value in Magi2PreviewModel(config).state_dict().items()
        }
        from_hf = adapter.from_hf(hf_dict)
        assert set(from_hf.keys()) == set(hf_dict.keys())
        # set_model_state_dict replaces the input dict's values with
        # DTensors as it distributes them; pin the full values first.
        full_state = {k: v.clone() for k, v in from_hf.items()}

        set_model_state_dict(
            model, from_hf, options=StateDictOptions(full_state_dict=True)
        )
        for name, param in model.named_parameters():
            assert torch.equal(_local_value(param), full_state[name]), name

        reference = Magi2PreviewModel(config)
        reference.load_state_dict(full_state)
        reference.train()
        x, inputs, _ = _model_inputs()
        pred = model(x, **inputs)
        pred_ref = reference(x, **inputs)
        assert torch.allclose(pred, pred_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Nightly: real 2-rank gloo process group (RUN_MODEL_PARALLEL_MULTI_RANK)
# ---------------------------------------------------------------------------


def _load_multi_rank_conventions():
    """Import ``tests/smoke_tests/model_parallel/_multi_rank.py``.

    Prefers the canonical package import (repo root leads sys.path, e.g.
    in CI); otherwise walks up from this file to find the repo ``tests``
    tree (harness layouts may shadow or detach it). Returns ``None``
    when the conventions module is unavailable.
    """
    try:
        from tests.smoke_tests.model_parallel import _multi_rank

        return _multi_rank
    except ImportError:
        import importlib.util
        import pathlib

        for parent in pathlib.Path(__file__).resolve().parents:
            path = (
                parent
                / "tests"
                / "smoke_tests"
                / "model_parallel"
                / "_multi_rank.py"
            )
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location(
                "magi2_tp_multi_rank_conventions", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return None


_multi_rank = _load_multi_rank_conventions()
if _multi_rank is not None:
    MULTI_RANK_AVAILABLE = _multi_rank.MULTI_RANK_AVAILABLE
    mark_multi_rank_nightly = _multi_rank.mark_multi_rank_nightly
    with_comms = _multi_rank.with_comms
else:
    MULTI_RANK_AVAILABLE = False

    def mark_multi_rank_nightly(test_obj):
        return test_obj

    with_comms = None

if MULTI_RANK_AVAILABLE:
    from torch.testing._internal.distributed._tensor.common_dtensor import (
        DTensorTestBase,
    )

    class _TwoRankCpuMultiRankTestBase(DTensorTestBase):
        @property
        def world_size(self):
            return 2

        @property
        def device_type(self):
            return "cpu"

    @mark_multi_rank_nightly
    class TestMagi2TpTwoRankMultiRank(_TwoRankCpuMultiRankTestBase):
        """TP=2 vs TP=1 fwd/bwd equivalence over a real gloo process group.

        Run with: torchrun --nproc_per_node=2 -m pytest \
            tests/unit_tests/models/test_magi2_tp.py -m nightly -k TwoRank
        """

        @with_comms
        def test_tp2_forward_backward_match_tp1(self):
            from torch.distributed.device_mesh import init_device_mesh

            from torchtitan_npu.models.magi2_preview.model import (
                Magi2PreviewModel,
            )
            from torchtitan_npu.models.magi2_preview.parallelize import (
                _apply_tensor_parallel,
            )

            mesh = init_device_mesh(
                self.device_type, (self.world_size,), mesh_dim_names=("tp",)
            )
            config = _small_model_config()
            torch.manual_seed(7)
            model = Magi2PreviewModel(config)
            model.init_weights()
            model.train()
            state = {k: v.clone() for k, v in model.state_dict().items()}
            _apply_tensor_parallel(model, tp_mesh=mesh)
            assert set(model.state_dict().keys()) == set(state.keys())

            reference = Magi2PreviewModel(config)
            reference.load_state_dict(state)
            reference.train()

            rank = mesh.get_local_rank()
            # The partition produced exactly this rank's slices of the
            # full weights (the distribute step the checkpoint load
            # reproduces via DTensor placements / the slicing helpers).
            ref_state = dict(reference.named_parameters())
            for name, param in model.named_parameters():
                expected = _expected_local_grad(
                    name, ref_state[name].detach(), rank, self.world_size, config
                )
                assert torch.equal(_local_value(param), expected), name

            x, inputs, labels = _model_inputs()
            x_tp = x.clone().requires_grad_(True)
            x_ref = x.clone().requires_grad_(True)
            pred = model(x_tp, **inputs)
            pred_ref = reference(x_ref, **inputs)
            assert torch.allclose(pred, pred_ref, atol=1e-4, rtol=1e-4)

            torch.nn.functional.mse_loss(pred, labels).backward()
            torch.nn.functional.mse_loss(pred_ref, labels).backward()
            # Sequence-replicated TP: the input gradient is complete on
            # every rank (module pre-hook all-reduce).
            assert torch.allclose(x_tp.grad, x_ref.grad, atol=1e-4, rtol=1e-4)

            ref_params = dict(reference.named_parameters())
            for name, param in model.named_parameters():
                assert param.grad is not None, name
                grad = _local_value(param.grad)
                expected = _expected_local_grad(
                    name, ref_params[name].grad, rank, self.world_size, config
                )
                assert torch.allclose(
                    grad, expected, atol=1e-4, rtol=1e-4
                ), name
