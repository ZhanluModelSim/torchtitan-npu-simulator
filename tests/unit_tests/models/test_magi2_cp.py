# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview Ulysses CP tests (see torchtitan_npu/.../cp_ulysses.py).

CI-safe single-process coverage:
- pure algebra of the Ulysses all-to-all swap layouts and the model
  entry/exit sequence sharding, with cp_degree simulated by chunking;
- end-to-end CP=2-vs-CP=1 equivalence (attention and full model, fwd and
  bwd) by emulating the funcol collectives across two virtual ranks run
  on threads (the emulated collectives implement the real autograd
  semantics: all-to-all is self-adjoint and the all-gather backward is a
  reduce-scatter with sum), exercising the real hooks/wiring code;
- single-rank gloo wiring (degree 1) exercising the real funcol path.

Nightly-gated real-collective coverage (RUN_MODEL_PARALLEL_MULTI_RANK,
following tests/smoke_tests/model_parallel/_multi_rank.py conventions):

    torchrun --nproc_per_node=2 -m pytest \
        tests/unit_tests/models/test_magi2_cp.py -m nightly -k TwoRank
"""

import contextlib
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.distributed.tensor import DTensor

CP = 2  # virtual CP degree used throughout the single-process tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_mesh(rank: int, degree: int = CP):
    """Duck-typed stand-in for a 1D DeviceMesh (size/local_rank/group API)."""
    return SimpleNamespace(
        ndim=1,
        size=lambda: degree,
        get_local_rank=lambda: rank,
        get_group=lambda: None,
    )


class _CollectiveEmulator:
    """In-process emulation of SPMD collectives between virtual CP ranks.

    Ranks call ``exchange`` with their send data in the same collective
    order (forward and backward alike); a call completes once every rank
    has staged its send, then every rank reads its result. Modes:
    - ``a2a``: all_to_all_single over dim 0 with even chunks (self-adjoint,
      used for both the Ulysses forward swaps and their backward);
    - ``gather``: dim-0 concatenation delivered to every rank;
    - ``reduce_scatter_sum``: the all-gather backward (sum of every rank's
      grad slice for this rank's shard).
    """

    def __init__(self, degree: int, timeout: float = 90.0):
        self.degree = degree
        self.timeout = timeout
        self._cond = threading.Condition()
        self._next_call = 0
        self._staged = {}
        self._results = {}

    def exchange(self, send: torch.Tensor, rank: int, mode: str) -> torch.Tensor:
        with self._cond:
            call_id = self._next_call
            staged = self._staged.setdefault(call_id, {})
            staged[rank] = send
            if len(staged) == self.degree:
                self._results[call_id] = self._resolve(staged, mode)
                self._next_call += 1
                self._cond.notify_all()
            while call_id not in self._results:
                if not self._cond.wait(timeout=self.timeout):
                    raise AssertionError(
                        f"emulated collective {call_id} timed out "
                        f"(rank {rank} waiting for peers)"
                    )
            return self._results[call_id][rank]

    def _resolve(self, staged: dict, mode: str) -> dict:
        degree = self.degree
        if mode == "gather":
            full = torch.cat([staged[i] for i in range(degree)], dim=0)
            return {r: full.clone() for r in range(degree)}
        if mode == "reduce_scatter_sum":
            shard_len = staged[0].shape[0] // degree
            return {
                r: torch.stack(
                    [
                        staged[i].narrow(0, r * shard_len, shard_len)
                        for i in range(degree)
                    ]
                ).sum(dim=0)
                for r in range(degree)
            }
        # mode == "a2a": chunk i of rank j's send goes to rank i
        return {
            r: torch.cat([staged[i][r] for i in range(degree)], dim=0)
            for r in range(degree)
        }


class _EmulatedDispatch(torch.autograd.Function):
    """(S, H, D) -> (cp*S, H/cp, D) via the emulated all-to-all."""

    @staticmethod
    def forward(ctx, x, rank, emulator, degree):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            _dispatch_send_layout,
        )

        ctx.rank = rank
        ctx.emulator = emulator
        ctx.degree = degree
        ctx.input_shape = x.shape
        send = _dispatch_send_layout(x, degree)
        recv = emulator.exchange(send, rank, "a2a")
        S, H, D = x.shape
        return recv.view(degree * S, H // degree, D)

    @staticmethod
    def backward(ctx, grad_output):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            _undispatch_send_layout,
        )

        S, H, D = ctx.input_shape
        degree = ctx.degree
        grad_send = _undispatch_send_layout(grad_output, degree)
        grad_recv = ctx.emulator.exchange(grad_send, ctx.rank, "a2a")
        return (
            grad_recv.view(degree, S, H // degree, D)
            .permute(1, 0, 2, 3)
            .reshape(S, H, D),
            None,
            None,
            None,
        )


class _EmulatedUndispatch(torch.autograd.Function):
    """(cp*S, H/cp, D) -> (S, H, D) via the emulated all-to-all."""

    @staticmethod
    def forward(ctx, x, rank, emulator, degree):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            _undispatch_send_layout,
        )

        ctx.rank = rank
        ctx.emulator = emulator
        ctx.degree = degree
        ctx.input_shape = x.shape
        send = _undispatch_send_layout(x, degree)
        recv = emulator.exchange(send, rank, "a2a")
        T, Hc, D = x.shape
        S = T // degree
        return (
            recv.view(degree, S, Hc, D)
            .permute(1, 0, 2, 3)
            .reshape(S, degree * Hc, D)
        )

    @staticmethod
    def backward(ctx, grad_output):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            _dispatch_send_layout,
        )

        T, Hc, D = ctx.input_shape
        degree = ctx.degree
        S = T // degree
        grad_send = _dispatch_send_layout(grad_output, degree)
        grad_recv = ctx.emulator.exchange(grad_send, ctx.rank, "a2a")
        return grad_recv.view(degree * S, Hc, D), None, None, None


class _EmulatedGather(torch.autograd.Function):
    """(S, ...) -> (cp*S, ...) all-gather; backward is reduce-scatter sum."""

    @staticmethod
    def forward(ctx, x, rank, emulator, degree):
        ctx.rank = rank
        ctx.emulator = emulator
        ctx.local_len = x.shape[0]
        return emulator.exchange(x.contiguous(), rank, "gather")

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.emulator.exchange(
            grad_output.contiguous(), ctx.rank, "reduce_scatter_sum"
        )
        return grad_input, None, None, None


@contextlib.contextmanager
def _patched_collectives(emulator: _CollectiveEmulator, degree: int):
    """Replace cp_ulysses funcol wrappers with emulated autograd swaps."""
    prefix = "torchtitan_npu.models.magi2_preview.cp_ulysses"

    def fake_dispatch(x, *, mesh):
        return _EmulatedDispatch.apply(x, mesh.get_local_rank(), emulator, degree)

    def fake_undispatch(x, *, mesh):
        return _EmulatedUndispatch.apply(
            x, mesh.get_local_rank(), emulator, degree
        )

    def fake_gather(x, *, mesh):
        return _EmulatedGather.apply(x, mesh.get_local_rank(), emulator, degree)

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch(f"{prefix}.ulysses_dispatch", fake_dispatch)
        )
        stack.enter_context(
            mock.patch(f"{prefix}.ulysses_undispatch", fake_undispatch)
        )
        stack.enter_context(mock.patch(f"{prefix}.gather_seq", fake_gather))
        yield


def _run_ranks(fn, degree: int = CP, timeout: float = 180.0):
    """Run fn(rank) on `degree` threads; surface the first worker error."""
    results, errors = {}, []

    def target(rank):
        try:
            results[rank] = fn(rank)
        except Exception as exc:  # re-raised in the main thread below
            errors.append(exc)

    threads = [
        threading.Thread(target=target, args=(rank,)) for rank in range(degree)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
    if any(thread.is_alive() for thread in threads):
        raise AssertionError("virtual CP ranks deadlocked")
    if errors:
        raise errors[0]
    return [results[rank] for rank in range(degree)]


def _init_attention(attn):
    """Fill a standalone Magi2Attention like Magi2PreviewModel.init_weights."""
    from torchtitan_npu.models.magi2_preview.grouped_linear import GroupedLinear
    from torchtitan_npu.models.magi2_preview.norms import MultiModalityRMSNorm

    with torch.no_grad():
        for module in attn.modules():
            if isinstance(module, MultiModalityRMSNorm):
                torch.nn.init.zeros_(module.weight)
            elif isinstance(module, GroupedLinear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        attn.sinks.normal_(mean=0.0, std=0.02)
    return attn


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
    return _init_attention(attn)


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


def _small_model_config():
    """Tiny MAGI-2-preview config: 4 attention heads, divisible by CP=2."""
    from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

    return Magi2PreviewModel.Config(
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


def _build_rank_model(config, state_dict, rank):
    """One virtual CP rank's model: shared weights, real style wiring."""
    from torchtitan_npu.models.magi2_preview.cp_ulysses import (
        apply_magi2_ulysses_cp,
    )
    from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

    model = Magi2PreviewModel(config)
    model.load_state_dict(state_dict)
    model.train()
    apply_magi2_ulysses_cp(model, cp_mesh=_fake_mesh(rank))
    return model


def _assert_summed_grads(models, ref_grads, factor: float, atol=1e-4, rtol=1e-4):
    """Cross-rank gradient accounting of the faithful emulation.

    Each rank's backward routes cross-rank gradients exactly like the real
    funcol collectives, so a replicated parameter's reference gradient is
    ``factor`` x the SUM of the per-rank gradients, and a head-sharded
    ``sinks`` grad matches the corresponding head slice of the reference
    directly (scaled by ``factor``). ``ref_grads`` maps parameter names to
    the reference-run gradients.
    """
    for name, expected in ref_grads.items():
        if name == "sinks" or name.endswith(".sinks"):
            heads_per_rank = expected.shape[1] // len(models)
            for rank, model in enumerate(models):
                shard = dict(model.named_parameters())[name]
                assert shard.grad is not None, name
                assert torch.allclose(
                    shard.grad,
                    factor
                    * expected[
                        :, rank * heads_per_rank : (rank + 1) * heads_per_rank
                    ],
                    atol=atol,
                    rtol=rtol,
                ), f"sinks grad mismatch (rank {rank})"
            continue
        total = None
        for model in models:
            grad = dict(model.named_parameters())[name].grad
            assert grad is not None, f"missing grad for {name}"
            total = grad if total is None else total + grad
        assert torch.allclose(
            total, factor * expected, atol=atol, rtol=rtol
        ), f"grad mismatch for {name}"


# ---------------------------------------------------------------------------
# Pure algebra of the Ulysses swaps and entry/exit sharding
# ---------------------------------------------------------------------------


class TestUlyssesSwapAlgebra:
    def test_dispatch_layout_maps_head_groups_to_ranks(self):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            _dispatch_send_layout,
        )

        T, H, D = 8, 4, 3
        S = T // CP
        full = torch.arange(T * H * D, dtype=torch.float32).reshape(T, H, D)
        sends = [
            _dispatch_send_layout(full.narrow(0, r * S, S), CP)
            for r in range(CP)
        ]
        heads_per_rank = H // CP
        for r in range(CP):
            recv = torch.cat([sends[i][r] for i in range(CP)], dim=0)
            assert torch.equal(
                recv, full[:, r * heads_per_rank : (r + 1) * heads_per_rank, :]
            )

    def test_undispatch_layout_returns_local_token_shards(self):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            _undispatch_send_layout,
        )

        T, H, D = 8, 4, 3
        S = T // CP
        heads_per_rank = H // CP
        full_out = torch.arange(T * H * D, dtype=torch.float32).reshape(T, H, D)
        sends = [
            _undispatch_send_layout(
                full_out[:, r * heads_per_rank : (r + 1) * heads_per_rank, :], CP
            )
            for r in range(CP)
        ]
        for r in range(CP):
            # same local token shard from every peer -> cat over heads
            recv = torch.cat([sends[i][r] for i in range(CP)], dim=1)
            assert torch.equal(recv, full_out.narrow(0, r * S, S))

    def test_dispatch_undispatch_round_trip_is_identity(self):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            _dispatch_send_layout,
            _undispatch_send_layout,
        )

        T, H, D = 8, 4, 3
        S = T // CP
        full = torch.randn(T, H, D)
        # dispatch: every rank's send layout, then each rank's receive
        # (full sequence in rank order, local head group)
        dispatch_sends = [
            _dispatch_send_layout(full.narrow(0, i * S, S), CP)
            for i in range(CP)
        ]
        dispatched = [
            torch.cat([dispatch_sends[i][r] for i in range(CP)], dim=0)
            for r in range(CP)
        ]
        # undispatch: back to the local token shard with full heads
        undispatch_sends = [
            _undispatch_send_layout(dispatched[r], CP) for r in range(CP)
        ]
        for r in range(CP):
            recv = torch.cat([undispatch_sends[i][r] for i in range(CP)], dim=1)
            assert torch.equal(recv, full.narrow(0, r * S, S))


class TestModelCpEntryExitAlgebra:
    def test_cp_shard_round_trip_and_divisibility_error(self):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import CpContext
        from torchtitan_npu.models.magi2_preview.model import _cp_shard

        tensor = torch.arange(32 * 5, dtype=torch.float32).reshape(32, 5)
        shards = [
            _cp_shard(tensor, CpContext(mesh=None, degree=CP, rank=r))
            for r in range(CP)
        ]
        assert all(shard.shape == (16, 5) for shard in shards)
        assert torch.equal(torch.cat(shards, dim=0), tensor)
        with pytest.raises(ValueError, match="divisible"):
            _cp_shard(torch.randn(7, 2), CpContext(mesh=None, degree=CP, rank=0))

    def test_local_modality_sorts_partition_global_counts(self):
        from torchtitan_npu.models.magi2_preview.model import Modality

        _, inputs, _ = _model_inputs()
        modality = inputs["modality_mapping"].clone()
        modality[modality == Modality.TIME] = Modality.TEXT
        global_counts = torch.bincount(modality, minlength=3)

        T = modality.shape[0]
        S = T // CP
        local_counts = torch.zeros(3, dtype=torch.long)
        for r in range(CP):
            shard = modality.narrow(0, r * S, S)
            sort_idx = torch.argsort(shard)
            # sorted rows invert back to the shard (per-rank bookkeeping)
            assert torch.equal(shard[sort_idx][torch.argsort(sort_idx)], shard)
            local_counts += torch.bincount(shard, minlength=3)
        assert torch.equal(local_counts, global_counts)

    def test_exit_gather_reconstructs_full_prediction(self):
        # gather_seq semantics: dim-0 concatenation in rank order, which is
        # the original token order because shards are contiguous.
        shards = [torch.randn(8, 64) for _ in range(CP)]
        full = torch.cat(shards, dim=0)
        assert full.shape == (8 * CP, 64)
        for r in range(CP):
            assert torch.equal(full.narrow(0, r * 8, 8), shards[r])


# ---------------------------------------------------------------------------
# Emulated CP=2 vs CP=1 equivalence (real hooks/wiring, virtual collectives)
# ---------------------------------------------------------------------------


class TestAttentionCpEmulatedEquivalence:
    @pytest.mark.parametrize("backend", ("sdpa", "flex"))
    @pytest.mark.parametrize(
        "cu_seqlens",
        [None, torch.tensor([0, 5, 12], dtype=torch.int32)],
        ids=["single-segment", "multi-segment"],
    )
    def test_cp2_matches_cp1_fwd_bwd(self, backend, cu_seqlens):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            Magi2UlyssesAttentionCP,
        )

        T, hidden = 12, 128
        S = T // CP
        x_orig, modality, rope = _attention_original_inputs(T, hidden)

        x_ref = x_orig.clone().requires_grad_(True)
        ref = _make_attention(backend)
        sort_idx, inv_sort_idx, m_splits = _sorted_args(modality)
        out_ref = ref(
            x_ref[sort_idx], rope, m_splits, sort_idx, inv_sort_idx, cu_seqlens
        )
        # per-rank local-sum losses partition the full sum loss used by ref
        out_ref.sum().backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

        style = Magi2UlyssesAttentionCP()
        rank_attns, x_ranks = [], []
        for r in range(CP):
            attn_r = _make_attention(backend)
            style._apply(attn_r, _fake_mesh(r))
            rank_attns.append(attn_r)
            x_ranks.append(x_orig.clone().requires_grad_(True))
        # head-sharded sinks partition the full parameter exactly
        assert torch.equal(
            torch.cat([attn.sinks.data for attn in rank_attns], dim=1),
            ref.sinks.data,
        )

        def run_rank(rank):
            x_shard = x_ranks[rank].narrow(0, rank * S, S)
            sort_idx_r, inv_sort_idx_r, m_splits_r = _sorted_args(
                modality.narrow(0, rank * S, S)
            )
            out = rank_attns[rank](
                x_shard[sort_idx_r],
                rope.narrow(0, rank * S, S),
                m_splits_r,
                sort_idx_r,
                inv_sort_idx_r,
                cu_seqlens,
            )
            out.sum().backward()
            return out.detach(), inv_sort_idx_r

        emulator = _CollectiveEmulator(CP)
        with _patched_collectives(emulator, CP):
            rank_outs = _run_ranks(run_rank)

        # fwd: per-rank rows (original order) match the CP=1 full output
        out_ref_orig = out_ref[inv_sort_idx]
        for r, (out_r, inv_r) in enumerate(rank_outs):
            assert torch.allclose(
                out_r[inv_r],
                out_ref_orig.narrow(0, r * S, S),
                atol=1e-4,
                rtol=1e-4,
            ), f"rank {r} fwd mismatch"

        # bwd: no exit gather at attention level, so the summed per-rank
        # gradients equal the CP=1 gradients exactly (factor 1).
        _assert_summed_grads(rank_attns, ref_grads, factor=1.0)
        x_grad_sum = sum(x.grad for x in x_ranks)
        assert torch.allclose(x_grad_sum, x_ref.grad, atol=1e-4, rtol=1e-4)


class TestModelCpEmulatedEquivalence:
    def _run_cp_ranks(self, config, state_dict, x_ranks, inputs, labels, loss_scale):
        """Forward+backward both virtual CP ranks (full batch on each)."""
        models = [_build_rank_model(config, state_dict, r) for r in range(CP)]
        emulator = _CollectiveEmulator(CP)

        def run_rank(rank):
            # every rank receives the FULL batch; forward slices at entry
            pred = models[rank](x_ranks[rank], **inputs)
            (
                torch.nn.functional.mse_loss(pred, labels) * loss_scale
            ).backward()
            return pred.detach()

        with _patched_collectives(emulator, CP):
            preds = _run_ranks(run_rank)
        return models, preds

    def test_cp2_matches_cp1_with_model_side_compensation(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        config = _small_model_config()
        torch.manual_seed(7)
        ref = Magi2PreviewModel(config)
        ref.init_weights()
        ref.train()
        state_dict = {k: v.clone() for k, v in ref.state_dict().items()}
        x, inputs, labels = _model_inputs()

        # Un-scaled CP=1 reference run.
        x_ref = x.clone().requires_grad_(True)
        pred_ref = ref(x_ref, **inputs)
        torch.nn.functional.mse_loss(pred_ref, labels).backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

        # Raw full-label loss on every rank: the model's exit-gather
        # gradient compensation restores the CP=1 gradient scale, so the
        # summed rank gradients equal the CP=1 gradients directly.
        x_ranks = [x.clone().requires_grad_(True) for _ in range(CP)]
        models, preds = self._run_cp_ranks(
            config, state_dict, x_ranks, inputs, labels, 1.0
        )
        assert models[0].cp_degree == CP and models[0].cp_rank == 0
        assert torch.allclose(preds[0], preds[1], atol=1e-4, rtol=1e-4)
        assert torch.allclose(preds[0], pred_ref, atol=1e-4, rtol=1e-4)
        _assert_summed_grads(models, ref_grads, factor=1.0)
        x_grad_sum = sum(x_r.grad for x_r in x_ranks)
        assert torch.allclose(x_grad_sum, x_ref.grad, atol=1e-4, rtol=1e-4)

        # An additional manual loss scale simply multiplies the (already
        # compensated) summed gradients, proving no hidden compensation
        # happens outside the model.
        x_ranks = [x.clone().requires_grad_(True) for _ in range(CP)]
        models, preds = self._run_cp_ranks(
            config, state_dict, x_ranks, inputs, labels, 0.5
        )
        assert torch.allclose(preds[0], pred_ref, atol=1e-4, rtol=1e-4)
        _assert_summed_grads(models, ref_grads, factor=0.5)
        x_grad_sum = sum(x_r.grad for x_r in x_ranks)
        assert torch.allclose(
            x_grad_sum, 0.5 * x_ref.grad, atol=1e-4, rtol=1e-4
        )


# ---------------------------------------------------------------------------
# Wiring: single-rank gloo (real funcol), guards, state-dict keys
# ---------------------------------------------------------------------------


class TestCpWiringSingleRank:
    @pytest.mark.usefixtures("single_rank_process_group")
    def test_degree1_wiring_matches_noncp_fwd_bwd(self):
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            apply_magi2_ulysses_cp,
        )
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("cp",))
        config = _small_model_config()
        torch.manual_seed(7)
        ref = Magi2PreviewModel(config)
        ref.init_weights()
        ref.train()
        state_dict = {k: v.clone() for k, v in ref.state_dict().items()}
        model = Magi2PreviewModel(config)
        model.load_state_dict(state_dict)
        model.train()

        apply_magi2_ulysses_cp(model, cp_mesh=mesh)
        assert model.cp_degree == 1 and model.cp_rank == 0
        assert model.cp_context is not None
        assert set(model.state_dict().keys()) == set(state_dict.keys())

        x, inputs, labels = _model_inputs()
        pred = model(x, **inputs)
        pred_ref = ref(x, **inputs)
        assert torch.allclose(pred, pred_ref, atol=1e-5, rtol=1e-5)
        torch.nn.functional.mse_loss(pred, labels).backward()
        torch.nn.functional.mse_loss(pred_ref, labels).backward()
        ref_params = dict(ref.named_parameters())
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            grad = param.grad
            if isinstance(grad, DTensor):
                # CP-sharded sinks carry a DTensor grad on the cp mesh.
                grad = grad.to_local()
            assert torch.allclose(
                grad, ref_params[name].grad, atol=1e-5, rtol=1e-5
            ), f"grad mismatch for {name}"

    @pytest.mark.usefixtures("single_rank_process_group")
    def test_inline_cp_context_matches_noncp_attention(self):
        """Hook-free forward(cp_context=...) runs the same swap primitives."""
        from torch.distributed.device_mesh import init_device_mesh

        from torchtitan_npu.models.magi2_preview.cp_ulysses import CpContext

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("cp",))
        ctx = CpContext(mesh=mesh, degree=1, rank=0)
        plain = _make_attention("sdpa")
        hooked = _make_attention("sdpa")
        x, modality, rope = _attention_original_inputs()
        sort_idx, inv_sort_idx, m_splits = _sorted_args(modality)
        args = (x[sort_idx], rope, m_splits, sort_idx, inv_sort_idx, None)
        out_plain = plain(*args)
        out_cp = hooked(*args, cp_context=ctx)
        assert torch.allclose(out_cp, out_plain, atol=1e-5, rtol=1e-5)


class TestWiringGuards:
    def test_cp_and_ep_together_wired(self):
        """CP+EP no longer raises: attention keeps the Ulysses swaps on the
        cp mesh and the combined cp x ep MoE dispatch (regime b) is wired
        by parallelize._apply_moe_parallel afterwards."""
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            apply_magi2_ulysses_cp,
        )
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        model = Magi2PreviewModel(_small_model_config())
        model = apply_magi2_ulysses_cp(
            model, cp_mesh=_fake_mesh(0), ep_degree=2
        )
        assert model.cp_degree == CP and model.cp_rank == 0
        for layer in model.block.layers.values():
            assert layer.attention.cp_context is not None

    def test_requires_head_divisibility(self):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            Magi2UlyssesAttentionCP,
        )

        # hidden 160 / head_dim 32 = 5 heads, not divisible by CP=2
        attn = _make_attention(hidden_size=160)
        with pytest.raises(ValueError, match="divisible"):
            Magi2UlyssesAttentionCP()._apply(attn, _fake_mesh(0))

    def test_rejects_double_application(self):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            Magi2UlyssesAttentionCP,
        )

        attn = _make_attention()
        style = Magi2UlyssesAttentionCP()
        style._apply(attn, _fake_mesh(0))
        with pytest.raises(ValueError, match="twice"):
            style._apply(attn, _fake_mesh(0))

    def test_rejects_non_magi2_module(self):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            Magi2UlyssesAttentionCP,
        )

        with pytest.raises(TypeError, match="Magi2Attention"):
            Magi2UlyssesAttentionCP()._apply(
                torch.nn.Linear(4, 4), _fake_mesh(0)
            )

    def test_wiring_keeps_state_dict_keys_and_shards_sinks(self):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            apply_magi2_ulysses_cp,
        )
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        torch.manual_seed(7)
        model = Magi2PreviewModel(_small_model_config())
        model.init_weights()
        full_sinks = [
            layer.attention.sinks.data.clone()
            for layer in model.block.layers.values()
        ]
        keys_before = set(model.state_dict().keys())

        apply_magi2_ulysses_cp(model, cp_mesh=_fake_mesh(0))
        assert set(model.state_dict().keys()) == keys_before
        for layer_id, layer in enumerate(model.block.layers.values()):
            attn = layer.attention
            # 4 attention heads / CP=2 -> 2 local heads per rank
            assert attn.sinks.shape == (1, 2)
            assert torch.equal(attn.sinks.data, full_sinks[layer_id][:, :2])
            assert attn.cp_context is not None


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
                "magi2_cp_multi_rank_conventions", path
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
    class TestMagi2CpTwoRankMultiRank(_TwoRankCpuMultiRankTestBase):
        """CP=2 vs CP=1 fwd/bwd equivalence over a real gloo process group.

        Run with: torchrun --nproc_per_node=2 -m pytest \
            tests/unit_tests/models/test_magi2_cp.py -m nightly -k TwoRank
        """

        @with_comms
        def test_cp2_forward_backward_match_cp1(self):
            import torch.distributed as dist
            from torch.distributed.device_mesh import init_device_mesh

            from torchtitan_npu.models.magi2_preview.cp_ulysses import (
                apply_magi2_ulysses_cp,
            )
            from torchtitan_npu.models.magi2_preview.model import (
                Magi2PreviewModel,
            )

            mesh = init_device_mesh(
                self.device_type, (self.world_size,), mesh_dim_names=("cp",)
            )
            config = _small_model_config()
            torch.manual_seed(7)
            model = Magi2PreviewModel(config)
            model.init_weights()
            model.train()
            state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            apply_magi2_ulysses_cp(model, cp_mesh=mesh)
            assert model.cp_degree == self.world_size

            ref = Magi2PreviewModel(config)
            ref.load_state_dict(state_dict)
            ref.train()

            x, inputs, labels = _model_inputs()
            pred = model(x, **inputs)
            pred_ref = ref(x, **inputs)
            assert torch.allclose(pred, pred_ref, atol=1e-4, rtol=1e-4)

            # v1 loss decision: plain full-label loss on every rank; the
            # model's exit-gather gradient compensation restores the CP=1
            # gradient scale, so no manual loss scaling is needed here.
            torch.nn.functional.mse_loss(pred, labels).backward()
            torch.nn.functional.mse_loss(pred_ref, labels).backward()

            rank = mesh.get_local_rank()
            ref_params = dict(ref.named_parameters())
            for name, param in model.named_parameters():
                assert param.grad is not None, name
                expected = ref_params[name].grad
                if name == "sinks" or name.endswith(".sinks"):
                    heads_per_rank = expected.shape[1] // self.world_size
                    expected = expected[
                        :, rank * heads_per_rank : (rank + 1) * heads_per_rank
                    ]
                    grad = param.grad
                    if isinstance(grad, DTensor):
                        grad = grad.to_local()
                    assert torch.allclose(
                        grad, expected, atol=1e-4, rtol=1e-4
                    ), name
                    continue
                grad_sum = param.grad.clone()
                dist.all_reduce(grad_sum, op=dist.ReduceOp.SUM)
                assert torch.allclose(
                    grad_sum, expected, atol=1e-4, rtol=1e-4
                ), name
