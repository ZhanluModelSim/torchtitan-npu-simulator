# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview CP x EP combination tests (phase-3 gap 1).

Verifies that every combination of Ulysses CP and head-parallel MoE EP
produces the same forward output and backward gradients as the unsharded
(cp=1, ep=1) reference:

* CP=2 x EP=1 (Ulysses CP only, no MoE sharding)
* CP=1 x EP=2 (head-parallel MoE regime (a), no sequence sharding)
* CP=2 x EP=2 (combined regime (b) on the flattened cp x ep head mesh)

CI-safe single-process coverage emulates collectives via the
``_CollectiveEmulator`` / ``_ExchangeHub`` pattern from ``test_magi2_cp.py``
and ``test_magi2_expert_parallel.py``: virtual ranks run on threads and the
collectives (all-to-all, all-gather, all-reduce) are replaced by
synchronized tensor exchanges with correct autograd semantics.

Nightly-gated real-collective coverage (``RUN_MODEL_PARALLEL_MULTI_RANK``,
following ``tests/smoke_tests/model_parallel/_multi_rank.py`` conventions):

* 2-rank gloo: CP=2 x EP=1 and CP=1 x EP=2 vs unsharded
* 4-rank gloo: CP=2 x EP=2 vs unsharded

Run with::

    torchrun --nproc_per_node=2 -m pytest \\
        tests/unit_tests/models/test_magi2_cp_ep.py -m nightly -k TwoRank
    torchrun --nproc_per_node=4 -m pytest \\
        tests/unit_tests/models/test_magi2_cp_ep.py -m nightly -k FourRank
"""

import contextlib
import copy
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from torchtitan_npu.models.magi2_preview.expert_parallel import (
    EXPERT_PARAM_NAMES,
    MoEDispatchContext,
    all_reduce_head_parallel_input_grad,
    all_reduce_head_parallel_output,
    head_range_for_rank,
    head_seq_dispatch_permute,
    head_seq_undispatch_permute,
    shard_moe_core_by_head,
)
from torchtitan_npu.models.magi2_preview.feed_forward import (
    CoreMultiHeadMoE,
    MultiHeadMoELayer,
)

CP = 2
EP = 2
CP_EP = CP * EP
_VRANK = threading.local()


# ---------------------------------------------------------------------------
# Helpers: fake meshes, model config, inputs
# ---------------------------------------------------------------------------


def _fake_mesh(rank, degree):
    """Duck-typed stand-in for a 1D DeviceMesh (size/local_rank/group)."""
    return SimpleNamespace(
        ndim=1,
        size=lambda: degree,
        get_local_rank=lambda: rank,
        get_group=lambda: None,
    )


def _cp_members(rank):
    """Ranks sharing this rank's EP coordinate (the CP group)."""
    return tuple(r for r in range(CP_EP) if r % EP == rank % EP)


def _fake_cp_mesh(rank):
    members = _cp_members(rank)
    return SimpleNamespace(
        ndim=1,
        size=lambda: len(members),
        get_local_rank=lambda: members.index(rank),
        get_group=lambda: members,
    )


def _fake_combined_mesh(rank):
    """Duck-typed flattened cp x ep mesh (all CP_EP virtual ranks)."""
    members = tuple(range(CP_EP))
    return SimpleNamespace(
        ndim=1,
        size=lambda: CP_EP,
        get_local_rank=lambda: rank,
        get_group=lambda: members,
    )


def _small_model_config():
    """Tiny MAGI-2-preview config: 4 attn heads + 4 MoE heads."""
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


def _packed_inputs(seq_len=16, seed=5):
    """Full packed batch with every modality and multi-segment."""
    from torchtitan_npu.models.magi2_preview.model import Modality

    gen = torch.Generator().manual_seed(seed)
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
    x = torch.randn(seq_len, 64, generator=gen)
    coords = torch.randn(seq_len, 9, generator=gen)
    time_embedding = torch.randn(seq_len, 64, generator=gen)
    cu_seqlens = torch.tensor([0, 6, seq_len], dtype=torch.int32)
    labels = torch.zeros(seq_len, 64)
    video_rows = modality == Modality.VIDEO
    audio_rows = modality == Modality.AUDIO
    labels[video_rows, :48] = torch.randn(
        int(video_rows.sum()), 48, generator=gen
    )
    labels[audio_rows, :64] = torch.randn(
        int(audio_rows.sum()), 64, generator=gen
    )
    return (
        x,
        dict(
            coords_mapping=coords,
            modality_mapping=modality,
            time_embedding=time_embedding,
            cu_seqlens=cu_seqlens,
        ),
        labels,
    )


def _moe_layer_config():
    return MultiHeadMoELayer.Config(
        hidden_size=64,
        num_modality=1,
        moe_num_heads=4,
        num_experts=6,
        moe_top_k=3,
        expert_intermediate_size=32,
        shared_expert_intermediate_size=32,
    )


def _make_moe_layer(seed=3):
    torch.manual_seed(seed)
    layer = MultiHeadMoELayer(_moe_layer_config())
    with torch.no_grad():
        for param in layer.parameters():
            param.normal_(0.0, 0.02)
        layer.moe_mlp.router.expert_bias.normal_(0.0, 0.05)
    return layer


# ---------------------------------------------------------------------------
# Collective emulators
# ---------------------------------------------------------------------------


class _CollectiveEmulator:
    """In-process emulation of SPMD collectives between virtual ranks.

    Ranks call ``exchange`` with their send data in the same collective
    order (forward and backward alike); a call completes once every rank
    has staged its send, then every rank reads its result. Modes:

    * ``a2a``: all_to_all_single over dim 0 with even chunks (self-adjoint,
      used for Ulysses swaps and regime-(b) MoE dispatch);
    * ``gather``: dim-0 concatenation delivered to every rank;
    * ``reduce_scatter_sum``: the all-gather backward (sum of every rank's
      grad slice for this rank's shard);
    * ``all_reduce``: element-wise sum delivered to every rank (regime (a)
      output assembly and input-gradient conjugate).
    """

    def __init__(self, degree, timeout=90.0):
        self.degree = degree
        self.timeout = timeout
        self._cond = threading.Condition()
        self._next_call = 0
        self._staged = {}
        self._results = {}

    def exchange(self, send, rank, mode):
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

    def _resolve(self, staged, mode):
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
        if mode == "all_reduce":
            total = sum(staged[i] for i in range(degree))
            return {r: total.clone() for r in range(degree)}
        # mode == "a2a": chunk i of rank j's send goes to rank i
        return {
            r: torch.cat([staged[i][r] for i in range(degree)], dim=0)
            for r in range(degree)
        }


class _ExchangeHub:
    """Per-member-set emulator registry (one emulator per process group)."""

    def __init__(self, timeout=90.0):
        self.timeout = timeout
        self._emulators = {}

    def exchange(self, send, rank, members, mode):
        members = tuple(members)
        emulator = self._emulators.get(members)
        if emulator is None:
            emulator = _CollectiveEmulator(
                len(members), timeout=self.timeout
            )
            self._emulators[members] = emulator
        return emulator.exchange(send, members.index(rank), mode)


# ---------------------------------------------------------------------------
# Emulated autograd functions
# ---------------------------------------------------------------------------


class _EmulatedSeqHeadSwap(torch.autograd.Function):
    """(S, H, D) -> (deg*S, H/deg, D) via emulated all-to-all."""

    @staticmethod
    def forward(ctx, x, rank, hub, members):
        ctx.rank = rank
        ctx.hub = hub
        ctx.members = tuple(members)
        ctx.input_shape = x.shape
        degree = len(members)
        send = head_seq_dispatch_permute(x, degree)
        recv = hub.exchange(send, rank, members, "a2a")
        seq_len, num_heads, d_head = x.shape
        return recv.view(degree * seq_len, num_heads // degree, d_head)

    @staticmethod
    def backward(ctx, grad_output):
        degree = len(ctx.members)
        seq_len, num_heads, d_head = ctx.input_shape
        heads = num_heads // degree
        send = grad_output.contiguous().view(degree, seq_len, heads, d_head)
        recv = ctx.hub.exchange(send, ctx.rank, ctx.members, "a2a")
        return (
            head_seq_undispatch_permute(
                recv.view(degree, seq_len, heads, d_head), degree
            ),
            None,
            None,
            None,
        )


class _EmulatedHeadSeqSwap(torch.autograd.Function):
    """(deg*S, H/deg, D) -> (S, H, D) via emulated all-to-all."""

    @staticmethod
    def forward(ctx, x, rank, hub, members):
        ctx.rank = rank
        ctx.hub = hub
        ctx.members = tuple(members)
        ctx.input_shape = x.shape
        degree = len(members)
        total, heads, d_head = x.shape
        seq_len = total // degree
        send = x.contiguous().view(degree, seq_len, heads, d_head)
        recv = hub.exchange(send, rank, members, "a2a")
        return head_seq_undispatch_permute(
            recv.view(degree, seq_len, heads, d_head), degree
        )

    @staticmethod
    def backward(ctx, grad_output):
        degree = len(ctx.members)
        total, heads, d_head = ctx.input_shape
        send = head_seq_dispatch_permute(grad_output, degree)
        recv = ctx.hub.exchange(send, ctx.rank, ctx.members, "a2a")
        return recv.view(total, heads, d_head), None, None, None


class _EmulatedGather(torch.autograd.Function):
    """(S, ...) -> (deg*S, ...) all-gather; backward reduce-scatter sum."""

    @staticmethod
    def forward(ctx, x, rank, hub, members):
        ctx.rank = rank
        ctx.hub = hub
        ctx.members = tuple(members)
        return hub.exchange(x.contiguous(), rank, members, "gather")

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = ctx.hub.exchange(
            grad_output.contiguous(),
            ctx.rank,
            ctx.members,
            "reduce_scatter_sum",
        )
        return grad_input, None, None, None


class _EmulatedAllReduceOutput(torch.autograd.Function):
    """Regime-(a) output: zero-pad + all-reduce forward; slice backward."""

    @staticmethod
    def forward(ctx, partial, head_start, head_end, num_heads, rank, hub):
        ctx.head_start = head_start
        ctx.head_end = head_end
        ctx.num_heads = num_heads
        ctx.rank = rank
        ctx.hub = hub
        tokens = partial.shape[0]
        local_heads = head_end - head_start
        d_head = partial.shape[-1] // local_heads
        full = partial.new_zeros(tokens, num_heads, d_head)
        full[:, head_start:head_end] = partial.view(
            tokens, local_heads, d_head
        )
        full = full.reshape(tokens, num_heads * d_head)
        return hub.exchange(full, rank, (0, 1), "all_reduce")

    @staticmethod
    def backward(ctx, grad_full):
        grad_local = grad_full.view(
            -1, ctx.num_heads, grad_full.shape[-1] // ctx.num_heads
        )[:, ctx.head_start : ctx.head_end]
        return (
            grad_local.reshape(grad_full.shape[0], -1),
            None,
            None,
            None,
            None,
            None,
        )


class _EmulatedAllReduceInputGrad(torch.autograd.Function):
    """Regime-(a) input: identity forward; all-reduce backward."""

    @staticmethod
    def forward(ctx, x, rank, hub):
        ctx.rank = rank
        ctx.hub = hub
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad = ctx.hub.exchange(
            grad_output.contiguous(), ctx.rank, (0, 1), "all_reduce"
        )
        return grad, None, None


# ---------------------------------------------------------------------------
# Patching helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _patched_cp_collectives(hub):
    """Replace cp_ulysses funcol with emulated all-to-all/gather."""
    prefix = "torchtitan_npu.models.magi2_preview.cp_ulysses"

    def fake_dispatch(x, *, mesh):
        return _EmulatedSeqHeadSwap.apply(
            x, _VRANK.rank, hub, tuple(mesh.get_group())
        )

    def fake_undispatch(x, *, mesh):
        return _EmulatedHeadSeqSwap.apply(
            x, _VRANK.rank, hub, tuple(mesh.get_group())
        )

    def fake_gather(x, *, mesh):
        return _EmulatedGather.apply(
            x, _VRANK.rank, hub, tuple(mesh.get_group())
        )

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch(f"{prefix}.ulysses_dispatch", fake_dispatch)
        )
        stack.enter_context(
            mock.patch(f"{prefix}.ulysses_undispatch", fake_undispatch)
        )
        stack.enter_context(
            mock.patch(f"{prefix}.gather_seq", fake_gather)
        )
        yield


@contextlib.contextmanager
def _patched_cp_ep_collectives(hub):
    """Replace both cp_ulysses and expert_parallel collectives."""
    cp_prefix = "torchtitan_npu.models.magi2_preview.cp_ulysses"
    ep_prefix = "torchtitan_npu.models.magi2_preview.expert_parallel"

    def fake_ulysses_dispatch(x, *, mesh):
        return _EmulatedSeqHeadSwap.apply(
            x, _VRANK.rank, hub, tuple(mesh.get_group())
        )

    def fake_ulysses_undispatch(x, *, mesh):
        return _EmulatedHeadSeqSwap.apply(
            x, _VRANK.rank, hub, tuple(mesh.get_group())
        )

    def fake_gather(x, *, mesh):
        return _EmulatedGather.apply(
            x, _VRANK.rank, hub, tuple(mesh.get_group())
        )

    def fake_ep_dispatch(x, group):
        if group is None or len(group) <= 1:
            return x
        return _EmulatedSeqHeadSwap.apply(
            x, _VRANK.rank, hub, tuple(group)
        )

    def fake_ep_undispatch(x, group):
        if group is None or len(group) <= 1:
            return x
        return _EmulatedHeadSeqSwap.apply(
            x, _VRANK.rank, hub, tuple(group)
        )

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch(
                f"{cp_prefix}.ulysses_dispatch", fake_ulysses_dispatch
            )
        )
        stack.enter_context(
            mock.patch(
                f"{cp_prefix}.ulysses_undispatch", fake_ulysses_undispatch
            )
        )
        stack.enter_context(
            mock.patch(f"{cp_prefix}.gather_seq", fake_gather)
        )
        stack.enter_context(
            mock.patch(f"{ep_prefix}.ep_dispatch", fake_ep_dispatch)
        )
        stack.enter_context(
            mock.patch(f"{ep_prefix}.ep_undispatch", fake_ep_undispatch)
        )
        yield


@contextlib.contextmanager
def _patched_regime_a(hub):
    """Replace regime-(a) all-reduce wrappers with emulated versions."""
    prefix = "torchtitan_npu.models.magi2_preview.expert_parallel"

    def fake_output(partial, head_range, num_heads, group):
        head_start, head_end = head_range
        return _EmulatedAllReduceOutput.apply(
            partial, head_start, head_end, num_heads, _VRANK.rank, hub
        )

    def fake_input_grad(x, group):
        return _EmulatedAllReduceInputGrad.apply(x, _VRANK.rank, hub)

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch(
                f"{prefix}.all_reduce_head_parallel_output", fake_output
            )
        )
        stack.enter_context(
            mock.patch(
                f"{prefix}.all_reduce_head_parallel_input_grad",
                fake_input_grad,
            )
        )
        yield


def _run_ranks(fn, degree, timeout=180.0):
    """Run fn(rank) on ``degree`` threads; surface the first worker error."""
    results, errors = {}, []

    def target(rank):
        try:
            _VRANK.rank = rank
            results[rank] = fn(rank)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=target, args=(r,)) for r in range(degree)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)
    if any(t.is_alive() for t in threads):
        raise AssertionError("virtual ranks deadlocked")
    if errors:
        raise errors[0]
    return [results[r] for r in range(degree)]


# ---------------------------------------------------------------------------
# Regime-(a) EP-only sharding helpers
# ---------------------------------------------------------------------------


def _shard_layer_regime_a(layer, rank, degree):
    """Shard one MoE layer for regime (a): hooks on the MoE core, not the
    layer (split_linear/merge_linear stay per-token local on the full
    hidden width; the all-reduce combines the core's head-sharded partial
    outputs across the EP mesh)."""
    core = layer.moe_mlp
    head_range = head_range_for_rank(rank, degree, core.num_heads)
    shard_moe_core_by_head(core, head_range)
    group = tuple(range(degree))
    core.register_forward_pre_hook(
        lambda mod, inputs: (
            all_reduce_head_parallel_input_grad(inputs[0], group),
        )
        + tuple(inputs[1:])
    )
    core.register_forward_hook(
        lambda mod, inputs, output: all_reduce_head_parallel_output(
            output, core.head_range, core.num_heads, group
        )
    )
    return layer


def _shard_layer_regime_a_emulated(layer, rank, degree, hub):
    """Shard one MoE layer for emulated regime (a): hooks on moe_mlp."""
    core = layer.moe_mlp
    head_range = head_range_for_rank(rank, degree, core.num_heads)
    shard_moe_core_by_head(core, head_range)

    def pre_hook(mod, inputs):
        x = _EmulatedAllReduceInputGrad.apply(inputs[0], rank, hub)
        return (x,) + tuple(inputs[1:])

    def post_hook(mod, inputs, output):
        return _EmulatedAllReduceOutput.apply(
            output,
            head_range[0],
            head_range[1],
            core.num_heads,
            rank,
            hub,
        )

    core.register_forward_pre_hook(pre_hook)
    core.register_forward_hook(post_hook)
    return layer


# ---------------------------------------------------------------------------
# Dispatch algebra: combined cp x ep degree
# ---------------------------------------------------------------------------


class TestCombinedDispatchAlgebra:
    """Pure-tensor algebra of the combined cp x ep dispatch/undispatch.

    The dispatch runs over the flattened cp x ep mesh (degree CP_EP).
    Each rank holds a CP-sharded token slice of ``S`` rows (keyed by its
    cp_coord = rank // EP). After the dispatch, each rank receives one
    head-group chunk from every peer; with ``CP_EP`` senders the result
    has ``CP_EP * S`` rows (each distinct CP token shard appears ``EP``
    times, once per EP peer sharing its CP group).
    """

    DEGREE = CP_EP
    S = 5
    H = 8
    D = 3

    def test_dispatch_delivers_full_sequence_of_local_heads(self):
        torch.manual_seed(11)
        full = torch.randn(self.S * CP, self.H, self.D)
        heads_per_rank = self.H // self.DEGREE
        # Each virtual rank's local token shard (CP sharding).
        shards = [
            full.narrow(0, (r // EP) * self.S, self.S)
            for r in range(self.DEGREE)
        ]
        permuted = [
            head_seq_dispatch_permute(shard, self.DEGREE)
            for shard in shards
        ]
        for rank in range(self.DEGREE):
            dispatched = torch.cat(
                [permuted[s][rank] for s in range(self.DEGREE)], dim=0
            ).view(self.DEGREE * self.S, heads_per_rank, self.D)
            # Expected: every sender's CP token shard for this rank's
            # head group, concatenated in sender order. Each CP shard
            # appears EP times (once per EP peer of its CP group).
            expected = torch.cat(
                [
                    full.narrow(0, (s // EP) * self.S, self.S)[
                        :,
                        rank
                        * heads_per_rank : (rank + 1)
                        * heads_per_rank,
                    ]
                    for s in range(self.DEGREE)
                ],
                dim=0,
            )
            assert torch.equal(dispatched, expected.contiguous())

    def test_dispatch_cp_peers_send_identical_data(self):
        """CP peers (same cp_coord = rank // EP, different ep_coord) hold
        the same token shard and send the same head-group chunks; the
        dispatched tensor therefore contains each CP shard EP times (one
        per EP peer). The total row count is CP_EP * S (not CP * S)."""
        torch.manual_seed(13)
        full = torch.randn(self.S * CP, self.H, self.D)
        shards = [
            full.narrow(0, (r // EP) * self.S, self.S)
            for r in range(self.DEGREE)
        ]
        permuted = [
            head_seq_dispatch_permute(shard, self.DEGREE)
            for shard in shards
        ]
        # CP peers: senders 0 and 1 share cp_coord=0, same token shard.
        assert torch.equal(shards[0], shards[1])
        # Their dispatch permutations must match for every destination.
        for dest in range(self.DEGREE):
            assert torch.equal(permuted[0][dest], permuted[1][dest])


# ---------------------------------------------------------------------------
# CP=2 x EP=1 equivalence (Ulysses CP only, no MoE sharding)
# ---------------------------------------------------------------------------


def _fake_cp_only_mesh(rank, cp_degree):
    """Duck-typed 1D CP mesh when EP=1 (no EP axis, all ranks in CP)."""
    members = tuple(range(cp_degree))
    return SimpleNamespace(
        ndim=1,
        size=lambda: cp_degree,
        get_local_rank=lambda: rank,
        get_group=lambda: members,
    )


def _build_cp_rank_model(config, state_dict, rank):
    """One virtual CP rank's model: Ulysses CP wiring, no EP."""
    from torchtitan_npu.models.magi2_preview.cp_ulysses import (
        apply_magi2_ulysses_cp,
    )
    from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

    model = Magi2PreviewModel(config)
    model.load_state_dict(state_dict)
    model.train()
    apply_magi2_ulysses_cp(model, cp_mesh=_fake_cp_only_mesh(rank, CP))
    return model


class TestCp2Ep1EmulatedEquivalence:
    """CP=2 x EP=1 vs CP=1 x EP=1 full-model fwd/bwd equivalence."""

    def test_cp2_ep1_matches_unsharded(self):
        from torchtitan_npu.models.magi2_preview.model import (
            Magi2PreviewModel,
        )

        config = _small_model_config()
        torch.manual_seed(7)
        ref = Magi2PreviewModel(config)
        ref.init_weights()
        ref.train()
        state_dict = {
            k: v.clone() for k, v in ref.state_dict().items()
        }
        x, inputs, labels = _packed_inputs()

        x_ref = x.clone().requires_grad_(True)
        pred_ref = ref(x_ref, **inputs)
        torch.nn.functional.mse_loss(pred_ref, labels).backward()
        ref_grads = {
            n: p.grad.clone() for n, p in ref.named_parameters()
        }

        models = [
            _build_cp_rank_model(config, state_dict, rank)
            for rank in range(CP)
        ]
        hub = _ExchangeHub()
        x_ranks = [x.clone().requires_grad_(True) for _ in range(CP)]

        def run_rank(rank):
            pred = models[rank](x_ranks[rank], **inputs)
            torch.nn.functional.mse_loss(pred, labels).backward()
            return pred.detach()

        # The CP group spans both ranks (EP=1 means each CP group has 1
        # member only at the EP level, but the CP mesh covers CP=2 ranks).
        with _patched_cp_collectives(hub):
            # _VRANK.rank is set by _run_ranks; CP group members are
            # ranks 0 and 1 (both share EP coord 0 when EP=1).
            preds = _run_ranks(run_rank, degree=CP)

        for rank, pred in enumerate(preds):
            assert torch.allclose(
                pred, pred_ref, atol=1e-4, rtol=1e-4
            ), f"rank {rank} fwd mismatch"

        # The model's exit-gather gradient compensation restores the CP=1
        # gradient scale, so summed per-rank gradients equal the reference.
        for name, expected in ref_grads.items():
            if name.endswith(".sinks"):
                heads_per_rank = expected.shape[1] // CP
                for rank, model in enumerate(models):
                    shard = dict(model.named_parameters())[name]
                    assert shard.grad is not None, name
                    assert torch.allclose(
                        shard.grad,
                        expected[
                            :,
                            rank
                            * heads_per_rank : (rank + 1)
                            * heads_per_rank,
                        ],
                        atol=1e-4,
                        rtol=1e-4,
                    ), f"sinks grad mismatch (rank {rank})"
                continue
            total = sum(
                dict(m.named_parameters())[name].grad for m in models
            )
            assert torch.allclose(
                total, expected, atol=1e-4, rtol=1e-4
            ), f"grad mismatch for {name}"


# ---------------------------------------------------------------------------
# CP=1 x EP=2 equivalence (regime (a) all-reduce MoE, no CP)
# ---------------------------------------------------------------------------


class TestCp1Ep2EmulatedEquivalence:
    """CP=1 x EP=2 vs CP=1 x EP=1 full-model fwd/bwd equivalence."""

    def _build_ep_rank_model(self, config, state_dict, rank, hub):
        from torchtitan_npu.models.magi2_preview.model import (
            Magi2PreviewModel,
        )

        model = Magi2PreviewModel(config)
        model.load_state_dict(state_dict)
        model.train()
        for layer in model.block.layers.values():
            if isinstance(layer.mlp, MultiHeadMoELayer):
                _shard_layer_regime_a_emulated(
                    layer.mlp, rank, EP, hub
                )
        return model

    def test_cp1_ep2_matches_unsharded(self):
        from torchtitan_npu.models.magi2_preview.model import (
            Magi2PreviewModel,
        )

        config = _small_model_config()
        torch.manual_seed(7)
        ref = Magi2PreviewModel(config)
        ref.init_weights()
        ref.train()
        state_dict = {
            k: v.clone() for k, v in ref.state_dict().items()
        }
        x, inputs, labels = _packed_inputs()

        x_ref = x.clone().requires_grad_(True)
        pred_ref = ref(x_ref, **inputs)
        torch.nn.functional.mse_loss(pred_ref, labels).backward()
        ref_grads = {
            n: p.grad.clone() for n, p in ref.named_parameters()
        }

        hub = _ExchangeHub()
        models = [
            self._build_ep_rank_model(config, state_dict, rank, hub)
            for rank in range(EP)
        ]
        x_ranks = [x.clone().requires_grad_(True) for _ in range(EP)]

        def run_rank(rank):
            pred = models[rank](x_ranks[rank], **inputs)
            torch.nn.functional.mse_loss(pred, labels).backward()
            return pred.detach()

        with _patched_regime_a(hub):
            preds = _run_ranks(run_rank, degree=EP)

        for rank, pred in enumerate(preds):
            assert torch.allclose(
                pred, pred_ref, atol=1e-4, rtol=1e-4
            ), f"rank {rank} fwd mismatch"

        # EP peers produce identical outputs and gradients (tokens are
        # replicated); every rank's gradient equals the reference.
        for name, expected in ref_grads.items():
            if name.split(".")[-1] in EXPERT_PARAM_NAMES and (
                "moe_mlp" in name
            ):
                # Expert params are head-sharded: local grad = ref rows.
                for rank, model in enumerate(models):
                    grad = dict(model.named_parameters())[name].grad
                    assert grad is not None, (rank, name)
                    moe = model.get_submodule(
                        name.rsplit(".", 1)[0]
                    )
                    head_start, head_end = moe.head_range
                    rows = slice(
                        head_start * moe.num_experts,
                        head_end * moe.num_experts,
                    )
                    assert torch.allclose(
                        grad, expected[rows], atol=1e-4, rtol=1e-4
                    ), f"expert grad mismatch (rank {rank}, {name})"
                continue
            # Replicated params: every rank holds the same gradient.
            for rank, model in enumerate(models):
                grad = dict(model.named_parameters())[name].grad
                assert grad is not None, (rank, name)
                assert torch.allclose(
                    grad, expected, atol=1e-4, rtol=1e-4
                ), f"grad mismatch (rank {rank}, {name})"


# ---------------------------------------------------------------------------
# CP=2 x EP=2 equivalence (regime (b) combined dispatch)
# ---------------------------------------------------------------------------


def _ep_peer(rank):
    """Same CP token shard, different EP head shard."""
    return rank ^ 1


def _cp_peer(rank):
    """Same EP head shard, different CP token shard."""
    return rank ^ 2


def _build_cp_ep_rank_model(config, state_dict, rank):
    """One virtual rank: Ulysses CP + regime-(b) MoE wiring."""
    from torchtitan_npu.models.magi2_preview.cp_ulysses import (
        apply_magi2_ulysses_cp,
    )
    from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
    from torchtitan_npu.models.magi2_preview.parallelize import (
        _wire_moe_regime_b_layer,
    )

    model = Magi2PreviewModel(config)
    model.load_state_dict(state_dict)
    model.train()
    apply_magi2_ulysses_cp(
        model, cp_mesh=_fake_cp_mesh(rank), ep_degree=EP
    )
    for layer in model.block.layers.values():
        if isinstance(layer.mlp, MultiHeadMoELayer):
            core = layer.mlp.moe_mlp
            shard_moe_core_by_head(
                core,
                head_range_for_rank(rank, CP_EP, core.num_heads),
            )
            _wire_moe_regime_b_layer(
                layer.mlp,
                mesh=_fake_combined_mesh(rank),
                cp_degree=CP,
            )
    return model


class TestCp2Ep2EmulatedEquivalence:
    """CP=2 x EP=2 vs CP=1 x EP=1 full-model fwd/bwd equivalence."""

    def test_cp2_ep2_matches_unsharded(self):
        from torchtitan_npu.models.magi2_preview.model import (
            Magi2PreviewModel,
        )

        config = _small_model_config()
        torch.manual_seed(7)
        ref = Magi2PreviewModel(config)
        ref.init_weights()
        ref.train()
        state_dict = {
            k: v.clone() for k, v in ref.state_dict().items()
        }
        x, inputs, labels = _packed_inputs()

        x_ref = x.clone().requires_grad_(True)
        pred_ref = ref(x_ref, **inputs)
        torch.nn.functional.mse_loss(pred_ref, labels).backward()
        ref_grads = {
            n: p.grad.clone() for n, p in ref.named_parameters()
        }

        models = [
            _build_cp_ep_rank_model(config, state_dict, rank)
            for rank in range(CP_EP)
        ]
        hub = _ExchangeHub()
        x_ranks = [
            x.clone().requires_grad_(True) for _ in range(CP_EP)
        ]

        def run_rank(rank):
            pred = models[rank](x_ranks[rank], **inputs)
            torch.nn.functional.mse_loss(pred, labels).backward()
            return pred.detach()

        with _patched_cp_ep_collectives(hub):
            preds = _run_ranks(run_rank, degree=CP_EP)

        for rank, pred in enumerate(preds):
            assert torch.allclose(
                pred, pred_ref, atol=1e-4, rtol=1e-4
            ), f"rank {rank} fwd mismatch"
            # EP peers produce identical outputs.
            assert torch.allclose(
                pred, preds[_ep_peer(rank)], atol=1e-5, rtol=1e-5
            ), "EP peers must produce identical outputs"

        # sinks: head-sharded on the CP mesh.
        heads_per_cp_rank = (
            config.hidden_size // config.head_dim // CP
        )
        for rank, model in enumerate(models):
            sink_index = _cp_members(rank).index(rank)
            for name, param in model.named_parameters():
                if not name.endswith(".sinks"):
                    continue
                expected = ref_grads[name][
                    :,
                    sink_index
                    * heads_per_cp_rank : (sink_index + 1)
                    * heads_per_cp_rank,
                ]
                assert torch.allclose(
                    param.grad, expected, atol=1e-4, rtol=1e-4
                ), f"sinks grad mismatch (rank {rank})"

        # Expert and replicated params: regime-(b) accounting.
        for rank, model in enumerate(models):
            grads = dict(model.named_parameters())
            for name, expected in ref_grads.items():
                if name.endswith(".sinks"):
                    continue
                grad = grads[name].grad
                assert grad is not None, (rank, name)
                parts = name.split(".")
                if (
                    "moe_mlp" in parts
                    and parts[-1] in EXPERT_PARAM_NAMES
                ):
                    moe = model.get_submodule(name.rsplit(".", 1)[0])
                    head_start, head_end = moe.head_range
                    rows = slice(
                        head_start * moe.num_experts,
                        head_end * moe.num_experts,
                    )
                    assert torch.allclose(
                        grad, expected[rows], atol=1e-4, rtol=1e-4
                    ), f"expert grad mismatch (rank {rank}, {name})"
                    continue
                # EP peers hold identical gradients.
                peer_grad = dict(
                    models[_ep_peer(rank)].named_parameters()
                )[name].grad
                assert torch.allclose(
                    grad, peer_grad, atol=1e-5, rtol=1e-5
                ), f"EP-peer grad mismatch ({name})"
                # CP peers sum to the reference gradient.
                cp_grad = dict(
                    models[_cp_peer(rank)].named_parameters()
                )[name].grad
                assert torch.allclose(
                    grad + cp_grad, expected, atol=1e-4, rtol=1e-4
                ), f"grad mismatch for {name} (rank {rank})"

        # Input gradients: EP peers duplicate, CP peers partition.
        x_grad_sum = torch.zeros_like(x_ref.grad)
        for rank in range(CP_EP):
            assert torch.allclose(
                x_ranks[rank].grad,
                x_ranks[_ep_peer(rank)].grad,
                atol=1e-5,
                rtol=1e-5,
            ), "EP peers must hold identical input gradients"
            x_grad_sum += x_ranks[rank].grad
        assert torch.allclose(
            x_grad_sum / EP, x_ref.grad, atol=1e-4, rtol=1e-4
        )


# ---------------------------------------------------------------------------
# MoE layer-level CP x EP equivalence (3-way)
# ---------------------------------------------------------------------------


class TestMoELayerThreeWayEquivalence:
    """MoE-layer fwd/bwd: cp=2xep=1 vs cp=1xep=2 vs cp=2xep=2 vs ref."""

    T = 8

    def _reference_run(self):
        layer = _make_moe_layer()
        torch.manual_seed(5)
        x_data = torch.randn(self.T, 64)
        x_ref = x_data.clone().requires_grad_(True)
        ref_out = layer(x_ref, [self.T])
        ref_out.sum().backward()
        ref_grads = {
            n: p.grad.clone() for n, p in layer.named_parameters()
        }
        return layer, x_data, ref_out, ref_grads

    def test_cp2_ep1_layer_matches_unsharded(self):
        """CP=2 EP=1 at layer level: Ulysses dispatch over the CP mesh."""
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _wire_moe_regime_b_layer,
        )

        reference, x_data, ref_out, ref_grads = self._reference_run()
        # CP=2 EP=1: combined degree = CP=2; token shard = T/CP.
        shards = []
        for rank in range(CP):
            layer = copy.deepcopy(reference)
            core = layer.moe_mlp
            shard_moe_core_by_head(
                core, head_range_for_rank(rank, CP, core.num_heads)
            )
            mesh = SimpleNamespace(
                ndim=1,
                size=lambda: CP,
                get_local_rank=lambda r=rank: r,
                get_group=lambda: tuple(range(CP)),
            )
            _wire_moe_regime_b_layer(layer, mesh=mesh, cp_degree=CP)
            shards.append(layer)

        hub = _ExchangeHub()
        seq_per_rank = self.T // CP

        def run_rank(rank):
            x_shard = (
                x_data.narrow(0, rank * seq_per_rank, seq_per_rank)
                .clone()
                .requires_grad_(True)
            )
            out = shards[rank](x_shard, [seq_per_rank])
            out.sum().backward()
            return out.detach(), x_shard.grad

        with _patched_cp_ep_collectives(hub):
            results = _run_ranks(run_rank, degree=CP)

        for rank, (out, _) in enumerate(results):
            expected = ref_out.narrow(
                0, rank * seq_per_rank, seq_per_rank
            )
            assert torch.allclose(
                out, expected, atol=1e-4, rtol=1e-4
            ), f"rank {rank} fwd mismatch"

    def test_cp1_ep2_layer_matches_unsharded(self):
        """CP=1 EP=2 at layer level: regime (a) all-reduce assembly."""
        reference, x_data, ref_out, ref_grads = self._reference_run()
        hub = _ExchangeHub()
        shards = []
        for rank in range(EP):
            layer = copy.deepcopy(reference)
            _shard_layer_regime_a_emulated(layer, rank, EP, hub)
            shards.append(layer)

        def run_rank(rank):
            x = x_data.clone().requires_grad_(True)
            out = shards[rank](x, [self.T])
            out.sum().backward()
            return out.detach(), x.grad

        with _patched_regime_a(hub):
            results = _run_ranks(run_rank, degree=EP)

        for rank, (out, _) in enumerate(results):
            assert torch.allclose(
                out, ref_out, atol=1e-4, rtol=1e-4
            ), f"rank {rank} fwd mismatch"

    def test_cp2_ep2_layer_matches_unsharded(self):
        """CP=2 EP=2 at layer level: regime (b) combined dispatch."""
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _wire_moe_regime_b_layer,
        )

        reference, x_data, ref_out, ref_grads = self._reference_run()
        shards = []
        for rank in range(CP_EP):
            layer = copy.deepcopy(reference)
            core = layer.moe_mlp
            shard_moe_core_by_head(
                core,
                head_range_for_rank(rank, CP_EP, core.num_heads),
            )
            _wire_moe_regime_b_layer(
                layer,
                mesh=_fake_combined_mesh(rank),
                cp_degree=CP,
            )
            shards.append(layer)

        hub = _ExchangeHub()
        seq_per_rank = self.T // CP

        def run_rank(rank):
            cp_coord = rank // EP
            x_shard = (
                x_data.narrow(
                    0, cp_coord * seq_per_rank, seq_per_rank
                )
                .clone()
                .requires_grad_(True)
            )
            out = shards[rank](x_shard, [seq_per_rank])
            out.sum().backward()
            return out.detach(), x_shard.grad

        with _patched_cp_ep_collectives(hub):
            results = _run_ranks(run_rank, degree=CP_EP)

        for rank, (out, _) in enumerate(results):
            cp_coord = rank // EP
            expected = ref_out.narrow(
                0, cp_coord * seq_per_rank, seq_per_rank
            )
            assert torch.allclose(
                out, expected, atol=1e-4, rtol=1e-4
            ), f"rank {rank} fwd mismatch"
            # EP peers produce identical outputs.
            assert torch.allclose(
                out, results[_ep_peer(rank)][0], atol=1e-5, rtol=1e-5
            ), "EP peers must produce identical outputs"


# ---------------------------------------------------------------------------
# Wiring guards and state-dict expectations
# ---------------------------------------------------------------------------


class TestWiringGuards:
    """Divisibility and configuration guards for the CP x EP combination."""

    def test_cp_times_ep_must_divide_moe_heads(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_moe_parallel,
        )
        from torchtitan_npu.models.magi2_preview.model import (
            Magi2PreviewModel,
        )

        # moe_num_heads=4 divides CP_EP=4 but a degree-3 mesh does not.
        model = Magi2PreviewModel(_small_model_config())
        mesh = SimpleNamespace(
            ndim=1, size=lambda: 3, get_local_rank=lambda: 0
        )
        with pytest.raises(ValueError, match="divisible"):
            _apply_moe_parallel(
                model, ep_mesh=mesh, etp_mesh=None, cp_degree=3
            )

    def test_cp_times_ep_must_divide_attention_heads(self):
        from torchtitan_npu.models.magi2_preview.cp_ulysses import (
            Magi2UlyssesAttentionCP,
        )
        from torchtitan_npu.models.magi2_preview.attention import (
            Magi2Attention,
        )

        # hidden=160 / head_dim=32 = 5 heads, not div by CP=2
        attn = Magi2Attention(
            Magi2Attention.Config(
                hidden_size=160,
                head_dim=32,
                num_modality=3,
                sink_token_num=1,
            )
        )
        with pytest.raises(ValueError, match="divisible"):
            Magi2UlyssesAttentionCP()._apply(attn, _fake_mesh(0, 2))

    def test_combined_dispatch_context_passthrough_at_degree1(self):
        """Degree-1 dispatch context is a passthrough (no communication)."""
        layer = _make_moe_layer()
        plain = copy.deepcopy(layer)
        core = layer.moe_mlp
        head_range = (0, core.num_heads)
        core.set_head_range(head_range, sharded_input=True)
        layer.moe_dispatch_context = MoEDispatchContext(
            mesh=_fake_mesh(0, 1), head_range=head_range
        )
        torch.manual_seed(9)
        x = torch.randn(6, 64)
        out = layer(x, [6])
        out_ref = plain(x, [6])
        assert torch.allclose(out, out_ref, atol=1e-5)

    def test_set_head_range_supports_combined_degree(self):
        """CoreMultiHeadMoE.set_head_range works for any valid subrange."""
        core = CoreMultiHeadMoE(
            CoreMultiHeadMoE.Config(
                hidden_size=64,
                num_heads=8,
                num_experts=4,
                top_k=2,
                expert_intermediate_size=16,
            )
        )
        # cp=2 x ep=2 = 4-way split of 8 heads -> 2 heads per rank
        for rank in range(4):
            test_core = copy.deepcopy(core)
            hr = head_range_for_rank(rank, 4, 8)
            test_core.set_head_range(hr, sharded_input=True)
            assert test_core.head_range == hr
            assert test_core.sharded_input is True


# ---------------------------------------------------------------------------
# Nightly: real multi-rank gloo process group
# ---------------------------------------------------------------------------


def _load_multi_rank_conventions():
    """Import ``tests/smoke_tests/model_parallel/_multi_rank.py``."""
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
                "magi2_cp_ep_multi_rank_conventions", path
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

    class _TwoRankCpuTestBase(DTensorTestBase):
        @property
        def world_size(self):
            return 2

        @property
        def device_type(self):
            return "cpu"

    class _FourRankCpuTestBase(DTensorTestBase):
        @property
        def world_size(self):
            return 4

        @property
        def device_type(self):
            return "cpu"

    @mark_multi_rank_nightly
    class TestCp2Ep1TwoRankMultiRank(_TwoRankCpuTestBase):
        """CP=2 x EP=1 vs unsharded over a real 2-rank gloo group.

        Run with::

            torchrun --nproc_per_node=2 -m pytest \\
                tests/unit_tests/models/test_magi2_cp_ep.py \\
                -m nightly -k Cp2Ep1TwoRank
        """

        @with_comms
        def test_cp2_ep1_forward_backward_match_unsharded(self):
            from torch.distributed.device_mesh import init_device_mesh

            from torchtitan_npu.models.magi2_preview.cp_ulysses import (
                apply_magi2_ulysses_cp,
            )
            from torchtitan_npu.models.magi2_preview.model import (
                Magi2PreviewModel,
            )

            mesh = init_device_mesh(
                self.device_type,
                (self.world_size,),
                mesh_dim_names=("cp",),
            )
            config = _small_model_config()
            torch.manual_seed(7)
            model = Magi2PreviewModel(config)
            model.init_weights()
            model.train()
            reference = copy.deepcopy(model)

            apply_magi2_ulysses_cp(model, cp_mesh=mesh)

            x, inputs, labels = _packed_inputs()
            pred = model(x, **inputs)
            pred_ref = reference(x, **inputs)
            assert torch.allclose(
                pred, pred_ref, atol=1e-4, rtol=1e-4
            )

            torch.nn.functional.mse_loss(pred, labels).backward()
            torch.nn.functional.mse_loss(pred_ref, labels).backward()

            rank = mesh.get_local_rank()
            ref_params = dict(reference.named_parameters())
            cp_group = mesh.get_group()
            for name, param in model.named_parameters():
                assert param.grad is not None, name
                expected = ref_params[name].grad
                if name.endswith(".sinks"):
                    heads_per_rank = expected.shape[1] // self.world_size
                    expected = expected[
                        :,
                        rank
                        * heads_per_rank : (rank + 1)
                        * heads_per_rank,
                    ]
                    grad = param.grad
                    if isinstance(grad, DTensor):
                        grad = grad.to_local()
                    assert torch.allclose(
                        grad, expected, atol=1e-4, rtol=1e-4
                    ), name
                    continue
                grad_sum = param.grad.clone()
                dist.all_reduce(grad_sum, group=cp_group)
                assert torch.allclose(
                    grad_sum, expected, atol=1e-4, rtol=1e-4
                ), name

    @mark_multi_rank_nightly
    class TestCp1Ep2TwoRankMultiRank(_TwoRankCpuTestBase):
        """CP=1 x EP=2 vs unsharded over a real 2-rank gloo group.

        Run with::

            torchrun --nproc_per_node=2 -m pytest \\
                tests/unit_tests/models/test_magi2_cp_ep.py \\
                -m nightly -k Cp1Ep2TwoRank
        """

        @with_comms
        def test_cp1_ep2_forward_backward_match_unsharded(self):
            from torch.distributed.device_mesh import init_device_mesh

            from torchtitan_npu.models.magi2_preview.model import (
                Magi2PreviewModel,
            )
            from torchtitan_npu.models.magi2_preview.parallelize import (
                _apply_moe_parallel,
            )

            mesh = init_device_mesh(
                self.device_type,
                (self.world_size,),
                mesh_dim_names=("ep",),
            )
            config = _small_model_config()
            torch.manual_seed(7)
            model = Magi2PreviewModel(config)
            model.init_weights()
            model.train()
            reference = copy.deepcopy(model)

            _apply_moe_parallel(model, ep_mesh=mesh, etp_mesh=None)

            x, inputs, labels = _packed_inputs()
            pred = model(x, **inputs)
            pred_ref = reference(x, **inputs)
            assert torch.allclose(
                pred, pred_ref, atol=1e-4, rtol=1e-4
            )

            torch.nn.functional.mse_loss(pred, labels).backward()
            torch.nn.functional.mse_loss(pred_ref, labels).backward()

            rank = dist.get_rank()
            ref_params = dict(reference.named_parameters())
            for name, param in model.named_parameters():
                assert param.grad is not None, name
                expected = ref_params[name].grad
                short_name = name.split(".")[-1]
                if (
                    ".moe_mlp." in name
                    and short_name in EXPERT_PARAM_NAMES
                ):
                    moe = model.get_submodule(name.rsplit(".", 1)[0])
                    head_start, head_end = moe.head_range
                    rows = slice(
                        head_start * config.num_experts,
                        head_end * config.num_experts,
                    )
                    grad = param.grad
                    if isinstance(grad, DTensor):
                        grad = grad.to_local()
                    assert torch.allclose(
                        grad, expected[rows], atol=1e-4, rtol=1e-4
                    ), name
                    continue
                grad = param.grad
                if isinstance(grad, DTensor):
                    grad = grad.to_local()
                assert torch.allclose(
                    grad, expected, atol=1e-4, rtol=1e-4
                ), name

    @mark_multi_rank_nightly
    class TestCp2Ep2FourRankMultiRank(_FourRankCpuTestBase):
        """CP=2 x EP=2 vs unsharded over a real 4-rank gloo group.

        Run with::

            torchrun --nproc_per_node=4 -m pytest \\
                tests/unit_tests/models/test_magi2_cp_ep.py \\
                -m nightly -k Cp2Ep2FourRank
        """

        @with_comms
        def test_cp2_ep2_forward_backward_match_unsharded(self):
            from torch.distributed.device_mesh import init_device_mesh

            from torchtitan_npu.models.magi2_preview.cp_ulysses import (
                apply_magi2_ulysses_cp,
            )
            from torchtitan_npu.models.magi2_preview.expert_parallel import (
                flatten_head_mesh,
            )
            from torchtitan_npu.models.magi2_preview.model import (
                Magi2PreviewModel,
            )
            from torchtitan_npu.models.magi2_preview.parallelize import (
                _apply_moe_parallel,
            )

            mesh = init_device_mesh(
                self.device_type,
                (CP, EP),
                mesh_dim_names=("cp", "ep"),
            )
            config = _small_model_config()
            # cp x ep must divide moe_num_heads (regime b head sharding).
            config.moe_num_heads = CP * EP
            torch.manual_seed(7)
            model = Magi2PreviewModel(config)
            model.init_weights()
            model.train()
            reference = copy.deepcopy(model)

            dims = SimpleNamespace(get_mesh=lambda name: mesh[name])
            moe_mesh = flatten_head_mesh(dims)
            apply_magi2_ulysses_cp(
                model, cp_mesh=mesh["cp"], ep_degree=EP
            )
            _apply_moe_parallel(
                model,
                ep_mesh=moe_mesh,
                etp_mesh=None,
                cp_degree=CP,
            )

            x, inputs, labels = _packed_inputs()
            pred = model(x, **inputs)
            pred_ref = reference(x, **inputs)
            assert torch.allclose(
                pred, pred_ref, atol=1e-4, rtol=1e-4
            )

            torch.nn.functional.mse_loss(pred, labels).backward()
            torch.nn.functional.mse_loss(pred_ref, labels).backward()

            rank = dist.get_rank()
            cp_group = mesh["cp"].get_group()
            cp_local_rank = mesh["cp"].get_local_rank()
            attn_heads = config.hidden_size // config.head_dim
            heads_per_cp_rank = attn_heads // CP
            ref_params = dict(reference.named_parameters())
            for name, param in model.named_parameters():
                assert param.grad is not None, name
                ref_grad = ref_params[name].grad
                short_name = name.split(".")[-1]
                if (
                    ".moe_mlp." in name
                    and short_name in EXPERT_PARAM_NAMES
                ):
                    heads_per_rank = config.moe_num_heads // (CP * EP)
                    rows = slice(
                        rank
                        * heads_per_rank
                        * config.num_experts,
                        (rank + 1)
                        * heads_per_rank
                        * config.num_experts,
                    )
                    assert torch.allclose(
                        param.grad.to_local(),
                        ref_grad[rows],
                        atol=1e-4,
                        rtol=1e-4,
                    ), name
                    continue
                if name.endswith(".sinks"):
                    expected = ref_grad[
                        :,
                        cp_local_rank
                        * heads_per_cp_rank : (cp_local_rank + 1)
                        * heads_per_cp_rank,
                    ]
                    assert torch.allclose(
                        param.grad,
                        expected,
                        atol=1e-4,
                        rtol=1e-4,
                    ), name
                    continue
                grad_sum = param.grad.clone()
                dist.all_reduce(grad_sum, group=cp_group)
                assert torch.allclose(
                    grad_sum, ref_grad, atol=1e-4, rtol=1e-4
                ), name
