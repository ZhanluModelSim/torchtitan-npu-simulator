# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Head-parallel MoE tests: head_range routing/expert equivalence, slicing
utilities, regime-(a) zero-pad/all-reduce assembly, regime-(b) Ulysses
dispatch algebra, CP x EP combination (single-process emulation of 4
virtual ranks: dispatch round trip, MoE-layer and full-model fwd/bwd vs
the unsharded reference), DTensor/checkpoint expectations, and nightly
multi-rank gloo coverage (gated like tests/smoke_tests/model_parallel)."""

import contextlib
import copy
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from torchtitan_npu.models.magi2_preview import magi2_preview_configs
from torchtitan_npu.models.magi2_preview.expert_parallel import (
    EXPERT_PARAM_NAMES,
    ROUTER_BUFFER_NAMES,
    MoEDispatchContext,
    all_reduce_head_parallel_output,
    ep_dispatch,
    ep_undispatch,
    head_range_for_rank,
    head_seq_dispatch_permute,
    head_seq_undispatch_permute,
    pad_head_partial,
    shard_moe_core_by_head,
    slice_expert_state_by_head,
)
from torchtitan_npu.models.magi2_preview.feed_forward import (
    CoreMultiHeadMoE,
    MultiHeadMoELayer,
)

try:
    from tests.smoke_tests.model_parallel._multi_rank import (
        MULTI_RANK_AVAILABLE,
        DTensorTestBase,
        mark_multi_rank_nightly,
        with_comms,
    )
except ImportError:
    # Harness runs without the repo test tree on the path; the nightly
    # multi-rank class is skipped there anyway.
    MULTI_RANK_AVAILABLE = False
    DTensorTestBase = object
    mark_multi_rank_nightly = None
    with_comms = None


@pytest.fixture
def single_rank_process_group():
    """Shared single-rank gloo group (mirrors tests/conftest.py, redefined
    here so the file also runs standalone in integration harnesses)."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://localhost:12356",
            world_size=1,
            rank=0,
        )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def _moe_config() -> CoreMultiHeadMoE.Config:
    return CoreMultiHeadMoE.Config(
        hidden_size=64,
        num_heads=4,
        num_experts=6,
        top_k=3,
        expert_intermediate_size=32,
        route_scale=4.9,
    )


def _make_core(seed: int = 0) -> CoreMultiHeadMoE:
    """Small deterministic MoE core with nonzero router bias."""
    torch.manual_seed(seed)
    core = CoreMultiHeadMoE(_moe_config())
    with torch.no_grad():
        for param in (core.gate, core.W_gate, core.W_up, core.W_down):
            param.normal_(0.0, 0.1)
        core.router.expert_bias.normal_(0.0, 0.05)
    return core


def _shard_core(core: CoreMultiHeadMoE, head_range) -> CoreMultiHeadMoE:
    shard = copy.deepcopy(core)
    shard_moe_core_by_head(shard, head_range)
    return shard


def _simulate_all_to_all(per_rank_inputs: list[torch.Tensor], receiver: int):
    """Chunk-exchange standing in for all_to_all_single: ``per_rank_inputs[s]``
    is rank s's send tensor with dim 0 = destination rank; the receiver gets
    chunk ``receiver`` from every sender, concatenated along dim 0."""
    return torch.cat([send[receiver] for send in per_rank_inputs], dim=0)


# ---------------------------------------------------------------------------
# head_range routing / expert-compute equivalence (single process)
# ---------------------------------------------------------------------------


class TestHeadRangeEquivalence:
    HEAD_RANGES = ((0, 2), (2, 4))

    def test_sharded_routing_matches_unsharded(self):
        core = _make_core()
        x = torch.randn(7, core.num_heads * core.d_head)
        x_heads = x.view(-1, core.num_heads, core.d_head)
        full_probs, full_indices = core.router(x_heads, core.gate)
        assert full_probs.shape == (core.num_heads, 7, core.top_k)

        for head_range in self.HEAD_RANGES:
            shard = _shard_core(core, head_range)
            head_start, head_end = head_range
            probs, indices = shard.router(
                x_heads[:, head_start:head_end], shard.gate
            )
            # topk selection, unbiased probs (incl. route_scale) are per-head:
            # the shard must reproduce its heads bit-exactly.
            assert torch.equal(probs, full_probs[head_start:head_end])
            assert torch.equal(indices, full_indices[head_start:head_end])

    def test_forward_partials_assemble_to_unsharded(self):
        core = _make_core()
        x = torch.randn(7, core.num_heads * core.d_head)
        out_full = core(x)

        combined = torch.zeros_like(out_full)
        for head_range in self.HEAD_RANGES:
            shard = _shard_core(core, head_range)
            partial = shard(x)
            local_num_heads = head_range[1] - head_range[0]
            assert partial.shape == (7, local_num_heads * core.d_head)
            combined = combined + pad_head_partial(
                partial, head_range, core.num_heads
            )

        assert torch.allclose(out_full, combined)

    def test_gradients_match_unsharded(self):
        core = _make_core()
        shards = [_shard_core(core, hr) for hr in self.HEAD_RANGES]

        x = torch.randn(7, core.num_heads * core.d_head, requires_grad=True)
        core(x).sum().backward()
        x_grad_full = x.grad.clone()

        x_grad_combined = torch.zeros_like(x_grad_full)
        for head_range, shard in zip(self.HEAD_RANGES, shards, strict=True):
            head_start, head_end = head_range
            x_shard = x.detach().clone().requires_grad_(True)
            shard(x_shard).sum().backward()

            assert x_shard.grad is not None
            # A shard only touches its own head columns of the input.
            d = core.d_head
            assert torch.count_nonzero(x_shard.grad[:, : head_start * d]) == 0
            assert torch.count_nonzero(x_shard.grad[:, head_end * d :]) == 0
            x_grad_combined += x_shard.grad
            rows = slice(head_start * core.num_experts, head_end * core.num_experts)
            for name in EXPERT_PARAM_NAMES:
                full_grad = core.get_parameter(name).grad
                shard_grad = shard.get_parameter(name).grad
                assert shard_grad is not None
                assert torch.allclose(shard_grad, full_grad[rows])
            # Router buffers carry no gradient in either form.
            assert shard.router.expert_bias.grad is None

        # The per-shard input gradients tile the unsharded one exactly.
        assert torch.allclose(x_grad_combined, x_grad_full)


class TestHeadRangeLayerEquivalence:
    def test_moe_layer_matches_when_partials_combined(self):
        torch.manual_seed(3)
        layer = MultiHeadMoELayer(
            MultiHeadMoELayer.Config(
                hidden_size=64,
                num_modality=1,
                moe_num_heads=4,
                num_experts=6,
                moe_top_k=3,
                expert_intermediate_size=32,
                shared_expert_intermediate_size=32,
            )
        )
        with torch.no_grad():
            for param in layer.parameters():
                param.normal_(0.0, 0.02)
            layer.moe_mlp.router.expert_bias.normal_(0.0, 0.05)
        x = torch.randn(6, 64)
        m_splits = [6]
        out_full = layer(x, m_splits)

        combined_moe_out = torch.zeros(6, 64)
        for head_range in ((0, 2), (2, 4)):
            shard = copy.deepcopy(layer)
            shard_moe_core_by_head(shard.moe_mlp, head_range)
            norm_output = shard.pre_norm(x, m_splits)
            partial = shard.moe_mlp(shard.split_linear(norm_output, None))
            combined_moe_out = combined_moe_out + pad_head_partial(
                partial, head_range, shard.moe_mlp.num_heads
            )
        out_sharded = shard.merge_linear(combined_moe_out, None)
        out_sharded = out_sharded + shard._shared_expert_forward(
            shard.pre_norm(x, m_splits), m_splits
        )

        assert torch.allclose(out_full, out_sharded)


# ---------------------------------------------------------------------------
# Slicing utilities
# ---------------------------------------------------------------------------


class TestSlicingUtilities:
    def test_head_range_for_rank(self):
        assert head_range_for_rank(0, 2, 4) == (0, 2)
        assert head_range_for_rank(1, 2, 4) == (2, 4)
        assert head_range_for_rank(0, 1, 4) == (0, 4)

    def test_head_range_for_rank_requires_divisibility(self):
        with pytest.raises(ValueError, match="divisible"):
            head_range_for_rank(0, 3, 4)

    def test_slice_expert_state_by_head(self):
        core = _make_core()
        state = core.state_dict()
        state["unrelated.weight"] = torch.randn(3)

        sliced = slice_expert_state_by_head(state, (1, 3), core.num_experts)

        rows = slice(1 * core.num_experts, 3 * core.num_experts)
        for name in EXPERT_PARAM_NAMES:
            assert sliced[name].shape[0] == 2 * core.num_experts
            assert torch.equal(sliced[name], state[name][rows])
        for name in ROUTER_BUFFER_NAMES:
            key = f"router.{name}"
            assert sliced[key].shape[0] == 2 * core.num_experts
            assert torch.equal(sliced[key], state[key][rows])
        assert sliced["unrelated.weight"] is state["unrelated.weight"]

    def test_slice_expert_state_rejects_bad_leading_dim(self):
        state = {"gate": torch.randn(7, 4)}
        with pytest.raises(ValueError, match="num_experts"):
            slice_expert_state_by_head(state, (0, 1), 3)

    def test_shard_moe_core_by_head(self):
        core = _make_core()
        full_gate = core.gate.detach().clone()

        shard_moe_core_by_head(core, (2, 4))

        assert core.head_range == (2, 4)
        assert core.num_heads == 4  # global shape knowledge is retained
        local_rows = 2 * core.num_experts
        for name in EXPERT_PARAM_NAMES:
            assert core.get_parameter(name).shape[0] == local_rows
        assert torch.equal(core.gate, full_gate[2 * core.num_experts :])
        for name in ROUTER_BUFFER_NAMES:
            assert getattr(core.router, name).shape[0] == local_rows

    def test_set_head_range_validation(self):
        core = _make_core()
        with pytest.raises(ValueError):
            core.set_head_range((2, 2))
        with pytest.raises(ValueError):
            core.set_head_range((-1, 2))
        with pytest.raises(ValueError):
            core.set_head_range((0, core.num_heads + 1))


# ---------------------------------------------------------------------------
# Regime (a): zero-pad / all-reduce assembly
# ---------------------------------------------------------------------------


class TestRegimeAAssembly:
    def test_pad_head_partial_placement_and_backward(self):
        d_head = 8
        partial = torch.randn(5, 2 * d_head, requires_grad=True)
        full = pad_head_partial(partial, (1, 3), 4)

        assert full.shape == (5, 4 * d_head)
        assert torch.equal(full[:, d_head : 3 * d_head], partial)
        assert torch.count_nonzero(full[:, :d_head]) == 0
        assert torch.count_nonzero(full[:, 3 * d_head :]) == 0

        grad_out = torch.randn_like(full)
        full.backward(grad_out)
        assert torch.equal(partial.grad, grad_out[:, d_head : 3 * d_head])

    def test_all_reduce_single_rank_matches_padding(self, single_rank_process_group):
        partial = torch.randn(5, 16)
        out = all_reduce_head_parallel_output(partial, (0, 2), 4, dist.group.WORLD)
        assert torch.allclose(out, pad_head_partial(partial, (0, 2), 4))

    def test_all_reduce_backward_slices(self, single_rank_process_group):
        partial = torch.randn(5, 16, requires_grad=True)
        out = all_reduce_head_parallel_output(partial, (1, 3), 4, dist.group.WORLD)
        grad_out = torch.randn_like(out)
        out.backward(grad_out)
        assert torch.equal(partial.grad, grad_out[:, 8:24])


# ---------------------------------------------------------------------------
# Regime (b): Ulysses seq<->head dispatch algebra (single process)
# ---------------------------------------------------------------------------


class TestUlyssesDispatchAlgebra:
    EP_SIZE = 2
    SEQ_PER_RANK = 5
    NUM_HEADS = 4
    D_HEAD = 3

    def _seq_shards(self):
        torch.manual_seed(11)
        full_seq = torch.randn(
            self.EP_SIZE * self.SEQ_PER_RANK, self.NUM_HEADS, self.D_HEAD
        )
        return full_seq, list(full_seq.chunk(self.EP_SIZE, dim=0))

    def test_dispatch_gives_full_sequence_of_local_heads(self):
        full_seq, seq_shards = self._seq_shards()
        heads_per_rank = self.NUM_HEADS // self.EP_SIZE

        permuted = [
            head_seq_dispatch_permute(shard, self.EP_SIZE) for shard in seq_shards
        ]
        for rank in range(self.EP_SIZE):
            dispatched = _simulate_all_to_all(permuted, rank).view(
                self.EP_SIZE * self.SEQ_PER_RANK, heads_per_rank, self.D_HEAD
            )
            expected = full_seq[
                :, rank * heads_per_rank : (rank + 1) * heads_per_rank
            ]
            assert torch.equal(dispatched, expected.contiguous())

    def test_undispatch_inverts_dispatch(self):
        _, seq_shards = self._seq_shards()
        heads_per_rank = self.NUM_HEADS // self.EP_SIZE

        dispatched = []
        permuted = [
            head_seq_dispatch_permute(shard, self.EP_SIZE) for shard in seq_shards
        ]
        for rank in range(self.EP_SIZE):
            dispatched.append(
                _simulate_all_to_all(permuted, rank).view(
                    self.EP_SIZE * self.SEQ_PER_RANK, heads_per_rank, self.D_HEAD
                )
            )

        send_chunks = [
            tensor.view(self.EP_SIZE, self.SEQ_PER_RANK, heads_per_rank, self.D_HEAD)
            for tensor in dispatched
        ]
        for rank in range(self.EP_SIZE):
            received = _simulate_all_to_all(send_chunks, rank).view(
                self.EP_SIZE, self.SEQ_PER_RANK, heads_per_rank, self.D_HEAD
            )
            out = head_seq_undispatch_permute(received, self.EP_SIZE)
            assert torch.equal(out, seq_shards[rank].contiguous())

    def test_passthrough_without_group(self):
        x = torch.randn(4, 6, 2)
        assert ep_dispatch(x, None) is x
        assert ep_undispatch(x, None) is x


# ---------------------------------------------------------------------------
# Regime (b) CP x EP combination: single-process emulation of 4 virtual
# ranks (CP=2 x EP=2). The virtual ranks run on threads; the collectives
# are emulated with the real all-to-all algebra and autograd semantics
# (self-adjoint swaps, reduce-scatter-sum all-gather backward), so the
# real wiring code (head sharding, dispatch context, layer forward, model
# entry/exit) runs unchanged. Rank layout is cp-major: flattened rank
# r = cp_coord * EP + ep_coord, token shards keyed by cp_coord.
# ---------------------------------------------------------------------------

CP = 2
EP = 2
CP_EP = CP * EP
_VRANK = threading.local()


def _cp_members(rank: int) -> tuple[int, ...]:
    """This rank's CP group: one rank per CP token shard (the ranks
    sharing its EP coordinate); the coordinate inside the group is the
    CP coord and keys the token shard (cp-major flattened layout)."""
    return tuple(r for r in range(CP_EP) if r % EP == rank % EP)


def _fake_cp_mesh(rank: int):
    members = _cp_members(rank)
    return SimpleNamespace(
        ndim=1,
        size=lambda: len(members),
        get_local_rank=lambda: members.index(rank),
        get_group=lambda: members,
    )


def _fake_combined_mesh(rank: int):
    """Duck-typed flattened cp x ep mesh (all virtual ranks)."""
    members = tuple(range(CP_EP))
    return SimpleNamespace(
        ndim=1,
        size=lambda: CP_EP,
        get_local_rank=lambda: rank,
        get_group=lambda: members,
    )


def _fake_degree1_mesh():
    return SimpleNamespace(
        ndim=1, size=lambda: 1, get_local_rank=lambda: 0, get_group=lambda: None
    )


class _CollectiveEmulator:
    """In-process emulation of SPMD collectives between virtual ranks.

    Ranks call ``exchange`` with their send data in the same collective
    order (forward and backward alike); a call completes once every rank
    has staged its send, then every rank reads its result. Modes:
    - ``a2a``: all_to_all_single over dim 0 with even chunks (self-adjoint,
      used for the Ulysses swaps and their backward);
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


class _ExchangeHub:
    """Per-member-set emulator registry (one process group per member set)."""

    def __init__(self, timeout: float = 90.0):
        self.timeout = timeout
        self._emulators = {}

    def exchange(self, send, rank: int, members, mode: str) -> torch.Tensor:
        members = tuple(members)
        emulator = self._emulators.get(members)
        if emulator is None:
            emulator = _CollectiveEmulator(
                len(members), timeout=self.timeout
            )
            self._emulators[members] = emulator
        return emulator.exchange(send, members.index(rank), mode)


class _EmulatedSeqHeadSwap(torch.autograd.Function):
    """(S, H, D) -> (deg*S, H/deg, D) seq->head swap via the emulated a2a.

    Shared by the CP attention swaps and the regime-(b) MoE dispatch; the
    member set selects the emulated process group.
    """

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
        # The emulator concatenates the received chunks along dim 0 in
        # sender order, restoring the (degree, S, ...) chunk view first.
        return (
            head_seq_undispatch_permute(
                recv.view(degree, seq_len, heads, d_head), degree
            ),
            None,
            None,
            None,
        )


class _EmulatedHeadSeqSwap(torch.autograd.Function):
    """(deg*S, H/deg, D) -> (S, H, D) head->seq swap via the emulated a2a."""

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
        # The emulator concatenates the received chunks along dim 0 in
        # sender order, restoring the (degree, S, ...) chunk view first.
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
            grad_output.contiguous(), ctx.rank, ctx.members, "reduce_scatter_sum"
        )
        return grad_input, None, None, None


@contextlib.contextmanager
def _patched_collectives(hub: _ExchangeHub):
    """Replace the funcol/collective wrappers with emulated swaps."""
    cp_prefix = "torchtitan_npu.models.magi2_preview.cp_ulysses"
    ep_prefix = "torchtitan_npu.models.magi2_preview.expert_parallel"

    def swap_dispatch(x, rank, members):
        return _EmulatedSeqHeadSwap.apply(x, rank, hub, members)

    def swap_undispatch(x, rank, members):
        return _EmulatedHeadSeqSwap.apply(x, rank, hub, members)

    def fake_ulysses_dispatch(x, *, mesh):
        return swap_dispatch(x, _VRANK.rank, tuple(mesh.get_group()))

    def fake_ulysses_undispatch(x, *, mesh):
        return swap_undispatch(x, _VRANK.rank, tuple(mesh.get_group()))

    def fake_gather(x, *, mesh):
        return _EmulatedGather.apply(x, _VRANK.rank, hub, tuple(mesh.get_group()))

    def fake_ep_dispatch(x, group):
        if group is None or len(group) <= 1:
            return x
        return swap_dispatch(x, _VRANK.rank, tuple(group))

    def fake_ep_undispatch(x, group):
        if group is None or len(group) <= 1:
            return x
        return swap_undispatch(x, _VRANK.rank, tuple(group))

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            mock.patch(f"{cp_prefix}.ulysses_dispatch", fake_ulysses_dispatch)
        )
        stack.enter_context(
            mock.patch(f"{cp_prefix}.ulysses_undispatch", fake_ulysses_undispatch)
        )
        stack.enter_context(mock.patch(f"{cp_prefix}.gather_seq", fake_gather))
        stack.enter_context(
            mock.patch(f"{ep_prefix}.ep_dispatch", fake_ep_dispatch)
        )
        stack.enter_context(
            mock.patch(f"{ep_prefix}.ep_undispatch", fake_ep_undispatch)
        )
        yield


def _run_virtual_ranks(fn, degree: int = CP_EP, timeout: float = 180.0):
    """Run fn(rank) on ``degree`` threads; surface the first worker error."""
    results, errors = {}, []

    def target(rank):
        try:
            _VRANK.rank = rank
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
        raise AssertionError("virtual CP x EP ranks deadlocked")
    if errors:
        raise errors[0]
    return [results[rank] for rank in range(degree)]


def _ep_peer(rank: int) -> int:
    return rank ^ 1  # same CP token shard, different EP head shard


def _cp_peer(rank: int) -> int:
    return rank ^ 2  # same EP head shard, different CP token shard


def _assert_regime_b_grads(
    shards, ref_grads, *, atol: float = 1e-4, rtol: float = 1e-4
) -> None:
    """Cross-rank gradient accounting of the faithful CP x EP emulation.

    Expert params are head-sharded: each rank's local shard grad matches
    the corresponding leading-dim slice of the reference directly. Every
    other param is replicated: EP peers (same token shard) hold identical
    gradients and a CP peer pair sums to the unsharded reference gradient
    (each peer back-propagates only its own token shard).
    """
    for rank, shard in enumerate(shards):
        grads = dict(shard.named_parameters())
        for name, expected in ref_grads.items():
            grad = grads[name].grad
            assert grad is not None, (rank, name)
            parts = name.split(".")
            if "moe_mlp" in parts and parts[-1] in EXPERT_PARAM_NAMES:
                moe = shard.get_submodule(name.rsplit(".", 1)[0])
                head_start, head_end = moe.head_range
                rows = slice(
                    head_start * moe.num_experts, head_end * moe.num_experts
                )
                assert torch.allclose(
                    grad, expected[rows], atol=atol, rtol=rtol
                ), f"expert grad mismatch (rank {rank}, {name})"
                continue
            peer_grad = dict(shards[_ep_peer(rank)].named_parameters())[name].grad
            assert torch.allclose(
                grad, peer_grad, atol=atol, rtol=rtol
            ), f"EP-peer grad mismatch ({name})"
            cp_grad = dict(shards[_cp_peer(rank)].named_parameters())[name].grad
            assert torch.allclose(
                grad + cp_grad, expected, atol=atol, rtol=rtol
            ), f"grad mismatch for {name} (rank {rank})"


class TestCpEpDispatchEmulated:
    """Dispatch/undispatch round trip across the 4 virtual ranks."""

    S = 5
    H = 8
    D = 3

    def test_dispatch_content_and_round_trip(self):
        from torchtitan_npu.models.magi2_preview import expert_parallel as ep_mod

        hub = _ExchangeHub()
        torch.manual_seed(21)
        full_seq = torch.randn(CP * self.S, self.H, self.D)
        heads_per_rank = self.H // CP_EP

        def run_rank(rank):
            cp_coord = rank // EP
            local = (
                full_seq.narrow(0, cp_coord * self.S, self.S)
                .clone()
                .requires_grad_(True)
            )
            group = tuple(range(CP_EP))
            dispatched = ep_mod.ep_dispatch(local, group)
            assert dispatched.shape == (
                CP_EP * self.S,
                heads_per_rank,
                self.D,
            )
            # Received rows: every sender's CP token shard for this rank's
            # head group, concatenated in sender order (each distinct token
            # appears EP times, once per EP peer of its CP group).
            expected = torch.cat(
                [
                    full_seq.narrow(0, (s // EP) * self.S, self.S)[
                        :, rank * heads_per_rank : (rank + 1) * heads_per_rank
                    ]
                    for s in range(CP_EP)
                ],
                dim=0,
            )
            assert torch.equal(dispatched, expected.contiguous())

            back = ep_mod.ep_undispatch(dispatched, group)
            assert torch.equal(back, local.detach())

            dispatched.sum().backward()
            return local.grad

        with _patched_collectives(hub):
            grads = _run_virtual_ranks(run_rank)
        # Every local element enters the dispatched tensor exactly once.
        for grad in grads:
            assert torch.equal(grad, torch.ones_like(grad))


def _moe_layer_config() -> MultiHeadMoELayer.Config:
    return MultiHeadMoELayer.Config(
        hidden_size=64,
        num_modality=1,
        moe_num_heads=4,
        num_experts=6,
        moe_top_k=3,
        expert_intermediate_size=32,
        shared_expert_intermediate_size=32,
    )


def _make_moe_layer(seed: int = 3) -> MultiHeadMoELayer:
    torch.manual_seed(seed)
    layer = MultiHeadMoELayer(_moe_layer_config())
    with torch.no_grad():
        for param in layer.parameters():
            param.normal_(0.0, 0.02)
        layer.moe_mlp.router.expert_bias.normal_(0.0, 0.05)
    return layer


def _shard_moe_layer_for_rank(layer: MultiHeadMoELayer, rank: int):
    """Emulated production wiring: plain head slice + regime-(b) state."""
    from torchtitan_npu.models.magi2_preview.parallelize import (
        _wire_moe_regime_b_layer,
    )

    core = layer.moe_mlp
    shard_moe_core_by_head(
        core, head_range_for_rank(rank, CP_EP, core.num_heads)
    )
    _wire_moe_regime_b_layer(layer, mesh=_fake_combined_mesh(rank), cp_degree=CP)
    return layer


class TestMoELayerCpEpEmulatedEquivalence:
    """MoE-layer fwd/bwd on (T/cp, C) shards vs the unsharded layer."""

    T = 8

    def test_layer_matches_unsharded_fwd_bwd(self):
        reference = _make_moe_layer()
        shards = [
            _shard_moe_layer_for_rank(copy.deepcopy(reference), rank)
            for rank in range(CP_EP)
        ]
        torch.manual_seed(5)
        x_data = torch.randn(self.T, 64)
        x_ref = x_data.clone().requires_grad_(True)
        ref_out = reference(x_ref, [self.T])
        ref_out.sum().backward()
        ref_grads = {
            n: p.grad.clone() for n, p in reference.named_parameters()
        }

        hub = _ExchangeHub()
        seq_per_rank = self.T // CP

        def run_rank(rank):
            cp_coord = rank // EP
            x_shard = (
                x_data.narrow(0, cp_coord * seq_per_rank, seq_per_rank)
                .clone()
                .requires_grad_(True)
            )
            out = shards[rank](x_shard, [seq_per_rank])
            out.sum().backward()
            return out.detach(), x_shard.grad

        with _patched_collectives(hub):
            results = _run_virtual_ranks(run_rank)

        for rank, (out, _) in enumerate(results):
            cp_coord = rank // EP
            expected = ref_out.narrow(
                0, cp_coord * seq_per_rank, seq_per_rank
            )
            assert torch.allclose(
                out, expected, atol=1e-4, rtol=1e-4
            ), f"rank {rank} fwd mismatch"
            assert torch.allclose(
                out, results[_ep_peer(rank)][0], atol=1e-5, rtol=1e-5
            ), "EP peers must produce identical outputs"

        _assert_regime_b_grads(shards, ref_grads)
        x_grad_sum = torch.zeros_like(x_ref.grad)
        for rank, (_, x_grad) in enumerate(results):
            assert torch.allclose(
                x_grad, results[_ep_peer(rank)][1], atol=1e-5, rtol=1e-5
            ), "EP peers must hold identical input gradients"
            start = (rank // EP) * seq_per_rank
            x_grad_sum.narrow(0, start, seq_per_rank).add_(x_grad)
        # EP peers duplicate the same shard contribution.
        assert torch.allclose(
            x_grad_sum / EP, x_ref.grad, atol=1e-4, rtol=1e-4
        )

    def test_dispatch_context_degree1_is_passthrough(self):
        layer = _make_moe_layer()
        plain = copy.deepcopy(layer)
        core = layer.moe_mlp
        head_range = (0, core.num_heads)
        core.set_head_range(head_range, sharded_input=True)
        layer.moe_dispatch_context = MoEDispatchContext(
            mesh=_fake_degree1_mesh(), head_range=head_range
        )

        torch.manual_seed(9)
        x = torch.randn(6, 64)
        out = layer(x, [6])
        out_ref = plain(x, [6])
        assert torch.allclose(out, out_ref, atol=1e-5)

        out.sum().backward()
        out_ref.sum().backward()
        for name, param in layer.named_parameters():
            ref_grad = dict(plain.named_parameters())[name].grad
            assert torch.allclose(param.grad, ref_grad, atol=1e-5), name


def _small_cp_ep_model_config():
    """Tiny MAGI-2-preview config; moe_num_heads=4 divides CP x EP = 4."""
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


def _packed_model_inputs(seq_len: int = 16, seed: int = 5):
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


def _build_cp_ep_rank_model(config, state_dict, rank):
    """One virtual rank's model: Ulysses CP wiring + regime-(b) MoE wiring
    (plain head slices stand in for the DTensor Shard(0) distribution)."""
    from torchtitan_npu.models.magi2_preview.cp_ulysses import (
        apply_magi2_ulysses_cp,
    )
    from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

    model = Magi2PreviewModel(config)
    model.load_state_dict(state_dict)
    model.train()
    apply_magi2_ulysses_cp(model, cp_mesh=_fake_cp_mesh(rank), ep_degree=EP)
    for layer in model.block.layers.values():
        if isinstance(layer.mlp, MultiHeadMoELayer):
            _shard_moe_layer_for_rank(layer.mlp, rank)
    return model


class TestModelCpEpEmulatedEquivalence:
    """Full-model CP=2 x EP=2 fwd/bwd vs the unsharded model."""

    def test_cp2_ep2_matches_unsharded_fwd_bwd(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        config = _small_cp_ep_model_config()
        torch.manual_seed(7)
        ref = Magi2PreviewModel(config)
        ref.init_weights()
        ref.train()
        state_dict = {k: v.clone() for k, v in ref.state_dict().items()}
        x, inputs, labels = _packed_model_inputs()

        x_ref = x.clone().requires_grad_(True)
        pred_ref = ref(x_ref, **inputs)
        torch.nn.functional.mse_loss(pred_ref, labels).backward()
        ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

        models = [
            _build_cp_ep_rank_model(config, state_dict, rank)
            for rank in range(CP_EP)
        ]
        hub = _ExchangeHub()
        x_ranks = [x.clone().requires_grad_(True) for _ in range(CP_EP)]

        def run_rank(rank):
            pred = models[rank](x_ranks[rank], **inputs)
            torch.nn.functional.mse_loss(pred, labels).backward()
            return pred.detach()

        with _patched_collectives(hub):
            preds = _run_virtual_ranks(run_rank)

        for rank, pred in enumerate(preds):
            assert torch.allclose(
                pred, pred_ref, atol=1e-4, rtol=1e-4
            ), f"rank {rank} prediction mismatch"

        # sinks: head-sharded on the CP mesh; the shard index is the rank's
        # coordinate inside its CP group. Expert and replicated params use
        # the shared regime-(b) accounting.
        heads_per_cp_rank = config.hidden_size // config.head_dim // CP
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
                peer = dict(models[_ep_peer(rank)].named_parameters())[name]
                assert torch.allclose(
                    param.grad, peer.grad, atol=1e-5, rtol=1e-5
                ), "EP peers share the same sinks shard"
        non_sink_ref_grads = {
            n: g for n, g in ref_grads.items() if not n.endswith(".sinks")
        }
        _assert_regime_b_grads(models, non_sink_ref_grads)

        x_grad_sum = torch.zeros_like(x_ref.grad)
        for rank in range(CP_EP):
            assert torch.allclose(
                x_ranks[rank].grad,
                x_ranks[_ep_peer(rank)].grad,
                atol=1e-5,
                rtol=1e-5,
            ), "EP peers must hold identical input gradients"
            x_grad_sum += x_ranks[rank].grad
        # EP peers duplicate the same shard contribution.
        assert torch.allclose(
            x_grad_sum / EP, x_ref.grad, atol=1e-4, rtol=1e-4
        )


# ---------------------------------------------------------------------------
# _apply_moe_parallel validation (no process group required)
# ---------------------------------------------------------------------------


def _debug_model():
    from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

    torch.manual_seed(0)
    model = Magi2PreviewModel(magi2_preview_configs["debug"]())
    model.init_weights()
    return model


def _ci_model_inputs(config, seq_len: int = 16, seed: int = 5):
    """Full packed batch sized for the debug config (all modalities).

    Module-level (not gated on the nightly multi-rank import) so the
    CI-safe single-rank tests can build inputs too.
    """
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
    channels = max(
        config.video_in_channels,
        config.audio_in_channels,
        config.text_in_channels,
    )
    x = torch.randn(seq_len, channels, generator=generator)
    coords = torch.randn(seq_len, 9, generator=generator)
    time_embedding = torch.randn(
        seq_len, config.time_channel_dim, generator=generator
    )
    cu_seqlens = torch.tensor([0, 6, seq_len], dtype=torch.int32)
    labels = torch.zeros(seq_len, channels)
    video_rows = modality == Modality.VIDEO
    audio_rows = modality == Modality.AUDIO
    labels[video_rows, : config.video_in_channels] = torch.randn(
        int(video_rows.sum()), config.video_in_channels, generator=generator
    )
    labels[audio_rows, : config.audio_in_channels] = torch.randn(
        int(audio_rows.sum()), config.audio_in_channels, generator=generator
    )
    inputs = dict(
        coords_mapping=coords,
        modality_mapping=modality,
        time_embedding=time_embedding,
        cu_seqlens=cu_seqlens,
    )
    return x, inputs, labels


class TestApplyMoeParallelValidation:
    def test_noop_without_meshes(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_moe_parallel,
        )

        model = _debug_model()
        before = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        _apply_moe_parallel(model, ep_mesh=None, etp_mesh=None)

        assert model.block.layers["1"].mlp.moe_mlp.head_range is None
        for name, param in model.named_parameters():
            assert torch.equal(param, before[name])

    def test_num_heads_divisibility(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_moe_parallel,
        )

        model = _debug_model()  # debug flavor: moe_num_heads=2
        mesh = SimpleNamespace(ndim=1, size=lambda: 3, get_local_rank=lambda: 0)
        with pytest.raises(ValueError, match="divisible"):
            _apply_moe_parallel(model, ep_mesh=mesh, etp_mesh=None)

    def test_combined_ep_and_etp_deferred(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_moe_parallel,
        )

        with pytest.raises(NotImplementedError):
            _apply_moe_parallel(
                _debug_model(), ep_mesh=object(), etp_mesh=object()
            )

    def test_multi_dim_mesh_rejected(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_moe_parallel,
        )

        mesh = SimpleNamespace(ndim=2)
        with pytest.raises(ValueError, match="1D mesh"):
            _apply_moe_parallel(_debug_model(), ep_mesh=mesh, etp_mesh=None)

    def test_regime_b_requires_flattened_cp_ep_mesh(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_moe_parallel,
        )

        model = _debug_model()
        # degree 3 cannot contain the CP degree 2.
        mesh = SimpleNamespace(ndim=1, size=lambda: 3, get_local_rank=lambda: 0)
        with pytest.raises(ValueError, match="flattened"):
            _apply_moe_parallel(
                model, ep_mesh=mesh, etp_mesh=None, cp_degree=2
            )

    def test_regime_b_keeps_num_heads_divisibility_check(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_moe_parallel,
        )

        model = _debug_model()  # debug flavor: moe_num_heads=2
        mesh = SimpleNamespace(ndim=1, size=lambda: 4, get_local_rank=lambda: 0)
        with pytest.raises(ValueError, match="divisible"):
            _apply_moe_parallel(
                model, ep_mesh=mesh, etp_mesh=None, cp_degree=2
            )


# ---------------------------------------------------------------------------
# _apply_moe_parallel wiring on a single-rank gloo mesh (CI-safe)
# ---------------------------------------------------------------------------


class TestApplyMoeParallelSingleRank:
    def _wire_debug_model(self):
        from torch.distributed.device_mesh import DeviceMesh
        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_moe_parallel,
        )

        config = magi2_preview_configs["debug"]()
        model = _debug_model()
        reference = copy.deepcopy(model)
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("ep",))
        _apply_moe_parallel(model, ep_mesh=mesh, etp_mesh=None)
        return config, model, reference, mesh

    def test_state_dict_keys_and_placements(self, single_rank_process_group):
        from torch.distributed.tensor import DTensor, Shard

        config, model, reference, _ = self._wire_debug_model()

        assert set(model.state_dict().keys()) == set(
            reference.state_dict().keys()
        )
        for layer_id in config.moe_layers:
            moe = model.block.layers[str(layer_id)].mlp.moe_mlp
            assert moe.head_range == (0, config.moe_num_heads)
            for name in EXPERT_PARAM_NAMES:
                param = getattr(moe, name)
                assert isinstance(param, DTensor)
                assert param.placements == (Shard(0),)
            for name in ROUTER_BUFFER_NAMES:
                buffer = getattr(moe.router, name)
                assert isinstance(buffer, DTensor)
                assert buffer.placements == (Shard(0),)

    def test_forward_backward_match_unsharded(self, single_rank_process_group):
        config, model, reference, _ = self._wire_debug_model()
        layer_id = str(config.moe_layers[0])
        layer = model.block.layers[layer_id].mlp
        ref_layer = reference.block.layers[layer_id].mlp
        m_splits = [2, 2, 2]

        x = torch.randn(6, config.hidden_size, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)
        out = layer(x, m_splits)
        ref_out = ref_layer(x_ref, m_splits)
        assert torch.allclose(out, ref_out)

        out.sum().backward()
        ref_out.sum().backward()
        assert torch.allclose(x.grad, x_ref.grad)
        moe = model.block.layers[layer_id].mlp.moe_mlp
        ref_moe = reference.block.layers[layer_id].mlp.moe_mlp
        for name in EXPERT_PARAM_NAMES:
            grad = getattr(moe, name).grad
            assert grad is not None
            assert torch.allclose(
                grad.to_local(), getattr(ref_moe, name).grad
            )

    def test_full_checkpoint_loads_into_sharded_model(
        self, single_rank_process_group
    ):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )
        from torch.distributed.tensor import Shard, distribute_tensor

        from torchtitan_npu.models.magi2_preview.state_dict_adapter import (
            Magi2PreviewStateDictAdapter,
        )

        config, model, reference, mesh = self._wire_debug_model()
        adapter = Magi2PreviewStateDictAdapter(model_config=config)

        # The adapter keeps working on gathered (full) tensors after sharding.
        torch.manual_seed(5)
        hf_dict = {
            key: torch.randn_like(value)
            for key, value in reference.state_dict().items()
        }
        from_hf = adapter.from_hf(hf_dict)
        assert set(from_hf.keys()) == set(hf_dict.keys())

        # Load the full checkpoint into the plain reference first and pin
        # the expected local shards: set_model_state_dict replaces the input
        # dict's values with DTensors as it distributes them.
        reference.load_state_dict(from_hf)
        expected_local = {}
        for layer_id in config.moe_layers:
            prefix = f"block.layers.{layer_id}.mlp.moe_mlp."
            for name in EXPERT_PARAM_NAMES + tuple(
                f"router.{buffer}" for buffer in ROUTER_BUFFER_NAMES
            ):
                expected_local[prefix + name] = distribute_tensor(
                    from_hf[prefix + name], mesh, [Shard(0)]
                ).to_local().clone()

        # Loading the full checkpoint into the sharded model distributes the
        # expert tensors through DTensor Shard(0).
        set_model_state_dict(
            model, from_hf, options=StateDictOptions(full_state_dict=True)
        )

        for layer_id in config.moe_layers:
            moe = model.block.layers[str(layer_id)].mlp.moe_mlp
            ref_moe = reference.block.layers[str(layer_id)].mlp.moe_mlp
            prefix = f"block.layers.{layer_id}.mlp.moe_mlp."
            for name in EXPERT_PARAM_NAMES:
                assert torch.equal(
                    getattr(moe, name).to_local(), expected_local[prefix + name]
                )
                assert torch.allclose(
                    getattr(moe, name).to_local(), getattr(ref_moe, name)
                )
            for name in ROUTER_BUFFER_NAMES:
                key = prefix + f"router.{name}"
                assert torch.equal(
                    getattr(moe.router, name).to_local(), expected_local[key]
                )

        layer_id = str(config.moe_layers[0])
        x = torch.randn(6, config.hidden_size)
        m_splits = [2, 2, 2]
        out = model.block.layers[layer_id].mlp(x, m_splits)
        ref_out = reference.block.layers[layer_id].mlp(x, m_splits)
        assert torch.allclose(out, ref_out)


class TestApplyFsdpEFSDPWiring:
    """CI-safe wiring decisions of ``_apply_fsdp``'s eFSDP path.

    Single-rank gloo only exercises the ep_degree=1 branch and the guard
    that eFSDP (ep_degree > 1) requires an edp_mesh; the actual ep x dp
    composition needs the nightly multi-rank tests below.
    """

    def _fsdp_kwargs(self):
        return dict(
            training=SimpleNamespace(
                mixed_precision_param="float32",
                mixed_precision_reduce="float32",
                enable_cpu_offload=False,
            ),
            parallelism=SimpleNamespace(fsdp_reshard_after_forward="default"),
            pp_enabled=False,
        )

    def test_efsdP_requires_edp_mesh(self, single_rank_process_group):
        from torch.distributed.device_mesh import DeviceMesh

        from torchtitan_npu.models.magi2_preview.parallelize import _apply_fsdp

        model = _debug_model()
        dp_mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))
        with pytest.raises(ValueError, match="edp_mesh"):
            _apply_fsdp(
                model,
                dp_mesh,
                ep_degree=2,
                edp_mesh=None,
                **self._fsdp_kwargs(),
            )

    def test_ep1_fsdp_forward_backward_match_unsharded(
        self, single_rank_process_group
    ):
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor

        from torchtitan_npu.models.magi2_preview.parallelize import (
            _apply_fsdp,
            _apply_moe_parallel,
        )

        config = magi2_preview_configs["debug"]()
        model = _debug_model()
        model.train()
        reference = copy.deepcopy(model)

        ep_mesh = DeviceMesh("cpu", [0], mesh_dim_names=("ep",))
        _apply_moe_parallel(model, ep_mesh=ep_mesh, etp_mesh=None)
        dp_mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))
        _apply_fsdp(
            model, dp_mesh, ep_degree=1, edp_mesh=None, **self._fsdp_kwargs()
        )

        x, inputs, labels = _ci_model_inputs(config)
        pred = model(x, **inputs)
        pred_ref = reference(x, **inputs)
        assert torch.allclose(pred, pred_ref, atol=1e-4, rtol=1e-4)
        torch.nn.functional.mse_loss(pred, labels).backward()
        torch.nn.functional.mse_loss(pred_ref, labels).backward()
        ref_params = dict(reference.named_parameters())
        for name, param in model.named_parameters():
            assert param.grad is not None, name
            grad = param.grad
            if isinstance(grad, DTensor):
                grad = grad.to_local()
            assert torch.allclose(
                grad, ref_params[name].grad, atol=1e-4, rtol=1e-4
            ), name


# ---------------------------------------------------------------------------
# Nightly multi-rank gloo coverage
# ---------------------------------------------------------------------------


if MULTI_RANK_AVAILABLE:

    class TwoRankMultiRankTestBase(DTensorTestBase):
        @property
        def world_size(self):
            return 2

    @mark_multi_rank_nightly
    class TestHeadParallelMoEMultiRank(TwoRankMultiRankTestBase):
        @with_comms
        def test_regime_a_matches_unsharded(self):
            from torch.distributed.device_mesh import init_device_mesh

            from torchtitan_npu.models.magi2_preview.model import (
                Magi2PreviewModel,
            )
            from torchtitan_npu.models.magi2_preview.parallelize import (
                _apply_moe_parallel,
            )

            config = magi2_preview_configs["debug"]()
            torch.manual_seed(1234)
            model = Magi2PreviewModel(config)
            model.init_weights()
            reference = copy.deepcopy(model)

            mesh = init_device_mesh(
                self.device_type, (2,), mesh_dim_names=("ep",)
            )
            _apply_moe_parallel(model, ep_mesh=mesh, etp_mesh=None)

            rank = dist.get_rank()
            heads_per_rank = config.moe_num_heads // 2
            local_rows = heads_per_rank * config.num_experts
            for layer_id in config.moe_layers:
                moe = model.block.layers[str(layer_id)].mlp.moe_mlp
                assert moe.head_range == (
                    rank * heads_per_rank,
                    (rank + 1) * heads_per_rank,
                )
                assert moe.gate.to_local().shape[0] == local_rows

            layer_id = str(config.moe_layers[0])
            layer = model.block.layers[layer_id].mlp
            ref_layer = reference.block.layers[layer_id].mlp
            m_splits = [2, 3, 3]
            torch.manual_seed(7)
            x = torch.randn(8, config.hidden_size)

            out = layer(x, m_splits)
            ref_out = ref_layer(x, m_splits)
            assert torch.allclose(out, ref_out, atol=1e-5)

            x1 = x.clone().requires_grad_(True)
            x2 = x.clone().requires_grad_(True)
            layer(x1, m_splits).sum().backward()
            ref_layer(x2, m_splits).sum().backward()
            assert torch.allclose(x1.grad, x2.grad, atol=1e-5)

            moe = model.block.layers[layer_id].mlp.moe_mlp
            ref_moe = reference.block.layers[layer_id].mlp.moe_mlp
            head_start = moe.head_range[0]
            rows = slice(
                head_start * config.num_experts,
                (head_start + heads_per_rank) * config.num_experts,
            )
            for name in EXPERT_PARAM_NAMES:
                grad = getattr(moe, name).grad
                assert grad is not None
                assert torch.allclose(
                    grad.to_local(),
                    getattr(ref_moe, name).grad[rows],
                    atol=1e-5,
                )

        @with_comms
        def test_dispatch_undispatch_round_trip(self):
            ep_size = dist.get_world_size()
            rank = dist.get_rank()
            seq_per_rank, num_heads, d_head = 6, 4, 3

            torch.manual_seed(21)
            full_seq = torch.randn(ep_size * seq_per_rank, num_heads, d_head)
            local = full_seq[rank * seq_per_rank : (rank + 1) * seq_per_rank]
            local = local.contiguous().clone().requires_grad_(True)

            group = dist.group.WORLD
            dispatched = ep_dispatch(local, group)
            heads_per_rank = num_heads // ep_size
            expected = full_seq[:, rank * heads_per_rank : (rank + 1) * heads_per_rank]
            assert dispatched.shape == (
                ep_size * seq_per_rank,
                heads_per_rank,
                d_head,
            )
            assert torch.equal(dispatched, expected.contiguous())

            combined = ep_undispatch(dispatched, group)
            assert torch.equal(combined, local.detach())

            dispatched.float().sum().backward()
            assert local.grad is not None
            assert torch.equal(local.grad, torch.ones_like(local))

    class FourRankMultiRankTestBase(DTensorTestBase):
        @property
        def world_size(self):
            return 4

        @property
        def device_type(self):
            return "cpu"

    def _debug_model_inputs(config, seq_len: int = 16, seed: int = 5):
        """Full packed batch sized for the debug config (all modalities)."""
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
        channels = max(
            config.video_in_channels,
            config.audio_in_channels,
            config.text_in_channels,
        )
        x = torch.randn(seq_len, channels, generator=generator)
        coords = torch.randn(seq_len, 9, generator=generator)
        time_embedding = torch.randn(
            seq_len, config.time_channel_dim, generator=generator
        )
        cu_seqlens = torch.tensor([0, 6, seq_len], dtype=torch.int32)
        labels = torch.zeros(seq_len, channels)
        video_rows = modality == Modality.VIDEO
        audio_rows = modality == Modality.AUDIO
        labels[video_rows, : config.video_in_channels] = torch.randn(
            int(video_rows.sum()), config.video_in_channels, generator=generator
        )
        labels[audio_rows, : config.audio_in_channels] = torch.randn(
            int(audio_rows.sum()), config.audio_in_channels, generator=generator
        )
        inputs = dict(
            coords_mapping=coords,
            modality_mapping=modality,
            time_embedding=time_embedding,
            cu_seqlens=cu_seqlens,
        )
        return x, inputs, labels

    @mark_multi_rank_nightly
    class TestCpEpMoEFourRankMultiRank(FourRankMultiRankTestBase):
        """CP=2 x EP=2 regime (b) vs unsharded over a real gloo group.

        Wires Ulysses CP + the flattened cp x ep MoE dispatch (the
        production wiring order, without FSDP) and compares the full
        prediction plus every parameter gradient against the cp=1/ep=1
        run. Run with:

            torchrun --nproc_per_node=4 -m pytest \
                tests/unit_tests/models/test_magi2_expert_parallel.py \
                -m nightly -k FourRank
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
                self.device_type, (CP, EP), mesh_dim_names=("cp", "ep")
            )
            config = magi2_preview_configs["debug"]()
            # cp x ep must divide moe_num_heads (regime b head sharding).
            config.moe_num_heads = CP * EP
            torch.manual_seed(1234)
            model = Magi2PreviewModel(config)
            model.init_weights()
            model.train()
            reference = copy.deepcopy(model)

            dims = SimpleNamespace(get_mesh=lambda name: mesh[name])
            moe_mesh = flatten_head_mesh(dims)
            assert moe_mesh.size() == CP * EP
            # Matches torchtitan's own flattening of the cp x ep submesh.
            official_flat = mesh["cp", "ep"]._flatten()
            assert dist.get_process_group_ranks(
                moe_mesh.get_group()
            ) == dist.get_process_group_ranks(official_flat.get_group())

            apply_magi2_ulysses_cp(model, cp_mesh=mesh["cp"], ep_degree=EP)
            _apply_moe_parallel(
                model, ep_mesh=moe_mesh, etp_mesh=None, cp_degree=CP
            )

            assert set(model.state_dict().keys()) == set(
                reference.state_dict().keys()
            )
            rank = dist.get_rank()
            heads_per_rank = config.moe_num_heads // (CP * EP)
            local_rows = heads_per_rank * config.num_experts
            for layer_id in config.moe_layers:
                layer = model.block.layers[str(layer_id)].mlp
                moe = layer.moe_mlp
                assert moe.head_range == (
                    rank * heads_per_rank,
                    (rank + 1) * heads_per_rank,
                )
                assert layer.moe_dispatch_context is not None
                # DTensor Shard(0): global shape kept, local head shard.
                assert moe.gate.shape[0] == config.moe_num_heads * config.num_experts
                assert moe.gate.to_local().shape[0] == local_rows
                assert moe.router.expert_bias.to_local().shape[0] == local_rows

            x, inputs, labels = _debug_model_inputs(config)
            pred = model(x, **inputs)
            pred_ref = reference(x, **inputs)
            assert torch.allclose(pred, pred_ref, atol=1e-4, rtol=1e-4)

            torch.nn.functional.mse_loss(pred, labels).backward()
            torch.nn.functional.mse_loss(pred_ref, labels).backward()

            cp_group = mesh["cp"].get_group()
            cp_local_rank = mesh["cp"].get_local_rank()
            attn_heads = config.hidden_size // config.head_dim
            heads_per_cp_rank = attn_heads // CP
            ref_params = dict(reference.named_parameters())
            for name, param in model.named_parameters():
                assert param.grad is not None, name
                ref_grad = ref_params[name].grad
                short_name = name.split(".")[-1]
                if ".moe_mlp." in name and short_name in EXPERT_PARAM_NAMES:
                    # Head-sharded expert params: local DTensor shard grad
                    # equals the reference's head rows (1/ep compensated).
                    rows = slice(
                        rank * heads_per_rank * config.num_experts,
                        (rank + 1) * heads_per_rank * config.num_experts,
                    )
                    assert torch.allclose(
                        param.grad.to_local(),
                        ref_grad[rows],
                        atol=1e-4,
                        rtol=1e-4,
                    ), name
                    continue
                if name.endswith(".sinks"):
                    # Head-sharded on the cp mesh (Ulysses attention).
                    expected = ref_grad[
                        :,
                        cp_local_rank
                        * heads_per_cp_rank : (cp_local_rank + 1)
                        * heads_per_cp_rank,
                    ]
                    assert torch.allclose(
                        param.grad, expected, atol=1e-4, rtol=1e-4
                    ), name
                    continue
                # Replicated params: each rank back-propagates only its CP
                # token shard, so summing over the cp mesh restores the
                # unsharded gradient (EP peers hold identical gradients).
                grad_sum = param.grad.clone()
                dist.all_reduce(grad_sum, group=cp_group)
                assert torch.allclose(
                    grad_sum, ref_grad, atol=1e-4, rtol=1e-4
                ), name
