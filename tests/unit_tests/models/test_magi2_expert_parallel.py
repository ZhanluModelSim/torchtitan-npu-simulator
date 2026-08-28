# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Head-parallel MoE tests: head_range routing/expert equivalence, slicing
utilities, regime-(a) zero-pad/all-reduce assembly, regime-(b) Ulysses
dispatch algebra, DTensor/checkpoint expectations, and nightly multi-rank
gloo coverage (gated like tests/smoke_tests/model_parallel)."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from torchtitan_npu.models.magi2_preview import magi2_preview_configs
from torchtitan_npu.models.magi2_preview.expert_parallel import (
    EXPERT_PARAM_NAMES,
    ROUTER_BUFFER_NAMES,
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
# _apply_moe_parallel validation (no process group required)
# ---------------------------------------------------------------------------


def _debug_model():
    from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

    torch.manual_seed(0)
    model = Magi2PreviewModel(magi2_preview_configs["debug"]())
    model.init_weights()
    return model


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
