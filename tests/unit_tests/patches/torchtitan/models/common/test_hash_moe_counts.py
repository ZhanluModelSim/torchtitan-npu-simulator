# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed expert configuration, native router statistics, and score transport.

The grouped GEMM seam uses portable CPU matmul. Router, dispatch, clamp,
score placement, combine, gradients, and the two-rank SPMD boundaries stay real.
This does not validate NPU kernels or HCCL.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.linear import GroupedLinear, RouterGateLinear
from torchtitan.models.common.moe import MoE, register_moe_load_balancing_hook
from torchtitan.models.deepseek_v4.moe import DeepSeekV4Router

from tests.unit_tests.models.spmd_test_utils import gloo_ep_group_context
from torchtitan_npu.models.deepseek_v4 import model_registry
from torchtitan_npu.models.deepseek_v4_1.moe import DeepSeekV41Router
from torchtitan_npu.patches.torchtitan.models.common.moe import (
    ClampedSwiGLU,
    _RouterScoreAbsorbingRoutedExperts,
)
from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
)


class _RoutedExpertsBoundaryStub(torch.nn.Module):
    """Only isolate expert math when checking router counts and the bias hook."""

    def forward(self, x, scores, expert_ids, counts):
        return x


def _small_config(*, route="hash", absorb=True, limit=1.0, enable_ep=False):
    # Start at the actual recipe so this test detects factory/patch selection
    # regressions, then shrink dimensions without changing expert ownership.
    recipe = model_registry("debugmodel").layers[0].moe
    router_type = DeepSeekV4Router if route == "hash" else DeepSeekV41Router
    router = router_type.Config(
        num_experts=4,
        gate=RouterGateLinear.Config(in_features=2, out_features=4),
        score_func=Sigmoid.Config(),
        top_k=2,
        **({"vocab_size": 8, "layer_id": 0, "n_hash_layers": 1} if route == "hash" else {}),
        **({"vision_enabled": route == "vision", "sorted_topk": True} if route != "hash" else {}),
    )
    experts = replace(
        recipe.routed_experts,
        w13=GroupedLinear.Config(group_size=4, in_features=2, out_features=3, num_linears=2),
        w2=GroupedLinear.Config(group_size=4, in_features=3, out_features=2),
        activation_fn=ClampedSwiGLU.Config(swiglu_limit=limit),
        token_dispatcher=(AllToAllTokenDispatcher if enable_ep else LocalTokenDispatcher).Config(
            num_experts=4, top_k=2, absorb_router_scores=absorb
        ),
    )
    return replace(
        recipe, num_experts=4, router=router, routed_experts=experts, shared_experts=None, load_balance_coeff=0.1
    )


def _cpu_grouped_mm(self, *, input_RI, weight_EOI, offsets_E):
    """Portable replacement of the device primitive; no routing logic is mocked."""
    boundaries = [0, *offsets_E.tolist()]
    return torch.cat(
        [
            input_RI[start:end] @ weight.bfloat16().T
            for start, end, weight in zip(boundaries[:-1], boundaries[1:], weight_EOI, strict=True)
        ]
    )


def _initialize(moe):
    with torch.no_grad():
        moe.router.gate.weight.copy_(torch.tensor([[1.0, -0.5], [-0.2, 1.0], [0.4, 0.7], [-0.8, -0.3]]))
        for i, parameter in enumerate(moe.routed_experts.parameters()):
            parameter.copy_(torch.linspace(-1.5 + i * 0.1, 1.6, parameter.numel()).reshape_as(parameter))
        if getattr(moe.router, "hash", False):
            moe.router.tid2eid.copy_(torch.tensor([[3, 1], [0, 2], [2, 0], [1, 3], [3, 0], [1, 2], [2, 3], [0, 1]]))


def test_v4_recipe_preserves_typed_grouped_linears_clamp_scores_and_balancing_buffer():
    config = model_registry("debugmodel").layers[0].moe
    experts = config.routed_experts
    assert experts.__class__ is _RouterScoreAbsorbingRoutedExperts.Config
    assert (experts.w13.group_size, experts.w13.num_linears) == (config.num_experts, 2)
    assert (experts.w13.in_features, experts.w13.out_features) == (experts.w2.out_features, experts.w2.in_features)
    assert (experts.w2.group_size, experts.w2.num_linears) == (config.num_experts, 1)
    assert experts.activation_fn.swiglu_limit == 10.0
    assert experts.token_dispatcher.absorb_router_scores is True
    with torch.random.fork_rng(devices=[]):
        moe = config.build()
    assert type(moe) is MoE
    assert moe.router.hash
    assert moe.load_balance_coeff == config.load_balance_coeff
    torch.testing.assert_close(moe.expert_bias_E, torch.zeros(config.num_experts))
    assert "expert_bias_E" in moe.state_dict()
    assert "router.tokens_per_expert_E" not in moe.state_dict()


def _reference(x, gate, w13, w2, ids, *, limit, absorb):
    # Independent token/expert loops: no dispatcher, grouped GEMM helper, or
    # activation implementation is reused to obtain expected values.
    scores = torch.sigmoid(x @ gate.T).gather(-1, ids)
    output = []
    for token, selected, selected_scores in zip(x, ids, scores, strict=True):
        contributions = []
        for expert, score in zip(selected.tolist(), selected_scores, strict=True):
            gate_up = token.bfloat16() @ w13[expert].flatten(0, 1).bfloat16().T
            gate_value, up = gate_up.reshape(2, -1).unbind(0)
            if limit > 0:
                gate_value = gate_value.clamp(max=limit)
                up = up.clamp(min=-limit, max=limit)
            hidden = F.silu(gate_value) * up
            if absorb:
                hidden = (hidden.float() * score).to(hidden.dtype)
            contribution = (hidden @ w2[expert].bfloat16().T).to(x.dtype)
            contributions.append(contribution if absorb else contribution * score)
        output.append(torch.stack(contributions).sum(0))
    return torch.stack(output)


@pytest.mark.parametrize("absorb", [False, True])
@pytest.mark.parametrize("limit", [0.0, 1.0])
def test_expert_score_placement_clamp_and_gradients_match_independent_reference(monkeypatch, absorb, limit):
    monkeypatch.setattr(GroupedLinear, "_grouped_mm", _cpu_grouped_mm)
    with torch.random.fork_rng(devices=[]):
        moe = _small_config(absorb=absorb, limit=limit).build()
    _initialize(moe)
    x = torch.tensor([[3.0, -1.0], [2.0, 0.5], [-1.0, 2.0], [0.0, 3.0]], requires_grad=True)
    token_ids = torch.tensor([0, 2, 5, 7])
    ids = torch.tensor([[3, 1], [2, 0], [1, 2], [0, 1]])
    reference_inputs = [
        value.detach().clone().requires_grad_()
        for value in (x, moe.router.gate.weight, moe.routed_experts.w13.weight, moe.routed_experts.w2.weight)
    ]

    actual = moe(x, input_ids_T=token_ids)
    expected = _reference(*reference_inputs, ids, limit=limit, absorb=absorb)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()

    for actual_tensor, reference_tensor in zip(
        (x, moe.router.gate.weight, moe.routed_experts.w13.weight, moe.routed_experts.w2.weight),
        reference_inputs,
        strict=True,
    ):
        assert actual_tensor.grad is not None
        assert torch.count_nonzero(actual_tensor.grad) > 0
        torch.testing.assert_close(actual_tensor.grad, reference_tensor.grad, rtol=0.02, atol=0.02)
    torch.testing.assert_close(moe.router.tokens_per_expert_E, torch.tensor([2.0, 3.0, 2.0, 1.0]))


def test_moe_forwards_image_mask_to_router():
    with torch.random.fork_rng(devices=[]):
        moe = _small_config(route="vision").build()
    moe.routed_experts = _RoutedExpertsBoundaryStub()
    _initialize(moe)
    mask = torch.tensor([False, True])
    observed = []
    handle = moe.router.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs), with_kwargs=True
    )
    try:
        x = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
        torch.testing.assert_close(moe(x, image_mask=mask), x)
    finally:
        handle.remove()
    assert observed[0]["image_mask"] is mask
    assert moe.router.tokens_per_expert_E.sum().item() == 4


def _spmd_worker(rank, rendezvous, *, check_experts, uneven_peer_splits=False):
    import torchtitan.distributed.parallelism_context as context_module
    from torchtitan.distributed import ParallelismContext
    from torchtitan.models.common.decoder_sharding import dense_sequence_parallel_placement, token_id_placement

    from torchtitan_npu.models.deepseek_v4.sharding import set_deepseek_v4_layer_sharding

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        context = ParallelismContext(
            dp_replicate=1, dp_shard=1, cp=1, tp=2, pp=1, ep=2, world_size=2, enable_sequence_parallel=True
        )
        with patch.object(context_module, "device_type", "cpu"):
            context.build_mesh()
        layer = model_registry("debugmodel").layers[0]
        layer.moe = _small_config(enable_ep=True)
        set_deepseek_v4_layer_sharding(layer, enable_sp=True, enable_ep=True)
        with torch.random.fork_rng(devices=[]):
            moe = layer.moe.build()
        _initialize(moe)
        reference_weights = [
            value.detach().clone().requires_grad_()
            for value in (moe.router.gate.weight, moe.routed_experts.w13.weight, moe.routed_experts.w2.weight)
        ]
        if not check_experts:
            moe.routed_experts = _RoutedExpertsBoundaryStub()
        with gloo_ep_group_context(), patch.object(GroupedLinear, "_grouped_mm", _cpu_grouped_mm):
            moe._parallelize(context)
            full_x = torch.tensor([[3.0, -1.0], [2.0, 0.5], [-1.0, 2.0], [0.0, 3.0]])
            x = full_x.chunk(2)[rank].clone().requires_grad_()
            # Gloo list-alltoall requires equal peer split sizes. The balanced
            # peer case still has unequal counts among individual experts.
            use_equal_peer_splits = check_experts and not uneven_peer_splits
            token_ids = torch.tensor([0, 2, 3, 4] if use_equal_peer_splits else [0, 2, 5, 7])
            with context.activate_spmd(typechecking=True):
                spmd.assert_type(x, dense_sequence_parallel_placement())
                spmd.assert_type(token_ids, token_id_placement())
                output = moe(x, input_ids_T=token_ids)
                output.square().sum().backward()
            assert output.shape == x.shape
            assert torch.isfinite(output).all()
            assert x.grad is not None and torch.isfinite(x.grad).all()
            if check_experts:
                reference_x = full_x.clone().requires_grad_()
                reference_ids = torch.tensor(
                    [[3, 1], [2, 0], [1, 3], [3, 0]] if use_equal_peer_splits else [[3, 1], [2, 0], [1, 2], [0, 1]]
                )
                expected = _reference(reference_x, *reference_weights, reference_ids, limit=1.0, absorb=True)
                torch.testing.assert_close(output, expected.chunk(2)[rank], rtol=0, atol=0)
                expected.square().sum().backward()
                torch.testing.assert_close(x.grad, reference_x.grad.chunk(2)[rank], rtol=0.02, atol=0.02)
                for actual, reference in zip(
                    (moe.routed_experts.w13.weight, moe.routed_experts.w2.weight), reference_weights[1:], strict=True
                ):
                    assert actual.grad is not None and torch.count_nonzero(actual.grad) > 0
                    torch.testing.assert_close(actual.grad, reference.grad.chunk(2)[rank], rtol=0.02, atol=0.02)
                # The router gate is replicated; FSDP's later gradient reduce
                # is outside this unit. Sum the actual local contributions
                # explicitly to compare them with the full-token reference.
                dist.all_reduce(moe.router.gate.weight.grad)
                torch.testing.assert_close(moe.router.gate.weight.grad, reference_weights[0].grad, rtol=0.02, atol=0.02)
            else:
                torch.testing.assert_close(output, x)
            expected_local_counts = torch.tensor(
                [1.0, 1.0, 1.0, 1.0]
                if rank == 0
                else ([1.0, 1.0, 0.0, 2.0] if use_equal_peer_splits else [1.0, 2.0, 1.0, 0.0])
            )
            torch.testing.assert_close(moe.router.tokens_per_expert_E, expected_local_counts)
            block = torch.nn.Module()
            block.moe_enabled, block.moe = True, moe
            model = torch.nn.Module()
            model.layers = torch.nn.ModuleList([block])
            optimizer = torch.optim.SGD(moe.parameters(), lr=0.0, foreach=False, fused=False)
            register_moe_load_balancing_hook(optimizer, [model], context)
            optimizer.step()
            expected_bias = [0.0, 0.0, 0.1, -0.1] if use_equal_peer_splits else [0.0, -0.1, 0.0, 0.1]
            torch.testing.assert_close(moe.expert_bias_E, torch.tensor(expected_bias))
            torch.testing.assert_close(moe.router.tokens_per_expert_E, torch.zeros(4))
    finally:
        dist.destroy_process_group()


def _count_worker(rank, rendezvous):
    _spmd_worker(rank, rendezvous, check_experts=False)


def _parallelized_routing_worker(rank, rendezvous, uneven_peer_splits):
    _spmd_worker(rank, rendezvous, check_experts=True, uneven_peer_splits=uneven_peer_splits)


def test_hash_moe_keeps_counts_and_balancing_updates_on_two_spmd_ranks(tmp_path):
    mp.spawn(_count_worker, args=(str(tmp_path / "counts"),), nprocs=2, join=True)


@pytest.mark.parametrize("uneven_peer_splits", [False, True])
def test_hash_moe_parallelized_token_metadata_dispatch_gradients_and_bias_hook_on_two_spmd_ranks(
    tmp_path, uneven_peer_splits
):
    try:
        mp.spawn(
            _parallelized_routing_worker, args=(str(tmp_path / "experts"), uneven_peer_splits), nprocs=2, join=True
        )
    except mp.ProcessRaisedException as error:
        gloo_uneven_split_errors = (
            "ProcessGroupGloo::alltoall: invalid tensor size at index 1 (expected (2, 2), got (3, 2))",
            "ProcessGroupGloo::alltoall: invalid tensor size at index 1 (expected (3, 2), got (1, 2))",
        )
        if uneven_peer_splits and any(message in str(error) for message in gloo_uneven_split_errors):
            pytest.xfail("spmd_types 0.2.5 uses list-alltoall for uneven splits; Gloo requires equal peer sizes")
        raise
