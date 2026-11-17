# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models.deepseek_v4_1 import _LINEAR_INIT, _make_v41_moe_config
from torchtitan_npu.models.deepseek_v4_1.config_registry import deepseek_v4_1_debugmodel_multimodal


def test_plain_weights_use_routed_experts_and_track_empty_experts():
    config = _make_v41_moe_config(
        layer_id=0,
        dim=8,
        moe_inter_dim=8,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        route_scale=1.5,
        route_norm=True,
        load_balance_coeff=1e-3,
        moe_comm_backend="standard",
        non_blocking_capacity_factor=None,
        vision_enabled=False,
    )
    with torch.random.fork_rng(devices=[]):
        moe = build_cpu_model(config)
    with torch.no_grad():
        moe.router.gate.weight.zero_()
        moe.expert_bias_E.copy_(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    calls = []
    handle = moe.routed_experts.register_forward_hook(lambda module, args, output: calls.append(output))
    try:
        x = torch.linspace(-1, 1, 24).reshape(1, 3, 8).requires_grad_()
        out = moe(x)
        assert len(calls) == 1
        torch.testing.assert_close(moe.tokens_per_expert_E, torch.tensor([3.0, 3.0, 0.0, 0.0]))
        out.square().mean().backward()
        for parameter in (
            moe.routed_experts.inner_experts.w1_EFD,
            moe.routed_experts.inner_experts.w2_EDF,
            moe.routed_experts.inner_experts.w3_EFD,
        ):
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad[2:]) == 0
        assert x.grad is not None and torch.isfinite(x.grad).all()
    finally:
        handle.remove()


def test_v41_moe_rides_the_common_stack_with_an_opt_in_vision_bias():
    """V4.1 builds the common MoE factories; the vision bias is a router opt-in.

    The two extensions the plugin carries (the image-masked ``bias_vl`` and the
    reference's descending selection order) must stay off by default so every other
    user of the shared router -- ``deepseek_v4`` included -- keeps the pinned
    behaviour.  ``bias_vl`` is a plain parameter, so the MoE balancing hook, which
    updates ``expert_bias_E`` only, can never touch it.
    """
    from torchtitan.models.common.config_utils import make_router_config
    from torchtitan.models.common.feed_forward import FeedForward
    from torchtitan.models.common.moe import GroupedExperts, RoutedExperts, TokenChoiceTopKRouter

    config = _make_v41_moe_config(
        layer_id=0,
        dim=8,
        moe_inter_dim=8,
        num_experts=4,
        num_shared_experts=1,
        top_k=2,
        route_scale=1.5,
        route_norm=True,
        load_balance_coeff=1e-3,
        moe_comm_backend="standard",
        non_blocking_capacity_factor=None,
        vision_enabled=True,
    )
    assert config.router.vision_enabled and config.router.sorted_topk

    with torch.random.fork_rng(devices=[]):
        moe = build_cpu_model(config)
    assert isinstance(moe.router, TokenChoiceTopKRouter)
    assert isinstance(moe.routed_experts, RoutedExperts)
    assert isinstance(moe.routed_experts.inner_experts, GroupedExperts)
    assert isinstance(moe.shared_experts, FeedForward)
    named_parameters = dict(moe.router.named_parameters())
    assert "bias_vl" in named_parameters
    assert named_parameters["bias_vl"].requires_grad
    assert "bias_vl" not in dict(moe.router.named_buffers())
    assert "tokens_per_expert_E" in dict(moe.named_buffers())

    # The plain factory -- what the other models build -- stays bias-free and its
    # selection ignores an image mask entirely.
    plain = make_router_config(
        dim=8,
        num_experts=4,
        gate_param_init=_LINEAR_INIT,
        top_k=2,
        score_func="sigmoid",
    )
    assert not plain.vision_enabled and not plain.sorted_topk
    with torch.random.fork_rng(devices=[]):
        plain_router = build_cpu_model(plain)
    assert plain_router.bias_vl is None

    x_BLD = torch.linspace(-1, 1, 24).reshape(1, 3, 8)
    image_mask = torch.tensor([[True, False, True]])
    _, masked_ids, _ = plain_router(x_BLD, image_mask=image_mask)
    _, plain_ids, _ = plain_router(x_BLD)
    torch.testing.assert_close(masked_ids, plain_ids, rtol=0, atol=0)


def test_baseline_rejects_tensor_parallel():
    config = deepseek_v4_1_debugmodel_multimodal()
    config.parallelism.tensor_parallel_degree = 2
    with pytest.raises(NotImplementedError, match="TP=1"):
        config.model_spec.model.update_from_config(config=config)


def test_trainer_config_rejects_a_ratio_table_shorter_than_the_stack():
    # Positive control: the registered table is accepted by the same entry point, so a
    # check that always raised would not pass this test.
    registered = deepseek_v4_1_debugmodel_multimodal()
    registered.training.seq_len = 512
    registered.model_spec.model.update_from_config(config=registered)

    config = deepseek_v4_1_debugmodel_multimodal()
    config.training.seq_len = 512
    model_config = config.model_spec.model
    model_config.compress_ratios = model_config.compress_ratios[:-1]
    with pytest.raises(ValueError, match="compress_ratios must match n_layers"):
        model_config.update_from_config(config=config)


def test_vision_bias_follows_the_factory_switch():
    """``vision_enabled`` is the MoE factory's parameter, and it is the only thing that
    decides whether the router builds a ``bias_vl`` parameter at all."""
    from torchtitan_npu.models.deepseek_v4_1 import _make_v41_moe_config

    def build(vision_enabled: bool):
        config = _make_v41_moe_config(
            layer_id=0,
            dim=8,
            moe_inter_dim=8,
            num_experts=4,
            num_shared_experts=1,
            top_k=2,
            route_scale=1.5,
            route_norm=True,
            load_balance_coeff=1e-3,
            moe_comm_backend="standard",
            non_blocking_capacity_factor=None,
            vision_enabled=vision_enabled,
        )
        with torch.random.fork_rng(devices=[]):
            return config, build_cpu_model(config)

    config, moe = build(True)
    assert config.router.vision_enabled
    assert moe.router.bias_vl is not None and moe.router.bias_vl.requires_grad

    config, moe = build(False)
    assert not config.router.vision_enabled
    assert moe.router.bias_vl is None
    # The selection order is not a modality property: every V4.1 layer keeps it.
    assert config.router.sorted_topk
    named = dict(moe.router.named_parameters())
    assert "bias_vl" not in named
