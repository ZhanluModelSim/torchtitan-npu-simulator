# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 dynamic selection and upstream V4 hash routing on CPU."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torchtitan.config import derive
from torchtitan.models.common.activation import Sigmoid, Softmax, SqrtSoftplus
from torchtitan.models.deepseek_v4.moe import DeepSeekV4Router

from torchtitan_npu.models.deepseek_v4_1.moe import DeepSeekV41Router

_WEIGHT = torch.tensor(
    [
        [3.0, 0.2, -0.3],
        [-3.0, -0.7, 0.9],
        [-4.0, 1.4, -1.1],
        [-5.0, 0.6, 0.5],
        [0.9, -0.8, 1.2],
        [0.7, 0.3, -0.5],
        [0.5, 1.1, 0.4],
        [0.2, -0.2, -1.3],
    ]
)
_INPUT = torch.tensor([[1.0, 0.0, 0.0], [0.2, 1.2, -0.4], [-0.5, 0.3, 1.1], [0.7, -0.9, 0.6]])
_BALANCING_BIAS = torch.tensor([0.05, 0.7, -0.4, 0.3, -0.2, 0.1, -0.5, 0.4])
_VISION_BIAS = torch.tensor([0.8, -0.5, 0.2, -0.1, 1.1, -0.8, 0.4, -0.3])
_IMAGE_MASK = torch.tensor([False, True, False, True])
_TOKEN_IDS = torch.tensor([4, 1, 4, 6])
_HASH_TABLE = torch.tensor([[0, 1], [6, 2], [7, 4], [1, 3], [5, 0], [2, 7], [3, 6], [4, 1]])


def _router(settings):
    from torchtitan.models.common.config_utils import make_router_config

    settings = dict(settings)
    score_func = settings.pop("score_func", "sigmoid")
    activation = {"sigmoid": Sigmoid, "softmax": Softmax, "sqrtsoftplus": SqrtSoftplus}[score_func]
    config = make_router_config(dim=3, num_experts=8, gate_param_init={}, top_k=2, score_func=activation.Config())
    is_hash = settings.pop("hash", False)
    vocab_size = settings.pop("vocab_size", None)
    if is_hash:
        settings.pop("vision_enabled", None)
        settings.pop("sorted_topk", None)
        config = replace(config, **settings)
        config = derive(config, DeepSeekV4Router.Config, vocab_size=vocab_size, layer_id=0, n_hash_layers=1)
    else:
        config = derive(config, DeepSeekV41Router.Config, **settings)
    with torch.random.fork_rng(devices=[]):
        router = config.build()
    assert type(router) is (DeepSeekV4Router if is_hash else DeepSeekV41Router)
    with torch.no_grad():
        router.gate.weight.copy_(_WEIGHT)
        if getattr(router, "bias_vl", None) is not None:
            router.bias_vl.copy_(_VISION_BIAS)
        if getattr(router, "hash", False):
            router.tid2eid.copy_(_HASH_TABLE)
    return router


def _reference_expert_ids(scores, settings, balancing_bias):
    """Independent row/group ranking; the fixed map and debug order are fixtures."""
    top_k = settings.get("top_k", 2)
    if settings.get("hash"):
        chosen = [_HASH_TABLE[token].tolist() for token in _TOKEN_IDS.tolist()]
    else:
        choice = []
        chosen = []
        for token, row in enumerate(scores.detach().tolist()):
            bias = _VISION_BIAS if settings.get("vision_enabled") and _IMAGE_MASK[token] else balancing_bias
            values = [
                score + (0.0 if bias is None else float(bias.detach()[expert])) for expert, score in enumerate(row)
            ]
            eligible = list(range(8))
            if settings.get("num_expert_groups"):
                group_scores = [sum(sorted(values[group : group + 4], reverse=True)[:2]) for group in (0, 4)]
                winning_group = max(range(2), key=lambda group: group_scores[group])
                eligible = list(range(winning_group * 4, winning_group * 4 + 4))
                values = [value if expert in eligible else float("-inf") for expert, value in enumerate(values)]
            choice.append(values)
            chosen.append(sorted(eligible, key=lambda expert: values[expert], reverse=True)[:top_k])
        if not settings.get("sorted_topk", False):
            # Legacy unsorted ordering is a PyTorch topk contract, not descending
            # Python order. Independently ranked membership is checked as well.
            unsorted = torch.tensor(choice, dtype=scores.dtype).topk(top_k, dim=-1, sorted=False).indices
            assert torch.equal(unsorted.sort(-1).values, torch.tensor(chosen).sort(-1).values)
            chosen = unsorted.tolist()
    if settings.get("_debug_force_load_balance"):
        chosen = [[(token * top_k + slot) % 8 for slot in range(top_k)] for token in range(scores.shape[0])]
    return torch.tensor(chosen, dtype=torch.long)


def _reference_forward(inputs, weight, settings, balancing_bias):
    gate = F.linear(inputs, weight)
    score_func = settings.get("score_func", "sigmoid")
    if score_func == "softmax":
        scores = gate.softmax(-1)
    elif score_func == "sqrtsoftplus":
        scores = F.softplus(gate).sqrt()
    else:
        scores = gate.sigmoid()
    expert_ids = _reference_expert_ids(scores, settings, balancing_bias)
    selected = scores.gather(-1, expert_ids)
    if settings.get("route_norm", False):
        selected = selected / (selected.sum(-1, keepdim=True) + 1e-20)
    routing_map = torch.tensor([[expert in row for expert in range(8)] for row in expert_ids.tolist()])
    return selected * settings.get("route_scale", 1.0), expert_ids, routing_map


@pytest.mark.parametrize(
    ("settings", "with_balancing_bias"),
    [
        pytest.param({}, False, id="sigmoid-no-bias-unsorted"),
        pytest.param(
            {"score_func": "softmax", "sorted_topk": True, "top_k": 4}, True, id="softmax-balancing-bias-sorted"
        ),
        pytest.param(
            {
                "score_func": "sqrtsoftplus",
                "vision_enabled": True,
                "sorted_topk": True,
                "route_norm": True,
                "route_scale": 1.7,
            },
            True,
            id="vision-replaces-balancing-bias",
        ),
        pytest.param({"vision_enabled": True, "route_scale": 0.65}, False, id="vision-without-balancing-bias"),
        pytest.param(
            {
                "vision_enabled": True,
                "num_expert_groups": 2,
                "num_limited_groups": 1,
                "sorted_topk": True,
                "route_norm": True,
            },
            False,
            id="group-top2-sum-and-image-bias",
        ),
        pytest.param(
            {
                "hash": True,
                "vocab_size": 8,
                "vision_enabled": True,
                "sorted_topk": True,
                "score_func": "sqrtsoftplus",
                "route_norm": True,
                "route_scale": 1.3,
            },
            True,
            id="fixed-hash-ignores-choice-bias",
        ),
        pytest.param(
            {
                "hash": True,
                "vocab_size": 8,
                "vision_enabled": True,
                "_debug_force_load_balance": True,
                "route_norm": True,
                "route_scale": 0.6,
            },
            True,
            id="debug-round-robin-overrides-hash",
        ),
    ],
)
def test_plain_router_scores_routes_and_gate_gradients_match_reference(settings, with_balancing_bias):
    router = _router(settings)
    assert DeepSeekV41Router.forward is DeepSeekV41Router.__bases__[0].forward
    inputs = _INPUT.clone().requires_grad_()
    balancing_bias = _BALANCING_BIAS.clone().requires_grad_() if with_balancing_bias else None
    actual = router(
        inputs,
        balancing_bias,
        **({"input_ids_T": _TOKEN_IDS} if getattr(router, "hash", False) else {"image_mask": _IMAGE_MASK}),
    )
    reference_inputs = inputs.detach().clone().requires_grad_()
    reference_weight = router.gate.weight.detach().clone().requires_grad_()
    expected = _reference_forward(reference_inputs, reference_weight, settings, balancing_bias)

    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    top_k = settings.get("top_k", 2)
    assert actual[0].shape == actual[1].shape == (4, top_k)
    assert actual[0].dtype == torch.float32
    assert actual[1].dtype == torch.int64
    assert actual[2].dtype == torch.bool
    assert actual[2].shape == (4, 8)
    torch.testing.assert_close(actual[2].sum(-1), torch.full((4,), top_k))
    torch.testing.assert_close(router.tokens_per_expert_E, expected[2].sum(0).float())
    if settings.get("num_expert_groups") and not settings.get("hash"):
        # For row 0 the strongest single expert is in group 0, but the sum
        # of the best two experts correctly chooses group 1.
        assert actual[1][0].tolist() == [4, 5]
    if settings.get("hash") and not settings.get("_debug_force_load_balance"):
        assert actual[1][0].tolist() == actual[1][2].tolist() == [5, 0]

    selected_upstream = torch.tensor(
        [[0.6, -0.4, 1.2, -0.7], [-1.2, 0.3, 0.8, 0.5], [0.8, 1.5, -0.9, 0.2], [-0.7, 0.9, 0.4, -1.1]]
    )[:, :top_k]
    actual_loss = (actual[0] * selected_upstream).sum()
    expected_loss = (expected[0] * selected_upstream).sum()
    choice_parameters = tuple(
        value for value in (balancing_bias, getattr(router, "bias_vl", None)) if value is not None
    )
    gradients = torch.autograd.grad(actual_loss, (inputs, router.gate.weight, *choice_parameters), allow_unused=True)
    expected_gradients = torch.autograd.grad(expected_loss, (reference_inputs, reference_weight))
    for observed, reference in zip(gradients[:2], expected_gradients, strict=True):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert all(value is None for value in gradients[2:])
