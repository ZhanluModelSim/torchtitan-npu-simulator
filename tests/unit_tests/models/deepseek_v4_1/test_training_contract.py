# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib

import pytest
import torch
from torch.utils.checkpoint import DefaultDeviceType
from torchtitan.distributed.activation_checkpoint import FullAC

from tests.unit_tests.models.deepseek_v4_1.preprocess_utils import preprocess
from tests.unit_tests.models.mtp_test_utils import build_cpu_model
from torchtitan_npu.models.deepseek_v4_1.vision.data import build_image_token_layout


@pytest.mark.parametrize("with_engram", [False, True], ids=["baseline", "engram"])
def test_full_ac_preserves_image_routing(monkeypatch, with_engram):
    # CPU-only checkpoint inputs otherwise inherit the registered NPU backend.
    monkeypatch.setattr(DefaultDeviceType, "_default_device_type", "cpu")
    registry = importlib.import_module("torchtitan_npu.models.deepseek_v4_1")
    make_config = registry._make_v41_config

    def tiny_config(**kwargs):
        kwargs.update(
            dim=8,
            n_heads=2,
            head_dim=8,
            rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=4,
            n_groups=1,
            index_n_heads=2,
            index_head_dim=4,
            moe_inter_dim=16,
            vision_dim=8,
            vision_heads=2,
            vision_inter_dim=16,
        )
        return make_config(**kwargs)

    monkeypatch.setattr(registry, "_make_v41_config", tiny_config)
    config = registry.model_registry("deepseek_v4_1_debugmodel").model
    if not with_engram:
        for layer in config.layers:
            layer.engram = None
    for layer in config.layers:
        if layer.engram is not None:
            layer.engram.table.vocab_size = 64
    config.vocab_size = 64
    config.tok_embeddings.num_embeddings = 64
    config.lm_head.out_features = 64
    with torch.random.fork_rng(devices=[]):
        model = build_cpu_model(config)
    with torch.no_grad():
        for layer in model.layers.values():
            layer.moe.router.gate.weight.zero_()
            bias = torch.arange(layer.moe.router.num_experts, dtype=torch.float32)
            layer.moe.expert_bias_E.copy_(bias)
            layer.moe.router.bias_vl.copy_(bias.flip(0))
    checkpointed = copy.deepcopy(model)
    FullAC.Config().build().apply(checkpointed)

    # Real image protocol without depending on the synthetic RNG fallback.
    ids, types, feature_ids = build_image_token_layout([(3, 3)], span_start=2, vocab_size=64)
    tokens = torch.arange(128).remainder(32).unsqueeze(0)
    tokens[:, 2 : ids.numel()] = ids[2:]
    token_types = torch.full_like(tokens, -1)
    token_types[:, : types.numel()] = types
    indices = torch.full_like(tokens, -1)
    indices[:, : feature_ids.numel()] = feature_ids
    inputs = dict(
        positions=torch.arange(128).unsqueeze(0),
        pixel_values=torch.linspace(-1, 1, 9 * 588).reshape(1, 9, 588),
        image_grid=torch.tensor([[3, 3]]),
        image_feature_indices=indices,
        token_types=token_types,
    )
    # The modality tensors are dataloader keys the model's forward consumes, so they ride
    # through the seam exactly as the trainer hands them over.
    modality = {key: value for key, value in inputs.items() if key != "positions"}
    expected_mask = token_types >= 0
    route_calls = []

    def capture_route(module, args, kwargs, output):
        mask = kwargs.get("image_mask")
        route_calls.append((None if mask is None else mask.clone(), output[1].detach().clone()))

    engram_calls = []
    engram_handles = []
    if with_engram:

        def capture_engram(module, args, kwargs, output):
            torch.testing.assert_close(kwargs["image_mask"], expected_mask)
            torch.testing.assert_close(output[expected_mask], args[0][expected_mask], rtol=0, atol=0)
            engram_calls.append(module)

        for layer in checkpointed.layers.values():
            if layer.engram is not None:
                engram_handles.append(layer.engram.register_forward_hook(capture_engram, with_kwargs=True))
    handle = checkpointed.layers["0"].moe.router.register_forward_hook(capture_route, with_kwargs=True)
    try:
        _, _, kwargs = preprocess(model, tokens, tokens, inputs["positions"], **modality)
        expected = model(tokens, **kwargs)
        _, _, kwargs = preprocess(checkpointed, tokens, tokens, inputs["positions"], **modality)
        actual = checkpointed(tokens, **kwargs)
        assert route_calls[0][0] is not None
        torch.testing.assert_close(route_calls[0][0], expected_mask)
        selected = route_calls[0][1]
        assert selected[expected_mask][0].tolist() == [0, 1, 2, 3, 4, 5]
        assert selected[~expected_mask][0].tolist() == [15, 14, 13, 12, 11, 10]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().mean().backward()
        actual.square().mean().backward()
        for mask, _ in route_calls:
            torch.testing.assert_close(mask, expected_mask)
        for expected_parameter, actual_parameter in zip(model.parameters(), checkpointed.parameters(), strict=True):
            assert (expected_parameter.grad is None) == (actual_parameter.grad is None)
            if expected_parameter.grad is not None:
                torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=1e-5, atol=1e-7)
        if with_engram:
            assert engram_calls
            for index in ("1", "14"):
                table = model.layers[index].engram.table
                actual_table = checkpointed.layers[index].engram.table
                expected_grad = table.pending_sparse_grad().to_dense()
                actual_grad = actual_table.pending_sparse_grad().to_dense()
                assert torch.count_nonzero(expected_grad) > 0
                torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-7)
            for hook in engram_handles:
                hook.remove()
        route_calls.clear()
        # A batch with no modality keys at all: the same stack, and no ``image_mask``.
        _, _, kwargs = preprocess(checkpointed, tokens, tokens, inputs["positions"])
        checkpointed(tokens, **kwargs)
        assert route_calls[0][0] is None
    finally:
        handle.remove()
        for hook in engram_handles:
            hook.remove()
