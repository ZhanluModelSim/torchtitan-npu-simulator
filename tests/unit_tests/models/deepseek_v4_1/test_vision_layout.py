# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Folded multimodal consumer: image placement, gradients and dummy-image path."""

from copy import deepcopy

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from torchtitan_npu.models.deepseek_v4_1 import build_model_config
from torchtitan_npu.models.deepseek_v4_1.vision.data import TEXT, build_image_token_layout


def _multimodal_model():
    config = build_model_config("deepseek_v4_1_debugmodel").model
    config.dim = config.tok_embeddings.embedding_dim = config.lm_head.in_features = 8
    config.vocab_size = config.tok_embeddings.num_embeddings = config.lm_head.out_features = 64
    config.norm.normalized_shape = 8
    config.n_layers = 0
    config.layers = []
    config.compress_ratios = ()
    config.vision_encoder.dim = config.vision_encoder.inter_dim = 8
    config.vision_encoder.num_layers = 1
    config.vision_encoder.num_heads = 2
    config.vision_encoder.text_dim = 8
    config.image_marker_embeddings.dim = 8
    with torch.random.fork_rng(devices=[]):
        model = config.build()
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model


@pytest.mark.parametrize("grids", ((), ((3, 3),), ((3, 6), (6, 3))), ids=("text-dummy", "single-image", "two-grids"))
def test_folded_modality_consumer_matches_batched_reference_and_updates(grids):
    model = _multimodal_model()
    reference = deepcopy(model)
    image_ids, image_types, feature_ids = build_image_token_layout(grids, span_start=0, vocab_size=64)
    tokens = torch.cat((torch.tensor([1, 2]), image_ids, torch.tensor([3, 4])))
    types = torch.cat((torch.full((2,), TEXT), image_types, torch.full((2,), TEXT)))
    indices = torch.cat((torch.full((2,), -1), feature_ids, torch.full((2,), -1)))
    if len(grids) > 1:
        boundary = 2 + image_types.tolist().index(3) + 1
        tokens = torch.cat((tokens[:boundary], torch.tensor([5, 6]), tokens[boundary:]))
        types = torch.cat((types[:boundary], torch.full((2,), TEXT), types[boundary:]))
        indices = torch.cat((indices[:boundary], torch.full((2,), -1), indices[boundary:]))
    generator = torch.Generator().manual_seed(17)
    image_grid = torch.tensor(grids or ((1, 1),))
    patches = pad_sequence(
        [torch.randn(int(h * w), 588, generator=generator) for h, w in image_grid.tolist()], batch_first=True
    )
    actual, image_mask = model._modality_inputs(
        tokens, pixel_values=patches, image_grid=image_grid, image_feature_indices=indices, token_types=types
    )
    hidden = reference.tok_embeddings(tokens.unsqueeze(0))
    visual = reference.vision_encoder(patches, image_grid)
    expected = hidden.clone()
    for kind, parameter in (
        (0, reference.image_marker_embeddings.image_start),
        (2, reference.image_marker_embeddings.image_newline),
        (3, reference.image_marker_embeddings.image_end),
    ):
        expected[types.unsqueeze(0) == kind] = parameter
    features = torch.cat(
        [row[: ((h + 2) // 3) * ((w + 2) // 3)] for row, (h, w) in zip(visual, image_grid.tolist(), strict=True)]
    )
    if grids:
        for token, feature in enumerate(indices.tolist()):
            if feature >= 0:
                expected[0, token] = features[feature]
    else:
        expected = expected + features.sum() * 0
    torch.testing.assert_close(actual, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(image_mask, types >= 0)
    upstream = torch.randn(actual.shape, generator=generator)
    (actual * upstream).sum().backward()
    (expected * upstream.unsqueeze(0)).sum().backward()
    for (name, parameter), (reference_name, expected_parameter) in zip(
        model.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert name == reference_name
        assert (parameter.grad is None) == (expected_parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0, msg=name)
            if not grids and name.startswith("vision_encoder."):
                assert torch.count_nonzero(parameter.grad) == 0, name
    # CPU-only step: torch_npu replaces torch.optim's foreach-device probe with
    # one that queries the current NPU, so the default ``foreach=None`` would
    # initialize a driver that a CPU-only host does not have.  The single-tensor
    # and multi-tensor SGD form the same update; this test compares the update.
    torch.optim.SGD(model.parameters(), lr=0.01, foreach=False).step()
    torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False).step()
    for actual_parameter, expected_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_parameter, expected_parameter, rtol=0, atol=0)
