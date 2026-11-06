# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.models.common.text_data_utils import IntegerTextTokenizer, aligned_text_config
from torchtitan_npu.models.deepseek_v4_1 import model_registry
from torchtitan_npu.models.deepseek_v4_1.config_registry import (
    deepseek_v4_1_debugmodel_multimodal,
    deepseek_v4_1_debugmodel_text,
)


def test_multimodal_preprocess_annotates_vision_inputs(monkeypatch):
    from torchtitan.distributed import spmd_types as spmd_module
    from torchtitan.models.common.decoder_sharding import token_id_placement
    from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding

    from torchtitan_npu.patches.torchtitan.models.common import decoder as decoder_patch

    calls = []

    def annotate(_dims, batch, layouts):
        calls.append((batch.copy(), layouts))
        return batch

    monkeypatch.setattr(decoder_patch, "annotate_input_spmd_types", annotate)
    monkeypatch.setattr(spmd_module, "annotate_input_spmd_types", annotate)
    recipe = deepseek_v4_1_debugmodel_multimodal()
    with torch.device("meta"):
        model = recipe.model.build()
    monkeypatch.setattr(model, "_get_attention_metadata", lambda *_args, **_kwargs: {})
    batch = {
        "input": torch.arange(4),
        "labels": torch.arange(4),
        "pixel_values": torch.ones(1, 2, 3),
        "image_grid": torch.tensor([[1, 2]]),
        "image_feature_indices": torch.arange(4),
        "token_types": torch.arange(4),
    }
    inputs, labels, extras = model.preprocess_inputs(
        batch,
        parallelism_context=SimpleNamespace(cp_enabled=False),
        parallelism=SimpleNamespace(spmd_backend="spmd_types"),
    )

    torch.testing.assert_close(inputs, batch["input"])
    torch.testing.assert_close(labels, batch["labels"])
    assert set(calls[0][0]) == {"input", "labels"}
    assert set(calls[1][0]) == set(extras) == {"pixel_values", "image_grid", "image_feature_indices", "token_types"}
    assert calls[1][1]["pixel_values"] == multimodal_input_sharding()["pixel_values"]
    assert calls[1][1]["token_types"] == token_id_placement()


def test_long_document_continuation_uses_context_boundaries(tmp_path):
    dataset = tmp_path / "data.json"
    dataset.write_text('{"text":"3"}\n{"text":"4 5 6 7 8"}\n')
    loader = aligned_text_config(dataset).build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=IntegerTextTokenizer(),
        max_context_length=4,
        num_tokens_per_microbatch=4,
    )
    try:
        iterator = iter(loader)
        next(iterator)
        inputs = next(iterator).as_input_dict()
    finally:
        loader.close()

    config = model_registry("deepseek_v4_1_debugmodel_text")
    attention = config.layers[2].attention.inner_attention
    metadata = attention.build_attention_metadata(inputs["positions"])

    torch.testing.assert_close(metadata.cu_seq_q, torch.tensor([0, 2, 4], dtype=torch.int32))
    torch.testing.assert_close(metadata.cu_seq_k, torch.tensor([0, 1, 2], dtype=torch.int32))
    torch.testing.assert_close(metadata.position_k, torch.tensor([0, 1, 0, 1]))


@pytest.mark.parametrize("token_budget", (4, 8), ids=("one-context", "two-contexts"))
def test_text_recipe_preprocess_builds_folded_metadata_map(tmp_path, token_budget, monkeypatch):
    from torchtitan_npu.patches.torchtitan.models.common import decoder as decoder_patch

    # SPMD annotation requires a process group; this case checks data/metadata,
    # while the decoder seam tests cover the named annotation contract.
    monkeypatch.setattr(decoder_patch, "annotate_input_spmd_types", lambda _context, batch, _shardings: batch)
    dataset = tmp_path / "data.json"
    dataset.write_text('{"text":"3 4"}\n{"text":"5 6"}\n{"text":"7"}\n')
    recipe = deepseek_v4_1_debugmodel_text()
    recipe.dataloader = replace(
        recipe.dataloader, dataset=aligned_text_config(dataset).dataset, repeat=False, shuffle=False
    )
    with torch.device("meta"):
        model = recipe.model.build()
    context = SimpleNamespace(cp_enabled=False)
    loader = recipe.dataloader.build(
        dp_world_size=1,
        dp_rank=0,
        tokenizer=IntegerTextTokenizer(),
        max_context_length=4,
        num_tokens_per_microbatch=token_budget,
    )
    try:
        batches = list(loader)
    finally:
        loader.close()

    # Two aligned four-token documents fill the rows; the final two-token
    # document is dropped at finite EOF, as in the release text loader.
    assert len(batches) == 8 // token_budget
    tokens = torch.tensor([1, 3, 4, 0, 1, 5, 6, 0])
    labels = torch.tensor([3, 4, 2, -100, 5, 6, 2, -100])
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    for index, microbatch in enumerate(batches):
        batch = microbatch.as_input_dict()
        inputs, actual_labels, kwargs = model.preprocess_inputs(
            batch, parallelism_context=context, parallelism=recipe.parallelism
        )
        row = slice(index * token_budget, (index + 1) * token_budget)
        torch.testing.assert_close(inputs, tokens[row], rtol=0, atol=0)
        torch.testing.assert_close(actual_labels, labels[row], rtol=0, atol=0)
        torch.testing.assert_close(kwargs["positions"], positions[row], rtol=0, atol=0)
        assert microbatch.num_valid_tokens == 3 * token_budget // 4
        assert set(kwargs) == {"positions", "padding_mask", "attention_metadata"}
        # The native mask marks padding, while num_valid_tokens counts its complement.
        torch.testing.assert_close(kwargs["padding_mask"], labels[row] == -100)
        backends = [layer.attention.inner_attention for layer in recipe.model.layers]
        assert set(kwargs["attention_metadata"]) == {backend.attention_metadata_key for backend in backends}
        for backend in backends:
            metadata = kwargs["attention_metadata"][backend.attention_metadata_key]
            bounds = torch.tensor([0, 4] if token_budget == 4 else [0, 4, 8], dtype=torch.int32)
            pooled = bounds // max(backend.compress_ratio, 1)
            torch.testing.assert_close(metadata.cu_seq_q, bounds, rtol=0, atol=0)
            torch.testing.assert_close(metadata.cu_seq_k, pooled, rtol=0, atol=0)
            torch.testing.assert_close(metadata.position_k, positions[row], rtol=0, atol=0)
