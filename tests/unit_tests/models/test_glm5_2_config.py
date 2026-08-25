# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

import dataclasses

import torch

from torchtitan_npu.models.glm5_2 import glm5_2_configs
from torchtitan_npu.models.glm5_2.config_overrides import GLM52ModelOverrides


def test_glm5_2_official_topology_and_indexshare_schedule():
    config = glm5_2_configs["78layers_1mtp"]()

    assert config.vocab_size == 154880
    assert config.dim == 6144
    assert len(config.layers) == 79
    assert config.num_mtp_modules == 1
    assert config.layers[0].attention.n_heads == 64
    assert config.layers[0].attention.index_n_heads == 32
    assert config.layers[0].attention.index_topk == 2048
    assert config.layers[0].attention.qk_nope_head_dim == 192
    assert config.layers[0].attention.v_head_dim == 256
    assert sum(layer.feed_forward is not None for layer in config.layers[:78]) == 3
    assert sum(layer.moe is not None for layer in config.layers[:78]) == 75

    main_schedule = config.indexer_types
    assert main_schedule[:7] == ["full", "full", "full", "shared", "shared", "shared", "full"]
    assert sum(item == "full" for item in main_schedule) == 21
    assert sum(layer.attention.skip_topk for layer in config.layers) == 58
    assert config.layers[-1].attention.skip_topk is True
    assert config.layers[0].attention.indexer_rope_interleave is True


def test_glm5_2_overrides_round_trip():
    original = glm5_2_configs["smoketest"]()
    overrides = GLM52ModelOverrides.from_model_config(original)
    rebuilt = overrides.to_model_config()

    assert rebuilt.dim == original.dim
    assert rebuilt.vocab_size == original.vocab_size
    assert rebuilt.num_mtp_modules == original.num_mtp_modules
    assert rebuilt.indexer_types == original.indexer_types
    assert [layer.attention.skip_topk for layer in rebuilt.layers] == [
        layer.attention.skip_topk for layer in original.layers
    ]
    assert dataclasses.asdict(overrides)["index_topk_freq"] == 4


def test_glm5_2_meta_forward_returns_main_and_mtp_logits():
    from torchtitan_npu.simulator.meta_env import patch_device_type_to_meta

    patch_device_type_to_meta()
    config = glm5_2_configs["smoketest"]()
    meta_context = torch.device("meta")
    meta_context.__enter__()
    try:
        model = config.build()
        tokens = torch.empty((1, 7), dtype=torch.long, device="meta")
        outputs = model(tokens)
    finally:
        meta_context.__exit__(None, None, None)

    assert isinstance(outputs, list)
    assert [tuple(output.shape) for output in outputs] == [(1, 6, 320), (1, 6, 320)]
    assert model.layers["3"].attention.pre_attention.indexer is None
    assert model.layers["2"].attention.pre_attention.indexer is not None
