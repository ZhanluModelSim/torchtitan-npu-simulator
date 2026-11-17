# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def test_debugmodel_uses_the_documented_per_layer_compression_ratios():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_spec = model_registry("debugmodel")

    assert model_spec.model.compress_ratios == (1, 1, 4, 128)
    assert len(model_spec.model.layers) == len(model_spec.model.compress_ratios)


def test_flash_mtp_4k_model_flops():
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_config = model_registry(
        "deepseek_v4_flash",
        num_mtp_layers=1,
    ).model

    with torch.device("meta"):
        model = model_config.build()

    assert model_config.get_nparams_and_flops(model, seq_len=4096) == (
        290_942_278_866,
        92_762_352_876,
    )


def test_debugmodel_uses_cann_profiler_extension():
    from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
    from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel

    config = deepseek_v4_debugmodel()

    assert isinstance(config.profiler, CANNProfiler.Config)
    assert isinstance(config.profiler.build(), CANNProfiler)


def test_every_layer_pins_the_core_config_the_overrides_target():
    """The sparse core's config type is the anchor the fused overrides replace.

    ``asc``/``pypto``/``golden`` are registered against exactly this type, so a model that
    moved to another Config class would quietly run the reference core instead of the fused
    one.  The core's implementation may change; this anchor may not.
    """
    from torchtitan_npu.models.deepseek_v4 import model_registry
    from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention

    model_config = model_registry("debugmodel").model
    cores = [layer.attention.compressed_sparse_attention.inner_attention for layer in model_config.layers]

    assert len(cores) == len(model_config.compress_ratios) == 4
    for core in cores:
        assert type(core) is CompressedSparseInnerAttention.Config
        assert isinstance(core, CompressedSparseInnerAttention.Config)


def test_flash_rope_configs_pin_split_per_site():
    """Every rope site's config carries the split matching its tensor width."""
    from torchtitan_npu.models.deepseek_v4 import model_registry

    model_config = model_registry("deepseek_v4_flash", num_mtp_layers=0).model
    rd = 64

    for layer in model_config.layers:
        attn = layer.attention
        # Attention q/kv/o rotate the tail of a head_dim-wide (512) tensor.
        assert attn.rope.dim == rd
        assert attn.rope.split == attn.head_dim - rd
        if attn.compress_ratio > 1:
            assert attn.compressor.rope.split == attn.compressor.head_dim - rd
        if attn.compress_ratio == 4:
            # Indexer q is index_head_dim-wide (128): a different split even
            # though it reuses the same base rope flavor.
            indexer = attn.indexer
            assert indexer.rope.dim == rd
            assert indexer.rope.split == indexer.index_head_dim - rd
            return
    raise AssertionError("flash spec must contain compress_ratio == 4 layers")
