# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
import torch
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.converters import get_model_converter_config
from torchtitan_npu.models.deepseek_v4 import model as deepseek_v4_model
from torchtitan_npu.models.deepseek_v4.config_registry import debug_deepseek_v4_single_node_1b


@pytest.mark.parametrize("with_compressor", [False, True])
def test_precompute_rope_cache_preserves_complex_bits(with_compressor):
    config = deepseek_v4_model.DeepSeekV4Model.Config(
        max_seq_len=128,
        rope_head_dim=8,
        compress_ratios=(1, 1, 4, 128),
    )
    expected = deepseek_v4_model.precompute_freqs_cis(config, with_compressor)

    complex_cache = deepseek_v4_model.precompute_rope_cache(config, with_compressor)

    assert complex_cache.shape == (128, 4)
    assert complex_cache.dtype == torch.complex64
    assert torch.equal(complex_cache, expected)

    config.use_npu_rope = True
    actual = deepseek_v4_model.precompute_rope_cache(config, with_compressor)

    packed_seq_len = 128 + 128 // 4 + 128 // 128 if with_compressor else 128
    assert actual.shape == (2, packed_seq_len, 8)
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    full_cache = actual.narrow(1, 0, config.max_seq_len)
    assert torch.equal(full_cache[0], expected.real.repeat_interleave(2, dim=-1))
    assert torch.equal(full_cache[1], expected.imag.repeat_interleave(2, dim=-1))


def test_packed_npu_rope_cache_supports_configured_compressor_ratios():
    config = deepseek_v4_model.DeepSeekV4Model.Config(
        max_seq_len=129,
        rope_head_dim=8,
        n_layers=4,
        compress_ratios=(1, 2, 4, 2),
        use_npu_rope=True,
    )

    cache = deepseek_v4_model.precompute_rope_cache(config, with_compressor=True)
    full_cache = cache.narrow(1, 0, config.max_seq_len)
    ratio2_len = (config.max_seq_len + 2 - 1) // 2
    ratio4_len = (config.max_seq_len + 4 - 1) // 4
    ratio2_cache = cache.narrow(1, config.max_seq_len, ratio2_len)
    ratio4_cache = cache.narrow(
        1,
        config.max_seq_len + ratio2_len,
        ratio4_len,
    )

    assert cache.shape == (2, 129 + 65 + 33, 8)
    assert torch.equal(ratio2_cache, full_cache[:, ::2])
    assert torch.equal(ratio4_cache, full_cache[:, ::4])


@pytest.mark.parametrize("with_compressor", [False, True])
def test_npu_rope_cache_packs_mtp_compression_ratio_for_both_theta_caches(with_compressor):
    config = deepseek_v4_model.DeepSeekV4Model.Config(
        max_seq_len=128,
        rope_head_dim=8,
        n_layers=2,
        compress_ratios=(1, 1),
        num_mtp_modules=1,
        mtp_layer_compress_ratio=4,
        use_npu_rope=True,
    )

    cache = deepseek_v4_model.precompute_rope_cache(config, with_compressor)
    full_cache = cache.narrow(1, 0, config.max_seq_len)
    mtp_cache = cache.narrow(1, config.max_seq_len, config.max_seq_len // 4)

    assert cache.shape == (2, 128 + 32, 8)
    assert torch.equal(mtp_cache, full_cache[:, ::4])


def test_mtp_compressor_uses_packed_non_compressor_theta_cache(monkeypatch):
    config = deepseek_v4_model.DeepSeekV4Model.Config(
        dim=8,
        max_seq_len=128,
        rope_head_dim=4,
        n_layers=2,
        compress_ratios=(1, 1),
        num_mtp_modules=1,
        mtp_layer_compress_ratio=4,
        use_npu_rope=True,
    )
    compressor = deepseek_v4_model.Compressor.Config(
        args=config,
        compress_ratio=config.mtp_layer_compress_ratio,
        head_dim=8,
    ).build()
    compressor.init_weights(0.02)
    cache = deepseek_v4_model.precompute_rope_cache(config, with_compressor=False)
    captured = {}

    def capture_rotary(x, freqs_cis, inverse=False, positions=None):
        captured["freqs_cis"] = freqs_cis
        captured["positions"] = positions
        return x

    monkeypatch.setattr(deepseek_v4_model, "apply_rotary_emb", capture_rotary)
    output = compressor(torch.randn(1, config.max_seq_len, config.dim), cache)

    segment = captured["freqs_cis"]
    full_cache = cache.narrow(1, 0, config.max_seq_len)
    assert output.shape == (1, config.max_seq_len // 4, 8)
    assert captured["positions"] is None
    assert torch.equal(segment, full_cache[:, ::4])


def test_compressor_requires_packed_real_cache(monkeypatch):
    config = deepseek_v4_model.DeepSeekV4Model.Config(
        dim=8,
        max_seq_len=128,
        rope_head_dim=4,
        compress_ratios=(1, 1, 4, 128),
        use_npu_rope=True,
    )
    compressor = deepseek_v4_model.Compressor.Config(
        args=config,
        compress_ratio=4,
        head_dim=8,
    ).build()
    compressor.init_weights(0.02)
    cache = torch.randn(2, config.max_seq_len, config.rope_head_dim)

    def passthrough(x, freqs_cis, inverse=False, positions=None):
        return x

    monkeypatch.setattr(deepseek_v4_model, "apply_rotary_emb", passthrough)
    with pytest.raises(AssertionError, match="packed compression segment"):
        compressor(torch.randn(1, config.max_seq_len, config.dim), cache)


def test_compressor_keeps_complex_cache_semantics(monkeypatch):
    config = deepseek_v4_model.DeepSeekV4Model.Config(
        dim=8,
        max_seq_len=16,
        rope_head_dim=4,
        compress_ratios=(1, 1, 4, 4),
    )
    compressor = deepseek_v4_model.Compressor.Config(
        args=config,
        compress_ratio=4,
        head_dim=8,
    ).build()
    compressor.init_weights(0.02)
    cache = deepseek_v4_model.precompute_rope_cache(config, with_compressor=True)
    captured = {}

    def capture_rotary(x, freqs_cis, inverse=False, positions=None):
        captured["freqs_cis"] = freqs_cis
        captured["positions"] = positions
        return x

    monkeypatch.setattr(deepseek_v4_model, "apply_rotary_emb", capture_rotary)
    compressor(torch.randn(1, config.max_seq_len, config.dim), cache)

    assert captured["positions"] is None
    assert torch.equal(captured["freqs_cis"], cache[::4])


def test_update_from_config_derives_use_npu_rope(monkeypatch):
    trainer_config = debug_deepseek_v4_single_node_1b()
    trainer_config = replace(
        trainer_config,
        model_converters=ModelConvertersContainer.Config(converters=[get_model_converter_config("npu_rope")]),
    )
    model_config = trainer_config.model_spec.model
    monkeypatch.setattr(deepseek_v4_model, "get_npu_device_type", lambda: "A3")

    model_config.update_from_config(trainer_config=trainer_config)

    assert model_config.use_npu_rope is True


def test_rope_cache_is_rebuilt_after_to_empty():
    config = deepseek_v4_model.DeepSeekV4Model.Config(
        vocab_size=8,
        max_seq_len=128,
        n_heads=4,
        dim=128,
        moe_inter_dim=64,
        head_dim=32,
        rope_head_dim=16,
        q_lora_rank=64,
        o_lora_rank=32,
        o_groups=4,
        window_size=32,
        hc_mult=2,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=16,
        use_npu_rope=True,
    )
    with torch.device("meta"):
        model = config.build()

    assert model.freqs_cis.is_meta
    assert model.freqs_cis_wo_compressor.is_meta

    expected_freqs_cis = deepseek_v4_model.precompute_rope_cache(config, True)
    expected_freqs_cis_wo_compressor = deepseek_v4_model.precompute_rope_cache(config, False)

    model.to_empty(device="cpu")
    model.init_weights(torch.device("cpu"))

    assert torch.equal(model.freqs_cis, expected_freqs_cis)
    assert torch.equal(model.freqs_cis_wo_compressor, expected_freqs_cis_wo_compressor)

    state_dict = model.state_dict()
    assert "freqs_cis" not in state_dict
    assert "freqs_cis_wo_compressor" not in state_dict
