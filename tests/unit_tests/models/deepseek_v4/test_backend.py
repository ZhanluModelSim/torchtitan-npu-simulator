# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4 import build_model_config
from torchtitan.models.deepseek_v4.attention import DeepSeekV4Attention
from torchtitan.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter as NativeStateDictAdapter

from torchtitan_npu.models.deepseek_v4.attention import NPUAttention
from torchtitan_npu.models.deepseek_v4.backend import (
    NPUBackendDeepSeekV4Model,
    to_npu_attention_config,
    with_npu_attention_backend,
)


def test_npu_backend_selection_preserves_native_source_and_adapts_main_and_mtp():
    native = build_model_config(
        "debugmodel", seq_len=128, n_mtp_layers=1, parallelism=ParallelismConfig(), indexer_loss_coeff=0
    )
    native_attention = native.layers[0].attention
    backend = with_npu_attention_backend(native)

    assert type(native_attention) is DeepSeekV4Attention.Config
    assert native.layers[0].attention is native_attention
    assert native_attention.compress_ratio == 0
    assert native_attention.rope.split == 0
    assert isinstance(backend, NPUBackendDeepSeekV4Model.Config)
    assert all(isinstance(layer.attention, NPUAttention.Config) for layer in (*backend.layers, *backend.mtp_layers))
    assert backend.layers[0].attention.compress_ratio == 1
    assert backend.layers[0].attention.rope.split == native_attention.head_dim - native_attention.rope_head_dim
    assert backend.layers[3].attention.compressor is not None
    assert backend.layers[3].attention.compressor.compress_ratio == 128
    assert backend.layers[2].attention.lightning_indexer is not None
    assert backend.layers[0].attention.lightning_indexer is None


def test_npu_backend_attention_keeps_projection_storage_and_selects_bare_sink():
    config = build_model_config("debugmodel", seq_len=128, parallelism=ParallelismConfig(), indexer_loss_coeff=0)
    native_config = config.layers[0].attention
    backend_config = to_npu_attention_config(native_config)
    with torch.device("meta"):
        native = native_config.build()
        backend = backend_config.build()

    assert native.wo_a.weight.shape == backend.wo_a.weight.shape
    assert native.attn_sink.weight.shape == (native_config.n_heads, 1)
    assert backend.attn_sink.shape == (native_config.n_heads,)
    assert "attn_sink.weight" in native.state_dict()
    assert "attn_sink" in backend.state_dict()


def test_npu_backend_selects_matching_hf_checkpoint_names_and_sink_rank():
    native = build_model_config("debugmodel", seq_len=128, parallelism=ParallelismConfig(), indexer_loss_coeff=0)
    backend = with_npu_attention_backend(native)
    sink = torch.arange(native.layers[0].attention.n_heads, dtype=torch.float32)
    ape = torch.arange(4, dtype=torch.float32)
    hf_state = {"layers.0.attn.attn_sink": sink, "layers.3.attn.compressor.ape": ape}
    native_state = NativeStateDictAdapter(native, None).from_hf(hf_state)
    backend_adapter = NPUBackendDeepSeekV4Model.state_dict_adapter_cls(backend, None)
    backend_state = backend_adapter.from_hf(hf_state)

    assert torch.equal(native_state["layers.0.attention.attn_sink.weight"], sink[:, None])
    assert torch.equal(backend_state["layers.0.attention.attn_sink"], sink)
    assert torch.equal(native_state["layers.3.attention.compressor_128.ape"], ape)
    assert torch.equal(backend_state["layers.3.attention.compressor.ape"], ape)
    round_trip = backend_adapter.to_hf(backend_state)
    assert set(round_trip) == set(hf_state)
    assert all(torch.equal(round_trip[name], value) for name, value in hf_state.items())


def test_npu_backend_rejects_untranslated_indexer_objective():
    config = build_model_config("debugmodel", seq_len=128, parallelism=ParallelismConfig())
    with pytest.raises(ValueError, match="indexer_loss_coeff=0"):
        to_npu_attention_config(config.layers[2].attention)
