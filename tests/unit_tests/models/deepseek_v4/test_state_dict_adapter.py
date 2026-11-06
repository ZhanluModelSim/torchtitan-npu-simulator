# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from tests.unit_tests.models.deepseek_v4.lora_test_utils import _build_model_config
from tests.unit_tests.models.test_deepseek_v4_mtp_model import _build_model_config as _compressed_config
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter


@pytest.mark.parametrize("prefix", ["layers.0", "mtp.0"], ids=["main", "mtp"])
def test_sink_preserves_bare_parameter_shape_and_values(prefix):
    config = _build_model_config(num_mtp_layers=1)
    adapter = DeepSeekV4StateDictAdapter(config, None)
    hf_key = f"{prefix}.attn.attn_sink"
    local_key = f"{prefix.replace('mtp.', 'mtp_layers.')}.attention.attn_sink"
    sink = torch.arange(4, dtype=torch.float32)
    restored = adapter.from_hf({hf_key: sink})
    assert restored.keys() == {local_key}
    torch.testing.assert_close(restored[local_key], sink, rtol=0, atol=0)
    exported = adapter.to_hf({local_key: sink})
    assert exported.keys() == {hf_key}
    torch.testing.assert_close(exported[hf_key], sink, rtol=0, atol=0)


def test_full_model_checkpoint_roundtrip_preserves_fused_experts_and_compressors():
    config = _compressed_config(compress_ratios=(1, 4, 128), rope_head_dim=8, original_seq_len=65536)
    with torch.device("meta"):
        model = config.build()
    state = {
        key: torch.arange(value.numel(), dtype=torch.float32).reshape(value.shape).to(value.dtype)
        for key, value in model.state_dict().items()
    }
    adapter = DeepSeekV4StateDictAdapter(config, None)
    exported = adapter.to_hf(state)
    for prefix in ("layers.0", "mtp.0"):
        local_prefix = prefix.replace("mtp.", "mtp_layers.")
        for expert in range(2):
            for index, projection in enumerate(("w1", "w3")):
                torch.testing.assert_close(
                    exported[f"{prefix}.ffn.experts.{expert}.{projection}.weight"],
                    state[f"{local_prefix}.moe.routed_experts.w13.weight"][expert, index],
                    rtol=0,
                    atol=0,
                )
    for layer in (1, 2):
        torch.testing.assert_close(
            exported[f"layers.{layer}.attn.compressor.wkv.weight"],
            state[f"layers.{layer}.attention.compressor.wkv.weight"],
            rtol=0,
            atol=0,
        )
    restored = adapter.from_hf(exported)
    assert restored.keys() == state.keys()
    for key in state:
        torch.testing.assert_close(restored[key], state[key], rtol=0, atol=0)
