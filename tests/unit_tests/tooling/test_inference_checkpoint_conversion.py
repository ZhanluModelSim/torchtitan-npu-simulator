# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU checks for inference-reference checkpoint preparation."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.checkpoint_conversion.convert_dsv4_hf_to_dcp import _read_hf_state_dict
from scripts.checkpoint_conversion.dequant_save_hf_dsv4 import dequantize_checkpoint

pytestmark = pytest.mark.tooling


def test_dequantize_checkpoint_fp4_preserves_hf_names_and_values(tmp_path):
    source = tmp_path / "quantized"
    source.mkdir()
    # Low/high nibbles encode +1/-1 and +2/-2, with row scales 2 and 4.
    save_file(
        {
            "layers.0.ffn.experts.0.w1.weight": torch.tensor([[-94], [-60]], dtype=torch.int8),
            "layers.0.ffn.experts.0.w1.scale": torch.tensor([[2.0], [4.0]]),
            "embed.weight": torch.tensor([[1.25, -0.5]], dtype=torch.bfloat16),
        },
        source / "model-00001-of-00001.safetensors",
    )
    (source / "config.json").write_text(
        json.dumps({"model_type": "deepseek_v4", "quantization_config": {"quant_method": "fp4"}})
    )
    output = tmp_path / "bf16"

    dequantize_checkpoint(source, output)

    restored = load_file(output / "model-00001-of-00001.safetensors")
    torch.testing.assert_close(
        restored["layers.0.ffn.experts.0.w1.weight"],
        torch.tensor([[2.0, -2.0], [8.0, -8.0]], dtype=torch.bfloat16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        restored["embed.weight"], torch.tensor([[1.25, -0.5]], dtype=torch.bfloat16), rtol=0, atol=0
    )
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(restored)
    assert "quantization_config" not in json.loads((output / "config.json").read_text())


def test_dequantize_checkpoint_fp8_scales_partial_blocks(tmp_path):
    source = tmp_path / "quantized"
    source.mkdir()
    weight = torch.ones((129, 129), dtype=torch.float32).to(torch.float8_e4m3fn)
    save_file(
        {"layers.0.attn.wq_a.weight": weight, "layers.0.attn.wq_a.scale": torch.tensor([[0.5, 2.0], [4.0, 8.0]])},
        source / "model-00001-of-00001.safetensors",
    )
    output = tmp_path / "bf16"
    expected = torch.full((129, 129), 0.5, dtype=torch.bfloat16)
    expected[:128, 128] = 2.0
    expected[128, :128] = 4.0
    expected[128, 128] = 8.0

    dequantize_checkpoint(source, output)

    restored = load_file(output / "model-00001-of-00001.safetensors")
    torch.testing.assert_close(restored["layers.0.attn.wq_a.weight"], expected, rtol=0, atol=0)


def test_read_hf_state_dict_reduced_view_preserves_selected_weights(tmp_path):
    kept = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
    router = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    save_file(
        {
            "layers.0.ffn.experts.0.w1.weight": kept,
            "layers.0.ffn.experts.4.w1.weight": kept + 1,
            "layers.1.ffn.experts.0.w1.weight": kept + 2,
            "layers.0.ffn.gate.weight": router,
            "mtp.0.ffn.experts.0.w1.weight": kept + 3,
        },
        tmp_path / "model.safetensors",
    )

    restored = _read_hf_state_dict(tmp_path, 1, num_experts=4, top_k=3)

    assert set(restored) == {"layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.gate.weight"}
    torch.testing.assert_close(restored["layers.0.ffn.experts.0.w1.weight"], kept, rtol=0, atol=0)
    torch.testing.assert_close(restored["layers.0.ffn.gate.weight"], router[:4], rtol=0, atol=0)
