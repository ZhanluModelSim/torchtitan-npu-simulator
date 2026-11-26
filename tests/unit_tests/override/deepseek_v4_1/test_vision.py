# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import torch_npu
from torchtitan.config import OverrideConfig, apply_overrides
from torchtitan.config.loader import ConfigLoader

import torchtitan_npu  # noqa: F401


@pytest.mark.cpu
@pytest.mark.parametrize("a5", [False, True], ids=["a3", "a5"])
def test_recipe_vision_attention_matches_cpu_outputs_and_gradients(monkeypatch, a5):
    for name in (
        "WORLD_SIZE",
        "NGPU",
        "NNODES",
        "NODE_IPS",
        "CLI_OVERRIDES",
        "OPTIMIZER_OVERRIDES",
        "DATALOADER_OVERRIDES",
        "MHC_POST_OVERRIDE",
        "USE_GOLDEN",
        "USE_TILELANG",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_ASSETS_PATH", str(Path(__file__).resolve().parents[4] / "tests/assets/deepseek_v3"))
    name = f"graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_{'a5' if a5 else 'a3'}_multimodal"
    config = ConfigLoader().load(
        ["--module", "torchtitan_npu_recipes.tests.graph_trainer.deepseek_v4_1", "--config", name]
    )
    vision = config.model.vision_encoder
    vision.dim, vision.inter_dim, vision.num_layers, vision.num_heads, vision.text_dim = 8, 16, 1, 2, 8
    reference_config = deepcopy(vision)
    target = "torchtitan_npu.override.deepseek_v4_1.vision.asc_unmasked_sdpa"
    entries = [item for item in config.override.imports if (item[0] if isinstance(item, tuple) else item) == target]
    assert len(entries) == 1
    # Exercise the vision leaf from the real recipe without unrelated A5 kernels.
    apply_overrides(OverrideConfig(imports=entries), vision)
    calls = []

    def cpu_attention(q, k, v, head_num, input_layout, *, scale, keep_prob):
        assert input_layout == "BNSD"
        assert q.shape == k.shape == v.shape == (1, 2, 15, 4)
        assert head_num == 2 and scale == 0.5 and keep_prob == 1.0
        assert all(not tensor.is_contiguous() for tensor in (q, k, v))
        assert q.stride(-2) == k.stride(-2) == head_num * q.shape[-1]
        assert v.stride(-2) == 3 * head_num * v.shape[-1]
        calls.append(input_layout)
        return (F.scaled_dot_product_attention(q, k, v, scale=scale, dropout_p=0.0),)

    monkeypatch.setattr(torch_npu, "npu_fusion_attention", cpu_attention)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(42)
        actual_model = vision.build()
        actual_model.init_states(buffer_device=torch.device("cpu"))
        reference_model = reference_config.build()
        reference_model.init_states(buffer_device=torch.device("cpu"))
    reference_model.load_state_dict(actual_model.state_dict())
    generator = torch.Generator().manual_seed(17)
    patches = torch.randn(1, 15, 588, generator=generator).requires_grad_()
    reference_patches = patches.detach().clone().requires_grad_()
    grid = torch.tensor([[3, 5]])
    actual = actual_model(patches, grid)
    expected = reference_model(reference_patches, grid)
    assert calls == ["BNSD"]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn(actual.shape, generator=generator)
    actual_grads = torch.autograd.grad(actual, (patches, *actual_model.parameters()), upstream)
    expected_grads = torch.autograd.grad(expected, (reference_patches, *reference_model.parameters()), upstream)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)
