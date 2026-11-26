# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtitan.config import ConfigLoader
from torchtitan.config.override import OverrideConfig, apply_overrides

from torchtitan_npu.models.common.rope import HalfRotation
from torchtitan_npu.models.deepseek_v4_1.attention import CompressedSparseInnerAttention2
from torchtitan_npu.models.deepseek_v4_1.mhc import HcPre
from torchtitan_npu.override.common.rope import (
    AscComplexRoPE,
    AscHalfRotation,
    AscPartialComplexRoPE,
    WorkaroundComplexRoPE,
)
from torchtitan_npu.override.common.swiglu_group import AscSwiGLU
from torchtitan_npu.override.common.token_dispatcher import AscDeepEPTokenDispatcher
from torchtitan_npu.override.deepseek_v4_1.lightning_indexer.ascendc import AscInnerIndexer
from torchtitan_npu.override.deepseek_v4_1.mhc import AscV41HcPre
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41SparseAttention


@pytest.fixture
def recipe_config():
    """Load the selected recipe without entering torchrun or reserving NPUs."""

    def parse(name="deepseek_v4_1_flash_40layers_16experts_text"):
        args = [
            "--module",
            "torchtitan_npu_recipes.tests.models.deepseek_v4_1",
            "--config",
            name,
        ]
        return ConfigLoader().load(args)

    return parse


def test_recipe_selects_shared_rope_for_text_compressor_and_indexer(recipe_config):
    trainer = recipe_config()
    overrides = trainer.override
    cfg = trainer.model
    apply_overrides(overrides, cfg)
    assert type(cfg.layers[0].attention.rope) is AscComplexRoPE.Config
    assert type(cfg.layers[2].attention.compressor.rope) is AscComplexRoPE.Config
    assert type(cfg.layers[2].attention.indexer.rope) is AscComplexRoPE.Config


def test_a3_text_rope_override_preserves_the_split_site_contract(recipe_config):
    """Keep every field the model's builder set on the public ComplexRoPE.Config."""
    trainer = recipe_config()
    cfg = trainer.model
    attention_before = cfg.layers[0].attention.rope
    compress_before = cfg.layers[2].attention.compressor.rope
    apply_overrides(trainer.override, cfg)
    for before, after in (
        (attention_before, cfg.layers[0].attention.rope),
        (compress_before, cfg.layers[2].attention.compressor.rope),
    ):
        assert type(after) is AscComplexRoPE.Config
        assert after.split == before.split
        assert after.dim == before.dim
        assert after.theta == before.theta
        assert after.max_context_length == before.max_context_length
        assert after.scaling == before.scaling


def test_rope_entries_are_mutually_exclusive(recipe_config):
    """Reference and fused text-rope entries must not be listed together."""
    trainer = recipe_config()
    cfg = trainer.model
    both = OverrideConfig(
        imports=[
            "torchtitan_npu.override.common.rope.workaround",
            "torchtitan_npu.override.common.rope.asc_complex",
        ]
    )
    with pytest.raises(ValueError, match="both claim"):
        apply_overrides(both, cfg)


def test_a5_recipe_selects_the_full_fused_stack(recipe_config, monkeypatch):
    """Apply the shipped A5 eager overrides without loading CANN SwiGLU kernels."""
    monkeypatch.setattr("torchtitan_npu.override.common.swiglu_group._ensure_cann_ops_loaded", lambda: None)
    trainer = recipe_config("deepseek_v4_1_flash_40layers_24experts_multimodal_block_fp8")
    assert trainer.extension.quantization.enable_quantized_training
    assert trainer.extension.quantization.recipe == "all_block_fp8"
    assert trainer.training.num_tokens_per_train_step == 64 * trainer.training.max_context_length

    model = trainer.model
    apply_overrides(trainer.override, trainer)
    layer = model.layers[0]
    attention = model.layers[2].attention
    for rope in (layer.attention.rope, attention.compressor.rope, attention.indexer.rope):
        assert type(rope) is AscPartialComplexRoPE.Config
    assert type(model.vision_encoder.rotary) is AscHalfRotation.Config
    assert type(layer.feed_forward.routed_experts.activation_fn) is AscSwiGLU.Config
    assert type(layer.feed_forward.shared_experts.activation_fn) is AscSwiGLU.Config
    assert type(layer.feed_forward.routed_experts.token_dispatcher) is AscDeepEPTokenDispatcher.Config
    assert type(attention.inner_attention) is AscV41SparseAttention.Config
    assert attention.inner_attention.compress_ratio == attention.compress_ratio
    assert type(attention.indexer.inner_indexer) is AscInnerIndexer.Config
    assert type(layer.hc_attn_pre) is AscV41HcPre.Config


def test_reference_recipe_keeps_the_unfused_multimodal_stack(recipe_config):
    trainer = recipe_config("deepseek_v4_1_debugmodel_multimodal")
    model = trainer.model
    apply_overrides(trainer.override, trainer)
    layer = model.layers[0]
    attention = model.layers[2].attention
    for rope in (layer.attention.rope, attention.compressor.rope, attention.indexer.rope):
        assert type(rope) is WorkaroundComplexRoPE.Config
    assert type(model.vision_encoder.rotary) is HalfRotation.Config
    assert type(layer.feed_forward.routed_experts.activation_fn) is not AscSwiGLU.Config
    assert type(layer.feed_forward.shared_experts.activation_fn) is not AscSwiGLU.Config
    assert type(attention.inner_attention) is CompressedSparseInnerAttention2.Config
    assert type(layer.hc_attn_pre) is HcPre.Config
