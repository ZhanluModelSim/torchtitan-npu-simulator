# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
from pathlib import Path

import pytest
from torchtitan.config.manager import ConfigManager
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
from torchtitan_npu.override.common.swiglu_group import AscFeedForward, AscGroupedExperts
from torchtitan_npu.override.deepseek_v4_1.mhc import AscV41HcPre
from torchtitan_npu.override.deepseek_v4_1.sparse_attn.ascendc import AscV41SparseAttention


@pytest.fixture
def launcher_config(tmp_path):
    """Capture launcher arguments without entering torchrun or reserving NPUs."""
    root = Path(__file__).resolve().parents[4]
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_train.sh").write_text('printf "%s\\0" --module "$MODULE" --config "$CONFIG" "$@"\n')

    def parse(hardware="a3", *, golden=False):
        env = os.environ | {
            "CONFIG": "deepseek_v4_1_debugmodel_multimodal",
            "MODULE": "torchtitan_npu.models.deepseek_v4_1",
            "USE_GOLDEN": str(int(golden)),
            "NGPU": "8",
            "CLI_OVERRIDES": "",
            "HF_ASSETS_PATH": str(root / "tests/assets/deepseek_v3"),
        }
        script = root / f"examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_{hardware}.sh"
        captured = subprocess.check_output(["bash", str(script)], cwd=tmp_path, env=env)
        args = captured.decode().rstrip("\0").split("\0")
        return ConfigManager().parse_args(args)

    return parse


@pytest.mark.parametrize(
    "golden,rope_type,vision_type",
    [
        (True, WorkaroundComplexRoPE, HalfRotation),
        (False, AscComplexRoPE, AscHalfRotation),
    ],
    ids=["reference", "a3"],
)
def test_launcher_selects_shared_rope_for_text_compressor_indexer_and_vision(
    launcher_config, golden, rope_type, vision_type
):
    trainer = launcher_config(golden=golden)
    overrides = trainer.override
    cfg = trainer.model_spec.model
    apply_overrides(overrides, cfg)
    assert type(cfg.layers[0].attention.rope) is rope_type.Config
    assert type(cfg.layers[2].attention.compressor.rope) is rope_type.Config
    assert type(cfg.layers[2].attention.indexer.rope) is rope_type.Config
    assert type(cfg.vision_encoder.rotary) is vision_type.Config


def test_a3_text_rope_override_preserves_the_split_site_contract(launcher_config):
    """The fused derive keeps every field the model's builder set on the
    public ComplexRoPE.Config (split prefix width, theta, YaRN, length)."""
    trainer = launcher_config()
    cfg = trainer.model_spec.model
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
        assert after.max_seq_len == before.max_seq_len
        assert after.scaling == before.scaling


def test_rope_entries_are_mutually_exclusive(launcher_config):
    """The reference and fused text-rope entries claim the same public
    ComplexRoPE.Config nodes, so listing two of them must fail loudly."""
    trainer = launcher_config()
    cfg = trainer.model_spec.model
    both = OverrideConfig(
        imports=[
            "torchtitan_npu.override.common.rope.workaround",
            "torchtitan_npu.override.common.rope.asc_complex",
        ]
    )
    with pytest.raises(ValueError, match="both claim"):
        apply_overrides(both, cfg)


def test_a5_launcher_selects_the_full_fused_stack(launcher_config):
    """The A5 launcher's override list actually replaces every fused target:
    partial text RoPE, SwiGLUGroup experts, sparse attention and Sinkhorn."""
    trainer = launcher_config("a5")
    cfg = trainer.model_spec.model
    apply_overrides(trainer.override, cfg)

    # Text rope: asc_partial replaces the A3 asc_complex choice; vision keeps
    # the half-rotation layout from the A3 base list.
    for rope_cfg in (
        cfg.layers[0].attention.rope,
        cfg.layers[2].attention.compressor.rope,
        cfg.layers[2].attention.indexer.rope,
    ):
        assert type(rope_cfg) is AscPartialComplexRoPE.Config
    assert type(cfg.vision_encoder.rotary) is AscHalfRotation.Config

    # SwiGLUGroup: routed grouped experts plus the FQN-limited shared experts.
    moe = cfg.layers[0].moe
    assert type(moe.routed_experts.inner_experts) is AscGroupedExperts.Config
    assert type(moe.shared_experts) is AscFeedForward.Config

    inner = cfg.layers[2].attention.inner_attention
    assert type(inner) is AscV41SparseAttention.Config
    assert issubclass(AscV41SparseAttention, CompressedSparseInnerAttention2)
    assert inner.compress_ratio == cfg.layers[2].attention.compress_ratio

    # field preservation through derive() is covered by test_mhc_adapters;
    # here the recipe must actually select the fused targets
    for layer in cfg.layers[:6]:
        assert type(layer.hc_attn_pre) is AscV41HcPre.Config
        assert issubclass(AscV41HcPre, HcPre)


def test_a5_golden_launches_no_fused_defaults(launcher_config):
    """USE_GOLDEN=1 through the A5 wrapper injects none of the fused entries."""
    trainer = launcher_config("a5", golden=True)
    imports = [entry if isinstance(entry, str) else entry[0] for entry in trainer.override.imports]
    assert [i for i in imports if ".asc" in i or "swiglu" in i] == []

    cfg = trainer.model_spec.model
    apply_overrides(trainer.override, cfg)
    assert type(cfg.layers[0].attention.rope) is WorkaroundComplexRoPE.Config
    assert type(cfg.layers[2].attention.compressor.rope) is WorkaroundComplexRoPE.Config
    assert type(cfg.layers[2].attention.indexer.rope) is WorkaroundComplexRoPE.Config
    assert type(cfg.vision_encoder.rotary) is HalfRotation.Config
    # The experts keep whatever reference config the builders produced; the
    # point here is that no SwiGLUGroup replacement was applied.
    assert type(cfg.layers[0].moe.shared_experts) is not AscFeedForward.Config
    assert type(cfg.layers[0].moe.routed_experts.inner_experts) is not AscGroupedExperts.Config
    assert type(cfg.layers[2].attention.inner_attention) is CompressedSparseInnerAttention2.Config
