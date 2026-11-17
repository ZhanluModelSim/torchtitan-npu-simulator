# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU coverage for the V4.1 Muon profile.

Real model parameters determine pattern/layout coverage; placement checks
protect per-head and expert sharding. NPU execution belongs to the smoke suite.
"""

import re

import pytest
from torch.distributed.tensor import Shard
from torchtitan.distributed.flex_shard import BlockShard, Owned
from torchtitan.distributed.parallel_dims import MeshAxisName

from torchtitan_npu.models.deepseek_v4_1 import model_registry
from torchtitan_npu.models.deepseek_v4_1.config_registry import (
    _v41_muon_profile,
    deepseek_v4_1_debugmodel_multimodal,
    deepseek_v4_1_flash_40layers_16experts_multimodal,
)


@pytest.fixture(scope="module")
def spec():
    return model_registry("deepseek_v4_1_flash_40layers_16experts_vision")


@pytest.fixture(scope="module")
def profile(spec):
    return _v41_muon_profile(spec)


@pytest.fixture(scope="module")
def layouts(profile):
    return profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]


class TestDefaultRecipe:
    def test_default_is_adamw(self):
        cfg = deepseek_v4_1_flash_40layers_16experts_multimodal()
        assert cfg.optimizer.name == "AdamW"
        assert [group.optimizer_name for group in cfg.optimizer.param_groups] == ["SparseAdam", "AdamW"]


class TestMuonPlacement:
    def test_wq_b_block_shard_all_dense_axes(self, layouts):
        layout = layouts["layers.0.attention.wq_b.weight"]
        dp = MeshAxisName.DP_SHARD.value
        dp_cp = f"{dp}_{MeshAxisName.CP.value}"
        fsdp = MeshAxisName.FSDP.value
        axes = set(layout.shardings_by_mesh_axis)
        assert {dp, dp_cp, fsdp} <= axes
        for axis in (dp, dp_cp, fsdp):
            p = layout.shardings_by_mesh_axis[axis]
            assert isinstance(p, BlockShard), f"wq_b {axis}: {type(p).__name__}"
            assert p.dim == 0
            assert p.block_size == 512

    def test_wo_a_block_shard(self, layouts, spec):
        layout = layouts["layers.0.attention.wo_a.weight"]
        bs = spec.model.layers[0].attention.wo_a.out_features
        dp = MeshAxisName.DP_SHARD.value
        dp_cp = f"{dp}_{MeshAxisName.CP.value}"
        assert set(layout.shardings_by_mesh_axis) == {dp, dp_cp, MeshAxisName.FSDP.value}
        for axis in (dp, dp_cp):
            p = layout.shardings_by_mesh_axis[axis]
            assert isinstance(p, BlockShard), f"wo_a {axis}: {type(p).__name__}"
            assert p.dim == 0
            assert p.block_size == bs

    def test_dense_layouts_cover_the_partial_dtensor_fsdp_axis(self, layouts):
        """``partial_dtensor`` stores dense parameters on the ``fsdp`` mesh
        (see parallelize.py); every dense layout must declare that axis or
        DistMuon rejects the parameter with ``no axis in storage mesh``."""
        for fqn in (
            "layers.0.attention.wq_a.weight",
            "layers.0.attention.wq_b.weight",
            "layers.0.attention.wo_a.weight",
            "layers.0.moe.router.gate.weight",
            "layers.0.hc_attn_pre.hc_fn",
        ):
            assert MeshAxisName.FSDP.value in layouts[fqn].shardings_by_mesh_axis, fqn

    def test_routed_shard_all_axes(self, layouts):
        layout = layouts["layers.0.moe.routed_experts.inner_experts.w1_EFD"]
        dp = MeshAxisName.DP_SHARD.value
        dp_cp = f"{dp}_{MeshAxisName.CP.value}"
        # the expert layout inherits the dense axes (including fsdp) and
        # adds the expert-parallel storage axes
        expected = {dp, dp_cp, MeshAxisName.FSDP.value, MeshAxisName.EFSDP.value, MeshAxisName.EP.value}
        assert set(layout.shardings_by_mesh_axis) == expected
        for axis, p in layout.shardings_by_mesh_axis.items():
            assert isinstance(p, Shard), f"routed {axis}: {type(p).__name__}"
            assert p.dim == 0

    def test_owned_dense_projections(self, layouts):
        for fqn in (
            "layers.0.attention.wq_a.weight",
            "layers.0.attention.wkv.weight",
            "layers.1.engram.gate.wkv",
            "layers.14.engram.gate.wkv",
            "layers.0.moe.router.gate.weight",
            "layers.0.hc_attn_pre.hc_fn",
        ):
            layout = layouts[fqn]
            assert MeshAxisName.DP_SHARD.value in layout.shardings_by_mesh_axis
            for axis, p in layout.shardings_by_mesh_axis.items():
                assert isinstance(p, Owned), f"{fqn} {axis}: {type(p).__name__}"


class TestRealModelReconciliation:
    """The pattern and layouts cover exactly the real debug model's weights."""

    def test_pattern_layouts_and_buckets_partition_the_debugmodel(self):
        from tests.unit_tests.models.mtp_test_utils import build_cpu_model

        trainer = deepseek_v4_1_debugmodel_multimodal()
        model = build_cpu_model(trainer.model_spec.model)
        fqns = [name for name, _ in model.named_parameters()]

        profile = trainer.optimizer._muon_profile
        assert profile is not None
        pattern = re.compile(profile.muon_pattern)
        distmuon = profile.optimizer_factory_kwargs["DistMuon"]
        layout_fqns = set(distmuon["compute_sharding_by_fqn"])
        bucket_entries = [p for b in distmuon["bucket_configs"] for p in b.patterns]
        bucket_fqns = set(bucket_entries)
        assert len(bucket_entries) == len(bucket_fqns)
        assert layout_fqns == bucket_fqns

        hits = {fqn for fqn in fqns if pattern.search(fqn)}
        assert hits == layout_fqns, (
            f"missing from layouts: {sorted(hits - layout_fqns)[:3]}; "
            f"declared but unmatched: {sorted(layout_fqns - hits)[:3]}"
        )
        # The AdamW fallback covers exactly the rest: the norm weights
        # (attention/ffn/final and the vision tower), the image marker
        # embeddings, the embeddings, the head, the sink, the indexer weights
        # (LI-trained but kept on the fallback by policy), and every 1-D
        # parameter (hc_base/hc_scale, the router's discrete vision-language
        # bias).
        rest = {fqn for fqn in fqns if fqn not in hits}
        unexpected = {
            fqn
            for fqn in rest
            if not (
                fqn.startswith(("vision_encoder", "tok_embeddings", "lm_head", "norm.", "image_marker_embeddings"))
                or fqn.endswith((".attn_sink", "_norm.weight", ".norm.weight", ".hc_base", ".hc_scale", ".bias_vl"))
                or ".indexer." in fqn
                or fqn.endswith((".engram.table.weight", ".engram.gate.q_weight", ".engram.gate.k_weight"))
            )
        }
        assert not unexpected, sorted(unexpected)[:5]

    def test_materialize_swaps_param_groups_to_distmuon_plus_adamw(self):
        trainer = deepseek_v4_1_debugmodel_multimodal()
        trainer.optimizer.name = "Muon"
        trainer.optimizer.lr = 1.23e-4
        trainer.optimizer.materialize()
        groups = trainer.optimizer.param_groups
        assert [group.optimizer_name for group in groups] == ["SparseAdam", "DistMuon", "AdamW"]
        sparse, distmuon, adamw = groups
        assert sparse.optimizer_kwargs["lr"] == 5e-5
        trainer.optimizer.materialize()
        assert trainer.optimizer.param_groups[0] is sparse
        assert len(trainer.optimizer.param_groups) == 3
        assert distmuon.pattern == trainer.optimizer._muon_profile.muon_pattern
        assert adamw.pattern == r".*"
        # Top-level CLI hyperparams land in both factories (param-groups.0
        # overrides from a launcher are dead after materialize).
        assert distmuon.optimizer_kwargs["lr"] == 1.23e-4
        assert adamw.optimizer_kwargs["lr"] == 1.23e-4

        # Check first-match assignment, including 2-D normalization scales.
        for layer_id in (1, 14):
            prefix = f"layers.{layer_id}.engram"
            for suffix, expected in (
                ("table.weight", "SparseAdam"),
                ("gate.wkv", "DistMuon"),
                ("gate.q_weight", "AdamW"),
                ("gate.k_weight", "AdamW"),
            ):
                selected = next(group for group in groups if re.search(group.pattern, f"{prefix}.{suffix}"))
                assert selected.optimizer_name == expected
