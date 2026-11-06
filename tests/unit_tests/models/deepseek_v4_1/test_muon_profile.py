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
from torchtitan.components.optim import AdamW
from torchtitan.distributed.flex_shard import BlockShard, Owned
from torchtitan.distributed.parallelism_context import MeshAxisName

from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon
from torchtitan_npu.models.deepseek_v4_1 import model_registry
from torchtitan_npu.models.deepseek_v4_1.config_registry import (
    EngramTableOptimizerConfig,
    _v41_muon_profile,
    deepseek_v4_1_debugmodel_text,
    deepseek_v4_1_flash_40layers_24experts_text,
)


@pytest.fixture(scope="module")
def spec():
    return model_registry("deepseek_v4_1_flash_40layers_24experts_text")


@pytest.fixture(scope="module")
def profile(spec):
    return _v41_muon_profile(spec)


@pytest.fixture(scope="module")
def layouts(profile):
    return profile.optimizer_factory_kwargs["DistMuon"]["compute_sharding_by_fqn"]


class TestDefaultRecipe:
    def test_default_is_adamw(self):
        cfg = deepseek_v4_1_flash_40layers_24experts_text()
        assert cfg.optim.optimizer.name == "AdamW"
        assert [type(group).__name__ for group in cfg.optim.optimizer.optimizers] == [
            "EngramTableOptimizerConfig",
            "Config",
        ]


class TestMuonPlacement:
    def test_wq_b_block_shard_all_dense_axes(self, layouts):
        layout = layouts["layers.0.attention.wq_b.weight"]
        dp = MeshAxisName.DP_SHARD.value
        axes = set(layout.shardings_by_mesh_axis)
        assert axes == {dp}
        for axis in (dp,):
            p = layout.shardings_by_mesh_axis[axis]
            assert isinstance(p, BlockShard), f"wq_b {axis}: {type(p).__name__}"
            assert p.dim == 0
            assert p.block_sizes == (512,)

    def test_wo_a_block_shard(self, layouts, spec):
        layout = layouts["layers.0.attention.wo_a.weight"]
        bs = spec.layers[0].attention.wo_a.out_features
        dp = MeshAxisName.DP_SHARD.value
        assert set(layout.shardings_by_mesh_axis) == {dp}
        for axis in (dp,):
            p = layout.shardings_by_mesh_axis[axis]
            assert isinstance(p, BlockShard), f"wo_a {axis}: {type(p).__name__}"
            assert p.dim == 0
            assert p.block_sizes == (bs,)

    def test_dense_layouts_cover_the_parameter_storage_axis(self, layouts):
        """Dense compute layouts must admit the native parameter storage axis."""
        for fqn in (
            "layers.0.attention.wq_a.weight",
            "layers.0.attention.wq_b.weight",
            "layers.0.attention.wo_a.weight",
            "layers.0.moe.router.gate.weight",
            "layers.0.hc_attn_pre.hc_fn",
        ):
            assert MeshAxisName.DP_SHARD.value in layouts[fqn].shardings_by_mesh_axis, fqn

    def test_routed_shard_all_axes(self, layouts):
        layout = layouts["layers.0.moe.routed_experts.w13.weight"]
        dp = MeshAxisName.DP_SHARD.value
        expected = {dp, MeshAxisName.EDP_SHARD.value, MeshAxisName.EP.value}
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

        trainer = deepseek_v4_1_debugmodel_text()
        model = build_cpu_model(trainer.model)
        fqns = [name for name, _ in model.named_parameters()]

        profile = trainer.optim.optimizer._muon_profile
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
        trainer = deepseek_v4_1_debugmodel_text()
        trainer.optim.optimizer.name = "Muon"
        trainer.optim.optimizer.lr = 1.23e-4
        trainer.optim.optimizer.materialize()
        groups = trainer.optim.optimizer.optimizers
        assert [type(group) for group in groups] == [EngramTableOptimizerConfig, DistMuon.Config, AdamW.Config]
        sparse, distmuon, adamw = groups
        assert sparse.lr == 5e-5
        trainer.optim.optimizer.materialize()
        assert trainer.optim.optimizer.optimizers[0] is sparse
        assert len(trainer.optim.optimizer.optimizers) == 3
        assert distmuon.pattern == trainer.optim.optimizer._muon_profile.muon_pattern
        assert adamw.pattern == r".*"
        # Top-level CLI hyperparams land in both factories (param-groups.0
        # overrides from a launcher are dead after materialize).
        assert distmuon.lr == 1.23e-4
        assert adamw.lr == 1.23e-4

        # Check first-match assignment, including 2-D normalization scales.
        for layer_id in (1, 14):
            prefix = f"layers.{layer_id}.engram"
            for suffix, expected in (
                ("table.weight", EngramTableOptimizerConfig),
                ("gate.wkv", DistMuon.Config),
                ("gate.q_weight", AdamW.Config),
                ("gate.k_weight", AdamW.Config),
            ):
                selected = next(group for group in groups if re.search(group.pattern, f"{prefix}.{suffix}"))
                assert isinstance(selected, expected)
