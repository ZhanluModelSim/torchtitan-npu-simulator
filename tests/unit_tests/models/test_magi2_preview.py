# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for MAGI-2-preview model: config registry, model instantiation, forward pass."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_model_override_schema_is_stable(self):
        from torchtitan_npu.models.magi2_preview.config_overrides import (
            Magi2PreviewModelOverrides,
        )

        assert {
            field.name for field in dataclasses.fields(Magi2PreviewModelOverrides)
        } == {
            "param_init",
            "num_layers",
            "hidden_size",
            "head_dim",
            "num_stream",
            "video_in_channels",
            "audio_in_channels",
            "text_in_channels",
            "time_channel_dim",
            "dense_intermediate_size",
            "mm_layers",
            "moe_layers",
            "moe_num_heads",
            "num_experts",
            "moe_top_k",
            "expert_intermediate_size",
            "shared_expert_intermediate_size",
            "route_scale",
            "sink_token_num",
            "norm_eps",
            "alpha_init",
            "attn_backend",
        }

    @pytest.mark.parametrize("flavor", ("debug", "full"))
    def test_model_overrides_round_trip_registered_presets(self, flavor):
        from torchtitan_npu.models.magi2_preview import model_registry
        from torchtitan_npu.models.magi2_preview.config_overrides import (
            Magi2PreviewModelOverrides,
        )

        original = model_registry(flavor).model
        overrides = Magi2PreviewModelOverrides.from_model_config(original)
        rebuilt = overrides.to_model_config()

        assert dataclasses.asdict(rebuilt) == dataclasses.asdict(original)

    @pytest.mark.parametrize(
        "module",
        ("torchtitan_npu.models.magi2_preview", "torchtitan_npu.simulator"),
    )
    def test_smoketest_cli_builds_debug_model(self, module):
        from torchtitan.config import ConfigManager

        config = ConfigManager().parse_args(
            ["--module", module, "--config", "magi2_preview_smoketest"]
        )

        model = config.model_spec.model
        assert model.num_layers == 4
        assert model.hidden_size == 512
        assert model.head_dim == 128
        assert model.num_stream == 4
        assert model.mm_layers == [0, 3]
        assert model.moe_layers == [1, 2]
        assert model.text_in_channels == 64
        assert model.num_experts == 8
        assert model.moe_top_k == 2

    @pytest.mark.parametrize(
        "module",
        ("torchtitan_npu.models.magi2_preview", "torchtitan_npu.simulator"),
    )
    def test_model_overrides_cli_rebuilds_model(self, module):
        from torchtitan.config import ConfigManager

        config = ConfigManager().parse_args(
            [
                "--module",
                module,
                "--config",
                "magi2_preview_smoketest",
                "--model-overrides.num-layers",
                "3",
                "--model-overrides.hidden-size",
                "256",
                "--model-overrides.mm-layers",
                "0",
                "--model-overrides.moe-layers",
                "1",
                "2",
                "--model-overrides.moe-top-k",
                "4",
            ]
        )

        model = config.model_spec.model
        assert model.num_layers == 3
        assert model.hidden_size == 256
        assert model.mm_layers == [0]
        assert model.moe_layers == [1, 2]
        assert model.moe_top_k == 4

    def test_smoketest_uses_synthetic_dataloader_and_two_steps(self):
        from torchtitan_npu.models.magi2_preview.config_registry import (
            magi2_preview_smoketest,
        )
        from torchtitan_npu.models.magi2_preview.dataset import (
            Magi2SyntheticDataLoader,
        )

        config = magi2_preview_smoketest()
        assert config.training.steps == 2
        assert config.parallelism.data_parallel_shard_degree == -1
        assert config.parallelism.expert_parallel_degree == 1
        assert config.checkpoint.enable is False
        assert isinstance(config.dataloader, Magi2SyntheticDataLoader.Config)

    def test_simulator_smoketest_reuses_training_parallelism(self):
        from torchtitan_npu.models.magi2_preview.config_registry import (
            magi2_preview_smoketest as training_smoketest,
        )
        from torchtitan_npu.simulator.config_registry import (
            magi2_preview_smoketest as simulator_smoketest,
        )

        training_config = training_smoketest()
        simulation_config = simulator_smoketest()

        assert simulation_config.parallelism == training_config.parallelism
        assert simulation_config.compile.enable is False
        assert (
            simulation_config.simulation.output_dir
            == "./simulator_output/magi2_preview_smoketest"
        )

    def test_debug_flavor_returns_model_spec(self):
        from torchtitan.components.loss import build_mse_loss
        from torchtitan_npu.models.magi2_preview import model_registry

        spec = model_registry("debug")
        assert spec.name == "magi2_preview"
        assert spec.flavor == "debug"
        assert spec.model is not None
        assert spec.parallelize_fn is not None
        assert spec.state_dict_adapter is not None
        assert spec.build_loss_fn is build_mse_loss

    def test_full_flavor_returns_model_spec(self):
        from torchtitan_npu.models.magi2_preview import model_registry

        spec = model_registry("full")
        assert spec.name == "magi2_preview"
        assert spec.flavor == "full"

    def test_invalid_flavor_raises(self):
        from torchtitan_npu.models.magi2_preview import model_registry

        with pytest.raises(KeyError):
            model_registry("nonexistent-flavor")

    def test_debug_flavor_layer_structure(self):
        """Debug model: 4 layers, mm=[0, 3] dense, moe=[1, 2]."""
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs

        config = magi2_preview_configs["debug"]()
        assert config.num_layers == 4
        assert config.hidden_size == 512
        assert config.head_dim == 128
        assert config.num_stream == 4
        assert config.mm_layers == [0, 3]
        assert config.moe_layers == [1, 2]
        assert config.num_experts == 8
        assert config.moe_top_k == 2
        assert config.expert_intermediate_size == 64
        assert config.shared_expert_intermediate_size == 64
        assert config.dense_intermediate_size == 512
        assert config.text_in_channels == 64
        assert config.video_in_channels == 48
        assert config.audio_in_channels == 64

    def test_full_flavor_layer_structure(self):
        """Full model: 40 layers, mm={0, 1, 38, 39}, moe=2..37."""
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs

        config = magi2_preview_configs["full"]()
        assert config.num_layers == 40
        assert config.hidden_size == 3072
        assert config.head_dim == 128
        assert set(config.mm_layers) == {0, 1, 38, 39}
        assert config.moe_layers == list(range(2, 38))
        assert not set(config.mm_layers) & set(config.moe_layers)
        assert config.moe_num_heads == 12
        assert config.num_experts == 256
        assert config.moe_top_k == 6

    def test_update_from_config_is_noop(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs

        config = magi2_preview_configs["debug"]()
        before = dataclasses.asdict(config)
        config.update_from_config(trainer_config=SimpleNamespace())
        assert dataclasses.asdict(config) == before


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------


class TestModelInstantiation:
    def test_debug_model_instantiates(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        config = magi2_preview_configs["debug"]()
        model = Magi2PreviewModel(config)
        assert model is not None
        assert len(model.block.layers) == 4
        assert model.pre_adapter is not None
        assert model.post_adapter is not None

    def test_debug_model_parameter_count(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        config = magi2_preview_configs["debug"]()
        model = Magi2PreviewModel(config)
        total_params = sum(p.numel() for p in model.parameters())
        # Debug model should be small (< 100M params)
        assert total_params < 100_000_000
        assert total_params > 0

    def test_init_weights_leaves_no_meta_tensors(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        model = Magi2PreviewModel(magi2_preview_configs["debug"]())
        model.init_weights()

        states = list(model.parameters()) + list(model.buffers())
        assert all(not state.is_meta for state in states)
        assert all(torch.isfinite(state).all() for state in states)
        # Zero-init MHC phis and router biases; zero weights give norm gain 1.
        assert torch.count_nonzero(model.block.layers["0"].mhc_phi_fused_attn) == 0
        assert torch.count_nonzero(model.block.layers["0"].mhc_norm.weight) == 0
        assert (
            torch.count_nonzero(
                model.block.layers["1"].mlp.moe_mlp.router.expert_bias
            )
            == 0
        )


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    def test_debug_model_forward(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.dataset import (
            Magi2SyntheticDataLoader,
        )
        from torchtitan_npu.models.magi2_preview.model import (
            Magi2PreviewModel,
            Modality,
        )

        model = Magi2PreviewModel(magi2_preview_configs["debug"]())
        model.init_weights()
        model.eval()

        loader = Magi2SyntheticDataLoader.Config().build()
        inputs, labels = next(iter(loader))
        x = inputs.pop("input")
        with torch.no_grad():
            pred = model(x, **inputs)

        assert pred.shape == (x.shape[0], 64)
        assert pred.shape == labels.shape
        assert torch.isfinite(pred).all()
        # PostAdapter keeps text rows zero so the sum-MSE loss is masked.
        text_rows = inputs["modality_mapping"] == Modality.TEXT
        assert text_rows.any()
        assert torch.equal(pred[text_rows], torch.zeros_like(pred[text_rows]))

    def test_debug_model_backward(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.dataset import (
            Magi2SyntheticDataLoader,
        )
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        model = Magi2PreviewModel(magi2_preview_configs["debug"]())
        model.init_weights()
        model.train()

        loader = Magi2SyntheticDataLoader.Config().build()
        inputs, labels = next(iter(loader))
        x = inputs.pop("input")
        pred = model(x, **inputs)
        torch.nn.functional.mse_loss(pred, labels).backward()

        # Representative params across the pre-adapter, attention, routed MoE
        # experts, and the post-adapter must receive finite gradients.
        representative_params = (
            model.pre_adapter.video_embedder.weight,
            model.block.layers["0"].attention.linear_qkv.weight,
            model.block.layers["1"].mlp.moe_mlp.W_down,
            model.post_adapter.final_linear_video.weight,
        )
        for param in representative_params:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all()
            assert param.grad.abs().sum() > 0

        # Verified init caveat: with zero-init MHC phis, mhc_alpha_*,
        # mhc_bias_res_* and mhc_norm.weight receive exactly zero gradient
        # until the phis become nonzero; that is expected, not a bug.
        layer = model.block.layers["0"]
        for param in (
            layer.mhc_alpha_pre_attn,
            layer.mhc_bias_res_attn,
            layer.mhc_norm.weight,
        ):
            if param.grad is not None:
                assert torch.count_nonzero(param.grad) == 0


# ---------------------------------------------------------------------------
# State dict adapter
# ---------------------------------------------------------------------------


class TestStateDictAdapter:
    def test_state_dict_adapter_round_trip_is_identity(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.state_dict_adapter import (
            Magi2PreviewStateDictAdapter,
        )

        config = magi2_preview_configs["debug"]()
        model = Magi2PreviewModel(config)
        model.init_weights()
        adapter = Magi2PreviewStateDictAdapter(model_config=config)
        # Official (HF) layout: multi-expert grouped weights are stored
        # fused expert-major 2D, so draw the HF tensors in the official
        # shapes derived from the model's internal state dict via to_hf.
        official = adapter.to_hf(model.state_dict())
        generator = torch.Generator().manual_seed(0)
        hf_dict = {
            key: torch.randn(value.shape, generator=generator).to(value.dtype)
            for key, value in official.items()
        }

        result = adapter.to_hf(adapter.from_hf(hf_dict))

        assert result.keys() == hf_dict.keys()
        for key, value in hf_dict.items():
            assert torch.equal(result[key], value)

    def test_from_hf_drops_unmapped_keys(self):
        from torchtitan_npu.models.magi2_preview import magi2_preview_configs
        from torchtitan_npu.models.magi2_preview.state_dict_adapter import (
            Magi2PreviewStateDictAdapter,
        )

        adapter = Magi2PreviewStateDictAdapter(
            model_config=magi2_preview_configs["debug"]()
        )
        hf_dict = {
            "pre_adapter.video_embedder.weight": torch.randn(2048, 48),
            "pre_adapter.video_embedder.bias": torch.randn(2048),
            "not_a_magi2_module.weight": torch.randn(4),
        }
        result = adapter.from_hf(hf_dict)
        assert "pre_adapter.video_embedder.weight" in result
        assert "pre_adapter.video_embedder.bias" in result
        assert "not_a_magi2_module.weight" not in result


# ---------------------------------------------------------------------------
# Parallelize
# ---------------------------------------------------------------------------


def _mock_parallel_dims(**overrides) -> SimpleNamespace:
    """All parallelism disabled except FSDP, mirroring the kimi_k3 mocks."""
    dims = dict(
        pp_enabled=False,
        tp_enabled=False,
        cp_enabled=False,
        ep_enabled=False,
        fsdp_enabled=True,
        dp_replicate_enabled=False,
        pp=1,
        tp=1,
        cp=1,
        ep=1,
        fsdp_gradient_divide_factor=None,
        get_mesh=lambda dims: MagicMock(),
        get_optional_mesh=lambda dims: None,
    )
    dims.update(overrides)
    return SimpleNamespace(**dims)


def _parallelize_kwargs(parallel_dims, ac_config) -> dict:
    return dict(
        parallel_dims=parallel_dims,
        training=SimpleNamespace(
            seq_len=64,
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            enable_cpu_offload=False,
        ),
        model_converters=MagicMock(),
        parallelism=SimpleNamespace(
            disable_loss_parallel=False,
            fsdp_reshard_after_forward="default",
        ),
        compile_config=SimpleNamespace(enable=False, components=[]),
        ac_config=ac_config,
        dump_folder="/tmp/test",
    )


class TestParallelize:
    def test_parallelize_applies_ac_and_returns_model(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        mock_model = MagicMock()
        ac_config = SimpleNamespace(mode="selective")
        with patch(
            "torchtitan_npu.models.magi2_preview.parallelize.apply_moe_ac"
        ) as apply_moe_ac, patch(
            "torchtitan_npu.models.magi2_preview.parallelize._apply_fsdp"
        ) as apply_fsdp:
            result = parallelize_magi2_preview(
                mock_model,
                **_parallelize_kwargs(_mock_parallel_dims(), ac_config),
            )

        assert result is mock_model
        # MAGI-2 layers live at model.block.layers, so AC wraps the block.
        apply_moe_ac.assert_called_once_with(
            mock_model.block,
            ac_config,
            model_compile_enabled=False,
            base_folder="/tmp/test",
        )
        apply_fsdp.assert_called_once()

    # pp_enabled is no longer deferred: pipeline_magi2 implements it
    # (see tests/unit_tests/models/test_magi2_pp.py, including the
    # pp-alone-runs path and the pp+cp/tp/ep combination guards).
    # tp_enabled is no longer deferred either: _apply_tensor_parallel
    # implements sequence-replicated TP v1 (see tests/unit_tests/models/
    # test_magi2_tp.py, including the tp+cp/ep/etp combination guards).

    def test_tp_enabled_invokes_tensor_parallel(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        tp_mesh = MagicMock()
        parallel_dims = _mock_parallel_dims(
            tp_enabled=True,
            tp=2,
            get_mesh=lambda dims: tp_mesh if dims == "tp" else MagicMock(),
        )
        mock_model = MagicMock()
        with patch(
            "torchtitan_npu.models.magi2_preview.parallelize."
            "_apply_tensor_parallel"
        ) as apply_tp, patch(
            "torchtitan_npu.models.magi2_preview.parallelize._apply_fsdp"
        ):
            parallelize_magi2_preview(
                mock_model,
                **_parallelize_kwargs(
                    parallel_dims, SimpleNamespace(mode="none")
                ),
            )

        apply_tp.assert_called_once_with(mock_model, tp_mesh=tp_mesh, sequence_parallel=False)

    def test_cp_enabled_invokes_ulysses_wiring(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        cp_mesh = MagicMock()
        parallel_dims = _mock_parallel_dims(
            cp_enabled=True,
            cp=2,
            get_mesh=lambda dims: cp_mesh if dims == "cp" else MagicMock(),
        )
        mock_model = MagicMock()
        with patch(
            "torchtitan_npu.models.magi2_preview.parallelize."
            "apply_magi2_ulysses_cp"
        ) as apply_cp, patch(
            "torchtitan_npu.models.magi2_preview.parallelize._apply_fsdp"
        ):
            parallelize_magi2_preview(
                mock_model,
                **_parallelize_kwargs(
                    parallel_dims, SimpleNamespace(mode="none")
                ),
            )

        apply_cp.assert_called_once_with(
            mock_model, cp_mesh=cp_mesh, ep_degree=1
        )

    def test_ep_enabled_invokes_moe_parallel_with_meshes(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        ep_mesh, etp_mesh = MagicMock(), MagicMock()
        meshes = {"ep": ep_mesh, "etp": etp_mesh}
        parallel_dims = _mock_parallel_dims(
            ep_enabled=True,
            # get_optional_mesh accepts a string or a list of strings
            # (eFSDP mesh names); EP test doesn't need the eFSDP mesh.
            get_optional_mesh=lambda name: (
                meshes.get(name) if isinstance(name, str) else None
            ),
        )
        mock_model = MagicMock()
        with patch(
            "torchtitan_npu.models.magi2_preview.parallelize._apply_moe_parallel"
        ) as apply_moe_parallel, patch(
            "torchtitan_npu.models.magi2_preview.parallelize._apply_fsdp"
        ):
            parallelize_magi2_preview(
                mock_model,
                **_parallelize_kwargs(
                    parallel_dims, SimpleNamespace(mode="none")
                ),
            )

        apply_moe_parallel.assert_called_once_with(
            mock_model, ep_mesh=ep_mesh, etp_mesh=etp_mesh
        )

    def test_ep_disabled_passes_no_moe_meshes(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        mock_model = MagicMock()
        with patch(
            "torchtitan_npu.models.magi2_preview.parallelize._apply_moe_parallel"
        ) as apply_moe_parallel, patch(
            "torchtitan_npu.models.magi2_preview.parallelize._apply_fsdp"
        ):
            parallelize_magi2_preview(
                mock_model,
                **_parallelize_kwargs(
                    _mock_parallel_dims(), SimpleNamespace(mode="none")
                ),
            )

        # Baseline keeps EP/ETP disabled: no sharding work may happen.
        apply_moe_parallel.assert_called_once_with(
            mock_model, ep_mesh=None, etp_mesh=None
        )
