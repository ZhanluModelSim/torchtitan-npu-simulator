# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _run_isolated(source: str, root: Path) -> None:
    """Run import-sensitive checks in a child process to avoid global leaks."""
    env = os.environ.copy()
    pythonpath = [str(root), *sys.path]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
    subprocess.run([sys.executable, "-c", textwrap.dedent(source)], check=True, env=env)


def test_qwen3_5_launcher_delegates_to_run_train_and_uses_current_gdn_override():
    root = Path(__file__).resolve().parents[4]
    script = (root / "examples/qwen3_5/qwen3_5_debugmodel_1p_cpt_512.sh").read_text(encoding="utf-8")
    eager_flex = (root / "torchtitan_npu/extensions/torchtitan/eager_flex.py").read_text(encoding="utf-8")


    assert "bash scripts/run_train.sh" in script
    assert "TORCHTITAN_REPO:-${TORCHTITAN_DIR:-" in script
    assert 'if [[ ${OVERRIDE_IMPORTS+x} ]]' in script
    assert "HF_ASSETS_PATH:-${TORCHTITAN_REPO}/tests/assets/tokenizer" in script
    assert "DATASET_PATH:-${TORCHTITAN_REPO}/tests/assets/cc12m_test" in script

    assert "torchtitan_npu.models.qwen3_5" in script
    assert "torchtitan_npu.override.qwen3_5.gated_delta.npu" in script
    assert "override.common.attention" not in script
    assert "torchtitan_npu.override.common.token_dispatcher.asc" in script
    assert "ENABLE_NPU_MOE_DISPATCHER" in script
    assert 'ENABLE_NPU_MOE_DISPATCHER:-0' in script
    assert '--override.imports "${OVERRIDE_IMPORTS}"' in script
    assert 'if [[ -n "${OVERRIDE_IMPORTS}" ]]' not in script
    assert "return_lse=_requests_auxiliary_outputs(return_aux)" in eager_flex
    assert "return_aux=_normalize_aux_request(return_aux)" in eager_flex
    assert "use_legacy_npu_compile" in eager_flex


def test_qwen3_5_adapter_applies_only_early_compatibility_patches():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import torchtitan_npu.models.qwen3_5
        from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
        from torchtitan.models.qwen3_5 import vision_encoder as upstream_vision
        from torchtitan_npu.patches.torchtitan.hf_datasets.multimodal.mm_collator import patched_build_mrope_positions
        from torchtitan_npu.patches.torchtitan.models.qwen3_5.vision_encoder import (
            eager_create_block_mask,
            owning_vision_encoder_forward,
        )
        assert MultiModalCollator._build_mrope_positions is patched_build_mrope_positions
        assert upstream_vision.compiled_create_block_mask is eager_create_block_mask
        assert upstream_vision.Qwen35VisionEncoder.forward is owning_vision_encoder_forward
        from torchtitan_npu.extensions.torchtitan import eager_flex
        assert callable(eager_flex.mark_dense_sdpa_mask_mod)
        assert callable(eager_flex._run_eager_flex_attention)
        """,
        root,
    )


def test_qwen3_5_config_registry_forwards_upstream_factories():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import torchtitan_npu.models.qwen3_5
        from torchtitan.models.qwen3_5 import config_registry as upstream
        from torchtitan_npu.models.qwen3_5 import config_registry as adapter
        assert adapter.qwen35_debugmodel is upstream.qwen35_debugmodel
        assert adapter.qwen35_debugmodel().override.imports == []
        """,
        root,
    )


def test_qwen3_5_moe_factory_preserves_unpatched_dispatcher_schemas():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import torchtitan_npu.models.qwen3_5
        from torchtitan.models.common.config_utils import make_routed_experts_config

        for backend in ("standard", "deepep", "hybridep", "minimal_async_ep"):
            config = make_routed_experts_config(
                dim=8,
                hidden_dim=16,
                num_experts=4,
                top_k=2,
                param_init={},
                comm_backend=backend,
            )
            has_absorb = hasattr(config.token_dispatcher, "absorb_router_scores")
            assert has_absorb is (backend in ("standard", "deepep")), (backend, config.token_dispatcher)
        """,
        root,
    )


def test_qwen3_5_video_mrope_uses_contiguous_temporal_coordinates():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import torch
        import torchtitan_npu.models.qwen3_5
        from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
        collator = MultiModalCollator(
            batch_size=1,
            seq_len=4,
            max_images_per_batch=8,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            tokenizer=object(),
            build_mrope_positions=True,
        )
        positions = collator._build_mrope_positions(
            torch.tensor([[100, 11, 11, 101]]),
            grid_thw=None,
            grid_thw_videos=torch.tensor([[2, 2, 2]]),
            positions=torch.arange(4).unsqueeze(0),
            image_token_id=10,
            video_token_id=11,
        )
        torch.testing.assert_close(positions, torch.tensor([[[0, 0, 0], [1, 1, 1], [2, 1, 1], [3, 3, 3]]]))
        """,
        root,
    )


def test_qwen3_5_gdn_override_targets_upstream_kernel_config():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import inspect
        from torchtitan.config import apply_overrides
        from torchtitan_npu.models.qwen3_5.config_registry import qwen35_debugmodel
        from torchtitan_npu.override.qwen3_5.gated_delta import TritonGatedDeltaKernel, npu
        config = qwen35_debugmodel()
        config.override.imports = ["torchtitan_npu.override.qwen3_5.gated_delta.npu"]
        apply_overrides(config.override, config)
        delta_layers = [layer.delta_net for layer in config.model_spec.model.layers if layer.delta_net is not None]
        assert delta_layers
        assert all(isinstance(layer.kernel, TritonGatedDeltaKernel.Config) for layer in delta_layers)
        assert inspect.isfunction(npu)
        """,
        root,
    )


def test_qwen3_5_gdn_override_is_directly_importable_without_fla():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import torchtitan_npu.override.qwen3_5.gated_delta as gated_delta
        assert callable(gated_delta._causal_conv1d_varlen)
        """,
        root,
    )


def test_qwen3_5_varlen_conv_rejects_truncated_metadata():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import torch
        from torchtitan_npu.override.qwen3_5.gated_delta import _causal_conv1d_varlen
        # The helper must remain traceable inside checkpoint HOP; a disabled
        # callable is rejected by the target Torch version.
        assert not getattr(_causal_conv1d_varlen, "_torchdynamo_disable", False)
        x = torch.ones(1, 3, 2)
        weight = torch.ones(2, 3)
        try:
            _causal_conv1d_varlen(x, weight, torch.tensor([0, 2], dtype=torch.int32))
        except ValueError as error:
            assert "sequence_length=3" in str(error)
        else:
            raise AssertionError("truncated cu_seqlens metadata was accepted")
        output = _causal_conv1d_varlen(x, weight, torch.tensor([0, 1, 3], dtype=torch.int32))
        assert output.shape == x.shape
        """,
        root,
    )


def test_qwen3_5_flex_attention_preserves_aux_output_protocol():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import torch
        import torchtitan_npu.models.qwen3_5
        from torch.nn.attention.flex_attention import AuxOutput, AuxRequest, create_block_mask
        from torchtitan_npu.extensions.torchtitan.eager_flex import _run_eager_flex_attention

        q = torch.randn(1, 1, 4, 4)
        k = torch.randn(1, 1, 4, 4)
        v = torch.randn(1, 1, 4, 4)
        mask = create_block_mask(
            lambda _b, _h, q_idx, _kv_idx: q_idx >= 0,
            1,
            1,
            4,
            4,
            device="cpu",
            _compile=False,
        )
        _, no_lse = _run_eager_flex_attention(
            q,
            k,
            v,
            score_mod=None,
            block_mask=mask,
            scale=None,
            enable_gqa=False,
            return_aux=AuxRequest(lse=False),
            kernel_options={},
        )
        _, with_lse = _run_eager_flex_attention(
            q,
            k,
            v,
            score_mod=None,
            block_mask=mask,
            scale=None,
            enable_gqa=False,
            return_aux=AuxRequest(lse=True),
            kernel_options={},
        )
        assert isinstance(no_lse, AuxOutput) and no_lse.lse is None
        assert isinstance(with_lse, AuxOutput) and with_lse.lse is not None
        """,
        root,
    )


def test_qwen3_5_parallelizer_uses_v030_spmd_storage_mesh():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        from types import SimpleNamespace
        from unittest.mock import patch
        from torchtitan_npu.override.qwen3_5 import parallelize as adapter
        class ParallelDims:
            cp_enabled = False
            tp_enabled = False
            ep_enabled = False
            dp_replicate_enabled = False
            pp_enabled = False
            ep = 1
            def get_mesh(self, _dims):
                raise AssertionError("spmd_types must use the v0.3.0 storage mesh resolver")
        class Model:
            vision_encoder = None
            def __init__(self): self.parallelized = False
            def parallelize(self, _parallel_dims): self.parallelized = True
        parallel_dims = ParallelDims()
        model = Model()
        dense_mesh, dense_dims = object(), object()
        training = SimpleNamespace(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            enable_cpu_offload=False,
        )
        parallelism = SimpleNamespace(spmd_backend="spmd_types", fsdp_reshard_after_forward="default")
        compile_config = SimpleNamespace(enable=False, components=())
        with (
            patch.object(adapter, "resolve_fsdp_mesh", return_value=(dense_mesh, dense_dims)) as resolve_dense,
            patch.object(adapter, "resolve_sparse_fsdp_mesh", return_value=(None, None)) as resolve_sparse,
            patch.object(adapter, "apply_fsdp_to_decoder") as apply_decoder,
        ):
            result = adapter.parallelize_qwen3_5_npu(
                model,
                parallel_dims=parallel_dims,
                training=training,
                parallelism=parallelism,
                compile_config=compile_config,
                ac_config=None,
                dump_folder="unused",
            )
        assert result is model and model.parallelized
        resolve_dense.assert_called_once_with(parallel_dims)
        resolve_sparse.assert_called_once_with(parallel_dims)
        assert apply_decoder.call_args.args[1] is dense_mesh
        assert apply_decoder.call_args.kwargs["dp_mesh_dims"] is dense_dims
        assert apply_decoder.call_args.kwargs["edp_mesh_dims"] is None
        """,
        root,
    )


def test_qwen3_5_cp_rejects_expert_parallel():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        from types import SimpleNamespace
        from unittest.mock import patch
        from torchtitan_npu.override.qwen3_5 import parallelize as adapter

        parallel_dims = SimpleNamespace(cp_enabled=True, ep_enabled=True, tp_enabled=False)
        parallelism = SimpleNamespace(spmd_backend="spmd_types")
        compile_config = SimpleNamespace(enable=False, components=())
        with patch.object(adapter, "parallelize_qwen3_5_cp") as cp:
            try:
                adapter.parallelize_qwen3_5_npu(
                    object(),
                    parallel_dims=parallel_dims,
                    training=SimpleNamespace(),
                    parallelism=parallelism,
                    compile_config=compile_config,
                    ac_config=None,
                    dump_folder="unused",
                )
            except NotImplementedError as error:
                assert "Expert Parallel" in str(error)
            else:
                raise AssertionError("CP + EP was not rejected")
            cp.assert_not_called()
        """,
        root,
    )


def test_qwen3_5_cp_helper_rejects_unsupported_modes_before_mesh_setup():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        from types import SimpleNamespace
        from torchtitan_npu.override.qwen3_5 import parallelize as adapter

        common = dict(
            training=SimpleNamespace(),
            parallelism=SimpleNamespace(),
            compile_config=SimpleNamespace(),
            ac_config=None,
            dump_folder="unused",
        )
        for field, message in (("ep_enabled", "Expert Parallel"), ("tp_enabled", "Tensor Parallel")):
            dims = SimpleNamespace(cp_enabled=True, ep_enabled=False, tp_enabled=False)
            setattr(dims, field, True)
            try:
                adapter.parallelize_qwen3_5_cp(object(), parallel_dims=dims, **common)
            except NotImplementedError as error:
                assert message in str(error)
            else:
                raise AssertionError(f"CP + {field} was not rejected")

        dims = SimpleNamespace(cp_enabled=True, ep_enabled=False, tp_enabled=False)
        model = SimpleNamespace(vision_encoder=object())
        try:
            adapter.parallelize_qwen3_5_cp(model, parallel_dims=dims, **common)
        except NotImplementedError as error:
            assert "multimodal" in str(error)
        else:
            raise AssertionError("VL CP was not rejected")
        """,
        root,
    )


def test_qwen3_5_cp_parallelizer_uses_v030_spmd_storage_mesh():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        from types import SimpleNamespace
        from unittest.mock import patch
        from torchtitan_npu.override.qwen3_5 import parallelize as adapter
        cp_mesh = object()
        class ParallelDims:
            cp_enabled = True
            tp_enabled = False
            ep_enabled = False
            dp_replicate_enabled = False
            pp_enabled = False
            ep = 1
            def get_mesh(self, dims):
                if dims == "cp":
                    return cp_mesh
                raise AssertionError("spmd_types CP must use the v0.3.0 storage mesh resolver")
        class Model:
            vision_encoder = None
            layers = {}
            def register_forward_pre_hook(self, hook, *, with_kwargs):
                self.hook = hook
                self.with_kwargs = with_kwargs
            # torchtitan module protocol: distribute params to DTensors
            # before FSDP2 wrapping under the spmd_types backend.
            def parallelize(self, parallel_dims):
                self.parallelized_dims = parallel_dims
        parallel_dims = ParallelDims()
        model = Model()
        dense_mesh, dense_dims = object(), object()
        training = SimpleNamespace(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            enable_cpu_offload=False,
        )
        parallelism = SimpleNamespace(spmd_backend="spmd_types", fsdp_reshard_after_forward="default")
        compile_config = SimpleNamespace(enable=False, components=())
        with (
            patch.object(adapter, "resolve_fsdp_mesh", return_value=(dense_mesh, dense_dims)) as resolve_dense,
            patch.object(adapter, "apply_fsdp_to_decoder") as apply_decoder,
        ):
            result = adapter.parallelize_qwen3_5_npu(
                model,
                parallel_dims=parallel_dims,
                training=training,
                parallelism=parallelism,
                compile_config=compile_config,
                ac_config=None,
                dump_folder="unused",
            )
        assert result is model
        assert model.context_parallel_mesh is cp_mesh
        assert model.hook is adapter.prepare_sequence_metadata and model.with_kwargs
        assert model.parallelized_dims is parallel_dims
        resolve_dense.assert_called_once_with(parallel_dims)
        assert apply_decoder.call_args.args[1] is dense_mesh
        assert apply_decoder.call_args.kwargs["dp_mesh_dims"] is dense_dims
        assert apply_decoder.call_args.kwargs["edp_mesh_dims"] is None
        """,
        root,
    )


def test_qwen3_5_does_not_vendor_a_second_gdn_implementation():
    root = Path(__file__).resolve().parents[4]
    override = (root / "torchtitan_npu/override/qwen3_5/gated_delta.py").read_text(encoding="utf-8")
    assert "torchtitan_npu.ops.triton.gdn" in override
    assert "torchtitan_npu.ops.triton.gated_delta_rule" not in override
    assert (root / "torchtitan_npu/ops/triton/gdn/gated_delta.py").is_file()
    old_ops = root / "torchtitan_npu/ops/triton/gated_delta_rule"
    assert not any(old_ops.rglob("*.py"))


def test_presplit_pipeline_bridge_handles_older_npu_schedule():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import torch
        from torchtitan_npu.patches.torchtitan.trainer import _run_presplit_pipeline_schedule
        class Stage:
            def __init__(self): self.has_backward = False; self.cleared = 0
            def clear_runtime_states(self): self.cleared += 1
        class Schedule:
            _has_backward = True
            def __init__(self): self._stage = Stage(); self.received = None
            def _step_microbatches(self, **kwargs): self.received = kwargs
        schedule = Schedule()
        _run_presplit_pipeline_schedule(
            schedule,
            arg_mbs=[(torch.ones(1),)],
            kwarg_mbs=[{}],
            target_mbs=[torch.ones(1)],
            losses=[],
            loss_kwargs={},
            return_outputs=False,
        )
        assert schedule._stage.has_backward and schedule._stage.cleared == 1 and schedule.received["arg_mbs"]
        """,
        root,
    )
