# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _run_isolated(source: str, root: Path) -> None:
    """Isolate CPU checks and reject execution of the upstream CUDA short conv."""
    env = os.environ.copy()
    pythonpath = [str(root), *sys.path]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath))
    # Qwen's package imports this CUDA-only export while loading model configs.
    # These checks use real recipes/loaders and the NPU conv implementation;
    # the upstream CUDA kernel is outside their scope and must never execute.
    cuda_boundary = """
        import attn_gym.linear as attention_gym_linear

        def unexpected_cuda_short_conv(*args, **kwargs):
            raise AssertionError("CPU adapter/data checks must not execute the upstream CUDA short convolution")

        attention_gym_linear.causal_conv1d = unexpected_cuda_short_conv
    """
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(cuda_boundary) + textwrap.dedent(source)],
        check=True,
        env=env,
    )


def _multimodal_loader_assets() -> tuple[Path, Path] | None:
    """The upstream tokenizer and this repo's cc12m fixture, when both are present."""
    torchtitan = Path(os.environ.get("TORCHTITAN_DIR", Path.home() / "workspace/torchtitan"))
    tokenizer_dir = torchtitan / "tests/assets/tokenizer"
    if not (tokenizer_dir / "tokenizer.json").exists():
        return None
    return tokenizer_dir, Path(__file__).resolve().parents[4] / "tests/assets/cc12m_test"


def test_qwen3_5_launcher_delegates_to_run_train_and_uses_current_gdn_override():
    root = Path(__file__).resolve().parents[4]
    script = (root / "examples/qwen3_5/qwen3_5_debugmodel_1p_cpt_512.sh").read_text(encoding="utf-8")
    eager_flex = (root / "torchtitan_npu/extensions/torchtitan/eager_flex.py").read_text(encoding="utf-8")

    assert "bash scripts/run_train.sh" in script
    assert "TORCHTITAN_REPO:-${TORCHTITAN_DIR:-" in script
    assert "if [[ ${OVERRIDE_IMPORTS+x} ]]" in script
    assert "HF_ASSETS_PATH:-${TORCHTITAN_REPO}/tests/assets/tokenizer" in script
    assert "DATASET_PATH:-${TORCHTITAN_REPO}/tests/assets/cc12m_test" in script

    assert "torchtitan_npu.models.qwen3_5" in script
    assert "torchtitan_npu.override.qwen3_5.gated_delta.npu" in script
    assert "override.common.attention" not in script
    assert "torchtitan_npu.override.common.token_dispatcher.asc" in script
    assert "ENABLE_NPU_MOE_DISPATCHER" in script
    assert "ENABLE_NPU_MOE_DISPATCHER:-0" in script
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
        from torchtitan.models.common import vision_encoder as common_vision
        from torchtitan.hf_datasets.multimodal.mm_collator import MultiModalCollator
        from torchtitan.models.qwen3_5 import vision_encoder as upstream_vision
        from torchtitan_npu.patches.torchtitan.models.qwen3_5.vision_encoder import (
            eager_create_block_mask,
            owning_vision_encoder_forward,
        )
        assert MultiModalCollator._build_mrope_positions.__module__ == "torchtitan.hf_datasets.multimodal.mm_collator"
        assert common_vision.compiled_create_block_mask is eager_create_block_mask
        assert upstream_vision.Qwen35VisionEncoder.forward is owning_vision_encoder_forward
        from torchtitan_npu.extensions.torchtitan import eager_flex
        assert callable(eager_flex.mark_dense_sdpa_mask_mod)
        assert callable(eager_flex._run_eager_flex_attention)
        """,
        root,
    )


def test_qwen3_5_debug_recipe_preserves_upstream_defaults():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import os

        import torchtitan_npu.models.qwen3_5
        from torchtitan.components.data import GrainDataLoader
        from torchtitan.hf_datasets.multimodal.mm_datasets import MM_DATASETS
        from torchtitan.models.qwen3_5 import config_registry as upstream
        from torchtitan_npu.models.qwen3_5 import config_registry as adapter

        os.environ.pop("DATASET", None)
        os.environ.pop("DATASET_PATH", None)
        recipes = (
            (adapter.qwen35_debugmodel, upstream.qwen35_debugmodel),
            (adapter.qwen35_debugmodel_moe, upstream.qwen35_debugmodel_moe),
        )
        for adapter_factory, upstream_factory in recipes:
            config = adapter_factory()
            expected = upstream_factory()
            assert config.training == expected.training
            assert isinstance(config.dataloader, GrainDataLoader.Config)
            assert config.dataloader.dataset == expected.dataloader.dataset
            assert config.dataloader.collator == expected.dataloader.collator
            assert config.dataloader.streaming_shuffle_buffer_size == expected.dataloader.streaming_shuffle_buffer_size
            # The loader collator describes the dataset processor's patch geometry.
            processor = config.dataloader.dataset.processor
            collator = config.dataloader.collator
            assert (collator.patch_size, collator.temporal_patch_size, collator.spatial_merge_size) == (
                processor.patch_size,
                processor.temporal_patch_size,
                processor.spatial_merge_size,
            )
            assert config.override.imports == []

        os.environ["DATASET_PATH"] = "/tmp/cc12m_override"
        overridden = adapter.qwen35_debugmodel()
        assert overridden.dataloader.dataset.source.path == "/tmp/cc12m_override"
        # The adapter edits its own copy, so the shared upstream dataset stays intact.
        assert MM_DATASETS["cc12m-test"].source.path == "tests/assets/cc12m_test"

        os.environ.pop("DATASET_PATH", None)
        os.environ["DATASET"] = "cc12m"
        for adapter_factory, _upstream_factory in recipes:
            selected = adapter_factory()
            assert selected.dataloader.dataset == MM_DATASETS["cc12m"]
        """,
        root,
    )


@pytest.mark.parametrize(
    "dp_rank,dp_world_size",
    [(0, 1), (0, 2), (1, 2)],
    ids=("single-rank", "dp2-rank0", "dp2-rank1"),
)
def test_qwen3_5_multimodal_loader_matches_upstream_batches(dp_rank, dp_world_size):
    root = Path(__file__).resolve().parents[4]
    assets = _multimodal_loader_assets()
    if assets is None:
        pytest.skip("the upstream tokenizer assets are needed to build the multimodal loader")
    tokenizer_dir, cc12m_dir = assets
    _run_isolated(
        f"""
        import os

        import torch

        os.environ["DATASET"] = "cc12m-test"
        os.environ["DATASET_PATH"] = r"{cc12m_dir.as_posix()}"

        from torchtitan.components.tokenizer import MultiModalTokenizer
        from torchtitan.models.qwen3_5 import QWEN3_5_SPECIAL_TOKENS
        from torchtitan.models.qwen3_5 import config_registry as upstream
        from torchtitan_npu.models.qwen3_5 import config_registry as adapter

        tokenizer = MultiModalTokenizer(
            MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
            tokenizer_path=r"{tokenizer_dir.as_posix()}",
        )
        build = dict(
            tokenizer=tokenizer,
            dp_rank={dp_rank},
            dp_world_size={dp_world_size},
            max_context_length=512,
            num_tokens_per_microbatch=512,
        )
        adapter_config = adapter.qwen35_debugmodel(seq_len=512)
        upstream_config = upstream.qwen35_debugmodel(seq_len=512)
        adapter_loader = adapter_config.dataloader.build(**build)
        upstream_loader = upstream_config.dataloader.build(**build)
        try:
            adapter_batches = iter(adapter_loader)
            upstream_batches = iter(upstream_loader)
            for _ in range(2):
                adapter_batch = next(adapter_batches)
                upstream_batch = next(upstream_batches)
                adapter_inputs, adapter_labels = adapter_batch.as_input_dict(), adapter_batch.labels
                upstream_inputs, upstream_labels = upstream_batch.as_input_dict(), upstream_batch.labels
                assert adapter_batch.num_valid_tokens == upstream_batch.num_valid_tokens
                assert adapter_inputs.keys() == upstream_inputs.keys()
                torch.testing.assert_close(adapter_labels, upstream_labels, rtol=0, atol=0)
                for key, value in adapter_inputs.items():
                    reference = upstream_inputs[key]
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, reference, rtol=0, atol=0)
                    else:
                        assert value == reference, key
        finally:
            adapter_loader.close()
            upstream_loader.close()
        """,
        root,
    )


def test_qwen3_5_multimodal_loader_resumes_from_state():
    root = Path(__file__).resolve().parents[4]
    assets = _multimodal_loader_assets()
    if assets is None:
        pytest.skip("the upstream tokenizer assets are needed to build the multimodal loader")
    tokenizer_dir, cc12m_dir = assets
    _run_isolated(
        f"""
        import os
        from copy import deepcopy

        import torch

        os.environ["DATASET"] = "cc12m-test"
        os.environ["DATASET_PATH"] = r"{cc12m_dir.as_posix()}"

        from torchtitan.components.tokenizer import MultiModalTokenizer
        from torchtitan.models.qwen3_5 import QWEN3_5_SPECIAL_TOKENS
        from torchtitan_npu.models.qwen3_5 import config_registry as adapter

        tokenizer = MultiModalTokenizer(
            MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
            tokenizer_path=r"{tokenizer_dir.as_posix()}",
        )
        build = dict(
            tokenizer=tokenizer,
            dp_rank=0,
            dp_world_size=1,
            max_context_length=512,
            num_tokens_per_microbatch=512,
        )
        config = adapter.qwen35_debugmodel(seq_len=512)
        original = config.dataloader.build(**build)
        restored = config.dataloader.build(**build)
        try:
            iterator = iter(original)
            for _ in range(3):
                next(iterator)
            restored.load_state_dict(deepcopy(original.state_dict()))
            resumed = iter(restored)
            for _ in range(3):
                expected_batch = next(iterator)
                actual_batch = next(resumed)
                expected_inputs, expected_labels = expected_batch.as_input_dict(), expected_batch.labels
                actual_inputs, actual_labels = actual_batch.as_input_dict(), actual_batch.labels
                assert actual_inputs.keys() == expected_inputs.keys()
                assert actual_batch.num_valid_tokens == expected_batch.num_valid_tokens
                torch.testing.assert_close(actual_labels, expected_labels, rtol=0, atol=0)
                for key, value in expected_inputs.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(actual_inputs[key], value, rtol=0, atol=0)
                    else:
                        assert actual_inputs[key] == value, key
        finally:
            original.close()
            restored.close()
        """,
        root,
    )


def test_qwen3_5_moe_factory_preserves_unpatched_dispatcher_schemas():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import pytest
        import torchtitan_npu.models.qwen3_5
        from torchtitan.models.common.config_utils import make_routed_experts_config

        for backend in ("standard", "deepep", "hybridep"):
            config = make_routed_experts_config(
                dim=8,
                hidden_dim=16,
                num_experts=4,
                top_k=2,
                param_init={},
                comm_backend=backend,
                absorb_router_scores=backend in ("standard", "deepep"),
            )
            has_absorb = hasattr(config.token_dispatcher, "absorb_router_scores")
            assert has_absorb is (backend in ("standard", "deepep")), (backend, config.token_dispatcher)

        with pytest.raises(ValueError, match="Unknown comm_backend: 'minimal_async_ep'"):
            make_routed_experts_config(
                dim=8,
                hidden_dim=16,
                num_experts=4,
                top_k=2,
                param_init={},
                comm_backend="minimal_async_ep",
            )
        """,
        root,
    )


def test_qwen3_5_video_mrope_uses_contiguous_temporal_coordinates():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        import torch
        import torchtitan_npu.models.qwen3_5
        from types import SimpleNamespace
        import grain.python as grain
        from torchtitan.components.data import DatasetBuildContext, GrainDataLoader
        from torchtitan_npu.models.qwen3_5.config_registry import qwen35_debugmodel
        config = qwen35_debugmodel(seq_len=4)
        assert isinstance(config.dataloader, GrainDataLoader.Config)
        # Only the collator's numeric token IDs are stubbed; patches and positions
        # are produced by the native collator selected by the actual recipe.
        tokenizer = SimpleNamespace(TOKEN_FIELDS=("image", "video"), image_id=10, video_id=11, pad_id=0)
        collator = config.dataloader.collator.build(
            context=DatasetBuildContext(
                tokenizer=tokenizer,
                max_context_length=4,
                num_tokens_per_microbatch=4,
                read_options=grain.ReadOptions(),
            )
        )
        microbatch = collator([{
            "input_ids": torch.tensor([100, 11, 11, 101]),
            "labels": torch.tensor([-100, -100, 101, -100]),
            "positions": torch.arange(4),
            "pixel_values_videos": [torch.zeros(4, 32, 32, 3)],
        }])
        inputs, labels = microbatch.as_input_dict(), microbatch.labels
        assert microbatch.num_valid_tokens == 1
        assert inputs["input"].shape == labels.shape == (4,)
        assert inputs["pixel_values_videos"].shape == (8, 1536)
        torch.testing.assert_close(inputs["grid_thw_videos"], torch.tensor([[2, 2, 2]]))
        torch.testing.assert_close(
            inputs["mrope_positions"], torch.tensor([[0, 0, 0], [1, 1, 1], [2, 1, 1], [3, 3, 3]])
        )
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
        delta_layers = [layer.delta_net for layer in config.model.layers if layer.delta_net is not None]
        assert delta_layers
        # The 0.4.0 GatedDeltaNet delegates to InnerGatedDeltaNet, which owns the kernel.
        assert all(
            isinstance(layer.inner_gated_delta_net.kernel, TritonGatedDeltaKernel.Config)
            for layer in delta_layers
        )
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


def test_qwen3_5_fsdp_uses_resolved_storage_mesh():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        from types import SimpleNamespace
        from unittest.mock import patch
        from torchtitan_npu.override.qwen3_5 import parallelize as adapter
        class ParallelismContext:
            cp_enabled = False
            pp_enabled = False
            ep = 1
            def get_mesh(self, _dims):
                raise AssertionError("FSDP must use the storage mesh resolver")
        model = SimpleNamespace(vision_encoder=None)
        context = ParallelismContext()
        dense_mesh, dense_dims = object(), object()
        training = SimpleNamespace(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            enable_cpu_offload=False,
        )
        parallelism = SimpleNamespace(fsdp_reshard_after_forward="default", fsdp_symm_mem_scope=None)
        with (
            patch.object(adapter, "resolve_fsdp_mesh", return_value=(dense_mesh, dense_dims)) as resolve_dense,
            patch.object(adapter, "resolve_sparse_fsdp_mesh", return_value=(None, None)) as resolve_sparse,
            patch.object(adapter, "apply_fsdp_to_decoder") as apply_decoder,
        ):
            adapter.apply_qwen3_5_npu_fsdp(
                model, parallelism_context=context, training=training, parallelism=parallelism
            )
        resolve_dense.assert_called_once_with(context)
        resolve_sparse.assert_called_once_with(context)
        assert apply_decoder.call_args.args[0] is model
        assert apply_decoder.call_args.args[1] is dense_mesh
        assert apply_decoder.call_args.kwargs["dp_mesh_dims"] is dense_dims
        assert apply_decoder.call_args.kwargs["edp_mesh_dims"] is None
        """,
        root,
    )


@pytest.mark.skip(
    reason="Qwen3.5 NPU context parallel is not migrated to the folded token layout yet; "
    "re-enable with the Qwen CP folded-token migration"
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


@pytest.mark.skip(
    reason="Qwen3.5 NPU context parallel is not migrated to the folded token layout yet; "
    "re-enable with the Qwen CP folded-token migration"
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


@pytest.mark.skip(
    reason="Qwen3.5 NPU context parallel is not migrated to the folded token layout yet; "
    "re-enable with the Qwen CP folded-token migration"
)
def test_qwen3_5_cp_parallelizer_rejects_unmigrated_folded_path():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        from types import SimpleNamespace
        import pytest
        from torchtitan_npu.override.qwen3_5 import parallelize as adapter

        model = SimpleNamespace(vision_encoder=None)
        dims = SimpleNamespace(cp_enabled=True, ep_enabled=False, tp_enabled=False)
        with pytest.raises(NotImplementedError, match="not migrated to the folded token layout"):
            adapter.parallelize_qwen3_5_npu(
                model,
                parallel_dims=dims,
                training=SimpleNamespace(),
                parallelism=SimpleNamespace(spmd_backend="spmd_types"),
                compile_config=SimpleNamespace(enable=False, components=()),
                ac_config=None,
                dump_folder="unused",
            )
        assert vars(model) == {"vision_encoder": None}
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


def test_native_engine_passes_presplit_pipeline_inputs_to_schedule():
    root = Path(__file__).resolve().parents[4]
    _run_isolated(
        """
        from contextlib import nullcontext
        from types import SimpleNamespace
        import torch
        from torchtitan.training_engine import TrainingEngine
        class Schedule:
            def __init__(self): self.received = None
            def step(self, **kwargs):
                self.received = kwargs
                kwargs["losses"].append(torch.tensor(2.0, requires_grad=True))
        schedule = Schedule()
        engine = object.__new__(TrainingEngine)
        engine.parallelism_context = SimpleNamespace(activate_spmd=lambda **kwargs: nullcontext())
        engine.config = SimpleNamespace(debug=SimpleNamespace(spmd_typechecking=False))
        engine.pp_has_last_stage = True
        engine.pp_schedule = schedule
        engine.device = torch.device("cpu")
        inputs, labels, model_kwargs = [(torch.ones(1),)], [torch.ones(1)], [{}]
        loss = engine._pp_forward_backward_microbatch_group(
            inputs=inputs, labels=labels, model_kwargs=model_kwargs,
            loss_kwargs={}, finalize_gradients=False,
        )
        assert schedule.received["arg_mbs"] is inputs
        assert schedule.received["target_mbs"] is labels
        assert schedule.received["kwarg_mbs"] is model_kwargs
        assert schedule.received["return_outputs"] is False
        assert schedule.received["finalize_gradients"] is False
        assert loss.item() == 2.0 and not loss.requires_grad
        """,
        root,
    )
