# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview pipeline parallelism tests (pipeline_parallel.py).

CI-safe single-process coverage:
- stage-split plan correctness (which layers/adapter on which stage for
  pp=2/4, debug and full flavors);
- module splitting with a mocked PipelineStage (pruning, bound stage
  forward, state-dict keys);
- stage-chain vs unsplit-model fwd/bwd equivalence (the stage boundary
  is emulated by detaching the inter-stage activations and manually
  handing the gradients back, exactly what the pipeline schedule does);
- guards: pp+cp/tp/ep/etp in parallelize, single-microbatch and
  layer-divisibility checks in pipeline_magi2;
- one-stage pipeline_magi2 construction over a real 1-rank gloo
  PipelineStage + 1F1B schedule (real fwd/bwd + loss protocol).

Nightly-gated real-P2P coverage (RUN_MODEL_PARALLEL_MULTI_RANK,
following tests/smoke_tests/model_parallel/_multi_rank.py conventions):

    torchrun --nproc_per_node=2 -m pytest \
        tests/unit_tests/models/test_magi2_pp.py -m nightly -k TwoRank
"""

import contextlib
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torchtitan.components.loss import build_mse_loss, mse_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _debug_config():
    """The debug flavor model config (4 layers: 2 mm + 2 MoE)."""
    from torchtitan_npu.models.magi2_preview import magi2_preview_configs

    return magi2_preview_configs["debug"]()


def _packed_sample(seed: int = 0):
    """One synthetic packed flow-matching sample.

    Returns ``(x, kwargs, labels)`` where ``kwargs`` carries exactly the
    forward kwargs the trainer forwards to every PP stage
    (coords_mapping/modality_mapping/time_embedding/cu_seqlens).
    """
    from torchtitan_npu.models.magi2_preview.dataset import Magi2SyntheticDataset

    input_dict, labels = Magi2SyntheticDataset(seed=seed)._build_sample(seed)
    x = input_dict.pop("input")
    return x, input_dict, labels


def _fake_pp_mesh(rank: int, degree: int):
    """Duck-typed stand-in for a 1D pp DeviceMesh."""
    return SimpleNamespace(
        ndim=1,
        size=lambda: degree,
        get_local_rank=lambda: rank,
        get_group=lambda name: None,
    )


class _RecordingPipelineStage:
    """PipelineStage stub: records the stage chunk, runs no P2P."""

    def __init__(
        self, module, stage_index, num_stages, device, group=None, **kwargs
    ):
        del device, kwargs
        self.submod = module
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.group = group
        self.is_first = stage_index == 0
        self.is_last = stage_index == num_stages - 1


def _split_model(model, degree: int, schedule: str = "1F1B"):
    """Split ``model`` into ``degree`` stage chunks without a real pp mesh.

    Runs magi2_pipeline_module_split once per virtual pp rank with a
    mocked PipelineStage and returns ``{stage_idx: stage_model}``.
    """
    from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
        generate_magi2_fqn_per_model_part,
        magi2_pipeline_module_split,
    )

    num_layers = model.config.num_layers
    module_names_per_stage = generate_magi2_fqn_per_model_part(
        degree, num_layers, 1, 1
    )
    chunks = {}
    with mock.patch(
        "torchtitan_npu.models.magi2_preview.pipeline_parallel.PipelineStage",
        _RecordingPipelineStage,
    ):
        for rank in range(degree):
            stages, models = magi2_pipeline_module_split(
                model,
                _fake_pp_mesh(rank, degree),
                schedule,
                torch.device("cpu"),
                module_names_per_stage,
            )
            assert len(stages) == 1, f"rank {rank} built {len(stages)} stages"
            chunks[stages[0].stage_index] = models[0]
    assert sorted(chunks) == list(range(degree))
    return chunks


def _identity_parallelize(model, **kwargs):
    del kwargs
    return model


def _mse_sum(pred, labels):
    return mse_loss(pred, labels)


def _duck_parallel_dims(
    *,
    pp_enabled: bool = True,
    cp_enabled: bool = False,
    tp_enabled: bool = False,
    fsdp_enabled: bool = False,
    dp_replicate_enabled: bool = False,
    ep_mesh=None,
    etp_mesh=None,
):
    """Duck-typed ParallelDims for the parallelize pp-branch guards."""
    meshes = {"ep": ep_mesh, "etp": etp_mesh}
    return SimpleNamespace(
        pp_enabled=pp_enabled,
        cp_enabled=cp_enabled,
        tp_enabled=tp_enabled,
        fsdp_enabled=fsdp_enabled,
        dp_replicate_enabled=dp_replicate_enabled,
        get_optional_mesh=lambda name: meshes.get(name),
    )


@contextlib.contextmanager
def _cpu_fork_rng():
    """Yield with the unwrapped ``torch.random.fork_rng`` restored.

    The repo's NPU patch (patches/torch/pipelining.py) forces
    ``device_type="npu"`` on every pipelining ``fork_rng`` call, which
    breaks CPU/gloo runs (``fork_rng(devices=[])`` still gets redirected
    because ``[]`` is not None); the unwrapped original is fine there.
    Only substitute when the NPU hook is actually applied: the stock
    ``@contextmanager``-decorated ``fork_rng`` also exposes ``__wrapped__``
    (its bare generator), which is not a valid replacement.
    """
    current = torch.random.fork_rng
    if getattr(current, "npu_pipeline_rng_patched", False):
        unwrapped = getattr(current, "__wrapped__", None)
        if unwrapped is not None:
            with mock.patch("torch.random.fork_rng", unwrapped):
                yield
            return
    yield


def _parallelize_kwargs(**overrides):
    """Keyword args for parallelize_magi2_preview beyond parallel_dims."""
    defaults = dict(
        training=SimpleNamespace(
            mixed_precision_param="float32",
            mixed_precision_reduce="float32",
            enable_cpu_offload=False,
        ),
        model_converters=SimpleNamespace(converters=[]),
        parallelism=SimpleNamespace(fsdp_reshard_after_forward="default"),
        compile_config=SimpleNamespace(enable=False, components=[]),
        ac_config=SimpleNamespace(mode="none"),
        dump_folder="",
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Stage-split plan
# ---------------------------------------------------------------------------


class TestStageSplitPlan:
    def test_pp2_debug_layout(self):
        from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
            generate_magi2_fqn_per_model_part,
        )

        # 4 layers + 2 adapter weights -> 3 effective layers per stage.
        plan = generate_magi2_fqn_per_model_part(2, 4, 1, 1)
        assert plan == [
            ["pre_adapter", "block.layers.0", "block.layers.1"],
            ["block.layers.2", "block.layers.3", "post_adapter"],
        ]

    def test_pp4_debug_layout(self):
        from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
            generate_magi2_fqn_per_model_part,
        )

        # 6 effective layers over 4 stages -> sizes [2, 2, 1, 1].
        plan = generate_magi2_fqn_per_model_part(4, 4, 1, 1)
        assert plan == [
            ["pre_adapter", "block.layers.0"],
            ["block.layers.1", "block.layers.2"],
            ["block.layers.3"],
            ["post_adapter"],
        ]

    def test_full_model_pp4_covers_every_layer_once(self):
        from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
            generate_magi2_fqn_per_model_part,
        )

        plan = generate_magi2_fqn_per_model_part(4, 40, 1, 1)
        assert plan[0][0] == "pre_adapter"
        assert "post_adapter" not in plan[0]
        assert plan[-1][-1] == "post_adapter"
        assert "pre_adapter" not in plan[-1]
        # Middle stages hold transformer layers only.
        for stage_modules in plan[1:-1]:
            assert all(name.startswith("block.layers.") for name in stage_modules)

        layer_ids = []
        for stage_modules in plan:
            ids = [
                int(name.split(".")[-1])
                for name in stage_modules
                if name.startswith("block.layers.")
            ]
            # Per-stage layers are contiguous and increasing.
            assert ids == list(range(ids[0], ids[0] + len(ids)))
            layer_ids.extend(ids)
        assert layer_ids == list(range(40))

    def test_single_stage_keeps_everything(self):
        from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
            generate_magi2_fqn_per_model_part,
        )

        plan = generate_magi2_fqn_per_model_part(1, 4, 1, 1)
        assert plan == [
            ["pre_adapter"]
            + [f"block.layers.{i}" for i in range(4)]
            + ["post_adapter"]
        ]

    def test_rejects_more_stages_than_effective_layers(self):
        from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
            generate_magi2_fqn_per_model_part,
        )

        with pytest.raises(ValueError, match="effective layers"):
            generate_magi2_fqn_per_model_part(8, 4, 1, 1)

    def test_rejects_weight_exceeding_layers_per_stage(self):
        from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
            generate_magi2_fqn_per_model_part,
        )

        # 6 effective layers over 6 stages -> 1 per stage < input_weight 2.
        with pytest.raises(ValueError, match="input_weight"):
            generate_magi2_fqn_per_model_part(6, 2, 2, 2)


# ---------------------------------------------------------------------------
# Module splitting (mocked PipelineStage)
# ---------------------------------------------------------------------------


class TestStageModuleSplit:
    def _build_model(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        torch.manual_seed(7)
        model = Magi2PreviewModel(_debug_config())
        model.init_weights()
        model.train()
        return model

    def test_pp2_split_prunes_modules_and_binds_stage_forward(self):
        from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
            _magi2_stage_forward,
        )

        model = self._build_model()
        full_keys = set(model.state_dict().keys())
        chunks = _split_model(model, degree=2)

        stage0, stage1 = chunks[0], chunks[1]
        assert stage0.pre_adapter is not None
        assert stage0.post_adapter is None
        assert set(stage0.block.layers.keys()) == {"0", "1"}
        assert stage1.pre_adapter is None
        assert stage1.post_adapter is not None
        assert set(stage1.block.layers.keys()) == {"2", "3"}

        # The PP-aware forward is bound on every stage part.
        assert stage0.forward.__func__ is _magi2_stage_forward
        assert stage1.forward.__func__ is _magi2_stage_forward

        # State-dict keys are exactly the kept modules' keys.
        kept0 = {
            key
            for key in full_keys
            if key.startswith(("pre_adapter.", "block.layers.0.", "block.layers.1."))
        }
        kept1 = {
            key
            for key in full_keys
            if key.startswith(("block.layers.2.", "block.layers.3.", "post_adapter."))
        }
        assert set(stage0.state_dict().keys()) == kept0
        assert set(stage1.state_dict().keys()) == kept1
        assert kept0 | kept1 == full_keys

    def test_pp4_split_edge_stages(self):
        model = self._build_model()
        chunks = _split_model(model, degree=4)

        assert chunks[0].pre_adapter is not None
        assert set(chunks[0].block.layers.keys()) == {"0"}
        assert set(chunks[1].block.layers.keys()) == {"1", "2"}
        assert chunks[1].pre_adapter is None and chunks[1].post_adapter is None
        assert set(chunks[2].block.layers.keys()) == {"3"}
        # Last stage owns post_adapter and no layers for this layout.
        assert chunks[3].post_adapter is not None
        assert chunks[3].pre_adapter is None
        assert len(chunks[3].block.layers) == 0


# ---------------------------------------------------------------------------
# Stage-chain equivalence with the unsplit model
# ---------------------------------------------------------------------------


class TestStageForwardEquivalence:
    def _build(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        torch.manual_seed(7)
        model = Magi2PreviewModel(_debug_config())
        model.init_weights()
        model.train()
        return model

    @staticmethod
    def _run_stage_chain(chunks, x, kwargs, labels):
        """Forward every stage in order + backward with manual grad handoff.

        The inter-stage activations are detached with requires_grad at
        each stage boundary, and the upstream backward is invoked with
        the downstream grads — exactly the dataflow the pipeline schedule
        implements across ranks. Returns ``(pred, loss, x_pp)``.
        """
        ordered = [chunks[stage_idx] for stage_idx in range(len(chunks))]
        x_pp = x.clone().requires_grad_(True)
        boundaries = []
        activations = ordered[0](x_pp, **kwargs)
        for chunk in ordered[1:]:
            hidden = activations[0].detach().requires_grad_(True)
            rope = activations[1].detach().requires_grad_(True)
            boundaries.append((activations, (hidden, rope)))
            activations = chunk(hidden, rope, **kwargs)
        pred = activations
        loss = _mse_sum(pred, labels)
        loss.backward()
        # Hand the inter-stage gradients back through each earlier
        # stage's ORIGINAL outputs (the detached copies only carried the
        # grads across the boundary, as received activations do).
        for originals, leaves in reversed(boundaries):
            outputs, grad_outputs = [], []
            for orig, leaf in zip(originals, leaves, strict=True):
                if orig.grad_fn is not None:
                    outputs.append(orig)
                    grad_outputs.append(leaf.grad)
            torch.autograd.backward(outputs, grad_outputs)
        return pred, loss, x_pp

    def test_pp2_stage_chain_matches_full_model_fwd_bwd(self):
        model = self._build()
        chunks = _split_model(model, degree=2)
        x, kwargs, labels = _packed_sample()

        x_ref = x.clone().requires_grad_(True)
        pred_ref = model(x_ref, **kwargs)
        loss_ref = _mse_sum(pred_ref, labels)
        loss_ref.backward()
        ref_grads = {n: p.grad.clone() for n, p in model.named_parameters()}

        pred, loss, x_pp = self._run_stage_chain(chunks, x, kwargs, labels)

        assert torch.allclose(pred, pred_ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(x_pp.grad, x_ref.grad, atol=1e-5, rtol=1e-5)

        # The two stage chunks partition the full parameter set and their
        # gradients match the unsplit-model gradients.
        seen = set()
        for chunk in chunks.values():
            for name, param in chunk.named_parameters():
                assert name in ref_grads, name
                assert param.grad is not None, name
                assert torch.allclose(
                    param.grad, ref_grads[name], atol=1e-4, rtol=1e-4
                ), f"grad mismatch for {name}"
                seen.add(name)
        assert seen == set(ref_grads)

    def test_pp4_stage_chain_matches_full_model_fwd_bwd(self):
        model = self._build()
        chunks = _split_model(model, degree=4)
        x, kwargs, labels = _packed_sample()

        x_ref = x.clone().requires_grad_(True)
        pred_ref = model(x_ref, **kwargs)
        loss_ref = _mse_sum(pred_ref, labels)
        loss_ref.backward()
        ref_grads = {n: p.grad.clone() for n, p in model.named_parameters()}

        pred, loss, x_pp = self._run_stage_chain(chunks, x, kwargs, labels)

        assert torch.allclose(pred, pred_ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(loss, loss_ref, atol=1e-5, rtol=1e-5)
        assert torch.allclose(x_pp.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
        seen = set()
        for chunk in chunks.values():
            for name, param in chunk.named_parameters():
                assert param.grad is not None, name
                assert torch.allclose(
                    param.grad, ref_grads[name], atol=1e-4, rtol=1e-4
                ), f"grad mismatch for {name}"
                seen.add(name)
        assert seen == set(ref_grads)

    def test_stage_output_shapes(self):
        model = self._build()
        chunks = _split_model(model, degree=2)
        x, kwargs, _ = _packed_sample()
        hidden, rope = chunks[0](x, **kwargs)
        config = _debug_config()
        seq_len = x.shape[0]
        assert hidden.shape == (seq_len, config.num_stream * config.hidden_size)
        # ElementWiseFourierEmbed(head_dim=128) emits 96 rope features.
        assert rope.shape == (seq_len, 96)


# ---------------------------------------------------------------------------
# pipeline_magi2 guards and single-stage schedule construction
# ---------------------------------------------------------------------------


class TestPipelineMagi2Guards:
    def _call(self, *, parallelism, training):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
            pipeline_magi2,
        )

        model = Magi2PreviewModel(_debug_config())
        parallel_dims = SimpleNamespace(
            pp=parallelism.pipeline_parallel_degree,
            get_mesh=lambda name: _fake_pp_mesh(0, parallel_dims.pp),
        )
        return pipeline_magi2(
            model,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=SimpleNamespace(converters=[]),
            parallelism=parallelism,
            compile_config=SimpleNamespace(enable=False, components=[]),
            ac_config=SimpleNamespace(mode="none"),
            dump_folder="",
            device=torch.device("cpu"),
            model_config=model.config,
            parallelize_fn=_identity_parallelize,
            loss_fn=build_mse_loss(SimpleNamespace(enable=False, components=[])),
        )

    def test_rejects_multiple_microbatches(self):
        from torchtitan.config import ParallelismConfig

        parallelism = ParallelismConfig(
            pipeline_parallel_degree=2,
            pipeline_parallel_microbatch_size=1,
        )
        with pytest.raises(ValueError, match="one microbatch"):
            self._call(
                parallelism=parallelism,
                training=SimpleNamespace(local_batch_size=2),
            )

    def test_rejects_nondivisible_layer_count(self):
        from torchtitan.config import ParallelismConfig

        # Debug flavor has 4 layers; pp=3 does not divide it.
        parallelism = ParallelismConfig(pipeline_parallel_degree=3)
        with pytest.raises(ValueError, match="divide num_layers"):
            self._call(
                parallelism=parallelism,
                training=SimpleNamespace(local_batch_size=1),
            )

    def test_rejects_indivisible_virtual_stages_with_layers_per_stage(self):
        from torchtitan.config import ParallelismConfig

        # 6 effective layers / 4 layers per stage -> ceil(6/4) = 2 virtual
        # stages, not divisible by pp=4.
        parallelism = ParallelismConfig(
            pipeline_parallel_degree=4,
            pipeline_parallel_layers_per_stage=4,
        )
        with pytest.raises(ValueError, match="divisible"):
            self._call(
                parallelism=parallelism,
                training=SimpleNamespace(local_batch_size=1),
            )


class TestPipelineMagi2SingleStage:
    @pytest.mark.usefixtures("single_rank_process_group")
    def test_single_stage_schedule_runs_fwd_bwd_and_loss(self):
        """Full pipeline_magi2 over a real 1-rank gloo PipelineStage."""
        from torch.distributed.device_mesh import init_device_mesh
        from torchtitan.config import ParallelismConfig

        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel
        from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
            pipeline_magi2,
        )

        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("pp",))
        parallel_dims = SimpleNamespace(pp=1, get_mesh=lambda name: mesh)

        torch.manual_seed(7)
        model = Magi2PreviewModel(_debug_config())
        model.init_weights()
        model.train()

        compile_config = SimpleNamespace(enable=False, components=[])
        schedule, model_parts, has_first, has_last = pipeline_magi2(
            model,
            parallel_dims=parallel_dims,
            training=SimpleNamespace(local_batch_size=1),
            model_converters=SimpleNamespace(converters=[]),
            parallelism=ParallelismConfig(pipeline_parallel_degree=1),
            compile_config=compile_config,
            ac_config=SimpleNamespace(mode="none"),
            dump_folder="",
            device=torch.device("cpu"),
            model_config=model.config,
            parallelize_fn=_identity_parallelize,
            loss_fn=build_mse_loss(compile_config),
        )
        assert has_first and has_last
        assert len(model_parts) == 1

        x, kwargs, labels = _packed_sample()
        with torch.no_grad():
            expected_loss = _mse_sum(model_parts[0](x, **kwargs), labels)

        losses = []
        with _cpu_fork_rng():
            schedule.step(
                x, **kwargs, target=labels, losses=losses, return_outputs=False
            )
        assert len(losses) == 1
        assert torch.allclose(losses[0], expected_loss, atol=1e-5, rtol=1e-5)
        # The schedule's backward populated gradients.
        grads = [p.grad for p in model_parts[0].parameters()]
        assert any(grad is not None for grad in grads)


# ---------------------------------------------------------------------------
# parallelize pp branch guards
# ---------------------------------------------------------------------------


class TestParallelizePpBranch:
    def _model(self):
        from torchtitan_npu.models.magi2_preview.model import Magi2PreviewModel

        return Magi2PreviewModel(_debug_config())

    def test_pp_with_cp_raises(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        with pytest.raises(NotImplementedError, match="PP \\+ CP"):
            parallelize_magi2_preview(
                self._model(),
                parallel_dims=_duck_parallel_dims(cp_enabled=True),
                **_parallelize_kwargs(),
            )

    def test_pp_with_tp_raises(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        with pytest.raises(NotImplementedError, match="PP \\+ TP"):
            parallelize_magi2_preview(
                self._model(),
                parallel_dims=_duck_parallel_dims(tp_enabled=True),
                **_parallelize_kwargs(),
            )

    def test_pp_with_ep_raises(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        with pytest.raises(NotImplementedError, match="EP/ETP"):
            parallelize_magi2_preview(
                self._model(),
                parallel_dims=_duck_parallel_dims(ep_mesh=object()),
                **_parallelize_kwargs(),
            )

    def test_pp_with_etp_raises(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        with pytest.raises(NotImplementedError, match="EP/ETP"):
            parallelize_magi2_preview(
                self._model(),
                parallel_dims=_duck_parallel_dims(etp_mesh=object()),
                **_parallelize_kwargs(),
            )

    def test_pp_alone_runs_on_stage_chunk(self):
        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        model = self._model()
        chunks = _split_model(model, degree=2)
        for chunk in chunks.values():
            out = parallelize_magi2_preview(
                chunk,
                parallel_dims=_duck_parallel_dims(),
                **_parallelize_kwargs(),
            )
            assert out is chunk

    def test_pp_stage_chunk_ac_wraps_block_layers(self):
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )
        from torchtitan.config import ActivationCheckpointConfig

        from torchtitan_npu.models.magi2_preview.parallelize import (
            parallelize_magi2_preview,
        )

        model = self._model()
        chunks = _split_model(model, degree=2)
        chunk = chunks[0]
        layer_ids = sorted(chunk.block.layers.keys())
        assert layer_ids, "stage chunk must own at least one layer"
        parallelize_magi2_preview(
            chunk,
            parallel_dims=_duck_parallel_dims(),
            **_parallelize_kwargs(ac_config=ActivationCheckpointConfig(mode="full")),
        )
        for layer_id in layer_ids:
            assert isinstance(chunk.block.layers[layer_id], CheckpointWrapper)


# ---------------------------------------------------------------------------
# Nightly: real 2-rank gloo pipeline (RUN_MODEL_PARALLEL_MULTI_RANK)
# ---------------------------------------------------------------------------


def _load_multi_rank_conventions():
    """Import ``tests/smoke_tests/model_parallel/_multi_rank.py``.

    Prefers the canonical package import (repo root leads sys.path, e.g.
    in CI); otherwise walks up from this file to find the repo ``tests``
    tree (harness layouts may shadow or detach it). Returns ``None``
    when the conventions module is unavailable.
    """
    try:
        from tests.smoke_tests.model_parallel import _multi_rank

        return _multi_rank
    except ImportError:
        import importlib.util
        import pathlib

        for parent in pathlib.Path(__file__).resolve().parents:
            path = (
                parent
                / "tests"
                / "smoke_tests"
                / "model_parallel"
                / "_multi_rank.py"
            )
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location(
                "magi2_pp_multi_rank_conventions", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return None


_multi_rank = _load_multi_rank_conventions()
if _multi_rank is not None:
    MULTI_RANK_AVAILABLE = _multi_rank.MULTI_RANK_AVAILABLE
    mark_multi_rank_nightly = _multi_rank.mark_multi_rank_nightly
    with_comms = _multi_rank.with_comms
else:
    MULTI_RANK_AVAILABLE = False

    def mark_multi_rank_nightly(test_obj):
        return test_obj

    with_comms = None

if MULTI_RANK_AVAILABLE:
    from torch.testing._internal.distributed._tensor.common_dtensor import (
        DTensorTestBase,
    )

    class _TwoRankCpuMultiRankTestBase(DTensorTestBase):
        @property
        def world_size(self):
            return 2

        @property
        def device_type(self):
            return "cpu"

    @mark_multi_rank_nightly
    class TestMagi2PpTwoRankMultiRank(_TwoRankCpuMultiRankTestBase):
        """PP=2 vs unsplit fwd/bwd equivalence over a real gloo pp group.

        Run with: torchrun --nproc_per_node=2 -m pytest \
            tests/unit_tests/models/test_magi2_pp.py -m nightly -k TwoRank
        """

        @with_comms
        def test_pp2_forward_backward_match_unsplit(self):
            from torch.distributed.device_mesh import init_device_mesh
            from torchtitan.config import ParallelismConfig

            from torchtitan_npu.models.magi2_preview.model import (
                Magi2PreviewModel,
            )
            from torchtitan_npu.models.magi2_preview.pipeline_parallel import (
                pipeline_magi2,
            )

            mesh = init_device_mesh(
                self.device_type, (self.world_size,), mesh_dim_names=("pp",)
            )
            rank = mesh.get_local_rank()
            parallel_dims = SimpleNamespace(pp=2, get_mesh=lambda name: mesh)

            # Identical weights on both ranks: seed, build, re-seed and
            # build the unsplit reference.
            torch.manual_seed(7)
            model = Magi2PreviewModel(_debug_config())
            model.init_weights()
            model.train()
            torch.manual_seed(7)
            ref = Magi2PreviewModel(_debug_config())
            ref.init_weights()
            ref.train()

            compile_config = SimpleNamespace(enable=False, components=[])
            # GPipe, not 1F1B: v1 runs a single microbatch, and 1F1B
            # requires n_microbatches >= num_stages.
            schedule, model_parts, has_first, has_last = pipeline_magi2(
                model,
                parallel_dims=parallel_dims,
                training=SimpleNamespace(local_batch_size=1),
                model_converters=SimpleNamespace(converters=[]),
                parallelism=ParallelismConfig(
                    pipeline_parallel_degree=2,
                    pipeline_parallel_schedule="GPipe",
                ),
                compile_config=compile_config,
                ac_config=SimpleNamespace(mode="none"),
                dump_folder="",
                device=torch.device(self.device_type),
                model_config=model.config,
                parallelize_fn=_identity_parallelize,
                loss_fn=build_mse_loss(compile_config),
            )
            assert has_first == (rank == 0)
            assert has_last == (rank == 1)
            assert len(model_parts) == 1

            x, kwargs, labels = _packed_sample()

            # Unsplit reference fwd/bwd with the same weights and inputs.
            x_ref = x.clone().requires_grad_(True)
            pred_ref = ref(x_ref, **kwargs)
            loss_ref = _mse_sum(pred_ref, labels)
            loss_ref.backward()
            ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

            # Trainer PP protocol: inputs only on the first stage, target
            # and the losses container only on the last stage; kwargs go
            # to every stage.
            targets, losses = (labels, []) if has_last else (None, None)
            with _cpu_fork_rng():
                if has_first:
                    schedule.step(
                        x,
                        **kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )
                else:
                    schedule.step(
                        **kwargs,
                        target=targets,
                        losses=losses,
                        return_outputs=False,
                    )

            if has_last:
                assert len(losses) == 1
                assert torch.allclose(
                    losses[0], loss_ref, atol=1e-4, rtol=1e-4
                )

            part_params = dict(model_parts[0].named_parameters())
            assert part_params, "stage part owns no parameters"
            for name, param in part_params.items():
                assert param.grad is not None, name
                assert torch.allclose(
                    param.grad, ref_grads[name], atol=1e-4, rtol=1e-4
                ), f"grad mismatch for {name} (rank {rank})"
