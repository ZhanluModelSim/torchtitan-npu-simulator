# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipeline parallelism for MAGI-2-preview.

Adapts torchtitan's ``pipeline_llm`` protocol (see the pinned torchtitan
``distributed/pipeline_parallel.py``) to the MAGI-2-preview layout:

- The transformer layers live at ``model.block.layers`` (a ModuleDict
  keyed ``"0"`` .. ``"N-1"``), not at a top-level ``model.layers``, and
  the input/output sides are ``model.pre_adapter`` / ``model.post_adapter``
  instead of ``tok_embeddings`` / ``norm`` / ``output``. Stage 0 owns
  ``pre_adapter`` + the first layers, the last stage owns the last
  layers + ``post_adapter``; module names per stage are therefore
  ``"pre_adapter"``, ``"block.layers.{i}"`` and ``"post_adapter"``.
- ``torch.distributed.pipelining`` forwards ``step()`` kwargs to every
  stage (only positional args are stage-0 activations), so the trainer's
  ``extra_inputs`` kwargs (``coords_mapping`` / ``modality_mapping`` /
  ``time_embedding`` / ``cu_seqlens``) reach all stages and each stage
  recomputes the per-token modality sort bookkeeping locally. The only
  inter-stage activations are the sorted-order stream ``h`` and the RoPE
  features ``rope`` (computed once by ``pre_adapter`` on stage 0 and
  passed through the pipeline, since every layer's attention needs it).
- Loss: the last stage's forward returns the ``(T, 64)`` prediction and
  torchtitan's PP protocol applies ``loss_fn(pred, target)`` there; the
  trainer's PP branch passes ``target=labels`` (full ``(T, 64)`` flow-
  matching labels), so the sum-MSE ``build_mse_loss`` function works
  unchanged and the trainer rescales by ``global_valid_tokens`` exactly
  as in the non-PP path.

v1 restrictions (asserted):
- A single microbatch (``local_batch_size ==
  pipeline_parallel_microbatch_size``): packed ``cu_seqlens`` /
  ``coords_mapping`` boundary tensors cannot be re-chunked across pack
  boundaries, so multi-microbatch PP needs pack-aware microbatching
  first. Consequently pp > 1 must use a schedule that runs with one
  microbatch (e.g. ``GPipe``): torch's 1F1B rejects ``n_microbatches <
  num_stages``.
- Combining PP with CP/TP/EP/ETP raises ``NotImplementedError`` in
  ``parallelize.parallelize_magi2_preview`` (called per stage part).
"""

import copy
import math
import types
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    get_schedule_class,
    PipelineScheduleSingle,
    ScheduleDualPipeV,
    ScheduleZBVZeroBubble,
)
from torchtitan.components.loss import LossFunction
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import pipeline_parallel as titan_pipeline_parallel
from torchtitan.protocols import ModelConvertersContainer
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.protocols.module import ModuleDict, ModuleList
from torchtitan.tools.logging import logger

from torchtitan_npu.distributed.process_group import is_fake_process_group

from .model import Modality

if TYPE_CHECKING:
    from .model import Magi2PreviewModel

__all__ = [
    "pipeline_magi2",
    "generate_magi2_fqn_per_model_part",
    "magi2_pipeline_module_split",
]


def generate_magi2_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """Programmatically generate module names per MAGI-2-preview stage.

    Mirrors torchtitan's ``generate_llm_fqn_per_model_part`` but with the
    MAGI-2-preview module names: ``pre_adapter`` replaces
    ``tok_embeddings`` and ``post_adapter`` replaces ``norm``/``output``;
    the transformer layers live at ``block.layers.{i}``.

    Args:
        num_stages: number of (virtual) pipeline stages.
        num_layers: total number of transformer layers.
        input_weight: ``pre_adapter`` weight in the layer distribution.
        output_weight: ``post_adapter`` weight in the layer distribution.

    Returns:
        List of lists of module names, one inner list per stage.
    """
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    layer_names = [f"block.layers.{i}" for i in range(num_layers)]
    if num_stages == 1:
        # Single stage gets everything.
        return [["pre_adapter"] + layer_names + ["post_adapter"]]

    # Calculate effective layers including weights.
    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than effective layers ({num_effective_layers})"
        )

    # Calculate layers per stage (distribute evenly).
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    # Feasibility check: ensure at least 1 layer in each PP stage.
    if layers_per_stage == 0:
        raise ValueError(
            f"Configuration would result in empty stages. "
            f"With {num_stages} stages and {num_effective_layers} effective layers "
            f"(num_layers={num_layers} + input_weight={input_weight} + output_weight={output_weight}), "
            f"each stage would get {layers_per_stage} layers on average. "
            f"Reduce num_stages or increase num_layers/weights."
        )

    # Balance check: ensure weights don't exceed minimum layers per stage.
    if input_weight > layers_per_stage:
        raise ValueError(
            f"input_weight ({input_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )
    if output_weight > layers_per_stage:
        raise ValueError(
            f"output_weight ({output_weight}) exceeds minimum layers per stage ({layers_per_stage})."
        )

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        # Calculate effective layers for this stage.
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        # First stage: pre_adapter with weighting.
        if stage_idx == 0:
            stage_modules.append("pre_adapter")
            remaining_layers_for_stage = effective_layers_for_stage - input_weight
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"block.layers.{current_layer}")
                    current_layer += 1

        # Last stage: post_adapter with weighting.
        elif stage_idx == num_stages - 1:
            remaining_layers_for_stage = effective_layers_for_stage - output_weight
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"block.layers.{current_layer}")
                    current_layer += 1
            stage_modules.append("post_adapter")

        # Middle stages: only transformer layers.
        else:
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"block.layers.{current_layer}")
                    current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def _stage_bookkeeping(
    modality_mapping: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-token modality sort bookkeeping shared by every stage.

    Mirrors the entry bookkeeping of ``Magi2PreviewModel.forward``:
    TIME->TEXT remap, stable modality sort and per-modality counts. Every
    stage recomputes it from the ``modality_mapping`` kwarg (torch
    pipelining forwards step kwargs to all stages), so no non-tensor
    bookkeeping crosses stage boundaries.
    """
    modality_mapping = modality_mapping.clone()
    modality_mapping[modality_mapping == Modality.TIME] = Modality.TEXT

    sort_idx = torch.argsort(modality_mapping)
    inv_sort_idx = torch.argsort(sort_idx)
    m_splits = [
        int(v) for v in torch.bincount(modality_mapping, minlength=3).tolist()
    ]
    video_idx = (modality_mapping == Modality.VIDEO).nonzero().flatten()
    audio_idx = (modality_mapping == Modality.AUDIO).nonzero().flatten()
    text_idx = (modality_mapping == Modality.TEXT).nonzero().flatten()
    return sort_idx, inv_sort_idx, m_splits, video_idx, audio_idx, text_idx


def _magi2_stage_forward(
    self: "Magi2PreviewModel",
    x: torch.Tensor,
    rope: torch.Tensor | None = None,
    *,
    coords_mapping: torch.Tensor | None = None,
    modality_mapping: torch.Tensor | None = None,
    time_embedding: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    """PP-aware forward bound onto each stage model part.

    Stage roles are detected from the pruned modules:

    - first stage (``pre_adapter`` kept): ``x`` is the packed input
      tokens; runs the adapter + sort + this stage's layers and returns
      ``(h, rope)`` with ``h`` the sorted-order stream (the only
      activations sent to the next stage; ``rope`` is passed through
      because every later stage's attention needs it too).
    - middle stages: ``x``/``rope`` are the received activations; run
      this stage's layers and forward ``(h, rope)`` unchanged.
    - last stage (``post_adapter`` kept): additionally un-sorts the
      stream and projects to the ``(T, 64)`` prediction, which the PP
      schedule feeds to the loss function with ``target=labels``.

    A single-stage split (num_stages == 1) keeps both adapters and
    reproduces the full model forward.
    """
    is_first = self.pre_adapter is not None
    is_last = self.post_adapter is not None

    if modality_mapping is None:
        raise ValueError(
            "magi2_preview PP stage forward requires the modality_mapping kwarg"
        )
    if is_first and coords_mapping is None:
        raise ValueError(
            "magi2_preview PP first stage forward requires the coords_mapping kwarg"
        )

    sort_idx, inv_sort_idx, m_splits, video_idx, audio_idx, text_idx = (
        _stage_bookkeeping(modality_mapping)
    )

    if is_first:
        x_emb, rope = self.pre_adapter(
            x, coords_mapping, video_idx, audio_idx, text_idx
        )
        if time_embedding is not None:
            x_emb[:, : self.time_channel_dim] = time_embedding.to(x_emb.dtype)
        # torch pipelining's backward wiring sends a gradient for every
        # received inter-stage input, so each activation must require
        # grad. rope is derived from the fixed Fourier bands buffer (no
        # grad_fn), so mark it a grad-requiring leaf; the gradient it
        # accumulates on stage 0 during backward is discarded (no
        # learnable parameter feeds rope).
        rope = rope.detach().requires_grad_(True)
        h = x_emb.index_select(0, sort_idx)
    else:
        h = x

    h = self.block(h, rope, sort_idx, inv_sort_idx, m_splits, cu_seqlens)

    if is_last:
        h = h.index_select(0, inv_sort_idx)
        return self.post_adapter(h, video_idx, audio_idx)
    return h, rope


def _prune_module(module: nn.Module, keep_names: set[str]) -> None:
    """Drop every submodule not listed in ``keep_names`` (in place).

    Mirrors torchtitan's ``pipeline_module_split`` pruning semantics
    (unlisted simple modules become None, unlisted ModuleDict/ModuleList
    containers become empty) but descends into nested containers so
    ``"block.layers.{i}"`` names reach the layer ModuleDict inside
    ``block`` (torch's pinned helper only handles top-level containers).
    """
    for child_name, child in list(module.named_children()):
        sub_keep = {
            name.split(".", 1)[1]
            for name in keep_names
            if name.startswith(f"{child_name}.")
        }
        if isinstance(child, (nn.ModuleDict, nn.ModuleList)):
            if sub_keep:
                if isinstance(child, nn.ModuleDict):
                    for layer_name in list(child.keys()):
                        if layer_name not in sub_keep:
                            del child[layer_name]
                else:
                    indices_to_keep = {
                        int(idx) for idx in sub_keep if idx.isdigit()
                    }
                    setattr(
                        module,
                        child_name,
                        ModuleList(
                            [
                                layer
                                for i, layer in enumerate(child)
                                if i in indices_to_keep
                            ]
                        ),
                    )
            elif isinstance(child, nn.ModuleDict):
                setattr(module, child_name, ModuleDict())
            else:
                setattr(module, child_name, ModuleList())
        elif sub_keep:
            _prune_module(child, sub_keep)
        elif child_name not in keep_names:
            layer_stack = getattr(child, "layers", None)
            if isinstance(layer_stack, (nn.ModuleDict, nn.ModuleList)):
                # Layer-stack container (MAGI-2's ``block``): keep it with
                # an empty layer list (the stage forward always calls it;
                # an empty stack is a no-op), same convention as the
                # pinned split's emptied top-level layer containers.
                _prune_module(child, set())
            else:
                # Replace with None; nn.Module keeps the attribute
                # readable as None (torchtitan's pipeline_llm convention).
                setattr(module, child_name, None)


def magi2_pipeline_module_split(
    whole_model: nn.Module,
    pp_mesh,
    pp_schedule: str,
    device: torch.device,
    module_names_per_stage: list[list[str]],
) -> tuple[list[PipelineStage], list[nn.Module]]:
    """Split a MAGI-2-preview model into pipeline stages by module names.

    Same protocol as torchtitan's ``pipeline_module_split`` (deepcopy the
    whole model per stage, prune the modules the stage does not own,
    build a ``PipelineStage`` over the pp mesh), extended with:

    - nested ``"block.layers.{i}"`` names (see ``_prune_module``);
    - the PP-aware stage forward bound onto every stage part, because the
      pruned model no longer matches ``Magi2PreviewModel.forward``;
    - fake process group support (all stages in one process), matching
      the simulator's patched ``pipeline_module_split`` behavior.
    """
    pp_rank = pp_mesh.get_local_rank()
    pp_degree = pp_mesh.size()
    pp_group = pp_mesh.get_group("pp")

    def _build_stage_from_modules(
        stage_idx: int, module_names: list[str], num_stages: int
    ) -> tuple[PipelineStage, nn.Module]:
        model = copy.deepcopy(whole_model)
        _prune_module(model, set(module_names))
        # Pruned stages no longer run the whole-model forward (missing
        # adapters / reduced layer set), so bind the stage-aware one.
        model.forward = types.MethodType(_magi2_stage_forward, model)

        stage = PipelineStage(
            model,
            stage_idx,
            num_stages,
            device,
            group=pp_group,
        )
        return stage, model

    num_stages = len(module_names_per_stage)
    stages = []
    models = []

    schedule_class = get_schedule_class(pp_schedule)
    style = (
        "v" if schedule_class in (ScheduleZBVZeroBubble, ScheduleDualPipeV) else "loop"
    )

    def _get_stage_indices() -> tuple[int, ...]:
        """Stage ids this pp rank runs (looped or V style schedules)."""
        assert (
            num_stages % pp_degree == 0
        ), f"num_stages {num_stages} must be evenly divisible by pp_degree {pp_degree}"
        # Under the simulator's fake process group every stage runs in
        # this one process so the capture sees all stages' ops.
        if is_fake_process_group(pp_group):
            return tuple(range(num_stages))
        stages_per_rank = num_stages // pp_degree
        if style == "loop":
            return tuple(pp_rank + s * pp_degree for s in range(stages_per_rank))
        elif style == "v":
            assert (
                stages_per_rank == 2
            ), f"v schedules assume 2 stages per rank, got {stages_per_rank}"
            stage_v_pairs = list(
                zip(
                    range(pp_degree),
                    range(num_stages - 1, pp_degree - 1, -1),
                    strict=True,
                )
            )
            return stage_v_pairs[pp_rank]
        else:
            raise ValueError(f"Unknown style {style}")

    for stage_idx in _get_stage_indices():
        module_names = module_names_per_stage[stage_idx]
        stage, model_chunk = _build_stage_from_modules(
            stage_idx,
            module_names,
            num_stages,
        )
        logger.info(
            f"PP rank {pp_rank} is building stage_idx {stage_idx} "
            f"with modules {module_names}"
        )
        stages.append(stage)
        models.append(model_chunk)

    return stages, models


def pipeline_magi2(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    device: torch.device,
    model_config: "Magi2PreviewModel.Config",
    parallelize_fn: ParallelizeFunction,
    loss_fn: LossFunction,
) -> tuple[_PipelineSchedule, list[nn.Module], bool, bool]:
    """MAGI-2-preview pipelining_fn (torchtitan ``pipeline_llm`` protocol).

    Splits ``model`` into stages over the pp mesh, applies the SPMD
    parallelisms per stage part through ``parallelize_fn`` and builds the
    pipeline schedule with the flow-matching loss on the last stage.
    See the module docstring for the stage dataflow and v1 restrictions.
    """
    pp_mesh = parallel_dims.get_mesh("pp")

    # Determine the number of virtual stages based on schedule type.
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    is_single_stage_schedule = issubclass(schedule_class, PipelineScheduleSingle)
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage
    num_layers = model_config.num_layers

    # Adapter weights in the layer distribution (pre/post adapter cost).
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers

    if layers_per_stage is not None:
        # Virtual stages needed (ceiling division); stages can differ by
        # at most 1 layer, mirroring pipeline_llm.
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )

        model_config_info = (
            f"Model has {num_layers} layers with "
            f"pipeline_parallel_layers_per_stage={layers_per_stage}"
        )
        stage_distribution_info = (
            f"resulting in {num_virtual_stages=} across {parallel_dims.pp} PP ranks"
        )

        if num_virtual_stages % parallel_dims.pp != 0:
            raise ValueError(
                f"Number of virtual stages ({num_virtual_stages}) must be divisible by "
                f"pipeline parallel size ({parallel_dims.pp}). "
                f"{model_config_info}. "
                f"Please adjust pipeline_parallel_layers_per_stage to a value that results in a number "
                f"of stages divisible by {parallel_dims.pp}."
            )

        stages_per_rank = num_virtual_stages // parallel_dims.pp

        if is_single_stage_schedule and stages_per_rank != 1:
            raise ValueError(
                f"Single stage schedule requires exactly 1 stage per rank, but got {stages_per_rank} "
                f"stages per rank. {model_config_info}, {stage_distribution_info}. "
                f"Please increase pipeline_parallel_layers_per_stage to "
                f"{num_layers // parallel_dims.pp} or higher to achieve 1 stage per rank."
            )

        if not is_single_stage_schedule and stages_per_rank < 2:
            raise ValueError(
                f"Multi-stage schedule requires at least 2 stages per rank, but got {stages_per_rank} "
                f"stages per rank. {model_config_info}, {stage_distribution_info}. "
                f"Please decrease pipeline_parallel_layers_per_stage to achieve at least 2 stages "
                f"per rank."
            )
    else:
        # Default: one virtual stage per rank for single-stage schedules,
        # two for looped schedules (same as pipeline_llm).
        stages_per_rank = 1 if is_single_stage_schedule else 2
        num_virtual_stages = parallel_dims.pp * stages_per_rank
        if num_layers % parallel_dims.pp != 0:
            raise ValueError(
                f"MAGI-2-preview PP requires pipeline_parallel_degree "
                f"({parallel_dims.pp}) to divide num_layers ({num_layers})"
            )

    # v1 microbatch restriction: the packed cu_seqlens/coords_mapping
    # boundary tensors cannot be re-chunked across pack boundaries, so PP
    # runs a single microbatch (one whole pack) per step.
    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    if (
        training.local_batch_size % microbatch_size != 0
        or training.local_batch_size // microbatch_size != 1
    ):
        raise ValueError(
            f"MAGI-2-preview PP v1 runs exactly one microbatch per step: "
            f"packed cu_seqlens/coords_mapping cannot be re-chunked across pack "
            f"boundaries. Set pipeline_parallel_microbatch_size "
            f"({microbatch_size}) equal to local_batch_size "
            f"({training.local_batch_size})."
        )

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_magi2_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )
    for i, stage_ms in enumerate(module_names_per_stage):
        logger.debug(f"Stage {i}: {stage_ms}")

    stages, model_parts = magi2_pipeline_module_split(
        model,
        pp_mesh,
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )

    # For PP with looped schedules, each item in model_parts is one
    # stage-model-chunk: apply the SPMD parallelisms (and compilation)
    # per chunk, mirroring pipeline_llm.
    for i, m in enumerate(model_parts):
        m = parallelize_fn(
            m,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        model_parts[i] = m
        # Update the model in the stage in case parallelize modified it.
        stages[i].submod = m

    # Attribute lookup (not a from-import) so the simulator's fake-PG
    # patch of torchtitan.distributed.pipeline_parallel applies here too.
    pp_schedule = titan_pipeline_parallel.build_pipeline_schedule(
        parallelism=parallelism,
        local_batch_size=training.local_batch_size,
        stages=stages,
        loss_fn=loss_fn,
    )

    # The train loop uses these to decide whether to pass inputs/labels.
    has_first_stage = False
    has_last_stage = False
    for stage in stages:
        if stage.is_first:
            has_first_stage = True
        if stage.is_last:
            has_last_stage = True

    return pp_schedule, model_parts, has_first_stage, has_last_stage
