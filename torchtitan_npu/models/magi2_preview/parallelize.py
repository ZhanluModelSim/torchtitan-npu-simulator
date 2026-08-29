# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed parallelization for MAGI-2-preview."""

import logging

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    ShardPlacementResult,
)
from torch.distributed.tensor import (
    DTensor,
    Shard,
    distribute_module,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import SequenceParallel
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.models.llama3.parallelize import disable_fsdp_gradient_division
from torchtitan.protocols import ModelConvertersContainer

from torchtitan_npu.models.common.activation_checkpoint import apply_moe_ac

from .attention import Magi2Attention
from .cp_ulysses import apply_magi2_ulysses_cp
from .expert_parallel import (
    EXPERT_PARAM_NAMES,
    ROUTER_BUFFER_NAMES,
    MoEDispatchContext,
    all_reduce_head_parallel_input_grad,
    all_reduce_head_parallel_output,
    flatten_head_mesh,
    head_range_for_rank,
    shard_moe_core_by_head,
)
from .feed_forward import CoreMultiHeadMoE, Magi2MLP, MultiHeadMoELayer
from .grouped_linear import (
    GroupedLinear,
    slice_grouped_linear_by_heads,
    slice_grouped_linear_by_pairs,
)
from .model import Magi2PreviewModel

logger = logging.getLogger(__name__)


def _apply_fsdp(
    model: Magi2PreviewModel,
    dp_mesh: DeviceMesh,
    *,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    pp_enabled: bool,
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
) -> None:
    """Shard every transformer layer with FSDP2, then the root model.

    The upstream llama4 ``apply_fsdp`` hardcodes ``tok_embeddings``/``norm``/
    ``output``/``layers`` attributes that MAGI-2-preview does not have, so
    ``fully_shard`` is applied directly here instead. Under PP this runs
    per stage model part (``pipeline_magi2`` calls the parallelize fn on
    each part), so ``model`` may be a pruned stage chunk.

    With ``ep_degree > 1`` (eFSDP) the routed MoE expert params are already
    DTensor ``Shard(0)`` on the head-parallel ep mesh (see
    ``_apply_moe_parallel``); mirroring llama4's ``apply_fsdp``, a
    ``shard_placement_fn`` routes those params to ``edp_mesh`` (the dp mesh
    the experts shard over) while every other param uses ``dp_mesh``. FSDP2
    then composes the dp shard with the existing ep head shard into a
    multi-dim DTensor, so experts shard over dp AND keep their ep head
    shard. Gradient division is disabled below exactly like llama4 (the
    training loop scales by the global valid-token count).
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if training.enable_cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # "default" keeps params unsharded across the PP forward bubble
    # (reshard_after_forward=False) and otherwise reshards after forward.
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, pp_enabled=pp_enabled
    )

    for layer in model.block.layers.values():
        moe_mlp = (
            layer.mlp.moe_mlp
            if isinstance(layer.mlp, MultiHeadMoELayer)
            else None
        )
        if moe_mlp is not None and ep_degree > 1:
            if edp_mesh is None:
                raise ValueError(
                    "MAGI-2-preview eFSDP (ep_degree > 1) requires an "
                    "edp_mesh to shard the routed experts over"
                )
            expert_params = set(moe_mlp.parameters())
            edp_mesh_info = FSDPMeshInfo(mesh=edp_mesh, shard_mesh_dim=0)
            dp_mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)

            def _shard_placement_fn(
                param: nn.Parameter,
                _expert_params: set = expert_params,
                _edp_info: FSDPMeshInfo = edp_mesh_info,
                _dp_info: FSDPMeshInfo = dp_mesh_info,
            ):
                if param in _expert_params:
                    return ShardPlacementResult(
                        placement=Shard(0), mesh_info=_edp_info
                    )
                return ShardPlacementResult(
                    placement=Shard(0), mesh_info=_dp_info
                )

            fully_shard(
                layer,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_shard_placement_fn,
            )
        else:
            fully_shard(
                layer,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division; the training loop scales
    # the loss by the global valid-token count instead.
    disable_fsdp_gradient_division(model)


def _apply_replicate(
    model: Magi2PreviewModel,
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
) -> None:
    """Replicate the layers across dp_replicate ranks (no FSDP sharding)."""
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype
    )
    replicate_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    for layer in model.block.layers.values():
        replicate(layer, **replicate_config)
    replicate(model, **replicate_config)
    disable_fsdp_gradient_division(model)


def _partition_head_parallel_moe(
    name: str, module: nn.Module, device_mesh: DeviceMesh
) -> None:
    """Shard the fused expert tensors as DTensors along the leading dim.

    Expert rows are head-major (global expert id ``h * E + e``), so
    ``Shard(0)`` over a mesh whose degree divides ``moe_num_heads`` splits
    exactly at head boundaries (asserted in :func:`_apply_moe_parallel`);
    each rank's local shard holds whole heads. The router buffers share
    the same leading ``H * E`` dim and are sharded identically.
    """
    del name
    # distribute_module visits every submodule; only the MoE core carries
    # the fused expert tensors.
    if not isinstance(module, CoreMultiHeadMoE):
        return
    moe_mlp = module
    for param_name in EXPERT_PARAM_NAMES:
        param = getattr(moe_mlp, param_name)
        moe_mlp.register_parameter(
            param_name,
            nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])),
        )
    for buffer_name in ROUTER_BUFFER_NAMES:
        buffer = getattr(moe_mlp.router, buffer_name)
        moe_mlp.router.register_buffer(
            buffer_name, distribute_tensor(buffer, device_mesh, [Shard(0)])
        )


def _prepare_head_parallel_input(
    moe_mlp: CoreMultiHeadMoE,
    inputs: tuple,
    device_mesh: DeviceMesh,
) -> tuple:
    """Regime-(a) input handling for the replicated token stream.

    The MoE input is replicated across the expert mesh and every rank only
    back-propagates its own heads, so the input gradient must be summed
    across the mesh before it reaches the (replicated) modules upstream of
    the MoE. The forward passes the input through untouched.
    """
    del moe_mlp
    x = all_reduce_head_parallel_input_grad(inputs[0], device_mesh.get_group())
    return (x,) + tuple(inputs[1:])


def _combine_head_parallel_output(
    moe_mlp: CoreMultiHeadMoE, output: torch.Tensor, device_mesh: DeviceMesh
) -> torch.Tensor:
    """Regime-(a) assembly of the local-head partial MoE output."""
    return all_reduce_head_parallel_output(
        output, moe_mlp.head_range, moe_mlp.num_heads, device_mesh.get_group()
    )


def _register_regime_b_expert_grad_compensation(
    moe_mlp: CoreMultiHeadMoE, scale: float
) -> None:
    """Restore unsharded expert-gradient scale under regime (b).

    The combined-mesh dispatch delivers every rank's token shard once per
    mesh member, so when EP peers replicate the CP token shard
    (``degree > cp_degree``) the routed core computes each token
    ``degree / cp_degree`` times and the expert parameter gradients
    accumulate that factor; scale them back down to the unsharded value.
    A no-op when the mesh spans only the CP axis (``scale == 1``).
    """
    if scale == 1.0:
        return

    def hook(param):
        if param.grad is None:
            return
        grad = param.grad
        if isinstance(grad, DTensor):
            grad = grad.to_local()
        grad.mul_(scale)

    for param_name in EXPERT_PARAM_NAMES:
        moe_mlp.get_parameter(param_name).register_post_accumulate_grad_hook(
            hook
        )


def _wire_moe_regime_b_layer(
    mlp: MultiHeadMoELayer, *, mesh: DeviceMesh, cp_degree: int
) -> None:
    """Regime-(b) state of one MoE layer on the combined cp x ep mesh.

    Sets the routed core's local head range (sharded-input mode: the
    dispatched tensor already carries only the local head columns),
    installs the layer's dispatch context and registers the expert
    gradient compensation. The expert params/buffers must be sharded
    separately (``distribute_module`` in production, the plain
    ``shard_moe_core_by_head`` slice in collective-free tests).
    """
    degree = mesh.size()
    moe_mlp = mlp.moe_mlp
    head_range = head_range_for_rank(
        mesh.get_local_rank(), degree, moe_mlp.num_heads
    )
    moe_mlp.set_head_range(head_range, sharded_input=True)
    mlp.moe_dispatch_context = MoEDispatchContext(
        mesh=mesh, head_range=head_range
    )
    _register_regime_b_expert_grad_compensation(moe_mlp, cp_degree / degree)


def _apply_moe_parallel(
    model: Magi2PreviewModel,
    *,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    cp_degree: int = 1,
) -> None:
    """Shard the routed MoE across the head axis on the EP (or ETP) mesh.

    MAGI-2 routes per head, so expert parallelism is head-parallel: every
    rank owns ``moe_num_heads / degree`` whole heads of each MoE layer's
    fused expert tensors. Two communication regimes exist (see
    ``expert_parallel.py``):

    * Regime (a) — tokens replicated (no CP, or an ETP mesh beside CP):
      the zero-padded local-head partial outputs are all-reduced over the
      mesh.
    * Regime (b) — CP and EP combined (``cp_degree > 1`` with an EP
      mesh): the mesh must be the FLATTENED cp x ep head mesh
      (``expert_parallel.flatten_head_mesh``); each layer runs its routed
      core between the official seq<->head dispatch/undispatch
      (``MoEDispatchContext``), computing its local heads over the
      received sequence while ``split_linear``/``merge_linear``/shared
      experts stay per-token local on the ``(T/cp, C)`` shard.

    State-dict keys never change: the expert tensors become ``Shard(0)``
    DTensors with unchanged global shapes, so loading a full checkpoint
    still distributes through DTensor.
    """
    if ep_mesh is None and etp_mesh is None:
        return
    if ep_mesh is not None and etp_mesh is not None:
        raise NotImplementedError(
            "MAGI-2-preview combined EP+ETP would shard the head axis "
            "across two meshes; combining them (with or without TP) is not "
            "supported in v1"
        )
    mesh = ep_mesh if ep_mesh is not None else etp_mesh
    if mesh.ndim != 1:
        raise ValueError(
            f"MAGI-2 head-parallel MoE expects a 1D mesh, got {mesh.ndim}D"
        )
    degree = mesh.size()
    regime_b = cp_degree > 1 and ep_mesh is not None
    if regime_b and degree % cp_degree != 0:
        raise ValueError(
            "MAGI-2-preview regime-(b) MoE dispatch expects the flattened "
            f"cp x ep head mesh (degree divisible by the CP degree "
            f"{cp_degree}), got degree {degree}"
        )
    for layer in model.block.layers.values():
        if not isinstance(layer.mlp, MultiHeadMoELayer):
            continue
        moe_mlp = layer.mlp.moe_mlp
        if moe_mlp.num_heads % degree != 0:
            raise ValueError(
                f"moe_num_heads={moe_mlp.num_heads} must be divisible by "
                f"the head-parallel degree={degree}"
            )
        if regime_b:
            distribute_module(
                moe_mlp,
                mesh,
                partition_fn=_partition_head_parallel_moe,
            )
            _wire_moe_regime_b_layer(
                layer.mlp, mesh=mesh, cp_degree=cp_degree
            )
        else:
            moe_mlp.set_head_range(
                head_range_for_rank(
                    mesh.get_local_rank(), degree, moe_mlp.num_heads
                )
            )
            distribute_module(
                moe_mlp,
                mesh,
                partition_fn=_partition_head_parallel_moe,
                input_fn=_prepare_head_parallel_input,
                output_fn=_combine_head_parallel_output,
            )
    if regime_b:
        logger.info(
            "Applied MAGI-2-preview head-parallel MoE (regime b, cp x ep "
            f"degree {degree})"
        )
    else:
        logger.info("Applied MAGI-2-preview head-parallel MoE (regime a)")


# ---------------------------------------------------------------------------
# Tensor parallelism (v1 scope: sequence replicated, no sequence parallel)
# ---------------------------------------------------------------------------


def _tp_require_divisible(value: int, tp_degree: int, what: str) -> None:
    if value % tp_degree != 0:
        raise ValueError(
            f"MAGI-2-preview TP requires {what}={value} to be divisible by "
            f"the TP degree={tp_degree} (a heads < tp fallback is not "
            f"implemented in v1)"
        )


def _tp_local_param(
    local: torch.Tensor, mesh: DeviceMesh, placement: Shard
) -> nn.Parameter:
    """Wrap an already rank-local slice as a DTensor parameter.

    ``DTensor.from_local`` is communication-free: every rank enters the
    partition holding the same full weights and slices them identically, so
    the placement only records the sharding for state-dict redistribution
    (loading a full checkpoint still distributes through DTensor). The
    input is a ``.data`` slice, so ``requires_grad`` is re-enabled here.
    """
    wrapped = DTensor.from_local(local, mesh, [placement], run_check=False)
    return nn.Parameter(wrapped)


def _out_shard_placement(weight: torch.Tensor) -> Shard:
    """Out-dim shard placement for a column-split GroupedLinear weight.

    The out dim is dim 0 of a single-expert 2D ``(out, in)`` weight and
    dim 1 of a multi-expert 3D ``(num_experts, out, in)`` weight (see
    ``grouped_linear.py``); in both cases the rank-local shard is a
    contiguous out-dim range, so a single ``Shard`` expresses it.
    """
    return Shard(0) if weight.ndim == 2 else Shard(1)


def _in_shard_placement(weight: torch.Tensor) -> Shard:
    """In-dim shard placement for a row-split GroupedLinear weight.

    The in dim is dim 1 of a single-expert 2D ``(out, in)`` weight and
    dim 2 of a multi-expert 3D ``(num_experts, out, in)`` weight.
    """
    return Shard(1) if weight.ndim == 2 else Shard(2)


def _shard_grouped_linear_columns(
    linear: GroupedLinear, row_start: int, row_end: int
) -> None:
    """Contiguous out-dim (column) split of a single-expert GroupedLinear."""
    weight = linear.weight.data
    linear.register_parameter(
        "weight", nn.Parameter(weight[row_start:row_end].contiguous())
    )
    linear.out_features = row_end - row_start


def _shard_grouped_linear_rows(
    linear: GroupedLinear, col_start: int, col_end: int
) -> None:
    """Input-dim (row) split of a GroupedLinear weight.

    The same input-column range is kept for every modality expert; the
    output is a partial sum over the mesh that the module-boundary hook
    all-reduces. Handles the single-expert 2D ``(out, in)`` layout and the
    multi-expert 3D ``(num_experts, out, in)`` layout alike.
    """
    weight = linear.weight.data
    if weight.ndim == 3:
        local = weight[:, :, col_start:col_end]
    else:
        local = weight[:, col_start:col_end]
    linear.register_parameter(
        "weight", nn.Parameter(local.contiguous())
    )
    linear.in_features = col_end - col_start


def _shard_grouped_linear_heads(
    linear: GroupedLinear,
    num_heads: int,
    head_dim: int,
    num_sections: int,
    head_range: tuple[int, int],
) -> None:
    """Head-wise out-dim split of a GroupedLinear weight (per expert)."""
    sliced = slice_grouped_linear_by_heads(
        linear.weight.data,
        linear.num_experts,
        num_heads,
        head_dim,
        num_sections,
        head_range,
    )
    linear.register_parameter("weight", nn.Parameter(sliced))
    head_start, head_end = head_range
    linear.out_features = num_sections * (head_end - head_start) * head_dim


def _shard_grouped_linear_pairs(
    linear: GroupedLinear, num_pairs: int, pair_range: tuple[int, int]
) -> None:
    """swiglu7 pair-preserving out-dim split of a GroupedLinear weight."""
    sliced = slice_grouped_linear_by_pairs(
        linear.weight.data, linear.num_experts, num_pairs, pair_range
    )
    linear.register_parameter("weight", nn.Parameter(sliced))
    pair_start, pair_end = pair_range
    linear.out_features = 2 * (pair_end - pair_start)


def _shard_attention_tp(
    attention: Magi2Attention, tp_degree: int, tp_rank: int
) -> None:
    """Slice one Magi2Attention to its TP rank's heads.

    ``linear_g`` (one out row per head) and ``linear_qkv`` (head-major
    q/k/v per head) are column-split per modality expert on the head axis,
    ``linear_proj`` is row-split on the head-concatenated input dim
    (partial output), ``sinks`` is sliced on its head dim, and
    ``num_heads`` becomes the rank-local count; the replicated pre/q/k
    norms are unchanged. DTensor wrapping and the boundary hooks are added
    by the wiring step.
    """
    num_heads = attention.num_heads
    _tp_require_divisible(num_heads, tp_degree, "num attention heads")
    head_range = head_range_for_rank(tp_rank, tp_degree, num_heads)
    head_start, head_end = head_range
    head_dim = attention.head_dim

    _shard_grouped_linear_heads(attention.linear_g, num_heads, 1, 1, head_range)
    _shard_grouped_linear_heads(
        attention.linear_qkv, num_heads, head_dim, 3, head_range
    )
    _shard_grouped_linear_rows(
        attention.linear_proj, head_start * head_dim, head_end * head_dim
    )
    attention.sinks = nn.Parameter(
        attention.sinks.data[:, head_start:head_end].contiguous()
    )
    attention.num_heads = head_end - head_start


def _shard_dense_mlp_tp(mlp: Magi2MLP, tp_degree: int, tp_rank: int) -> None:
    """Column-split ``up_gate_proj`` at swiglu7 pair granularity.

    The gate/up pairs stay together on every rank (even row offsets and
    even local widths), so the interleaved swiglu7 of the local output
    pairs exactly the same way as the unsharded one; ``down_proj`` is the
    conjugate row split.
    """
    num_pairs = mlp.up_gate_proj.out_features // 2
    _tp_require_divisible(num_pairs, tp_degree, "dense intermediate size")
    pairs_per_rank = num_pairs // tp_degree
    pair_range = (tp_rank * pairs_per_rank, (tp_rank + 1) * pairs_per_rank)
    _shard_grouped_linear_pairs(mlp.up_gate_proj, num_pairs, pair_range)
    _shard_grouped_linear_rows(mlp.down_proj, pair_range[0], pair_range[1])


def _shard_moe_layer_tp(
    mlp: MultiHeadMoELayer, tp_degree: int, tp_rank: int
) -> None:
    """Slice one MoE layer: head-sharded routed core + TP-split shared path.

    ``split_linear`` is column-split so its local output columns are
    exactly the routed core's local head columns (``moe_num_heads``
    divisible by the TP degree, asserted here), and ``merge_linear`` is
    the conjugate row split consuming the core's partial output. The
    shared experts split like the dense MLP (fc1 pair-preserving column
    split, fc2 row split). One module-boundary all-reduce combines the
    routed and shared partial outputs.
    """
    hidden = mlp.split_linear.out_features
    _tp_require_divisible(hidden, tp_degree, "hidden size")
    col_width = hidden // tp_degree
    _shard_grouped_linear_columns(
        mlp.split_linear, tp_rank * col_width, (tp_rank + 1) * col_width
    )
    _shard_grouped_linear_rows(
        mlp.merge_linear, tp_rank * col_width, (tp_rank + 1) * col_width
    )

    moe = mlp.moe_mlp
    _tp_require_divisible(moe.num_heads, tp_degree, "moe_num_heads")
    head_range = head_range_for_rank(tp_rank, tp_degree, moe.num_heads)
    shard_moe_core_by_head(moe, head_range)
    # TP regime: the column-split split_linear already emits only this
    # rank's head columns, so the core views its input directly.
    moe.set_head_range(head_range, sharded_input=True)

    num_pairs = mlp._shared_intermediate_size
    _tp_require_divisible(
        num_pairs, tp_degree, "shared expert intermediate size"
    )
    pairs_per_rank = num_pairs // tp_degree
    pair_range = (tp_rank * pairs_per_rank, (tp_rank + 1) * pairs_per_rank)
    _shard_grouped_linear_pairs(mlp.shared_expert_fc1, num_pairs, pair_range)
    _shard_grouped_linear_rows(mlp.shared_expert_fc2, pair_range[0], pair_range[1])
    _shard_grouped_linear_pairs(
        mlp.modality_specific_shared_expert_fc1, num_pairs, pair_range
    )
    _shard_grouped_linear_rows(
        mlp.modality_specific_shared_expert_fc2, pair_range[0], pair_range[1]
    )
    mlp._shared_intermediate_size = pairs_per_rank


def _wrap_attention_tp(attention: Magi2Attention, mesh: DeviceMesh) -> None:
    """Record honest DTensor placements over the TP-sliced attention weights.

    Every attention weight becomes a single-placement DTensor: ``linear_g``
    and ``linear_qkv`` shard the out dim (head-major, so each rank's heads
    are a contiguous out range), ``linear_proj`` shards the in dim, and
    ``sinks`` shards its head dim. Multi-expert weights use the 3D
    ``(num_experts, out, in)`` layout, single-expert weights the 2D
    ``(out, in)`` layout (see ``grouped_linear.py``); the shard dim
    follows accordingly.
    """
    attention.linear_g.weight = _tp_local_param(
        attention.linear_g.weight.data,
        mesh,
        _out_shard_placement(attention.linear_g.weight),
    )
    attention.linear_qkv.weight = _tp_local_param(
        attention.linear_qkv.weight.data,
        mesh,
        _out_shard_placement(attention.linear_qkv.weight),
    )
    attention.linear_proj.weight = _tp_local_param(
        attention.linear_proj.weight.data,
        mesh,
        _in_shard_placement(attention.linear_proj.weight),
    )
    attention.sinks = _tp_local_param(attention.sinks.data, mesh, Shard(1))


def _wrap_dense_mlp_tp(mlp: Magi2MLP, mesh: DeviceMesh) -> None:
    """DTensor placements over the TP-sliced dense MLP weights.

    ``up_gate_proj`` shards the out dim at swiglu7 pair granularity and
    ``down_proj`` the in dim; both are honest single placements on the
    per-expert layout (see ``grouped_linear.py``).
    """
    mlp.up_gate_proj.weight = _tp_local_param(
        mlp.up_gate_proj.weight.data,
        mesh,
        _out_shard_placement(mlp.up_gate_proj.weight),
    )
    mlp.down_proj.weight = _tp_local_param(
        mlp.down_proj.weight.data,
        mesh,
        _in_shard_placement(mlp.down_proj.weight),
    )


def _wrap_moe_layer_tp(mlp: MultiHeadMoELayer, mesh: DeviceMesh) -> None:
    """DTensor placements over the TP-sliced MoE layer weights.

    The routed core reuses the head-parallel ``Shard(0)`` sharding of
    ``_apply_moe_parallel`` (head-major rows divide at head boundaries).
    The shared experts use the grouped placements: ``fc1`` shards the out
    dim at pair granularity and ``fc2`` the in dim, honest single
    placements on the per-expert layout for the E=3 modality-specific
    weights as well as the single-expert shared/split/merge weights.
    """
    moe = mlp.moe_mlp
    for param_name in EXPERT_PARAM_NAMES:
        param = getattr(moe, param_name)
        moe.register_parameter(
            param_name, _tp_local_param(param.data, mesh, Shard(0))
        )
    for buffer_name in ROUTER_BUFFER_NAMES:
        buffer = getattr(moe.router, buffer_name)
        moe.router.register_buffer(
            buffer_name,
            DTensor.from_local(buffer, mesh, [Shard(0)], run_check=False),
        )
    mlp.split_linear.weight = _tp_local_param(
        mlp.split_linear.weight.data, mesh, Shard(0)
    )
    mlp.merge_linear.weight = _tp_local_param(
        mlp.merge_linear.weight.data, mesh, Shard(1)
    )
    mlp.shared_expert_fc1.weight = _tp_local_param(
        mlp.shared_expert_fc1.weight.data, mesh, Shard(0)
    )
    mlp.shared_expert_fc2.weight = _tp_local_param(
        mlp.shared_expert_fc2.weight.data, mesh, Shard(1)
    )
    mlp.modality_specific_shared_expert_fc1.weight = _tp_local_param(
        mlp.modality_specific_shared_expert_fc1.weight.data,
        mesh,
        _out_shard_placement(mlp.modality_specific_shared_expert_fc1.weight),
    )
    mlp.modality_specific_shared_expert_fc2.weight = _tp_local_param(
        mlp.modality_specific_shared_expert_fc2.weight.data,
        mesh,
        _in_shard_placement(mlp.modality_specific_shared_expert_fc2.weight),
    )


class _AllReduceTpOutput(torch.autograd.Function):
    """All-reduce forward (sum of rank-local partial outputs); identity backward.

    With the sequence replicated every rank computes the SAME downstream
    loss from the reduced output, so ``dL/d(partial_r) = dL/dy`` and the
    gradient passes through unchanged. Re-reducing it in backward (the
    global-loss semantics of ``funcol.all_reduce``'s autograd) would
    over-count by the mesh degree — the same rationale as
    ``expert_parallel.all_reduce_head_parallel_output``'s slice backward.
    """

    @staticmethod
    def forward(ctx, x, group):
        del ctx
        out = x.contiguous().clone()
        dist.all_reduce(out, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def _register_tp_boundary_hooks(module: nn.Module, group) -> None:
    """All-reduce a TP-sharded sublayer's partial output and input gradient.

    v1 keeps the sequence replicated, so the module input is identical on
    every rank and its gradient is the sum of the rank-local (column- or
    head-sharded) backward contributions; the output is the partial sum of
    the row-split projections. The pre-hook is the gradient-only conjugate
    of the post-hook's forward all-reduce (see ``expert_parallel``'s
    regime-(a) primitives for the same pair).
    """

    def pre_hook(mod, inputs):
        del mod
        x = all_reduce_head_parallel_input_grad(inputs[0], group)
        return (x,) + tuple(inputs[1:])

    def post_hook(mod, inputs, output):
        del mod, inputs
        return _AllReduceTpOutput.apply(output, group)

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)


def _register_tp_replicated_grad_reduce(param: nn.Parameter, group) -> None:
    """Complete a TP-replicated parameter's gradient across the tp mesh.

    Replicated parameters inside TP-sharded sublayers (the attention
    pre/q/k norms and the MLP pre_norms) only see the rank-local backward
    paths, so their accumulated gradient is partial; the hook all-reduces
    it. It is registered before ``fully_shard`` runs, so it fires on the
    full-size gradient ahead of FSDP's dp reduction.
    """

    def hook(param):
        if param.grad is None:
            return
        grad = param.grad
        if isinstance(grad, DTensor):
            grad = grad.to_local()
        dist.all_reduce(grad, group=group)

    param.register_post_accumulate_grad_hook(hook)


def _apply_tensor_parallel(
    model: Magi2PreviewModel, 
    *, 
    tp_mesh: DeviceMesh,
    sequence_parallel: bool = False
) -> None:
    """Shard attention heads and MLP intermediates across the tp mesh.

    When sequence_parallel=False (default): the sequence stays REPLICATED on 
    every TP rank. Column-split projections carry replicated inputs and every 
    sharded sublayer all-reduces its partial output at the module boundary 
    (and, conjugately, its input gradient); state-dict keys never change.

    When sequence_parallel=True: the sequence is sharded across the TP mesh
    (kimi-style). Norms (pre_norm, q_norm, k_norm, mhp_norm) operate on local
    sequence shards and their weights remain replicated (no sharding). Linear
    layers receive sequence-sharded inputs.

    Per-rank placements after the partition (all honest single-placement
    DTensors; multi-expert weights use the 3D ``(num_experts, out, in)``
    layout, single-expert weights the 2D ``(out, in)`` layout, and the
    shard dim follows):
    - attention: ``linear_g``/``linear_qkv`` shard the out dim on the head
      axis, ``linear_proj`` the in dim, ``sinks`` ``Shard(1)`` on the head
      dim, q/k norms replicated unchanged;
    - dense MLP: ``up_gate_proj`` shards the out dim at swiglu7 pair
      granularity, ``down_proj`` the in dim;
    - MoE layer: ``split_linear`` ``Shard(0)``, routed core head-sharded
      ``Shard(0)`` like ``_apply_moe_parallel`` (regime (a) compute; the
      TP communication moves to the ``merge_linear`` row split, which is
      ``Shard(1)``), shared/modality shared experts fc1 shard the out dim
      at pair granularity / fc2 the in dim.

    Divisibility (asserted with clear errors): attention heads,
    moe_num_heads, hidden size, dense intermediate and shared expert
    intermediate all divide the TP degree.
    """
    if tp_mesh.ndim != 1:
        raise ValueError(
            f"MAGI-2 TP expects a 1D mesh, got {tp_mesh.ndim}D"
        )
    tp_degree = tp_mesh.size()
    tp_rank = tp_mesh.get_local_rank()
    group = tp_mesh.get_group()

    # Validate every layer before touching any weight, so a divisibility
    # error never leaves the model partially sharded.
    for layer in model.block.layers.values():
        _tp_require_divisible(
            layer.attention.num_heads, tp_degree, "num attention heads"
        )
        if isinstance(layer.mlp, MultiHeadMoELayer):
            _tp_require_divisible(
                layer.mlp.split_linear.out_features, tp_degree, "hidden size"
            )
            _tp_require_divisible(
                layer.mlp.moe_mlp.num_heads, tp_degree, "moe_num_heads"
            )
            _tp_require_divisible(
                layer.mlp._shared_intermediate_size,
                tp_degree,
                "shared expert intermediate size",
            )
        else:
            _tp_require_divisible(
                layer.mlp.up_gate_proj.out_features // 2,
                tp_degree,
                "dense intermediate size",
            )

    for layer in model.block.layers.values():
        attention = layer.attention
        _shard_attention_tp(attention, tp_degree, tp_rank)
        _wrap_attention_tp(attention, tp_mesh)
        _register_tp_boundary_hooks(attention, group)
        
        # Apply sequence parallel to norms or register for replicated grad reduce
        if sequence_parallel:
            # SequenceParallel shards sequence dim on TP mesh, weights stay replicated
            for norm_name in ("pre_norm", "q_norm", "k_norm"):
                norm = getattr(attention, norm_name, None)
                if norm is not None:
                    parallelize_module(norm, tp_mesh, SequenceParallel())
        else:
            for replicated in (
                attention.pre_norm.weight,
                attention.q_norm.weight,
                attention.k_norm.weight,
            ):
                _register_tp_replicated_grad_reduce(replicated, group)

        if isinstance(layer.mlp, MultiHeadMoELayer):
            _shard_moe_layer_tp(layer.mlp, tp_degree, tp_rank)
            _wrap_moe_layer_tp(layer.mlp, tp_mesh)
        else:
            _shard_dense_mlp_tp(layer.mlp, tp_degree, tp_rank)
            _wrap_dense_mlp_tp(layer.mlp, tp_mesh)
        _register_tp_boundary_hooks(layer.mlp, group)
        
        # Apply sequence parallel to MLP pre_norm or register for replicated grad reduce
        if sequence_parallel:
            parallelize_module(layer.mlp.pre_norm, tp_mesh, SequenceParallel())
        else:
            _register_tp_replicated_grad_reduce(layer.mlp.pre_norm.weight, group)
        
        # Apply sequence parallel to mhc_norm if present
        if sequence_parallel and hasattr(layer, "mhc_norm"):
            parallelize_module(layer.mhc_norm, tp_mesh, SequenceParallel())

    logger.info(
        "Applied MAGI-2-preview tensor parallelism (sequence %s)" %
        ("parallel" if sequence_parallel else "replicated")
    )


def parallelize_magi2_preview(
    model: Magi2PreviewModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply TP, Ulysses CP, head-parallel MoE (EP/ETP), AC and FSDP.

    TP (v1) splits attention heads and MLP intermediates across the tp
    mesh with the sequence REPLICATED (no sequence parallel yet): every
    sharded sublayer all-reduces its partial output at the module
    boundary; see ``_apply_tensor_parallel``. Combining TP with CP (v1:
    replicated vs sequence-sharded streams) or with EP/ETP (both shard the
    MoE head axis on different meshes) raises.
    With PP enabled, ``pipeline_magi2`` splits the model into stage parts
    and calls this function on each part, so the MoE/AC/FSDP path below
    runs per stage chunk; v1 supports PP combined with FSDP/DP only
    (combining PP with CP/TP/EP/ETP raises).
    CP shards the packed sequence in original token order across the cp
    mesh (Ulysses head-split all-to-all inside attention, autograd exit
    gather with gradient compensation; see ``cp_ulysses.py``). EP shards
    the routed MoE along the head axis: regime (a) (replicated tokens +
    all-reduce) without CP, and regime (b) when CP and EP are combined —
    the head axis then shards over the flattened cp x EP mesh and each
    MoE layer dispatches its routed core around the official seq<->head
    all-to-all (see ``_apply_moe_parallel`` and
    ``expert_parallel.MoEDispatchContext``).
    """
    del model_converters

    if parallel_dims.pp_enabled:
        # PP stage parts reach this function via pipeline_magi2; only
        # pure PP + FSDP/DP is supported in v1.
        if parallel_dims.cp_enabled:
            raise NotImplementedError(
                "MAGI-2-preview PP + CP is not implemented in v1: CP "
                "shards the sequence inside the model while PP needs the "
                "per-stage sequence shards as inter-stage activations"
            )
        if parallel_dims.tp_enabled:
            raise NotImplementedError(
                "MAGI-2-preview PP + TP is not implemented in v1"
            )
        if (
            parallel_dims.get_optional_mesh("ep") is not None
            or parallel_dims.get_optional_mesh("etp") is not None
        ):
            raise NotImplementedError(
                "MAGI-2-preview PP + EP/ETP head-parallel MoE is not "
                "implemented in v1"
            )
    if parallel_dims.tp_enabled:
        # TP can work with CP (orthogonal: TP shards hidden/head dims, CP
        # shards sequence) and EP/ETP (orthogonal: TP shards attention/linear
        # dims, EP shards MoE head axis). SP (sequence parallel) can be
        # enabled via tp_sequence_parallel=True to shard the sequence dim
        # on the TP mesh, reducing activation memory.
        tp_sequence_parallel = getattr(parallelism, "tp_sequence_parallel", False)
        _apply_tensor_parallel(
            model, 
            tp_mesh=parallel_dims.get_mesh("tp"),
            sequence_parallel=tp_sequence_parallel
        )
    if parallel_dims.cp_enabled:
        # Ulysses CP before MoE/AC/FSDP: installs the model cp_context and
        # the attention hooks; torchtitan's "fsdp" mesh spans dp_shard x cp,
        # so CP-replicated parameter grads reduce there during FSDP. A CP+EP
        # combination is wired by the MoE step below (regime (b) on the
        # flattened cp x ep head mesh).
        apply_magi2_ulysses_cp(
            model,
            cp_mesh=parallel_dims.get_mesh("cp"),
            ep_degree=parallel_dims.ep,
        )
    if compile_config.enable and "model" in compile_config.components:
        logger.warning(
            "MAGI-2-preview model compilation has not been validated with "
            "distributed parallelism; continuing without applying torch.compile"
        )

    # Head-parallel MoE before AC/FSDP, mirroring kimi_k3's ordering of
    # expert parallelism ahead of activation checkpointing and FSDP. With
    # CP enabled the EP mesh becomes the flattened cp x ep head mesh and
    # the MoE layers run regime (b) dispatch (Ulysses all-to-all); the
    # cp_degree kwarg is only forwarded in that combined case, keeping the
    # regime-(a) call shape untouched.
    ep_mesh = parallel_dims.get_optional_mesh("ep")
    moe_kwargs = {}
    if ep_mesh is not None and parallel_dims.cp_enabled:
        ep_mesh = flatten_head_mesh(parallel_dims)
        moe_kwargs["cp_degree"] = parallel_dims.cp
    _apply_moe_parallel(
        model,
        ep_mesh=ep_mesh,
        etp_mesh=parallel_dims.get_optional_mesh("etp"),
        **moe_kwargs,
    )

    # Model compilation is deliberately skipped above, so activation
    # checkpointing must not assume it is wrapping a compiled model.
    model_compile_enabled = False
    if ac_config.mode != "none":
        # Upstream apply_ac looks up model.get_submodule("layers"); MAGI-2
        # layers live at model.block.layers, so checkpoint the block instead.
        apply_moe_ac(
            model.block,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    if parallel_dims.fsdp_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        # eFSDP: with EP enabled the routed experts shard over the efsdp
        # mesh (mirroring llama4's edp_mesh); otherwise no expert mesh.
        edp_mesh = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = (
                ["dp_replicate", "efsdp"]
                if parallel_dims.dp_replicate_enabled
                else ["efsdp"]
            )
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
        _apply_fsdp(
            model,
            dp_mesh,
            training=training,
            parallelism=parallelism,
            pp_enabled=parallel_dims.pp_enabled,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
        )
        logger.info("Applied MAGI-2-preview FSDP")
    elif parallel_dims.dp_replicate_enabled:
        _apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )
        logger.info("Applied MAGI-2-preview replicate")

    return model
