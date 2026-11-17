# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CV batch-chunk preparation and state-isolated graph calibration."""

from __future__ import annotations

import functools
import operator
from contextvars import ContextVar
from dataclasses import replace
from types import FunctionType
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.fx as fx
from torch._dynamo.decorators import mark_unbacked
from torch.utils import _pytree as pytree
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    CHUNK_SYMBOL_HINTS_META,
    chunk_symbol_hints_for_mode,
    free_symbols,
    tensor_meta,
)
from torchtitan.experiments.graph_trainer.passes import _get_pass_name
from torchtitan.experiments.graph_trainer.registry import register_post_init_hook
from torchtitan.tools.logging import logger

from . import _BATCH_CHUNK_PAIR_ATTR, CV_PARALLEL_PIPELINE
from .symbolic_meta import batch_chunk_symbolic_meta_context

if TYPE_CHECKING:
    from collections.abc import Callable

_RUNTIME_CONTEXT: ContextVar[Callable[[], dict[str, Any]] | None] = ContextVar("cv_parallel_context", default=None)


@register_post_init_hook(CV_PARALLEL_PIPELINE)
def configure_runtime_context(trainer):
    """Adapt only this opted-in trainer; leave upstream classes and registries intact."""
    from .batch_chunk_metadata import BatchChunkMetadataExtension

    for model in trainer.model_parts:
        metadata_extension = getattr(model, "_metadata_extension", None)
        if isinstance(metadata_extension, BatchChunkMetadataExtension):
            # Quantization selects the LI provider before model construction.
            # Local layouts must use that same provider's opaque kernel contract.
            metadata_extension._chunk_li_metadata = model._lightning_indexer_metadata

    make_step = trainer._make_fx_forward_backward_step
    prepare_inputs = trainer._prepare_trace_inputs

    @functools.wraps(make_step)
    def with_live_inputs(model, inputs, labels, global_valid_tokens, params, extra_kwargs):
        # Resolve traced_result lazily: the trainer creates it during make_step.
        token = _RUNTIME_CONTEXT.set(
            lambda: {
                "traced_result": trainer._traced_step,
                "module": model,
                "args": (inputs, labels, global_valid_tokens, extra_kwargs),
                "train_context": trainer.train_context,
            }
        )
        try:
            ep = trainer.config.compile.ep_overlap
            trace_batch_chunks = (
                trainer._traced_step is None and ep.enabled and ep.chunk_dim == "batch" and ep.strategy == "graph"
            )
            with batch_chunk_symbolic_meta_context(enabled=trace_batch_chunks):
                return make_step(model, inputs, labels, global_valid_tokens, params, extra_kwargs)
        finally:
            _RUNTIME_CONTEXT.reset(token)

    @functools.wraps(prepare_inputs)
    def with_full_batch_inputs(args, kwargs):
        prepare_inputs(args, kwargs)
        prepare_full_batch_trace_inputs(trainer.config.compile, args, kwargs)

    trainer._make_fx_forward_backward_step = with_live_inputs
    trainer._prepare_trace_inputs = with_full_batch_inputs


def prepare_full_batch_trace_inputs(config, args, kwargs):
    ep = config.ep_overlap
    if not (ep.enabled and ep.chunk_dim == "batch" and ep.strategy == "graph"):
        return
    if config.pass_pipeline != CV_PARALLEL_PIPELINE:
        return
    # The trace symbol denotes the full batch (>=2); the chunk pass rewrites
    # it to B//2. Do not constrain that derived half-batch to be non-singleton.
    # Leave the upper bound open: [2, 2] would erase the unbacked symbol.
    for tensor in pytree.tree_leaves((args, kwargs)):
        if (
            isinstance(tensor, torch.Tensor)
            and getattr(tensor, "_dynamo_shape_ids", {}).get(0) == "torchtitan_chunk_batch"
        ):
            criteria = getattr(tensor, "_specialize_on", {}).get(0)
            mark_unbacked(tensor, 0, min=2, specialize_on=criteria)


# These operators mutate process-global execution state while dispatching NPU
# work. They are not ordinary MIX/AIV kernels: letting another stream cross
# their host/device dispatch window can make that stream observe transient
# state. Keep the registry explicit and extensible instead of baking the rule
# into the Cube/Vector classifier.
_GLOBAL_STATE_BARRIER_TARGETS = {
    "torchtitan.deterministic_scatter_add",
}

# Reading an NPU value on the host synchronizes the current main stream and
# exposes the value to Python control flow. No auxiliary island may cross this
# boundary: doing so can let dynamic routing/split decisions observe unfinished
# device work even when the tensor data dependencies look otherwise legal.
_HOST_OBSERVATION_BARRIER_TARGETS = {
    torch.ops.aten._local_scalar_dense.default,
    torch.ops.aten.item.default,
    torch.ops.aten.is_nonzero.default,
}

_BACKWARD_SHAPE_PLUMBING_TARGETS = {
    torch.ops.aten.sym_size.int,
    torch.ops.aten.sym_stride.int,
    torch.sym_ite,
    operator.add,
    operator.and_,
    operator.eq,
    operator.floordiv,
    operator.ge,
    operator.gt,
    operator.le,
    operator.lt,
    operator.mod,
    operator.mul,
    operator.or_,
    operator.sub,
}


def _map_node_inputs(node: fx.Node, replacements: dict[fx.Node, Any]):
    return fx.map_arg((node.args, node.kwargs), lambda n: replacements.get(n, n))


def annotate_backward_shape_plumbing_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
) -> fx.GraphModule:
    """Repair missing AOT backward metadata on scalar shape dependencies.

    AOTAutograd can leave a forward tensor's ``sym_size`` node untagged even
    when that scalar is saved exclusively for a backward reshape. The EP chunk
    pass then sees the tensor as escaping to a full forward user and attempts
    to reconstruct a variable token extent from the two chunks. Propagate the
    backward tag only through side-effect-free scalar shape plumbing whose
    semantic users are already all backward nodes.
    """
    del example_inputs
    backward_nodes = {node for node in gm.graph.nodes if node.meta.get("autograd_backward", False)}
    annotated = 0
    for node in reversed(tuple(gm.graph.nodes)):
        if (
            node in backward_nodes
            or node.op != "call_function"
            or node.target not in _BACKWARD_SHAPE_PLUMBING_TARGETS
            or tensor_meta(node) is not None
        ):
            continue
        semantic_users = tuple(user for user in node.users if user.op == "output" or user.users or user.is_impure())
        if not semantic_users or not all(user in backward_nodes for user in semantic_users):
            continue
        node.meta["autograd_backward"] = True
        backward_nodes.add(node)
        annotated += 1
    if annotated:
        logger.info(
            "Annotated %d backward-only scalar shape node(s) before EP chunking",
            annotated,
        )
    return gm


def _runtime_flat_inputs(runtime_context: dict[str, Any]) -> list[Any]:
    """Flatten live inputs exactly as ``run_traced`` flattens graph inputs."""
    from torchtitan.experiments.graph_trainer.make_fx_tracer import (
        _unwrap_subclasses,
        extract_train_state,
    )

    model_state, optim_state = extract_train_state(runtime_context["module"])
    state_flat, _ = pytree.tree_flatten({"model": model_state, "optim": optim_state})
    user_flat, _ = pytree.tree_flatten((runtime_context["args"], {}))
    flat, _ = _unwrap_subclasses([*state_flat, *user_flat])
    return flat


def remap_prebuilt_batch_metadata_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    runtime_context: dict[str, Any] | None = None,
) -> fx.GraphModule:
    """Route duplicated layer nodes to eager-built local attention metadata."""
    del example_inputs
    if runtime_context is None:
        raise RuntimeError("batch chunk metadata remapping requires the trainer runtime context")
    placeholders = tuple(node for node in gm.graph.nodes if node.op == "placeholder")
    runtime_inputs = _runtime_flat_inputs(runtime_context)
    if len(placeholders) != len(runtime_inputs):
        raise RuntimeError(
            "batch chunk metadata remapping found a graph/runtime input count "
            f"mismatch: graph={len(placeholders)} runtime={len(runtime_inputs)}"
        )

    full_by_pair: dict[str, fx.Node] = {}
    chunk_by_pair: dict[tuple[str, int], fx.Node] = {}
    for placeholder, value in zip(placeholders, runtime_inputs, strict=True):
        marker = getattr(value, _BATCH_CHUNK_PAIR_ATTR, None)
        if not (
            isinstance(marker, tuple) and len(marker) == 2 and isinstance(marker[0], str) and marker[1] in (-1, 0, 1)
        ):
            continue
        pair_id, chunk_id = marker
        if chunk_id == -1:
            full_by_pair[pair_id] = placeholder
        else:
            chunk_by_pair[(pair_id, chunk_id)] = placeholder
    if not full_by_pair:
        raise ValueError(
            "batch chunking of full DeepSeek-V4 layers requires eager-built "
            "attention metadata; enable the batch_chunk_metadata.asc_metadata override"
        )
    missing = [
        (pair_id, chunk_id)
        for pair_id in full_by_pair
        for chunk_id in (0, 1)
        if (pair_id, chunk_id) not in chunk_by_pair
    ]
    if missing:
        raise RuntimeError(f"batch chunk metadata is missing pair(s): {missing}")

    scalar_maps = [
        {full_node: chunk_by_pair[pair_id, chunk_id] for pair_id, full_node in full_by_pair.items()}
        for chunk_id in (0, 1)
    ]
    # The upstream batch pass can already split a full metadata placeholder.
    # Those getitems must select the prebuilt local value directly, otherwise
    # cloning the split would try to split a one-row local tensor into [1, 1].
    preselected = 0
    for split in tuple(gm.graph.nodes):
        if (
            split.op != "call_function"
            or split.target != torch.ops.aten.split_with_sizes.default
            or split.meta.get("chunked_region_role") != "split_boundary"
        ):
            continue
        if len(split.args) < 2 or split.args[0] not in scalar_maps[0]:
            continue
        full_node, sizes = split.args[:2]
        dim = split.args[2] if len(split.args) > 2 else split.kwargs.get("dim", 0)
        if not isinstance(sizes, (list, tuple)) or len(sizes) != 2 or dim != 0:
            continue
        users = tuple(split.users)
        if not users or any(
            u.op != "call_function"
            or u.target != operator.getitem
            or len(u.args) != 2
            or u.args[1] not in (0, 1)
            or u.meta.get("chunk_id") != u.args[1]
            for u in users
        ):
            continue
        for user in users:
            user.replace_all_uses_with(scalar_maps[user.args[1]][full_node])
            gm.graph.erase_node(user)
            preselected += 1
        gm.graph.erase_node(split)

    logger.info("Selected %d prebuilt metadata values instead of batch splits", preselected)
    # Shape/SymInt and metadata-derived tensor plumbing are often hoisted out
    # of module bodies by AOT. Clone every pure, non-body descendant of full
    # metadata so duplicated layer nodes consume values derived from their
    # local placeholder. This follows dependencies and is operator-agnostic.
    original_nodes = tuple(gm.graph.nodes)
    cloned_metadata_nodes = 0
    for node in original_nodes:
        if node.op in {"placeholder", "output"} or node.is_impure() or node.meta.get("chunk_id") in (0, 1):
            continue
        for chunk_id in (0, 1):
            scalar_map = scalar_maps[chunk_id]
            new_args, new_kwargs = _map_node_inputs(node, scalar_map)
            if new_args == node.args and new_kwargs == node.kwargs:
                continue
            with gm.graph.inserting_after(node):
                copied = gm.graph.create_node(
                    node.op,
                    node.target,
                    new_args,
                    new_kwargs,
                    type_expr=node.type,
                )
            copied._rename(f"{node.name}_metadata_chunk{chunk_id}")
            copied.meta = dict(node.meta)
            copied.meta["chunk_id"] = chunk_id
            scalar_map[node] = copied
            cloned_metadata_nodes += 1

    remapped_nodes = 0
    remapped_inputs = preselected
    for node in original_nodes:
        chunk_id = node.meta.get("chunk_id")
        if chunk_id not in (0, 1):
            continue
        node_replacements = scalar_maps[chunk_id]
        before = tuple(node.all_input_nodes)
        node.args, node.kwargs = _map_node_inputs(node, node_replacements)
        changed = sum(input_node in node_replacements for input_node in before)
        if changed:
            remapped_nodes += 1
            remapped_inputs += changed
    if remapped_inputs == 0:
        raise ValueError("batch chunk metadata remapping found no duplicated layer users")
    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Remapped %d attention metadata input(s) across %d batch-chunk "
        "layer node(s), including %d cloned metadata-derived node(s)",
        remapped_inputs,
        remapped_nodes,
        cloned_metadata_nodes,
    )
    return gm


def _mask_unrelated_dynamic_placeholders(
    gm: fx.GraphModule,
    *,
    mode: str,
) -> dict[fx.Node, torch.Tensor]:
    """Hide content-dynamic input extents from the upstream chunk scanner.

    The upstream batch scanner treats every symbolic placeholder leading
    dimension as batch-owned. DSV4 sparse-attention plans also expose
    content-dependent one-dimensional tensors, so use the explicit shape id
    installed by ``mark_chunk_dynamic_dims`` as the source of truth while the
    upstream transform runs. Derived graph metadata remains dynamic.
    """
    expected_shape_id = f"torchtitan_chunk_{mode}"
    all_hints = chunk_symbol_hints_for_mode(gm, mode)
    selected_hints: dict[object, int] = {}
    selection_source = "explicit_shape_id"
    for node in gm.graph.nodes:
        if node.op != "placeholder" or (value := tensor_meta(node)) is None:
            continue
        shape_ids = getattr(value, "_dynamo_shape_ids", {})
        if not isinstance(shape_ids, dict):
            continue
        for dim, shape_id in shape_ids.items():
            if shape_id != expected_shape_id or dim >= value.dim():
                continue
            for symbol in free_symbols(value.shape[dim]):
                if symbol in all_hints:
                    selected_hints[symbol] = all_hints[symbol]
    if not selected_hints:
        # AOTAutograd currently drops FakeTensor Python attributes from joint
        # graph placeholders. The semantic token grid is still uniquely
        # identifiable: it is at least rank two and carries the requested
        # batch/sequence symbol. DSV4 content-dependent sparse-plan tensors
        # are rank one. Refuse ambiguity instead of guessing.
        selection_source = "unique_rank2_token_grid"
        dim = {"batch": 0, "seq": 1}.get(mode)
        if dim is None:
            raise ValueError(f"Unknown chunk mode: {mode!r}")
        token_grid_symbols = set()
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            value = tensor_meta(node)
            if value is None or value.dim() < 2 or dim >= value.dim():
                continue
            token_grid_symbols.update(symbol for symbol in free_symbols(value.shape[dim]) if symbol in all_hints)
        if len(token_grid_symbols) == 1:
            selected_hints = {symbol: all_hints[symbol] for symbol in token_grid_symbols}
    if len(selected_hints) != 1:
        raise ValueError(
            "NPU chunk CV requires one unambiguous token-grid chunk symbol; "
            f"shape_id={expected_shape_id!r} candidates={tuple(selected_hints)}"
        )

    originals: dict[fx.Node, torch.Tensor] = {}
    unrelated_hints = {symbol: hint for symbol, hint in all_hints.items() if symbol not in selected_hints}
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        value = tensor_meta(node)
        if value is None:
            continue
        node.meta[CHUNK_SYMBOL_HINTS_META] = {
            symbol: hint
            for symbol, hint in selected_hints.items()
            if any(symbol in free_symbols(extent) for extent in value.shape)
        }
        concrete = _concretize_selected_chunk_meta(value, unrelated_hints)
        if concrete is not value:
            originals[node] = value
            node.meta["val"] = concrete
    logger.info(
        "Isolated explicit %s chunk symbol(s): selected=%d source=%s masked_content_dynamic_placeholders=%d",
        mode,
        len(selected_hints),
        selection_source,
        len(originals),
    )
    return originals


def _relevant_symbol_substitutions(value, substitutions):
    """Keep the transitive substitution closure for a symbolic expression.

    The upstream chunk pass accumulates fresh symbols for the whole graph.
    Passing thousands of unrelated replacements to SymPy for every shape
    makes the 43-layer transform expensive. Retain chained replacements and
    fall back unchanged if the mapping contains non-symbol patterns.
    """
    if any(not getattr(key, "is_Symbol", False) for key in substitutions):
        return substitutions
    pending = list(free_symbols(value))
    selected = {}
    while pending:
        symbol = pending.pop()
        if symbol in substitutions and symbol not in selected:
            replacement = substitutions[symbol]
            selected[symbol] = replacement
            pending.extend(free_symbols(replacement))
    return selected


def _explicit_chunk_symbol_wrapper(
    chunk_pass: Callable,
    *,
    mode: str,
) -> Callable:
    """Run the upstream transform with only explicit chunk symbols visible."""

    def run(gm, example_inputs):
        import torchtitan.experiments.graph_trainer.ep_chunk_pass as ep_chunk_module

        originals = _mask_unrelated_dynamic_placeholders(gm, mode=mode)
        original_rewrite = ep_chunk_module._rewrite_symbolic_value

        def rewrite_symbolic_value(value, symbol_subs, symbol_hints):
            symbol_subs = _relevant_symbol_substitutions(value, symbol_subs)
            try:
                return original_rewrite(value, symbol_subs, symbol_hints)
            except AssertionError as error:
                # PyTorch rejects hints on SymFloat expressions containing an
                # unbacked symbol, while the upstream chunk pass supplies the
                # evaluated hint for all symbolic scalar types. The hint is
                # metadata only; preserve the rewritten expression unhinted.
                if not (isinstance(value, torch.SymFloat) and "hint must be None for unbacked symbol" in str(error)):
                    raise
                expr = value.node.expr.subs(symbol_subs)
                shape_env = value.node.shape_env
                if shape_env is None:
                    raise RuntimeError("SymFloat chunk rewrite has no ShapeEnv") from error
                return shape_env.create_symfloatnode(
                    cast("Any", expr),
                    hint=None,
                )

        # Helpers resolve one another through module globals. Clone that
        # function namespace so an ordinary EP transform never observes our
        # symbolic rewrite, even while this pass is running.
        namespace = dict(vars(ep_chunk_module))
        for name, value in vars(ep_chunk_module).items():
            if isinstance(value, FunctionType) and value.__globals__ is vars(ep_chunk_module):
                cloned = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, value.__closure__)
                cloned.__kwdefaults__ = value.__kwdefaults__
                namespace[name] = functools.update_wrapper(cloned, value)
        namespace["_rewrite_symbolic_value"] = rewrite_symbolic_value
        func = chunk_pass.func if isinstance(chunk_pass, functools.partial) else chunk_pass
        isolated_pass = namespace[func.__name__]
        if isinstance(chunk_pass, functools.partial):
            isolated_pass = functools.partial(isolated_pass, *chunk_pass.args, **chunk_pass.keywords)
        try:
            return isolated_pass(gm, example_inputs)
        finally:
            for node, value in originals.items():
                node.meta["val"] = value

    run.__name__ = _get_pass_name(chunk_pass)
    return run


def _concretize_selected_chunk_value(
    value: object,
    symbol_hints: dict[object, int],
) -> object:
    """Substitute chunk-owned symbols without specializing unrelated shapes."""
    symbols = free_symbols(value)
    selected = symbols & symbol_hints.keys()
    if not selected or not hasattr(value, "node"):
        return value
    expr = value.node.expr.subs({symbol: symbol_hints[symbol] for symbol in selected})
    if not free_symbols(expr):
        if isinstance(value, torch.SymBool):
            return bool(expr)
        if isinstance(value, torch.SymFloat):
            return float(expr)
        return int(expr)
    shape_env = value.node.shape_env
    if isinstance(value, torch.SymInt):
        return shape_env.create_symintnode(expr, hint=None)
    if isinstance(value, torch.SymFloat):
        return shape_env.create_symfloatnode(expr, hint=None)
    if isinstance(value, torch.SymBool):
        return shape_env.create_symboolnode(expr)
    return value


def _concretize_selected_chunk_meta(
    value: object,
    symbol_hints: dict[object, int],
) -> object:
    if isinstance(value, torch.Tensor):
        dimensions = (*value.shape, *value.stride())
        if not any(free_symbols(item) & symbol_hints.keys() for item in dimensions):
            return value
        shape = cast(
            "tuple[int | torch.SymInt, ...]",
            tuple(_concretize_selected_chunk_value(item, symbol_hints) for item in value.shape),
        )
        stride = cast(
            "tuple[int | torch.SymInt, ...]",
            tuple(_concretize_selected_chunk_value(item, symbol_hints) for item in value.stride()),
        )
        return value.new_empty_strided(shape, stride)
    if isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
        return _concretize_selected_chunk_value(value, symbol_hints)
    return value


def _node_has_device_tensor(node: fx.Node, device_type: str) -> bool:
    value = node.meta.get("val", node.meta.get("example_value"))
    return any(isinstance(leaf, torch.Tensor) and leaf.device.type == device_type for leaf in pytree.tree_leaves(value))


def _requires_global_state_barrier(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    target = node.target
    target_name = str(target)
    if target_name.endswith(".default"):
        target_name = target_name[: -len(".default")]
    qualified_name = ".".join(
        part
        for part in (
            getattr(target, "__module__", ""),
            getattr(target, "__name__", ""),
        )
        if part
    )
    return bool({target_name, qualified_name} & _GLOBAL_STATE_BARRIER_TARGETS)


def _requires_host_observation_barrier(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    if node.target in _HOST_OBSERVATION_BARRIER_TARGETS:
        source = node.args[0] if node.args else node.kwargs.get("self")
        value = source.meta.get("val", source.meta.get("example_value")) if isinstance(source, fx.Node) else source
        return not isinstance(value, torch.Tensor) or value.device.type != "cpu"
    if node.target != torch.ops.aten._to_copy.default:
        return False
    device = node.kwargs.get("device")
    if device is not None:
        if not isinstance(device, (str, torch.device)):
            return False
        try:
            return torch.device(device).type == "cpu"
        except (RuntimeError, TypeError):
            return str(device).startswith("cpu")
    return _node_has_device_tensor(node, "cpu") and any(
        _node_has_device_tensor(dependency, "npu") for dependency in node.all_input_nodes
    )


def _requires_full_stream_barrier(node: fx.Node) -> bool:
    # FX data dependencies do not model RNG state, alias mutation, or an
    # impure module's hidden state. Letting auxiliary work cross one of those
    # calls can change numerics even when the explicit tensor DAG is legal.
    implicit_side_effect = (
        node.op
        in {
            "call_function",
            "call_method",
            "call_module",
        }
        and node.is_impure()
    )
    return implicit_side_effect or _requires_global_state_barrier(node) or _requires_host_observation_barrier(node)


def _make_run_candidate(runtime_context):
    """Build the live graph runner used by the isolated profile round."""

    from torchtitan.experiments.graph_trainer.make_fx_tracer import run_traced

    model = runtime_context["module"]
    traced_result = runtime_context["traced_result"]
    training_args = runtime_context["args"]
    train_context = runtime_context["train_context"]

    class RunCandidate:
        """Separate calibration safety work from the measured graph call."""

        def __init__(self) -> None:
            self.calibration_args: Any | None = None
            self.parameter_states: list[tuple[torch.Tensor, int, torch.Tensor | None, int | None]] = []
            self.buffer_states: list[tuple[str, torch.Tensor, int, torch.Tensor]] = []

        def prepare(self) -> None:
            if self.calibration_args is not None:
                raise RuntimeError("NPU chunk CV calibration was already prepared")
            # Reuse the batch already fetched for the real first step, but
            # protect it from unexpected in-place graph operations. This hook
            # is called before the measured profiler scope opens.
            calibration_args = pytree.tree_map(
                lambda value: value.clone() if isinstance(value, torch.Tensor) else value,
                training_args,
            )
            self.parameter_states = [
                (param, param._version, param.grad, None if param.grad is None else param.grad._version)
                for param in model.parameters()
            ]
            self.buffer_states = [
                (name, buffer, buffer._version, buffer.clone()) for name, buffer in model.named_buffers()
            ]
            self.calibration_args = calibration_args

        def __call__(self, candidate_gm: fx.GraphModule):
            if self.calibration_args is None:
                raise RuntimeError("NPU chunk CV calibration runner must be prepared before use")
            candidate = replace(traced_result, gm=candidate_gm)
            with train_context():
                return run_traced(candidate, module=model)(*self.calibration_args)

        def finalize(self) -> None:
            if self.calibration_args is None:
                raise RuntimeError("NPU chunk CV calibration runner was not prepared")
            mutated_parameters = [
                index for index, (param, version, _, _) in enumerate(self.parameter_states) if param._version != version
            ]
            mutated_gradients = [
                index
                for index, (param, _, grad, version) in enumerate(self.parameter_states)
                if param.grad is not grad or (grad is not None and grad._version != version)
            ]
            mutated_buffers, version_only_buffers = [], []
            for name, buffer, version, snapshot in self.buffer_states:
                if buffer._version == version:
                    continue
                if torch.equal(buffer, snapshot):
                    version_only_buffers.append(name)
                else:
                    mutated_buffers.append((name, buffer, snapshot))
            with torch.no_grad():
                for _, buffer, snapshot in mutated_buffers:
                    buffer.copy_(snapshot)
            restored_buffer_names = [name for name, _, _ in mutated_buffers]
            failed_buffer_names = [
                name for name, buffer, snapshot in mutated_buffers if not torch.equal(buffer, snapshot)
            ]
            num_parameters = len(self.parameter_states)
            num_buffers = len(self.buffer_states)

            # Release cloned inputs and buffer snapshots before cross-rank
            # alignment or the formal first training step.
            self.calibration_args = None
            self.parameter_states = []
            self.buffer_states = []

            if mutated_parameters or mutated_gradients or failed_buffer_names:
                raise RuntimeError(
                    "NPU chunk CV calibration mutated training state: "
                    f"parameters={mutated_parameters[:8]}, "
                    f"gradients={mutated_gradients[:8]}, "
                    f"buffer_restore_failed={failed_buffer_names[:8]}"
                )
            logger.info(
                "NPU chunk CV calibration preserved training state: "
                "data_batches_consumed=0 parameters=%d gradients=%d buffers=%d "
                "buffers_restored=%s buffer_version_only=%s",
                num_parameters,
                num_parameters,
                num_buffers,
                restored_buffer_names,
                version_only_buffers,
            )

    return RunCandidate()


def configure_batch_chunk_passes(
    passes: list[Callable],
    config,
    *,
    runtime_context=None,
) -> list[Callable]:
    """Keep the chunk transform but remove all cross-chunk scheduling."""
    if config.compile.inductor_compilation != "regional":
        raise ValueError("NPU batch chunk requires compile.inductor_compilation='regional'")
    ep_config = config.compile.ep_overlap
    if not ep_config.enabled or ep_config.strategy != "graph" or ep_config.chunk_dim != "batch":
        raise ValueError("NPU batch chunk requires graph EP chunking with chunk_dim='batch'")
    pass_names = [_get_pass_name(pass_fn) for pass_fn in passes]
    if pass_names.count("ep_overlap_schedule_pass") != 1 or pass_names.count("ep_overlap_chunk_pass") != 1:
        raise ValueError(
            "NPU batch chunk requires exactly one ep_overlap_chunk_pass and "
            "ep_overlap_schedule_pass in the default pipeline"
        )

    configured = []
    for pass_name, pass_fn in zip(pass_names, passes, strict=True):
        if pass_name == "ep_overlap_schedule_pass":
            continue
        if pass_name == "ep_overlap_chunk_pass":
            configured.extend(
                [
                    annotate_backward_shape_plumbing_pass,
                    _explicit_chunk_symbol_wrapper(
                        functools.partial(pass_fn, module_pattern="layers"),
                        mode=ep_config.chunk_dim,
                    ),
                    functools.partial(remap_prebuilt_batch_metadata_pass, runtime_context=runtime_context),
                ]
            )
            continue
        configured.append(pass_fn)
    return configured
