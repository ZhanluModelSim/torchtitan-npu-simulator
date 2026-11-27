# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Meta-capture support for system3's host-offload Engram hash tables.

The production table is CPU-resident, shards itself over EP and routes
deduplicated row requests through all-to-all exchanges whose split sizes
are data-dependent (``torch.unique``/``bincount``), which Meta capture
cannot execute. This bridge replaces the meta face of
``System3HostHashTable._lookup_rows`` with a shape-visible decomposition
while keeping the formal ``_System3HostLookup`` autograd boundary: the
lookup rows still carry a Function backward, so the owner-local sparse
gradient actually accumulates in warmup and capture alike. At EP > 1 the
three exchanges run as native functional all-to-alls with balanced
(uniform) split geometry, mirroring the EP token dispatcher bridge in
``compat/routing.py``.

The host sparse-gradient side effects are bridged explicitly for Meta
(PR #19 review protocol): the pending sparse gradient accumulates on meta
tensors, the host clipping rescale keeps its op structure, and the
per-table SparseAdam update records through the declared
``fused.sparse_adam_update`` boundary so the captured F/B/O includes
the sparse work the dense gradient contract exempts. Real-device execution
is untouched.
"""

import torch
from torch.distributed._functional_collectives import all_reduce, all_to_all_single
from torch.profiler import record_function

from torchtitan_simulator.capture.dispatch_capture import (
    annotate_active_capture_events,
    host_only_work,
    record_host_boundary,
)
from torchtitan_simulator.capture.tensor_semantics import (
    OperatorTensorSemantics,
    TensorMeaning,
    register_tensor_semantics,
)


# The fused SparseAdam boundary (registration, fake and tensor semantics) is
# declared once in compat.deepseek_v4_1 and reused here so every model shares
# one capture name.
from torchtitan_simulator.compat.deepseek_v4_1 import sparse_adam_update  # noqa: E402


def _meta_accumulate_sparse_gradient(self, local_row_ids: torch.Tensor, grad_rows: torch.Tensor) -> None:
    """Meta twin of the CPU accumulation: no device moves, no value reads.

    Production moves rows to CPU, range-checks ids by value and coalesces the
    sparse gradient; SparseMeta has no coalesce kernel and meta has no data
    to move, read or deduplicate, so this twin keeps the FP32 sparse
    accumulation as index/value concatenation and marks the result coalesced
    by declaration -- every downstream sparse consumer requires the coalesced
    flag, and on meta the entries carry shapes only. The coalesce itself is
    host-only work in production and stays outside the device graph.
    """
    with host_only_work():
        local_row_ids = local_row_ids.reshape(-1).to(dtype=torch.int64)
        grad_rows = grad_rows.reshape(local_row_ids.numel(), self.weight.shape[1]).to(dtype=torch.float32)
        if self._pending_sparse_grad is None:
            self._pending_sparse_grad = torch.sparse_coo_tensor(
                local_row_ids.unsqueeze(0),
                grad_rows,
                size=self.weight.shape,
                dtype=torch.float32,
                check_invariants=False,
            )
        else:
            pending = self._pending_sparse_grad
            self._pending_sparse_grad = torch.sparse_coo_tensor(
                torch.cat((pending.indices(), local_row_ids.unsqueeze(0)), dim=1),
                torch.cat((pending.values(), grad_rows), dim=0),
                size=self.weight.shape,
                dtype=torch.float32,
                check_invariants=False,
            )
        self._pending_sparse_grad._coalesced_(True)
    # Remember the device-side gradient rows this pending gradient was
    # accumulated from (stashed on the weight, which is the identity the
    # SparseAdam step sees); the sparse update and the clip squares register
    # their cross-boundary dependency on them (see record_host_boundary).
    sources = getattr(self.weight, "_pending_grad_sources", None)
    if sources is None:
        sources = []
        self.weight._pending_grad_sources = sources
    sources.append(grad_rows)


def _meta_scale_pending_sparse_grad(self, scale) -> None:
    """Meta twin of the clipping rescale, keeping the coalesced declaration."""
    if self._pending_sparse_grad is None:
        return
    # The scale tensor feeds the cross-boundary dependency of the sparse
    # update (the extracted rows carry the applied coefficient).
    self.weight._last_sparse_scale = scale
    with host_only_work():
        self._pending_sparse_grad = torch.sparse_coo_tensor(
            self._pending_sparse_grad.indices(),
            self._pending_sparse_grad.values() * scale,
            self._pending_sparse_grad.shape,
            check_invariants=False,
        )
        self._pending_sparse_grad._coalesced_(True)


def _meta_prepare_sparse_optimizer_step(self) -> None:
    """Meta twin of the pending-to-weight.grad attach.

    Production converts the pending gradient to the weight dtype; on meta
    the conversion is FP32 identity but dispatches ``_to_copy`` with a
    sparse operand, which is host work and must not enter the graph.
    """
    if self._pending_sparse_grad is None:
        self.weight.grad = None
        return
    pending = self._pending_sparse_grad
    self._pending_sparse_grad = None
    if pending.device.type == self.weight.device.type:
        with host_only_work():
            self.weight.grad = pending.to(dtype=self.weight.dtype)
        return
    # Host residency keeps the weight on a real CPU tensor through the meta
    # build while the pending gradient is a meta structure; torch rejects
    # cross-device gradient assignment, so the capture face hands the meta
    # pending to the SparseAdam meta step through an explicit channel and
    # keeps weight.grad clear for the dense clipper.
    self.weight._meta_pending_grad = pending
    self.weight.grad = None


def _meta_reduce_sparse_gradient_across_replicas(self) -> None:
    """Meta twin of the replica reduction (single-process capture view).

    Production pads the owner-local rows to the group maximum and
    all-gathers them across the dp replicas of this shard; meta cannot read
    row counts by value, and the single-process capture carries one logical
    rank, so the peer replicas' rows do not exist in-process and the
    reduction collapses to the local pending gradient. The missing host
    collective is a documented single-process gap, not a claim that
    production skips it.
    """
    group = self._replica_group
    if group is None:
        return


def _meta_sparse_adam_step(self, closure=None):
    """Meta twin of the native SparseAdam step (PR #19 review).

    Mirrors the native parameter selection, dense-gradient rejection and
    lazy state materialization exactly; the per-table update runs through
    the declared virtual boundary because the native functional needs
    SparseMeta kernels that do not exist.
    """
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    with torch.no_grad():
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                # The prepare twin hands host tables their meta pending
                # gradient through this channel (the host-resident weight
                # rejects cross-device .grad assignment in capture).
                grad = getattr(parameter, "_meta_pending_grad", None)
                if grad is None:
                    grad = parameter.grad
                if grad is None:
                    continue
                if not grad.is_sparse:
                    raise RuntimeError("SparseAdam does not support dense gradients, please consider Adam instead")
                with host_only_work():
                    # State materialization and the sparse row extraction are
                    # CPU work in production; only the declared update boundary
                    # below belongs to the captured graph. The moments stay on
                    # the capture device like every other tracked tensor.
                    state = self.state[parameter]
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format, device="meta"
                        )
                        state["exp_avg_sq"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format, device="meta"
                        )
                    state["step"] += 1
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                # The extracted rows carry the backward gradient and the
                # applied clip coefficient across the host boundary; record
                # both so the update node stays connected to the graph.
                sources = list(getattr(parameter, "_pending_grad_sources", ()) or ())
                applied_scale = getattr(parameter, "_last_sparse_scale", None)
                if applied_scale is not None:
                    sources.append(applied_scale)
                record_host_boundary(sources, grad_indices)
                record_host_boundary(sources, grad_values)
                parameter._pending_grad_sources = []
                parameter._meta_pending_grad = None
                sparse_adam_update(
                    parameter,
                    grad_indices,
                    grad_values,
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["step"],
                    group["lr"],
                    beta1,
                    beta2,
                    group["eps"],
                )
    return loss


def _meta_host_sparse_container_step(self, closure=None):
    """Meta twin of ``HostSparseOptimizersContainer.step`` (PR #19 review).

    The native SparseAdam functional needs SparseMeta kernels that do not
    exist; the container twin keeps the production body -- prepare,
    per-optimizer steps, refresh, mark -- and routes the sparse table
    optimizers through the meta step, which records the per-table update
    through the declared virtual boundary.

    TorchTitan's ``OptimizersContainer._post_init`` runs
    ``Optimizer.__init__`` on the container instance, whose
    ``_patch_step_function`` rebinds the class ``step`` with
    ``profile_hook_step`` (its own escape hatch is the ``hooked`` marker on
    an already-wrapped step). The marker below keeps this binding intact
    through the first container construction; the profile wrapper only adds
    a ``record_function`` envelope that the twin already carries, and the
    step-boundary tracker still saves and restores the twin exactly.
    """
    assert closure is None, "OptimizersContainer does not support closures"
    with record_function("engram::sparse_grad_attach"):
        for table in self._iter_host_tables():
            table.prepare_sparse_optimizer_step()
    for optimizer in self.optimizers:
        if isinstance(optimizer, torch.optim.SparseAdam):
            _meta_sparse_adam_step(optimizer)
        else:
            optimizer.step()
    for table in self._iter_host_tables():
        pending = table.pending_sparse_grad()
        if pending is not None:
            with record_function("engram::refresh_lookup_storage"):
                table.refresh_lookup_storage(pending.indices()[0])
        table.mark_sparse_step_complete()
    return None


# Torch's Optimizer._patch_step_function skips its profile wrapper for steps
# already carrying the 'hooked' marker; without it the first container build
# would wrap this binding over and the patch ownership check would fire.
_meta_host_sparse_container_step.hooked = True  # pyrefly: ignore [missing-attribute]


def _meta_clip_grad_norm_with_host_sparse_tables(
    parameters,
    max_norm,
    norm_type=2.0,
    *args,
    tables,
    clipper,
    scale_dense_gradients=None,
    **kwargs,
):
    """Meta twin of ``clip_grad_norm_with_host_sparse_tables``.

    The production wrapper reads the norms by value (``float``/``isfinite``
    guards) to decide whether to rescale; on meta the norms are symbolic, so
    this twin keeps the exact op structure and applies the clamped coefficients
    unconditionally -- the same convention the native dense clipper already
    follows on meta (``_foreach_mul_`` with a clamped coefficient tensor).
    """
    parameters = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    dense_norm = clipper(parameters, max_norm, norm_type, *args, **kwargs)
    # WeakSet order depends on local object addresses. Replica collectives
    # must visit the same layer on every rank, and norm summation must be stable.
    registered_tables = sorted(tables, key=lambda table: table.layer_idx)
    for table in registered_tables:
        table.reduce_sparse_gradient_across_replicas()
    pending_tables = [table for table in registered_tables if table.pending_sparse_grad() is not None]
    if not pending_tables:
        return dense_norm
    if norm_type != 2.0:
        raise NotImplementedError(f"Host Engram gradient clipping supports the 2-norm only, got norm_type={norm_type}.")
    group_squares: dict = {}
    for table in pending_tables:
        lookup_mesh = table.lookup_mesh
        # Rows are owner-local at EP degree 1: the production all_reduce over
        # the trivial row-ownership group is the identity. At EP degree > 1
        # the mesh-based functional collective models the production group
        # reduction shape-visibly (single-process capture owns one logical
        # rank; the swap is declared, not executed).
        group = lookup_mesh.get_group() if lookup_mesh is not None else None
        if group not in group_squares:
            group_squares[group] = torch.zeros((), dtype=torch.float32)
        pending = table.pending_sparse_grad()
        assert pending is not None
        # The sparse square is host-computed in production; only its scalar
        # joins the device coefficient math below. The boundary dependency
        # keeps it connected to the backward gradient it was derived from.
        with host_only_work():
            squares = pending.values().detach().float().pow(2).sum()
        record_host_boundary(getattr(table.weight, "_pending_grad_sources", ()), squares)
        group_squares[group] = group_squares[group] + squares
    sparse_sq = torch.zeros((), dtype=torch.float32, device=dense_norm.device)
    for group, local_sq in group_squares.items():
        # Production all-reduces the row squares over the row-ownership group
        # and then moves the host scalar next to the dense norm. The capture
        # face models the group reduction with the same mesh-based functional
        # collective the routed path uses, and the trivial group skips it.
        if group is not None and local_sq.device.type == "meta":
            local_sq = all_reduce(local_sq, "sum", group=group)
        sparse_sq = sparse_sq + local_sq.to(dense_norm.device)
    dense_norm_f32 = dense_norm.detach().float()
    total_norm = torch.sqrt(dense_norm_f32.pow(2) + sparse_sq)
    dense_coef = torch.clamp(max_norm / (dense_norm_f32 + 1e-6), max=1.0)
    total_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    correction = total_coef / dense_coef
    # Production reads the correction scalar and rescales only when it is
    # not 1.0; meta cannot read values, so the rescale records
    # unconditionally under a distinct scope, keeping the dense gradients
    # at the combined coefficient like production's taken branch.
    with annotate_active_capture_events("gradient_clip_sparse_correction"):
        if scale_dense_gradients is not None:
            scale_dense_gradients(parameters, correction)
        else:
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
    for table in pending_tables:
        table.scale_pending_sparse_grad(total_coef)
    return total_norm.to(dense_norm.dtype)


def _meta_container_clip_grad_norm(self, parameters, max_norm, **kwargs):
    from torchtitan.distributed.utils import clip_grad_norm_

    return _meta_clip_grad_norm_with_host_sparse_tables(
        parameters,
        max_norm,
        tables=list(self._iter_host_tables()),
        clipper=clip_grad_norm_,
        scale_dense_gradients=self._scale_dense_gradients,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Lookup: keep the formal autograd boundary on meta.
# ---------------------------------------------------------------------------


def _balanced_ep_lookup_rows(ctx, weight, row_ids, mesh):
    """Meta decomposition of the routed lookup with balanced split geometry.

    Production deduplicates row requests (``torch.unique``/``bincount``) and
    exchanges data-dependent split sizes, which meta cannot execute; the twin
    exchanges balanced (uniform) splits of the same logical payload instead.
    The three exchanges of the official routed path each appear as native
    functional all-to-alls, the owner-local gather contributes the H2D
    transfer of the received rows, and the routing state (``inverse``,
    ``received_ids``, splits, group) is saved for the backward exchange.
    """
    device = row_ids.device
    flat = row_ids.reshape(-1)
    row_dim = weight.shape[1]
    total = flat.numel()
    ep_size = mesh.size()
    quotient, remainder = divmod(total, ep_size)
    splits = [quotient + int(rank < remainder) for rank in range(ep_size)]

    counts = torch.zeros(ep_size, dtype=torch.int64, device=device)
    all_to_all_single(counts, None, None, mesh)
    received_ids = all_to_all_single(flat.contiguous(), splits, splits, mesh)
    rows_owner_host = torch.empty(
        (received_ids.shape[0], row_dim),
        dtype=weight.dtype,
        device="cpu",
    )
    rows_owner = rows_owner_host.to(device=device)
    rows = all_to_all_single(rows_owner, splits, splits, mesh)

    # The single logical rank of a capture sends each request to its balanced
    # bucket and reads it back from the same bucket, so the inverse is the
    # identity over the balanced layout; the accumulated ids stay global
    # table rows (the system3 table keeps the full capacity on the meta path).
    inverse = torch.arange(total, dtype=torch.int64, device=device)
    ctx.save_for_backward(inverse, received_ids)
    ctx.send_splits = splits
    ctx.recv_splits = splits
    ctx.mesh = mesh
    return rows.view(*row_ids.shape, row_dim)


def _balanced_ep_lookup_grad_rows(ctx, grad_output: torch.Tensor):
    """Meta twin of the routed lookup backward exchange.

    Mirrors the production backward aggregation in fp32 (duplicate hits are
    summed before the exchange) and reverses the balanced split geometry, so
    the owner-local gradient rows and their ids reach the accumulator with
    the same op structure as production.
    """
    inverse, received_ids = ctx.saved_tensors
    row_dim = grad_output.shape[-1]
    grad_unique = torch.zeros(
        sum(ctx.send_splits),
        row_dim,
        dtype=torch.float32,
        device=grad_output.device,
    )
    grad_unique.index_add_(0, inverse, grad_output.reshape(-1, row_dim).float())
    received_grads = all_to_all_single(
        grad_unique.contiguous(),
        ctx.recv_splits,
        ctx.send_splits,
        ctx.mesh,
    )
    return received_grads, received_ids


def install_system3_engram_meta_support(scope):
    """Give the host hash table a Meta-visible official training path."""
    from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import (
        HostSparseOptimizersContainer,
    )
    from torchtitan_npu.models.system3.model import System3HostHashTable, _System3HostLookup

    original_lookup = System3HostHashTable._lookup_rows
    original_lookup_forward = _System3HostLookup.forward

    def _meta_lookup_rows(self, flat_ids):
        if flat_ids.device.type != "meta":
            return original_lookup(self, flat_ids)
        # Meta keeps the formal autograd boundary: the rows carry a
        # _System3HostLookup backward, so the owner-local sparse gradient
        # accumulates exactly as in production (PR #24 review).
        if self._ep_size <= 1:
            return _System3HostLookup.apply(self.weight, flat_ids, self, None, None)
        return _System3HostLookup.apply(
            self.weight, flat_ids, self, _balanced_ep_lookup_rows, _balanced_ep_lookup_grad_rows
        )

    def _meta_lookup_forward(ctx, weight, flat_ids, table, lookup_rows, lookup_grad_rows):
        # The table weight keeps its CPU residency marker through the meta
        # build (host-offload is a memory fact), so the capture face is keyed
        # on the lookup request, which is always a meta tensor in capture.
        if flat_ids.device.type != "meta":
            return original_lookup_forward(ctx, weight, flat_ids, table, lookup_rows, lookup_grad_rows)
        if table._ep_size > 1:
            rows = lookup_rows(ctx, weight, flat_ids, table.lookup_mesh)
            ctx.distributed = True
        else:
            # Official local path: host gather, then one rows H2D transfer.
            # The ids D2H (b*s*nh int64, a few hundred bytes) cannot be
            # materialized from meta and is too small to model; the captured
            # graph keeps the dominant rows transfer visible with its real
            # byte count.
            head_dim = weight.shape[1]
            rows_host = torch.empty((flat_ids.numel(), head_dim), dtype=weight.dtype, device="cpu")
            rows = rows_host.to(device=flat_ids.device).view(*flat_ids.shape, head_dim)
            ctx.save_for_backward(flat_ids)
            ctx.distributed = False
        ctx.table = table
        ctx.lookup_grad_rows = lookup_grad_rows
        return rows

    original_lookup_backward = _System3HostLookup.backward

    def _meta_lookup_backward(ctx, grad_rows):
        """Meta face of the formal lookup backward.

        Production moves the row ids to the CPU table before accumulating;
        meta has no data to move and the host residency marker keeps the
        weight on CPU, so the capture face keeps the ids on the capture
        device and lets the accumulator's meta twin build the FP32 sparse
        structure. The distributed branch delegates to the saved callable,
        which is the production exchange off-capture and the balanced twin
        in capture.
        """
        table = ctx.table
        if ctx.distributed:
            grad_local_rows, local_ids = ctx.lookup_grad_rows(ctx, grad_rows)
        else:
            (flat_ids,) = ctx.saved_tensors
            local_ids = flat_ids.reshape(-1)
            if flat_ids.device.type != "meta":
                local_ids = local_ids.to(device=table.weight.device, dtype=torch.int64, copy=True)
            grad_local_rows = grad_rows.reshape(-1, grad_rows.shape[-1])
        table.accumulate_sparse_gradient(local_ids, grad_local_rows)
        return None, None, None, None, None

    scope.replace(System3HostHashTable, "_lookup_rows", _meta_lookup_rows)
    scope.replace(_System3HostLookup, "forward", _meta_lookup_forward)
    scope.replace(_System3HostLookup, "backward", _meta_lookup_backward)
    # The production accumulator moves rows to CPU and reads ids by value,
    # neither of which exists on meta; the meta twin keeps the FP32 sparse
    # accumulation structure (PR #19/#24 review).
    scope.replace(System3HostHashTable, "accumulate_sparse_gradient", _meta_accumulate_sparse_gradient)
    # The production clipping rescale coalesces; SparseMeta cannot.
    scope.replace(System3HostHashTable, "scale_pending_sparse_grad", _meta_scale_pending_sparse_grad)
    # The production attach converts the pending gradient to the weight
    # dtype, which dispatches a sparse _to_copy; the meta twin keeps it as
    # host-only work (FP32 identity on the capture path).
    scope.replace(System3HostHashTable, "prepare_sparse_optimizer_step", _meta_prepare_sparse_optimizer_step)
    # The production replica reduction pads and all-gathers on CPU with
    # value-read row counts; the single-process capture has no peer rows to
    # gather (see the twin docstring).
    scope.replace(
        System3HostHashTable,
        "reduce_sparse_gradient_across_replicas",
        _meta_reduce_sparse_gradient_across_replicas,
    )
    # The native SparseAdam functional needs SparseMeta kernels that do not
    # exist; the container twin routes the sparse table optimizers through
    # the meta step, which records the per-table update through the declared
    # virtual boundary (see _meta_host_sparse_container_step).
    scope.replace(HostSparseOptimizersContainer, "step", _meta_host_sparse_container_step)
    # The production wrapper reads norms by value; the meta twin keeps the
    # op structure with symbolic scalars instead.
    scope.replace(HostSparseOptimizersContainer, "clip_grad_norm", _meta_container_clip_grad_norm)
    return (
        "host-table lookup keeps the formal autograd boundary with visible transfers (EP=1) / "
        "balanced functional a2a (EP>1); pending sparse gradient, host clip and SparseAdam "
        "updates record through the fused.sparse_adam_update boundary"
    )


__all__ = ["install_system3_engram_meta_support"]
