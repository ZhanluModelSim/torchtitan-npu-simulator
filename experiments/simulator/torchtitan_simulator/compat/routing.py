# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Use native functional EP exchanges with balanced host split geometry."""

from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

import torch
from torch.distributed._functional_collectives import all_to_all_single


@dataclass
class _Geometry:
    dispatcher_id: int
    num_tokens: int
    top_k: int
    received_counts: torch.Tensor | None = None
    num_experts: int | None = None


_geometry: ContextVar[_Geometry | None] = ContextVar("simulator_routing_geometry", default=None)

# Balanced-EP proofs certified by the instrumented dispatchers. Meta captures
# cannot read tensor values, so consumers (grouped-mm cost facts) derive the
# per-expert split from these certified geometries instead of inventing one.
# Geometries repeat across layers; the (num_tokens, top_k, num_experts) triple
# is what a consumer matches on.
_certified_balanced_proofs: list[dict] = []


def certified_balanced_proofs() -> list[dict]:
    """Return the balanced-EP proof facts certified this session."""
    return _certified_balanced_proofs


def _register_balanced_proof(geometry: _Geometry) -> None:
    if geometry.num_tokens is None or geometry.top_k is None or geometry.num_experts is None:
        return
    proof = {
        "kind": "torchtitan_balanced_ep_v1",
        "num_tokens_per_rank": [geometry.num_tokens],
        "top_k": geometry.top_k,
        "num_experts": geometry.num_experts,
        "all_routers_force_load_balance": True,
        "source_version": "torchtitan_simulator_balanced_ep_v1",
    }
    if proof not in _certified_balanced_proofs:
        _certified_balanced_proofs.append(proof)


def balanced_splits(num_tokens, top_k, num_experts, ep_size, ep_rank):
    if num_experts % ep_size or not 0 <= ep_rank < ep_size:
        raise ValueError("Balanced EP geometry requires complete expert partitions and a valid EP rank")
    quotient, remainder = divmod(num_tokens * top_k, num_experts)
    counts = [quotient + int(expert < remainder) for expert in range(num_experts)]
    local_experts = num_experts // ep_size
    sends = [sum(counts[start : start + local_experts]) for start in range(0, num_experts, local_experts)]
    return sends, [sends[ep_rank]] * ep_size


def _make_round_robin_fact_hook(module):
    """Record the certified round-robin routing geometry on every forward.

    Both router families that honor ``_debug_force_load_balance`` share the
    same contract: expert ids are ``arange(assignments) % num_experts``, so
    the per-expert counts are the balanced split of the assignments. Meta
    captures cannot read the ids tensor, so the hook derives the fact from
    the certified shapes instead.
    """

    def hook(_module, _inputs, output):
        if not getattr(module, "_debug_force_load_balance", False):
            return
        tensors = output if isinstance(output, tuple) else (output,)
        ids = next(
            (t for t in tensors if isinstance(t, torch.Tensor) and t.dtype == torch.int64),
            None,
        )
        if ids is None or ids.numel() == 0 or not ids.is_meta:
            return
        experts = getattr(module, "num_experts", None)
        top_k = ids.shape[-1] if ids.dim() else 0
        if type(experts) is not int or experts <= 0 or top_k <= 0:
            return
        proof = {
            "kind": "debug_round_robin_balanced_v1",
            "assignments": int(ids.numel()),
            "num_experts": int(experts),
            "all_routers_force_load_balance": True,
            "source_version": "debug_round_robin_balanced_v1",
        }
        if proof not in _certified_balanced_proofs:
            _certified_balanced_proofs.append(proof)

    return hook


def install_balanced_dispatch(scope, model, *, uniform_geometry_source: str | None = None):
    routers = [module for module in model.modules() if hasattr(module, "_debug_force_load_balance")]
    if not routers:
        # Dense decoders have no routed experts; EP split derivation does not
        # apply and there is no dispatch class to instrument.
        return {
            "ep_token_exchange": "not-applicable-dense",
            "ep_token_dispatcher_classes": 0,
        }
    if any(not router._debug_force_load_balance for router in routers):
        raise ValueError("Meta EP split derivation requires verified balanced routing on every router")
    _certified_balanced_proofs.clear()
    for router in routers:
        router.register_forward_hook(_make_round_robin_fact_hook(router))
    classes = {
        type(dispatcher)
        for module in model.modules()
        if (dispatcher := getattr(module, "token_dispatcher", None)) is not None
        and hasattr(dispatcher, "_sync_token_count_exchange")
    }
    for cls in classes:
        _install_dispatch_class(
            scope,
            cls,
            verified_router_source="all_model_routers_force_load_balance_v1",
            uniform_geometry_source=uniform_geometry_source,
        )
    return {
        "ep_token_exchange": "functional_all_to_all_single",
        "ep_token_dispatcher_classes": len(classes),
    }


def _install_dispatch_class(scope, cls, *, verified_router_source=None, uniform_geometry_source=None):
    original_dispatch = cls.dispatch
    original_sync = cls._sync_token_count_exchange
    original_count_exchange = cls._token_count_exchange
    original_exchange = cls._dispatch_token_exchange
    original_combine = cls._combine_token_exchange
    completed_geometry = {}

    def facts_for(self, geometry, tensor, *, direction):
        if verified_router_source is None or uniform_geometry_source is None or geometry.num_experts is None:
            return None
        from torchtitan_simulator.capture.comm_events import _resolve_comm_ranks

        groups = _resolve_comm_ranks(self.ep_mesh)
        if len(groups) != 1:
            return None
        group = groups[0]
        return {
            "source": "torchtitan_balanced_ep_v1",
            "a2a_proof": {
                "kind": "torchtitan_balanced_ep_v1",
                "num_tokens_per_rank": [geometry.num_tokens] * len(group),
                "top_k": geometry.top_k,
                "num_experts": geometry.num_experts,
                "ordered_group": group,
                "all_routers_force_load_balance": True,
                "direction": direction,
                "trailing_shape": list(tensor.shape[1:]),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "source_version": "torchtitan_simulator_balanced_ep_v1",
                "router_verification_source": verified_router_source,
                "uniform_geometry_source": uniform_geometry_source,
            },
        }

    def exchange_with_facts(tensor, output_splits, input_splits, mesh, facts):
        if facts is None:
            return all_to_all_single(tensor, output_splits, input_splits, mesh)
        from torchtitan_simulator.capture.collective_facts import collective_fact_scope, retain_backward_facts

        with collective_fact_scope(facts):
            result = all_to_all_single(tensor, output_splits, input_splits, mesh)
        retain_backward_facts(result, facts)
        return result

    @wraps(original_dispatch)
    def dispatch(self, x, scores, indices, counts, *args, **kwargs):
        if x.device.type != "meta":
            return original_dispatch(self, x, scores, indices, counts, *args, **kwargs)
        geometry = _Geometry(id(self), x.shape[0], scores.shape[-1])
        token = _geometry.set(geometry)
        try:
            result = original_dispatch(self, x, scores, indices, counts, *args, **kwargs)
            # Combine is called after dispatch has left this dynamic scope.
            # Retain the instance too, so a recycled Python id cannot match.
            completed_geometry[id(self)] = (
                self,
                _Geometry(
                    geometry.dispatcher_id,
                    geometry.num_tokens,
                    geometry.top_k,
                    num_experts=geometry.num_experts,
                ),
            )
            return result
        finally:
            _geometry.reset(token)

    @wraps(original_sync)
    def sync(self, local_counts, received_counts, ep_size):
        if local_counts.device.type != "meta":
            return original_sync(self, local_counts, received_counts, ep_size)
        geometry = _geometry.get()
        if geometry is None or geometry.dispatcher_id != id(self):
            raise RuntimeError("EP count synchronization has no matching dispatch geometry")
        received_counts = torch.ops._c10d_functional.wait_tensor(received_counts).reshape(-1)
        sends, receives = balanced_splits(
            geometry.num_tokens,
            geometry.top_k,
            local_counts.numel(),
            ep_size,
            self.ep_mesh.get_local_rank(),
        )
        geometry.received_counts = received_counts
        geometry.num_experts = local_counts.numel()
        _register_balanced_proof(geometry)
        return received_counts, sends, receives

    @wraps(original_count_exchange)
    def count_exchange(self, num_local_tokens_per_expert_E, pg, ep_size):
        if num_local_tokens_per_expert_E.device.type != "meta":
            return original_count_exchange(self, num_local_tokens_per_expert_E, pg, ep_size)
        tensor = num_local_tokens_per_expert_E.view(ep_size, -1)
        geometry = _geometry.get()
        facts = None
        if (
            geometry is not None
            and geometry.dispatcher_id == id(self)
            and verified_router_source is not None
            and uniform_geometry_source is not None
        ):
            geometry.num_experts = num_local_tokens_per_expert_E.numel()
            # The producer knows all SPMD ranks exchange the same static
            # expert-count geometry. This is not inferred from one split row.
            facts = {
                "source": "torchtitan_equal_expert_count_exchange_v1",
                "uniform_across_ranks": True,
                "send_rows": [[1] * ep_size for _ in range(ep_size)],
                "recv_rows": [[1] * ep_size for _ in range(ep_size)],
                "rank_input_shapes": [list(tensor.shape)] * ep_size,
                "rank_output_shapes": [list(tensor.shape)] * ep_size,
                "uniform_geometry_source": uniform_geometry_source,
            }
        return exchange_with_facts(tensor, None, None, self.ep_mesh, facts)

    @wraps(original_exchange)
    def exchange(self, routed_input_ND, pg, output_splits, input_splits):
        if routed_input_ND.device.type != "meta":
            return original_exchange(self, routed_input_ND, pg, output_splits, input_splits)
        from torchtitan_simulator.capture.comm_events import collective_dependencies

        geometry = _geometry.get()
        if geometry is None or geometry.dispatcher_id != id(self) or geometry.received_counts is None:
            raise RuntimeError("EP token exchange has no synchronized count dependency")
        with collective_dependencies(geometry.received_counts):
            return exchange_with_facts(
                routed_input_ND,
                output_splits,
                input_splits,
                self.ep_mesh,
                facts_for(self, geometry, routed_input_ND, direction="dispatch"),
            )

    @wraps(original_combine)
    def combine(self, routed_output_RD, pg, input_splits, output_splits):
        if routed_output_RD.device.type != "meta":
            return original_combine(self, routed_output_RD, pg, input_splits, output_splits)
        entry = completed_geometry.get(id(self))
        geometry = entry[1] if entry is not None and entry[0] is self else _geometry.get()
        facts = (
            facts_for(self, geometry, routed_output_RD, direction="combine")
            if geometry is not None and geometry.dispatcher_id == id(self)
            else None
        )
        return exchange_with_facts(routed_output_RD, input_splits, output_splits, self.ep_mesh, facts)

    scope.replace(cls, "dispatch", dispatch)
    scope.replace(cls, "_token_count_exchange", count_exchange)
    scope.replace(cls, "_sync_token_count_exchange", sync)
    scope.replace(cls, "_dispatch_token_exchange", exchange)
    scope.replace(cls, "_combine_token_exchange", combine)


_head_local: ContextVar[int | None] = ContextVar("simulator_head_parallel_head_local", default=None)


def install_head_parallel_facts(scope, model) -> dict[str, object]:
    """Producer facts for the head-parallel flavor's native exchanges.

    The HP dispatch/combine all-to-alls carry uniform, shape-derived split
    geometry, and the debug-balanced routing makes every grouped-mm group
    size statically computable. Neither op can express those values in its
    schema (splits and ``offs`` are tensors), so the capture facts and op
    attributes are bound around the calls from here, where the simulator
    side may depend on the model package. The dynamic-routing (top-k) path
    keeps the all-to-all facts but cannot provide group sizes, so grouped
    nodes stay unpredicted there.
    """
    from functools import wraps

    import torch

    from torchtitan_npu.models.multi_head_moe import head_parallel as hp
    from torchtitan_simulator.capture.collective_facts import (
        collective_fact_scope,
        op_attribute_scope,
        retain_backward_facts,
        retain_op_attributes,
    )

    hp_modules = [
        module
        for module in model.modules()
        if isinstance(module, hp.HeadParallelMultiHeadMoE)
    ]
    if not hp_modules:
        return {"head_parallel_exchanges": "not-applicable"}
    routers = [module.router for module in hp_modules]
    if any(not router._debug_force_load_balance for router in routers):
        # Dynamic routing: the exchange geometry stays uniform, but the
        # per-expert group sizes are data-dependent and unknowable on meta.
        provide_group_sizes = False
        k_per_head = experts_per_head = None
    else:
        provide_group_sizes = True
        k_per_head = routers[0].k_per_head
        experts_per_head = routers[0].experts_per_head
        if any(router.k_per_head != k_per_head or router.experts_per_head != experts_per_head for router in routers):
            raise ValueError("Head-parallel capture facts require a uniform per-head router geometry")
    group_sizes_cache: dict[tuple[int, int], list[int]] = {}

    def compute_group_sizes(total_rows: int, head_local: int) -> list[int]:
        cached = group_sizes_cache.get((total_rows, head_local))
        if cached is None:
            tokens_all = total_rows // (head_local * k_per_head)
            _, expert_ids = hp.static_balanced_expert_ids(
                tokens_all, head_local, k_per_head, experts_per_head, torch.device("cpu")
            )
            cached = torch.bincount(expert_ids.reshape(-1), minlength=head_local * experts_per_head).tolist()
            assert sum(cached) == total_rows
            group_sizes_cache[(total_rows, head_local)] = cached
        return cached

    original_dispatch = hp.head_dispatch
    original_combine = hp.head_combine

    @wraps(original_dispatch)
    def dispatch_with_facts(x_HTS, ep_mesh, head_local):
        ep = ep_mesh.size()
        rows = [[head_local] * ep for _ in range(ep)]
        facts = {
            "source": "head_parallel_uniform_exchange_v1",
            "uniform_across_ranks": True,
            "send_rows": rows,
            "recv_rows": rows,
        }
        # The grouped_mm calls that consume this dispatch happen later in the
        # same forward, so the context must stay open until combine resets it.
        token = _head_local.set(head_local)
        try:
            with collective_fact_scope(facts):
                result = original_dispatch(x_HTS, ep_mesh, head_local)
        except BaseException:
            _head_local.reset(token)
            raise
        retain_backward_facts(result, facts)
        return result

    @wraps(original_combine)
    def combine_with_facts(y_TSH, ep_mesh, t_local):
        ep = ep_mesh.size()
        rows = [[t_local] * ep for _ in range(ep)]
        facts = {
            "source": "head_parallel_uniform_exchange_v1",
            "uniform_across_ranks": True,
            "send_rows": rows,
            "recv_rows": rows,
        }
        try:
            with collective_fact_scope(facts):
                result = original_combine(y_TSH, ep_mesh, t_local)
        finally:
            _head_local.set(None)
        retain_backward_facts(result, facts)
        return result

    scope.replace(hp, "head_dispatch", dispatch_with_facts)
    scope.replace(hp, "head_combine", combine_with_facts)

    if not provide_group_sizes:
        return {"head_parallel_exchanges": "facts-bound", "grouped_mm_group_sizes": "dynamic-routing"}

    original_grouped_mm = torch._grouped_mm

    @wraps(original_grouped_mm)
    def grouped_mm_with_facts(a, b, *, offs=None, **kwargs):
        head_local = _head_local.get()
        if a.device.type != "meta" or head_local is None or offs is None:
            return original_grouped_mm(a, b, offs=offs, **kwargs)
        counts = compute_group_sizes(a.shape[0], head_local)
        attributes = {"group_sizes": counts}
        with op_attribute_scope(attributes, raw_name="aten._grouped_mm.default"):
            result = original_grouped_mm(a, b, offs=offs, **kwargs)
        retain_op_attributes(result, attributes, raw_name="aten._grouped_mm.default")
        return result

    scope.replace(torch, "_grouped_mm", grouped_mm_with_facts)
    return {"head_parallel_exchanges": "facts-bound", "grouped_mm_group_sizes": "static-balanced"}
