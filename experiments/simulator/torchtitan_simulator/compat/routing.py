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


_geometry: ContextVar[_Geometry | None] = ContextVar("simulator_routing_geometry", default=None)


def balanced_splits(num_tokens, top_k, num_experts, ep_size, ep_rank):
    if num_experts % ep_size or not 0 <= ep_rank < ep_size:
        raise ValueError("Balanced EP geometry requires complete expert partitions and a valid EP rank")
    quotient, remainder = divmod(num_tokens * top_k, num_experts)
    counts = [quotient + int(expert < remainder) for expert in range(num_experts)]
    local_experts = num_experts // ep_size
    sends = [sum(counts[start : start + local_experts]) for start in range(0, num_experts, local_experts)]
    return sends, [sends[ep_rank]] * ep_size


def install_balanced_dispatch(scope, model):
    routers = [module for module in model.modules() if hasattr(module, "_debug_force_load_balance")]
    if not routers or any(not router._debug_force_load_balance for router in routers):
        raise ValueError("Meta EP split derivation requires verified balanced routing on every router")
    classes = {
        type(dispatcher)
        for module in model.modules()
        if (dispatcher := getattr(module, "token_dispatcher", None)) is not None
        and hasattr(dispatcher, "_sync_token_count_exchange")
    }
    for cls in classes:
        _install_dispatch_class(scope, cls)
    return {
        "ep_token_exchange": "functional_all_to_all_single",
        "ep_token_dispatcher_classes": len(classes),
    }


def _install_dispatch_class(scope, cls):
    original_dispatch = cls.dispatch
    original_sync = cls._sync_token_count_exchange
    original_count_exchange = cls._token_count_exchange
    original_exchange = cls._dispatch_token_exchange
    original_combine = cls._combine_token_exchange

    @wraps(original_dispatch)
    def dispatch(self, x, scores, indices, counts, *args, **kwargs):
        if x.device.type != "meta":
            return original_dispatch(self, x, scores, indices, counts, *args, **kwargs)
        token = _geometry.set(_Geometry(id(self), x.shape[0], scores.shape[-1]))
        try:
            return original_dispatch(self, x, scores, indices, counts, *args, **kwargs)
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
        return received_counts, sends, receives

    @wraps(original_count_exchange)
    def count_exchange(self, num_local_tokens_per_expert_E, pg, ep_size):
        if num_local_tokens_per_expert_E.device.type != "meta":
            return original_count_exchange(self, num_local_tokens_per_expert_E, pg, ep_size)
        return all_to_all_single(num_local_tokens_per_expert_E.view(ep_size, -1), None, None, self.ep_mesh)

    @wraps(original_exchange)
    def exchange(self, routed_input_ND, pg, output_splits, input_splits):
        if routed_input_ND.device.type != "meta":
            return original_exchange(self, routed_input_ND, pg, output_splits, input_splits)
        from torchtitan_simulator.capture.comm_events import collective_dependencies

        geometry = _geometry.get()
        if geometry is None or geometry.dispatcher_id != id(self) or geometry.received_counts is None:
            raise RuntimeError("EP token exchange has no synchronized count dependency")
        with collective_dependencies(geometry.received_counts):
            return all_to_all_single(routed_input_ND, output_splits, input_splits, self.ep_mesh)

    @wraps(original_combine)
    def combine(self, routed_output_RD, pg, input_splits, output_splits):
        if routed_output_RD.device.type != "meta":
            return original_combine(self, routed_output_RD, pg, input_splits, output_splits)
        return all_to_all_single(routed_output_RD, input_splits, output_splits, self.ep_mesh)

    scope.replace(cls, "dispatch", dispatch)
    scope.replace(cls, "_token_count_exchange", count_exchange)
    scope.replace(cls, "_sync_token_count_exchange", sync)
    scope.replace(cls, "_dispatch_token_exchange", exchange)
    scope.replace(cls, "_combine_token_exchange", combine)
