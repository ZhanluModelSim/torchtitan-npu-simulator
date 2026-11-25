# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observe DeepSeek V4's global CP planner while preserving its native exchange and autograd."""

import logging
from functools import wraps
from importlib import metadata

import torch
import torch.distributed as dist

from torchtitan_simulator.capture.collective_facts import collective_fact_scope, retain_backward_facts


def observe_deepseek_v4_cp_exchange(scope) -> list[list[int]]:
    """Observe exchanges and retain the planner's actual host split lists for the Meta lift."""
    from torchtitan_npu.models.deepseek_v4 import token_dispatcher

    routing = token_dispatcher._routing_geometry
    exchange = token_dispatcher.CPTokenDispatcher._all_to_all
    plans = {}
    host_splits = []
    warned_missing_backward = False

    @wraps(routing)
    def observe_plan(*args, **kwargs):
        rows = routing(*args, **kwargs)
        sends = tuple(tuple(row[1]) for row in rows)
        receives = tuple(tuple(row[2]) for row in rows)
        for local_rank, (_, send, receive, _) in enumerate(rows):
            # The host lists survive the Meta lift. Keep both references so IDs cannot
            # be reused or confused between two plans with equal local split counts.
            plans[id(send), id(receive)] = send, receive, local_rank, sends, receives
            host_splits.extend((send, receive))
        return rows

    @wraps(exchange)
    def observe_exchange(dispatcher, x, in_splits, out_splits):
        nonlocal warned_missing_backward
        plan = plans.get((id(in_splits), id(out_splits)))
        if plan is None or plan[0] is not in_splits or plan[1] is not out_splits or x.device.type != "meta":
            return exchange(dispatcher, x, in_splits, out_splits)
        _, _, local, sends, receives = plan
        group = dist.get_process_group_ranks(dispatcher.cp_mesh.get_group())
        if (
            len(group) != len(sends)
            or group[local] != dist.get_rank()
            or tuple(in_splits) != sends[local]
            or tuple(out_splits) != receives[local]
            or x.shape[0] != sum(in_splits)
        ):
            raise ValueError("CP exchange contradicts its native routing plan or ordered process group")
        # All ranks exchange rows of the same projected model tensor for this CP
        # group. Its trailing dimensions are independent of the per-rank row totals.
        tail = x.shape[1:]
        facts = {
            "source": "deepseek_v4_cp_routing_geometry_v1",
            "send_rows": sends,
            "recv_rows": receives,
            "rank_input_shapes": [(sum(row), *tail) for row in sends],
            "rank_output_shapes": [(sum(row), *tail) for row in receives],
        }
        with collective_fact_scope(facts):
            result = exchange(dispatcher, x, in_splits, out_splits)
        if x.requires_grad and torch.is_grad_enabled():
            unmatched_nodes = retain_backward_facts(result, facts, source=x)
            if unmatched_nodes is not None and not warned_missing_backward:
                warned_missing_backward = True
                versions = []
                for package in ("torchtitan-npu", "spmd-types", "torch"):
                    try:
                        version = metadata.version(package)
                    except metadata.PackageNotFoundError:
                        version = "unknown"
                    versions.append(f"{package}={version}")
                logging.getLogger(__name__).warning(
                    "DeepSeek V4 CP exchange retained forward routing facts but could not identify the native "
                    "backward collective. Backward geometry is unavailable. Observed nodes: %s. Installed: %s",
                    ", ".join(unmatched_nodes) or "<none>",
                    ", ".join(versions),
                )
        return result

    scope.replace(token_dispatcher, "_routing_geometry", observe_plan)
    scope.replace(token_dispatcher.CPTokenDispatcher, "_all_to_all", observe_exchange)
    return host_splits
