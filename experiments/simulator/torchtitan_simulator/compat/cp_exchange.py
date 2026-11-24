# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observe DeepSeek V4's global CP planner while preserving its native exchange and autograd."""

from functools import wraps

import torch.distributed as dist

from torchtitan_simulator.capture.collective_facts import collective_fact_scope, retain_backward_facts


def observe_deepseek_v4_cp_exchange(scope):
    from torchtitan_npu.models.deepseek_v4 import token_dispatcher

    routing = token_dispatcher._routing_geometry
    exchange = token_dispatcher.CPTokenDispatcher._all_to_all
    plans = {}

    @wraps(routing)
    def observe_plan(*args, **kwargs):
        rows = routing(*args, **kwargs)
        sends = tuple(tuple(row[1]) for row in rows)
        receives = tuple(tuple(row[2]) for row in rows)
        for local_rank, (_, send, receive, _) in enumerate(rows):
            # The host lists survive the Meta lift. Keep both references so IDs cannot
            # be reused or confused between two plans with equal local split counts.
            plans[id(send)] = send, receive, local_rank, sends, receives
        return rows

    @wraps(exchange)
    def observe_exchange(dispatcher, x, in_splits, out_splits):
        plan = plans.get(id(in_splits))
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
        retain_backward_facts(result, facts, source=x)
        return result

    scope.replace(token_dispatcher, "_routing_geometry", observe_plan)
    scope.replace(token_dispatcher.CPTokenDispatcher, "_all_to_all", observe_exchange)
