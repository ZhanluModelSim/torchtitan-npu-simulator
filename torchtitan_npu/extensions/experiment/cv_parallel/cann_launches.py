# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Recover CV-chunk CANN launches without a torch-to-NPU flow."""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from decimal import Decimal
from typing import Any


def associate_cann_launches(
    events: list[dict[str, Any]],
    node_scopes: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Recover missing torch_to_npu links via ACL connection and queue IDs.

    Some custom launches have a device connection_id and AscendCL record but
    no HostToDevice start/torch_to_npu flow. Follow the concrete launch through
    Dequeue -> correlation_id -> Enqueue -> native FX scope. Host intervals
    establish ownership only; their durations are never device costs.
    """
    process_names = {
        event.get("pid"): event.get("args", {}).get("name")
        for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    enqueue: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    dequeues: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    scopes: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    scope_ids = {id(scope): stable_id for stable_id, scope in node_scopes.items()}
    for scope in node_scopes.values():
        scopes[scope.get("pid"), scope.get("tid")].append(scope)
    for event in events:
        if event.get("ph") != "X":
            continue
        if event.get("cat") == "enqueue":
            enqueue[event.get("pid"), event.get("args", {}).get("correlation_id")].append(event)
        elif event.get("cat") == "dequeue":
            dequeues[event.get("tid")].append(event)

    def index_ranges(groups):
        times = {}
        for lane, ranges in groups.items():
            ranges.sort(key=lambda event: Decimal(str(event["ts"])))
            times[lane] = [Decimal(str(event["ts"])) for event in ranges]
        return times

    dequeue_times = index_ranges(dequeues)
    scope_times = index_ranges(scopes)

    def containing(ranges, times, event):
        begin = Decimal(str(event["ts"]))
        end = begin + Decimal(str(event["dur"]))
        index = bisect_right(times, begin) - 1
        if index < 0:
            return None
        parent = ranges[index]
        if end <= Decimal(str(parent["ts"])) + Decimal(str(parent["dur"])):
            return parent
        return None

    connection_owners: dict[Any, set[str]] = defaultdict(set)
    for event in events:
        args = event.get("args", {})
        if (
            event.get("ph") != "X"
            or process_names.get(event.get("pid")) != "CANN"
            or args.get("level") != "acl"
            or args.get("connection_id") is None
        ):
            continue
        tid = event.get("tid")
        dequeue = containing(dequeues.get(tid, ()), dequeue_times.get(tid, ()), event)
        if dequeue is None:
            continue
        queue_key = dequeue.get("pid"), dequeue.get("args", {}).get("correlation_id")
        enqueues = enqueue.get(queue_key, ())
        if queue_key[1] is None or len(enqueues) != 1:
            continue
        launch = enqueues[0]
        lane = launch.get("pid"), launch.get("tid")
        scope = containing(scopes.get(lane, ()), scope_times.get(lane, ()), launch)
        if scope is not None:
            connection_owners[args["connection_id"]].add(scope_ids[id(scope)])
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("ph") != "X" or process_names.get(event.get("pid")) != "Ascend Hardware":
            continue
        owners = connection_owners.get(event.get("args", {}).get("connection_id"), ())
        if len(owners) == 1:
            result[next(iter(owners))].append(event)
    return dict(result)
