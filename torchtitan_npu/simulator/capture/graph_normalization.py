# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Normalize captured L0 graphs before exposing them to backend replay."""

from __future__ import annotations

from typing import Hashable

from torchtitan_npu.simulator.ir.op_node import OpNode


def fold_metadata_views(nodes: dict[Hashable, OpNode]) -> dict[Hashable, OpNode]:
    """Remove alias-only view nodes and preserve their transitive dependencies.

    Dispatcher capture retains views to maintain exact alias and mutation
    provenance for memory analysis. Backend replay consumes ``WorkloadGraph``
    instead, where these nodes have no device work. This pass contracts every
    metadata-view path, connecting each retained consumer to its closest
    retained producers. References to nodes outside ``nodes`` are retained as
    external dependencies.
    """
    removed_ids = {
        op_id
        for op_id, node in nodes.items()
        if node.annotations.get("metadata_view", False)
    }
    if not removed_ids:
        return nodes

    predecessor_cache: dict[Hashable, set[Hashable]] = {}

    def retained_predecessors(op_id: Hashable, visiting: set[Hashable]) -> set[Hashable]:
        if op_id not in removed_ids:
            return {op_id}
        if op_id in predecessor_cache:
            return predecessor_cache[op_id]
        if op_id in visiting:
            # The capture graph must be acyclic, but retain a conservative
            # result rather than recursing forever if a malformed trace is
            # handed to this normalization pass.
            return set()

        node = nodes[op_id]
        result: set[Hashable] = set()
        for predecessor_id in node.predecessors:
            if predecessor_id in nodes:
                result.update(retained_predecessors(predecessor_id, visiting | {op_id}))
            else:
                result.add(predecessor_id)
        predecessor_cache[op_id] = result
        return result

    retained_nodes = {
        op_id: node for op_id, node in nodes.items() if op_id not in removed_ids
    }
    for op_id, node in retained_nodes.items():
        predecessors: set[Hashable] = set()
        for predecessor_id in node.predecessors:
            if predecessor_id in removed_ids:
                predecessors.update(retained_predecessors(predecessor_id, set()))
            else:
                predecessors.add(predecessor_id)
        predecessors.discard(op_id)
        node.predecessors = sorted(predecessors, key=str)
        node.successors = []

    for op_id, node in retained_nodes.items():
        for predecessor_id in node.predecessors:
            if predecessor_id in retained_nodes:
                retained_nodes[predecessor_id].successors.append(op_id)

    return retained_nodes
