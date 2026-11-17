# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4516

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deduplicate ready nodes across EP-overlap chunk candidates.

The upstream ``_ready_nodes`` can return a node once for each chunk containing
it as a candidate. Keep its scheduling decisions and first-occurrence order,
but remove duplicate results before they become cyclic phase-order edges.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torchtitan.experiments.graph_trainer.ep_overlap_pass as ep_overlap_pass
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    import torch.fx as fx
    from torchtitan.experiments.graph_trainer.ep_pass_utils import ChunkedRegion, ChunkOwner


def apply() -> None:
    import torchtitan.experiments.graph_trainer.ep_overlap_pass

    current = torchtitan.experiments.graph_trainer.ep_overlap_pass._ready_nodes
    if getattr(current, "npu_dedups_ready_nodes", False):
        return

    def ready_nodes(
        *,
        candidates_by_chunk: dict[int, set[fx.Node]],
        emitted: set[fx.Node],
        region: ChunkedRegion,
        chunk_order: tuple[int, ...],
        order: dict[fx.Node, int],
        owner_by_node: dict[fx.Node, ChunkOwner],
        include_waits: bool,
    ) -> tuple[fx.Node, ...]:
        """Return schedulable body nodes, each candidate node at most once."""
        ready = []
        selected: set[fx.Node] = set()
        for chunk_id in chunk_order:
            body = region.bodies_by_chunk[chunk_id]
            candidates = sorted(candidates_by_chunk.get(chunk_id, set()) - emitted, key=order.__getitem__)
            for node in candidates:
                if node in selected:
                    continue
                if not include_waits and ep_overlap_pass._is_c10d_functional_node(node):
                    continue
                deps = ep_overlap_pass._body_deps(node, body=body, owner_by_node=owner_by_node)
                if all(dep in emitted for dep in deps):
                    ready.append(node)
                    selected.add(node)
        return tuple(ready)

    # pyrefly: ignore [missing-attribute]
    ready_nodes.npu_dedups_ready_nodes = True
    ep_overlap_pass._ready_nodes = ready_nodes
    logger.info("Enabled GraphTrainer EP overlap ready-node dedup patch")


apply()
