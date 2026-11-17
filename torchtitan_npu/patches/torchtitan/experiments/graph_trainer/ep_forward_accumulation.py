# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4708
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Materialize functionalized forward buffer accumulations across EP chunks.

Mutation functionalization rewrites an update such as ``buffer.add_(delta)``
into pure dataflow followed by a graph epilogue::

    updated = torch.ops.aten.add.Tensor(buffer, delta)
    torch.ops.aten.copy_.default(buffer, updated)

After EP chunking, the two pure adds both read the original buffer.  Adding
their results would count that buffer twice, while rejecting the chunkless
live-out loses the mutation semantics entirely.  This patch supplies the
missing forward-accumulation proof and reconstructs the value as::

    updated = (buffer + delta_chunk0) + delta_chunk1

The proof is deliberately limited to the exact functionalization epilogue;
all other chunkless forward live-outs retain TorchTitan's native rejection.
The mutation-functionalization pipeline installs this patch on demand.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import torch
import torch.fx as fx
import torchtitan.experiments.graph_trainer.ep_chunk_pass
from torchtitan.tools.logging import logger


def _forward_accumulation_delta(
    live_out: fx.Node,
    copies: tuple[fx.Node, fx.Node],
    users: tuple[fx.Node, ...],
) -> fx.Node | None:
    """Return chunk 1's delta when a functionalized buffer update is proven."""
    if (
        len(users) != 1
        or live_out.op != "call_function"
        or live_out.target is not torch.ops.aten.add.Tensor
        or len(live_out.args) != 2
    ):
        return None

    copy_user = users[0]
    if (
        copy_user.op != "call_function"
        or copy_user.target is not torch.ops.aten.copy_.default
        or len(copy_user.args) < 2
        or copy_user.args[1] is not live_out
    ):
        return None

    # Functionalization of ``buffer.add_(delta)`` keeps the mutation target as
    # the first operand of the pure add and as the destination of ``copy_``.
    buffer = copy_user.args[0]
    if not isinstance(buffer, fx.Node) or live_out.args[0] is not buffer:
        return None

    first, second = copies
    if any(
        node.op != "call_function"
        or node.target is not torch.ops.aten.add.Tensor
        or len(node.args) != 2
        or node.args[0] is not buffer
        or node.kwargs != live_out.kwargs
        for node in (first, second)
    ):
        return None

    delta = second.args[1]
    return delta if isinstance(delta, fx.Node) else None


def apply() -> None:
    current = torchtitan.experiments.graph_trainer.ep_chunk_pass._materialize_live_out
    if getattr(current, "npu_supports_forward_accumulation", False):
        return

    @wraps(current)
    def materialize_live_out(
        gm: fx.GraphModule,
        plan,
        live_out: fx.Node,
        copies: tuple[fx.Node, fx.Node],
        users: tuple[fx.Node, ...],
        *,
        mode,
        symbol_hints: dict[object, int],
        original_meta: dict[str, Any],
    ) -> fx.Node:
        delta = _forward_accumulation_delta(live_out, copies, users)
        if delta is None:
            return current(
                gm,
                plan,
                live_out,
                copies,
                users,
                mode=mode,
                symbol_hints=symbol_hints,
                original_meta=original_meta,
            )

        # Chunk 0 already computes ``buffer + delta0``. Accumulating only
        # chunk 1's delta preserves the original buffer exactly once.
        materialized = gm.graph.call_function(
            torch.ops.aten.add.Tensor,
            args=(copies[0], delta),
            kwargs=dict(live_out.kwargs),
        )
        original_val = original_meta.get("val")
        materialized.meta["val"] = original_val
        materialized.meta["chunked_materialization_proof"] = "forward_functionalized_buffer_accumulation"
        materialized._rename(f"{live_out.name}_chunk_materialized")
        torchtitan.experiments.graph_trainer.ep_chunk_pass._set_synthetic_meta(
            materialized,
            region=plan.region,
            role="materialization",
        )
        torchtitan.experiments.graph_trainer.ep_chunk_pass._validate_materialized(
            materialized,
            live_out,
            original_val=original_val if isinstance(original_val, torch.Tensor) else None,
            symbol_hints=symbol_hints,
        )
        logger.debug(
            "Chunk pass add-materialized %s: functionalized forward buffer accumulation",
            live_out.name,
        )
        return materialized

    # pyrefly: ignore [missing-attribute]
    materialize_live_out.npu_supports_forward_accumulation = True
    torchtitan.experiments.graph_trainer.ep_chunk_pass._materialize_live_out = materialize_live_out
    logger.info("Enabled GraphTrainer EP forward buffer accumulation patch")
