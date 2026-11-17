# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4650
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Normalize EP_token_exchange annotations on token-exchange shape queries.

``aten.sym_size.int`` can inherit the annotation of its all-to-all producer.
Remove only that key for queries of an in-body token-exchange launch, then let
the upstream collector perform its normal validation and wait normalization.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

from functools import wraps

import torch
import torch.fx as fx
import torchtitan.experiments.graph_trainer.ep_overlap_pass as ep_overlap_pass
from torchtitan.experiments.graph_trainer.common_utils import _EP_TOKEN_EXCHANGE
from torchtitan.tools.logging import logger


def _is_token_exchange_shape_query(node: fx.Node, node_set: set[fx.Node]) -> bool:
    """Return whether a node is a size query on an in-body token exchange."""
    if node.op != "call_function" or node.target != torch.ops.aten.sym_size.int or len(node.args) != 2:
        return False
    producer = node.args[0]
    return (
        isinstance(producer, fx.Node) and producer in node_set and ep_overlap_pass._is_token_exchange_launch(producer)
    )


def apply() -> None:
    current = ep_overlap_pass._collect_token_exchanges
    if getattr(current, "npu_normalizes_ep_shape_queries", False):
        return

    @wraps(current)
    def collect_token_exchanges(body, *, order):
        node_set = set(body.nodes)
        for node in body.nodes:
            if not _is_token_exchange_shape_query(node, node_set):
                continue
            custom = ep_overlap_pass._custom_meta(node)
            if _EP_TOKEN_EXCHANGE in custom:
                custom = dict(custom)
                del custom[_EP_TOKEN_EXCHANGE]
                node.meta["custom"] = custom
        return current(body, order=order)

    # pyrefly: ignore [missing-attribute]
    collect_token_exchanges.npu_normalizes_ep_shape_queries = True
    ep_overlap_pass._collect_token_exchanges = collect_token_exchanges
    logger.info("Enabled GraphTrainer EP overlap shape-query annotation normalization patch")


apply()
