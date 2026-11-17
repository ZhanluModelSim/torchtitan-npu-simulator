# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Insert checksum checks after supported NPU matmul nodes.

The pass instruments the first graph and leaves activation to the torch-npu gate::

    install pass -> first graph: matmul -> checksum op (gate off: no-op)
    gradient strikes -> gate on
    later step -> same graph -> checksum runs

Keeping the op in the original graph avoids a compiler reset or recompile.
"""

import torch
import torch._inductor.config as inductor_config
from torch._inductor.custom_graph_pass import (
    CustomInferenceAwareGraphPass,
    get_custom_graph_passes,
    get_hash_for_files,
)

import torchtitan_npu.ops.misc.sdc_checksum  # noqa: F401

_MATMUL_TARGETS = {
    torch.ops.aten.mm.default,
    torch.ops.aten.matmul.default,
    torch.ops.aten.bmm.default,
}
_CHECKSUM_OP = torch.ops.torchtitan_npu.sdc_matmul_checksum.default


def _is_checksum_candidate(node: torch.fx.Node) -> bool:
    if node.op != "call_function" or node.target not in _MATMUL_TARGETS or len(node.args) < 2:
        return False
    left, right = node.args[:2]
    return all(
        isinstance(value, torch.fx.Node)
        and isinstance(value.meta.get("val"), torch.Tensor)
        and value.meta["val"].dtype in (torch.bfloat16, torch.float32)
        and value.meta["val"].device.type == "npu"
        for value in (left, right, node)
    )


class _ChecksumPostGradPass(CustomInferenceAwareGraphPass):
    """Instrument NPU training matmuls supported by the native checksum runtime."""

    def __call__(self, graph: torch.fx.Graph, is_inference: bool) -> None:
        if is_inference:
            return
        for node in tuple(graph.nodes):
            if not _is_checksum_candidate(node):
                continue
            left, right = node.args[:2]
            with graph.inserting_after(node):
                check = graph.call_function(_CHECKSUM_OP, args=(left, right, node))
            # The ordered op has no data result; explicit fake metadata keeps
            # later passes from treating it as another tensor producer.
            check.meta["val"] = None

    def uuid(self) -> bytes | None:
        # Tie Inductor's cache key to this instrumentation implementation so a
        # changed pass cannot reuse a stale compiled graph.
        return get_hash_for_files((__file__,))


_CHECKSUM_POST_GRAD_PASS = _ChecksumPostGradPass()


def install_checksum_pass() -> None:
    """Compose the checksum pass once without replacing existing post-grad passes."""

    installed = get_custom_graph_passes(inductor_config.post_grad_custom_post_pass)
    if _CHECKSUM_POST_GRAD_PASS not in installed:
        inductor_config.post_grad_custom_post_pass = (
            *installed,
            _CHECKSUM_POST_GRAD_PASS,
        )
