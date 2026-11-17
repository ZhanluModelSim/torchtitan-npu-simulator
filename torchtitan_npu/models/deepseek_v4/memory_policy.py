# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4-specific GraphTrainer memory policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.distributed.fsdp import (
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_module_fqn,
    _is_backward_node,
    matches_module_fqn_pattern,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _find_fsdp_unshard_save_nodes,
    _make_full_memory_policy,
)
from torchtitan.experiments.graph_trainer.registry import (
    register_memory_policy,
)
from torchtitan.tools.logging import logger

from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import memory_policy

if TYPE_CHECKING:
    from collections.abc import Sequence

_GEMMS = frozenset(
    {
        "aten::linear",
        "aten::mm",
        "aten::matmul",
        "aten::bmm",
        "aten::addmm",
        "aten::_grouped_mm",
        "npu::npu_grouped_matmul",
    }
)
_SWIGLU_OP_NAMES = {"swiglu_group", "swiglu_group_forward"}
_SHARED_EXPERTS = ("layers.*.moe.shared_experts",)
_ROUTED_EXPERTS = ("layers.*.moe.routed_experts.inner_experts",)


def _op_name(node: torch.fx.Node) -> str:
    schema = getattr(node.target, "_schema", None)
    return schema.name if schema is not None else ""


def _root_for(node: torch.fx.Node, patterns: Sequence[str]) -> str | None:
    parts = _get_module_fqn(node).split(".")
    for pattern in patterns:
        root = ".".join(parts[: len(pattern.split("."))])
        if matches_module_fqn_pattern(pattern, root):
            return root
    return None


def _swiglu_w13_candidate(
    node: torch.fx.Node,
    module_patterns: Sequence[str],
) -> tuple[str, torch.fx.Node] | None:
    if node.op != "call_function" or _is_backward_node(node):
        return None
    if _op_name(node).split("::")[-1] not in _SWIGLU_OP_NAMES:
        return None

    root = _root_for(node, module_patterns)
    if root is None:
        return None

    output = node.meta.get("val")
    if isinstance(output, (list, tuple)):
        output = output[0] if output else None
    if not (isinstance(output, torch.Tensor) and output.ndim and isinstance(output.shape[-1], int)):
        return None

    candidates = []
    for inp in node.all_input_nodes:
        value = inp.meta.get("val")
        if (
            isinstance(value, torch.Tensor)
            and value.ndim
            and isinstance(value.shape[-1], int)
            and value.shape[-1] == 2 * output.shape[-1]
            and _root_for(inp, module_patterns) == root
        ):
            # The 2F tensor consumed by SwiGLU is the activation required by
            # its backward, regardless of whether its producer is an ATen
            # GEMM, a packed NPU GEMM, or a quantize/dequantize chain.
            candidates.append(inp)

    candidates = set(candidates)
    if len(candidates) != 1:
        raise ValueError(
            f"Cannot identify the W13 activation feeding {node.name} ({node.target}) in {root}. "
            "Expected exactly one tensor input of width 2F."
        )
    return root, next(iter(candidates))


def _fallback_w1w3_nodes(gm: torch.fx.GraphModule, root: str) -> set[torch.fx.Node]:
    nodes = [
        node
        for node in gm.graph.nodes
        if node.op == "call_function" and not _is_backward_node(node) and _op_name(node) in _GEMMS
    ]
    exact = {node for node in nodes if _get_module_fqn(node) in {root + ".w1", root + ".w3"}}
    if len(exact) == 2:
        return exact

    same_root = [node for node in nodes if _get_module_fqn(node) == root]
    return set(same_root[:2]) if len(same_root) >= 3 else set()


def _find_w13_save_nodes(
    gm: torch.fx.GraphModule,
    *,
    module_patterns: Sequence[str],
) -> dict[str, set[torch.fx.Node]]:
    roots = {
        root
        for node in gm.graph.nodes
        if not _is_backward_node(node) and (root := _root_for(node, module_patterns)) is not None
    }
    saves: dict[str, set[torch.fx.Node]] = {}

    for node in gm.graph.nodes:
        candidate = _swiglu_w13_candidate(node, module_patterns)
        if candidate is not None:
            root, save_node = candidate
            saves.setdefault(root, set()).add(save_node)

    for root in sorted(roots - saves.keys()):
        saves[root] = _fallback_w1w3_nodes(gm, root)
        if len(saves[root]) != 2:
            raise ValueError(f"Cannot identify W13 or W1/W3 outputs in {root}. Inspect the pre-remat graph.")

    for root, nodes in sorted(saves.items()):
        logger.info(
            "DSV4 W13 save candidates: root=%s nodes=%s",
            root,
            [node.name for node in sorted(nodes, key=lambda item: item.name)],
        )
    return saves


def _must_save_key(
    *,
    target: str,
    module_fqn: str,
    occurrence: tuple[int | str, ...] = (1,),
) -> memory_policy.NodePolicyKey:
    return memory_policy.NodePolicyKey(target=target, module_fqn=module_fqn, occurrence=occurrence)


_DSV4_MHC_SAVE_OVERRIDES = {
    # Do not save ``layers.*.attention.wo_b``. In graph_trainer this splits the
    # numerically sensitive attention -> hc_post chain and was observed to make
    # first-step grad norm become NaN, even without EP overlap.
    _must_save_key(
        target="aten.add.Tensor",
        module_fqn="layers.*.moe",
    ): CheckpointPolicy.MUST_SAVE,
}


def _fsdp_force_save_nodes(gm: torch.fx.GraphModule, *, config) -> set[torch.fx.Node] | None:
    fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        config.parallelism.fsdp_reshard_after_forward,
        pp_enabled=config.parallelism.pipeline_parallel_degree > 1,
    )
    return _find_fsdp_unshard_save_nodes(gm) if not fsdp_reshard_after_forward else None


def _make_dsv4_mhc_policy():
    return memory_policy.make_node_override_memory_policy(
        base_policy=_make_full_memory_policy(),
        overrides=_DSV4_MHC_SAVE_OVERRIDES,
    )


@register_memory_policy("dsv4-mhc")
def _dsv4_mhc_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config,
) -> torch.fx.GraphModule:
    """Apply full recomputation with the validated MoE exit save."""
    memory_policy.tag_sac_policy(
        gm,
        policy_fn=_make_dsv4_mhc_policy(),
        force_save_nodes=_fsdp_force_save_nodes(gm, config=config),
        save_input_every_n_layers=1,
    )
    return gm


def _tag_dsv4_mhc_moe_save(
    gm: torch.fx.GraphModule,
    *,
    config,
) -> torch.fx.GraphModule:
    shared = _find_w13_save_nodes(gm, module_patterns=_SHARED_EXPERTS)
    routed = _find_w13_save_nodes(gm, module_patterns=_ROUTED_EXPERTS)
    extra_saves = {node for nodes in (shared | routed).values() for node in nodes}

    # Keep the router as one coherent recompute region. Saving only the gate
    # GEMM output while recomputing score normalization, top-k selection, and
    # dispatch metadata can mix saved expert activations with a replayed route.
    base_policy = _make_dsv4_mhc_policy()
    full_policy = _make_full_memory_policy()

    def policy_fn(node: torch.fx.Node) -> CheckpointPolicy:
        fqn = node.meta.get("custom", {}).get(memory_policy._MODULE_FQN, "")
        if any(fqn == root or fqn.startswith(root + ".") for root in shared):
            return full_policy(node)
        return base_policy(node)

    logger.info("DSV4 MHC MoE save: %d W13 or W1/W3 tensors selected before EP chunking", len(extra_saves))
    memory_policy.tag_sac_policy(
        gm,
        policy_fn=policy_fn,
        force_save_nodes=(_fsdp_force_save_nodes(gm, config=config) or set()) | extra_saves,
        save_input_every_n_layers=1,
    )
    return gm


@register_memory_policy("dsv4-mhc-moe-save")
def _dsv4_mhc_moe_save_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config,
) -> torch.fx.GraphModule:
    """Save shared and routed W13 activations, packed or as separate W1/W3 outputs.

    With swiglu_group overrides this saves the packed W13 tensor. Without those
    overrides it saves the two projection outputs feeding the SwiGLU multiply.
    Use the normal remat-before-chunk pipeline, e.g. mutation-functionalization.
    """
    return _tag_dsv4_mhc_moe_save(gm, config=config)
