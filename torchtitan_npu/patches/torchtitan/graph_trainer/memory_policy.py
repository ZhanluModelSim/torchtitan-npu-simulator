# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3949

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
from collections import defaultdict
from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.experiments.graph_trainer.common_utils import (
    _MODULE_FQN,
    _NOT_IN_LAYERS,
    _get_layer_id,
    _is_backward_node,
    matches_module_fqn_pattern,
)
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _make_default_memory_policy,
)
from torchtitan.tools.logging import logger


@dataclass(frozen=True)
class NodePolicyKey:
    target: str
    module_fqn: str
    occurrence: tuple[int | str, ...]


def _force_boundary_saves(
    gm: torch.fx.GraphModule,
    save_input_every_n_layers: int,
) -> int:
    boundary_saves = 0

    def is_recomputable(node: torch.fx.Node) -> bool:
        return node.meta.get("recompute") in (
            CheckpointPolicy.PREFER_RECOMPUTE,
            CheckpointPolicy.MUST_RECOMPUTE,
        )

    for node in gm.graph.nodes:
        if _is_backward_node(node) or not is_recomputable(node):
            continue

        produce_layer_id = _get_layer_id(node)
        if int(produce_layer_id + 1) % save_input_every_n_layers != 0:
            continue

        if any(
            not _is_backward_node(user) and is_recomputable(user) and _get_layer_id(user) > produce_layer_id
            for user in node.users
        ):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            boundary_saves += 1

    return boundary_saves


def _get_wait_tensor_target() -> object | None:
    namespace = getattr(torch.ops, "_c10d_functional", None)
    wait_tensor = getattr(namespace, "wait_tensor", None)
    return getattr(wait_tensor, "default", None)


def _tag_sac_node(
    node: torch.fx.Node,
    policy_fn: Callable[[torch.fx.Node], CheckpointPolicy],
    force_save_nodes: set[torch.fx.Node],
    wait_tensor_target: object | None,
) -> None:
    if node.op != "call_function" or _is_backward_node(node):
        return

    fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
    if fqn.startswith(("lm_head", "loss")):
        return

    if node in force_save_nodes:
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        return

    is_passthrough = node.target == operator.getitem or (
        wait_tensor_target is not None and node.target == wait_tensor_target
    )
    if is_passthrough:
        parent = node.args[0]
        if isinstance(parent, torch.fx.Node) and "recompute" in parent.meta:
            node.meta["recompute"] = parent.meta["recompute"]
        return

    if isinstance(node.meta.get("val"), torch.SymInt):
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        return

    node.meta["recompute"] = policy_fn(node)


def _collect_layer_stats(
    gm: torch.fx.GraphModule,
) -> dict[int, dict[str, int]]:
    layer_stats: dict[int, dict[str, int]] = defaultdict(lambda: {"save": 0, "recompute": 0})
    for node in gm.graph.nodes:
        if "recompute" not in node.meta:
            continue
        key = "save" if node.meta["recompute"] == CheckpointPolicy.MUST_SAVE else "recompute"
        layer_stats[_get_layer_id(node)][key] += 1
    return layer_stats


def tag_sac_policy(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    policy_fn: Callable[[torch.fx.Node], CheckpointPolicy] | None = None,
    force_save_nodes: set[torch.fx.Node] | None = None,
    save_input_every_n_layers: int = 1,
) -> torch.fx.GraphModule:
    if policy_fn is None:
        policy_fn = _make_default_memory_policy()
    if force_save_nodes is None:
        force_save_nodes = set()

    wait_tensor_target = _get_wait_tensor_target()
    for node in gm.graph.nodes:
        _tag_sac_node(
            node,
            policy_fn,
            force_save_nodes,
            wait_tensor_target,
        )

    boundary_saves = _force_boundary_saves(
        gm,
        save_input_every_n_layers=save_input_every_n_layers,
    )

    gm.recompile()

    layer_stats = _collect_layer_stats(gm)

    logger.info("Applied selective activation checkpointing (SAC) graph pass.")
    if boundary_saves:
        logger.info(f"  Forced {boundary_saves} nodes to MUST_SAVE at layer boundaries")
    for layer_id in sorted(layer_stats):
        stats = layer_stats[layer_id]
        label = "non-layer" if layer_id == _NOT_IN_LAYERS else str(layer_id)
        logger.info(f"  Layer {label}: {stats['save']} MUST_SAVE, {stats['recompute']} RECOMPUTE")
    return gm


def _matches_occurrence(
    actual: int,
    expected: tuple[int | str, ...],
) -> bool:
    if "*" in expected:
        return True

    return actual in expected


def make_node_override_memory_policy(
    base_policy: Callable[[torch.fx.Node], CheckpointPolicy],
    overrides: dict[NodePolicyKey, CheckpointPolicy],
) -> Callable[[torch.fx.Node], CheckpointPolicy]:
    node_counts: defaultdict[tuple[object, str], int] = defaultdict(int)

    def policy_fn(node: torch.fx.Node) -> CheckpointPolicy:
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
        count_key = (node.target, fqn)
        node_counts[count_key] += 1
        occurrence = node_counts[count_key]

        for key, policy in overrides.items():
            if (
                str(node.target) == key.target
                and matches_module_fqn_pattern(key.module_fqn, fqn)
                and _matches_occurrence(occurrence, key.occurrence)
            ):
                return policy

        return base_policy(node)

    return policy_fn


def _patch_graph_trainer_memory_policy_cli_choices() -> None:
    memory_policy_type = str

    GraphTrainerCompileConfig.__annotations__["memory_policy"] = memory_policy_type

    field = GraphTrainerCompileConfig.__dataclass_fields__.get("memory_policy")
    if field is not None:
        field.type = memory_policy_type


_patch_graph_trainer_memory_policy_cli_choices()
