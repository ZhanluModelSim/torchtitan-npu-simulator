# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4934
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport live-output selection to the v0.3.0 unshard deduplication pass.

An unused property read may leave a dead BF16 gather beside a live quantized
preparation chain. Retain the live endpoint instead of replacing it with BF16.
This narrow backport intentionally retains the v0.3.0 structural matcher.
"""

import sys

import torch
import torch.fx as fx
from torchtitan.experiments.graph_trainer.fsdp_patterns import find_fsdp_unshard_outputs
from torchtitan.tools.logging import logger


def _chain_nodes_to_placeholder(
    output: fx.Node,
    placeholder: fx.Node,
) -> set[fx.Node]:
    """Return graph nodes in one FSDP unshard chain, excluding the placeholder."""
    chain_nodes: set[fx.Node] = set()
    queue = [output]
    while queue:
        node = queue.pop()
        if node is placeholder or node in chain_nodes:
            continue
        chain_nodes.add(node)
        queue.extend(node.all_input_nodes)
    return chain_nodes


def deduplicate_fsdp_unshard_chains_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Canonicalize duplicate SimpleFSDP unshard chains per flat parameter.

    A traced parametrized module can read the same FSDP parameter more than
    once. Each read materializes an equivalent all-gather/wait reconstruction
    chain from the same flat parameter placeholder. Downstream FSDP passes
    assume one unsharded value per flat parameter, so this pass rewrites all
    duplicate chains to a live chain and removes the now-dead duplicate
    collective infrastructure.
    """
    del example_inputs

    removable_nodes: set[fx.Node] = set()
    num_duplicate_chains = 0
    for placeholder in gm.graph.find_nodes(op="placeholder"):
        unshard_outputs = find_fsdp_unshard_outputs(placeholder)
        if len(unshard_outputs) <= 1:
            continue
        live_outputs = tuple(output for output in unshard_outputs if output.users)
        canonical_output = live_outputs[0] if live_outputs else unshard_outputs[0]
        for duplicate_output in unshard_outputs:
            if duplicate_output is canonical_output:
                continue
            removable_nodes.update(_chain_nodes_to_placeholder(duplicate_output, placeholder))
            if duplicate_output.users:
                duplicate_output.replace_all_uses_with(canonical_output)
            num_duplicate_chains += 1

    if num_duplicate_chains == 0:
        return gm

    def _is_impure_for_fsdp_dedup(node: fx.Node) -> bool:
        if node in removable_nodes:
            return False
        return node.is_impure()

    gm.graph.eliminate_dead_code(is_impure_node=_is_impure_for_fsdp_dedup)
    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Canonicalized %d duplicate FSDP unshard chain(s)",
        num_duplicate_chains,
    )
    return gm


def apply() -> None:
    """Replace the upstream pass and its already-imported consumer bindings."""
    import torchtitan.experiments.graph_trainer.fsdp_passes
    import torchtitan.experiments.graph_trainer.passes

    if (
        torchtitan.experiments.graph_trainer.fsdp_passes.deduplicate_fsdp_unshard_chains_pass
        is deduplicate_fsdp_unshard_chains_pass
    ):
        return
    torchtitan.experiments.graph_trainer.fsdp_passes.deduplicate_fsdp_unshard_chains_pass = (
        deduplicate_fsdp_unshard_chains_pass
    )
    torchtitan.experiments.graph_trainer.passes.deduplicate_fsdp_unshard_chains_pass = (
        deduplicate_fsdp_unshard_chains_pass
    )
    # A late GraphPP import receives the patched definition through its normal
    # from-import; refresh the alias if GraphPP was imported before this patch.
    if "torchtitan.experiments.graph_trainer.graph_pp.graph_builder" in sys.modules:
        import torchtitan.experiments.graph_trainer.graph_pp.graph_builder

        torchtitan.experiments.graph_trainer.graph_pp.graph_builder.deduplicate_fsdp_unshard_chains_pass = (
            deduplicate_fsdp_unshard_chains_pass
        )
    logger.info("Enabled live-output FSDP unshard deduplication (#4934)")


apply()
