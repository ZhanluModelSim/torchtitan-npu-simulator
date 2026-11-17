# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backport of the mutation functionalization pass from upstream TorchTitan
PR https://github.com/pytorch/torchtitan/pull/4708 (head ``a64792d2``, fixing
https://github.com/pytorch/torchtitan/issues/4688): functionalizing first
exposes forward mutation writes as dataflow, so the native SAR recomputes
them instead of dropping them. Delete this module and the ``pass_pipeline``
settings once the pinned TorchTitan ships an equivalent pass.
"""

from typing import Any

import torch
import torch.fx as fx
from torch._functorch.partitioners import has_recomputable_ops
from torch._guards import detect_fake_mode
from torch._higher_order_ops.effects import _get_effect, has_effects
from torch._subclasses import FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensorMode, dispatch_functionalize
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.experiments.graph_trainer.passes import (
    _get_pass_name,
    construct_default_graph_passes,
)
from torchtitan.experiments.graph_trainer.registry import register_pass_pipeline

__all__ = ["PASS_PIPELINE_NAME", "functionalize_recompute_mutations_pass"]

PASS_PIPELINE_NAME = "mutation-functionalization"


def functionalize_recompute_mutations_pass(
    gm: fx.GraphModule,
    example_inputs: Any,
) -> fx.GraphModule:
    """Expose forward mutation results as dataflow before activation rematerialization."""
    if not has_recomputable_ops(gm) or not any(
        isinstance(node.target, torch._ops.OpOverload)
        and node.target._schema.is_mutable
        and not _is_backward_node(node)
        for node in gm.graph.nodes
    ):
        return gm
    effect_types = {
        _get_effect(node.target)
        for module in gm.modules()
        if isinstance(module, fx.GraphModule)
        for node in module.graph.nodes
        if has_effects(node.target)
    }
    mode = FunctionalTensorMode()

    def functionalize(*args):
        mode._tokens = {effect: torch.ops.prims._make_token.default() for effect in effect_types}
        result = fx.Interpreter(gm).run(*args)
        if mode._tokens:
            torch.ops.prims._sink_tokens.default(list(mode._tokens.values()))
        return result

    fake_mode = detect_fake_mode(example_inputs) or FakeTensorMode(allow_non_fake_inputs=True)
    fake_inputs = tuple(
        fake_mode.from_tensor(value) if isinstance(value, torch.Tensor) else value for value in example_inputs
    )
    with fake_mode, preserve_node_meta():
        functional_gm = make_fx(
            dispatch_functionalize(
                functionalize,
                mode,
                propagate_input_mutations=True,  # pyrefly: ignore [unexpected-keyword]
            ),
        )(*fake_inputs)
    functional_gm.meta.update(gm.meta)
    return functional_gm


@register_pass_pipeline(PASS_PIPELINE_NAME)
def _mutation_functionalization_pipeline(
    traced_result: Any,
    config: Any,
    *,
    parallel_dims: Any = None,
    runtime_context: Any = None,
) -> list:
    """Default GraphTrainer passes with mutation functionalization inserted.

    Order (upstream #4708): memory-policy tagging -> functionalization ->
    CPU offload -> native SAR, so the SAR replay sees the mutation writes as
    ordinary dataflow.
    """
    from . import ep_forward_accumulation

    # This compatibility is needed only when mutation functionalization can
    # feed functionalized buffer updates into EP chunking.
    ep_forward_accumulation.apply()
    passes = construct_default_graph_passes(traced_result, config, parallel_dims=parallel_dims)
    names = [_get_pass_name(pass_fn) for pass_fn in passes]
    if "functionalize_recompute_mutations_pass" in names:
        # The pinned TorchTitan already ships the upstream pass; reuse it.
        return passes
    try:
        anchor = names.index("tag_with_memory_policy_pass")
    except ValueError:
        # Artifact-only or custom lists without the tagging stage.
        return passes
    passes.insert(anchor + 1, functionalize_recompute_mutations_pass)
    return passes
