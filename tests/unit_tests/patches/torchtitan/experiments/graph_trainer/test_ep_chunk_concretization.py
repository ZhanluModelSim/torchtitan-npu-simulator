# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sympy
import torch
from torch.fx.experimental.symbolic_shapes import free_symbols, ShapeEnv
from torch.utils._sympy.numbers import int_oo

import torchtitan.experiments.graph_trainer.ep_chunk_pass as ep_chunk_pass
import torchtitan.experiments.graph_trainer.ep_pass_utils as ep_pass_utils
import torchtitan.experiments.graph_trainer.passes as graph_passes
import torchtitan.experiments.graph_trainer.registry as registry
from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
    ep_chunk_concretization,
)


def test_patch_updates_cached_graph_trainer_bindings() -> None:
    assert (
        graph_passes.populate_chunk_dim_metadata_pass
        is ep_chunk_pass.populate_chunk_dim_metadata_pass
    )
    assert (
        registry.TRACE_INPUT_PREPARERS["ep_overlap"]
        is ep_chunk_pass.prepare_ep_overlap_trace_inputs
    )
    assert getattr(
        ep_chunk_pass.populate_chunk_dim_metadata_pass,
        "npu_owns_ep_chunk_symbols",
        False,
    )


def test_owned_chunk_symbols_exclude_independent_packed_metadata() -> None:
    shape_env = ShapeEnv()
    fake_mode = torch._subclasses.FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=shape_env,
    )
    with fake_mode:
        batch = shape_env.create_unbacked_symint()
        packed = shape_env.create_unbacked_symint()
        torch._dynamo.override_optimization_hint(batch, 2)
        torch._dynamo.override_optimization_hint(packed, 2048)
        inputs_meta = torch.empty(batch, 4096, device="cuda")
        packed_meta = torch.empty(packed, device="cuda")
    setattr(inputs_meta, "_dynamo_shape_ids", {0: "torchtitan_chunk_batch"})

    graph = torch.fx.Graph()
    inputs = graph.placeholder("inputs")
    metadata = graph.placeholder("metadata")
    graph.output((inputs, metadata))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    inputs.meta["val"] = inputs_meta
    metadata.meta["val"] = packed_meta

    hints = ep_chunk_concretization._owned_chunk_symbol_hints(gm, "batch")

    assert set(hints) == free_symbols(batch)
    assert not (set(hints) & free_symbols(packed))


def test_unrelated_packed_symbol_is_not_concretized() -> None:
    shape_env = ShapeEnv()
    fake_mode = torch._subclasses.FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=shape_env,
    )
    with fake_mode:
        batch = shape_env.create_unbacked_symint()
        packed = shape_env.create_unbacked_symint()
        torch._dynamo.override_optimization_hint(batch, 2)
        torch._dynamo.override_optimization_hint(packed, 2048)

    batch_symbol = next(iter(free_symbols(batch)))
    result = ep_pass_utils._concretize_value(packed, {batch_symbol: 2})

    assert result is packed
    assert free_symbols(result) == free_symbols(packed)


def test_half_size_bound_preserves_infinity() -> None:
    assert ep_chunk_concretization._half_size_bound(sympy.oo) == int_oo
    assert ep_chunk_concretization._half_size_bound(int_oo) == int_oo
    assert ep_chunk_concretization._half_size_bound(3) == 2
