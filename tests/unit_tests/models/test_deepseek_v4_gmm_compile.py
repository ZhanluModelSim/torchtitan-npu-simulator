# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan_npu.converters.kernels import gmm as gmm_module
from torchtitan_npu.models.deepseek_v4 import parallelize


def test_grouped_mm_rejects_offsets_that_exceed_int32_before_kernel_call():
    x = torch.empty((torch.iinfo(torch.int32).max + 1, 0), device="meta")
    num_tokens_per_expert = torch.empty((0,), dtype=torch.int64, device="meta")
    run_experts_grouped_mm = getattr(gmm_module, "_run_experts_grouped_mm")

    with pytest.raises(ValueError, match="int32 grouped_mm offsets"):
        run_experts_grouped_mm(
            None,
            torch.empty((0,), device="meta"),
            None,
            x,
            num_tokens_per_expert,
        )


def test_grouped_mm_compile_patches_npu_gmm_with_default_partitioner(monkeypatch):
    compile_calls = []
    dynamic_calls = []
    compiled_calls = []

    def base_grouped_mm(*args, **kwargs):
        raise AssertionError("base grouped_mm should be compiled before use")

    def fake_compile(fn, *, backend, fullgraph, options):
        compile_calls.append(
            {
                "fn": fn,
                "backend": backend,
                "fullgraph": fullgraph,
                "options": options,
            }
        )

        def compiled_fn(*args):
            compiled_calls.append(args)
            return "compiled-result"

        return compiled_fn

    def fake_maybe_mark_dynamic(tensor, dim):
        dynamic_calls.append((tensor, dim))

    monkeypatch.setattr(gmm_module, "_run_experts_grouped_mm", base_grouped_mm)
    monkeypatch.setattr(torch, "compile", fake_compile)
    torch_dynamo = getattr(torch, "_dynamo")
    monkeypatch.setattr(torch_dynamo, "maybe_mark_dynamic", fake_maybe_mark_dynamic)

    compile_config = SimpleNamespace(backend="inductor_npu")
    patch_grouped_mm_compile = getattr(parallelize, "_patch_grouped_mm_compile")
    patch_grouped_mm_compile(compile_config, ep_enabled=True)
    patch_grouped_mm_compile(compile_config, ep_enabled=True)

    assert len(compile_calls) == 1
    compile_call = compile_calls[0]
    assert compile_call["fn"] is base_grouped_mm
    assert compile_call["backend"] == "inductor_npu"
    assert compile_call["fullgraph"] is True
    partitioner = compile_call["options"]["custom_partitioner_fn"]
    assert partitioner.uuid() == "npu_gmm_aot_default_partition"

    x = torch.empty((4, 8), device="meta")
    num_tokens_per_expert = torch.empty((1,), dtype=torch.int64, device="meta")
    compiled_grouped_mm = getattr(gmm_module, "_run_experts_grouped_mm")
    result = compiled_grouped_mm(
        w13="w13",
        w2="w2",
        _w3=None,
        x=x,
        num_tokens_per_expert=num_tokens_per_expert,
        swiglu_limit=3.0,
        routed_scores="scores",
    )

    assert result == "compiled-result"
    assert len(dynamic_calls) == 1
    assert dynamic_calls[0][0] is x
    assert dynamic_calls[0][1] == 0

    assert len(compiled_calls) == 1
    compiled_args = compiled_calls[0]
    assert compiled_args[:3] == ("w13", "w2", None)
    assert compiled_args[3] is x
    assert compiled_args[4] is num_tokens_per_expert
    assert compiled_args[5:] == (3.0, "scores")
