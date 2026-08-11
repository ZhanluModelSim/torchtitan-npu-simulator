# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch._inductor.custom_graph_pass import get_custom_graph_passes

from torchtitan_npu.compile import pattern_replacement
from torchtitan_npu.compile.pattern_replacement import (
    PatternReplacement,
    register_pre_aot_patterns,
)


def _search_relu_add(x):
    return torch.relu(x) + 1


def _replace_relu_add(x):
    return torch.sigmoid(x)


def _search_cos_add(x):
    return torch.cos(x) + 2


def _replace_cos_add(x):
    return torch.sin(x)


def _make_search_relu_add(offset):
    def search_fn(x):
        return torch.relu(x) + offset

    return search_fn


def _pattern(search_fn, replacement_fn):
    return PatternReplacement(
        search_fn=search_fn,
        replacement_fn=replacement_fn,
    )


@pytest.fixture(autouse=True)
def pattern_pass(monkeypatch):
    pattern_pass = pattern_replacement._PreAOTPatternPass()
    monkeypatch.setattr(
        pattern_replacement,
        "_PRE_AOT_PATTERN_PASS",
        pattern_pass,
    )
    with torch._inductor.config.patch(pre_grad_custom_pass=None):
        yield pattern_pass
    torch._dynamo.reset()


def test_registration_installs_shared_pre_aot_pass_once(pattern_pass):
    def existing_pass(graph):
        return graph

    torch._inductor.config.pre_grad_custom_pass = existing_pass
    patterns = {
        "relu_add": _pattern(_search_relu_add, _replace_relu_add),
        "cos_add": _pattern(_search_cos_add, _replace_cos_add),
    }

    register_pre_aot_patterns(patterns)
    register_pre_aot_patterns(patterns)

    installed = get_custom_graph_passes(torch._inductor.config.pre_grad_custom_pass)
    assert installed == (existing_pass, pattern_pass)
    assert tuple(pattern_pass._patterns) == (
        "relu_add",
        "cos_add",
    )


def test_pre_aot_pass_rewrites_forward_graph(pattern_pass):
    register_pre_aot_patterns(
        {"relu_add": _pattern(_search_relu_add, _replace_relu_add)}
    )
    graph_module = torch.fx.symbolic_trace(_search_relu_add)
    x = torch.randn(4)

    pattern_pass(graph_module.graph)
    graph_module.graph.lint()
    graph_module.recompile()

    torch.testing.assert_close(graph_module(x), _replace_relu_add(x))


def test_pattern_pass_uuid_hashes_closure_values(pattern_pass):
    register_pre_aot_patterns(
        {"relu_add": _pattern(_make_search_relu_add(1), _replace_relu_add)}
    )

    first_uuid = pattern_pass.uuid()
    register_pre_aot_patterns(
        {"relu_add": _pattern(_make_search_relu_add(2), _replace_relu_add)}
    )

    assert first_uuid != pattern_pass.uuid()
