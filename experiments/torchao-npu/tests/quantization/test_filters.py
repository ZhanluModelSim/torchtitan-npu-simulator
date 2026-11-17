# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torchao_npu.quantization.filters import (
    _is_expert,
    _is_parameter,
    all_filters,
    any_filter,
    match_fqn_exact,
    match_fqn_regex,
    match_fqn_suffix,
    match_module_type,
    not_filter,
)


def test_module_filters():
    linear = nn.Linear(2, 2)

    assert match_module_type(nn.Linear)(linear, "layers.0.proj")
    assert not match_module_type(nn.Conv2d)(linear, "layers.0.proj")
    assert _is_expert(linear, "layers.0.experts")
    assert _is_expert(linear, "layers.0.shared_experts")
    assert not _is_expert(linear, "layers.0.router")


def test_fqn_filters_and_combinators():
    obj = object()
    suffix = match_fqn_suffix(".w1", ".w3")
    exact = match_fqn_exact("layers.0.moe.w1")
    regex = match_fqn_regex(r"layers\.\d+\.moe\.w[13]")

    assert suffix(obj, "layers.0.moe.w1")
    assert exact(obj, "layers.0.moe.w1")
    assert regex(obj, "layers.12.moe.w3")
    assert all_filters(suffix, regex)(obj, "layers.12.moe.w3")
    assert any_filter(exact, regex)(obj, "layers.12.moe.w3")
    assert not_filter(exact)(obj, "layers.1.moe.w1")


def test_default_parameter_filter_accepts_unwrapped_parameter_only():
    parameter = nn.Parameter(torch.ones(2, 2))

    assert _is_parameter(parameter, "weight")
    assert not _is_parameter(torch.ones(2, 2), "weight")
