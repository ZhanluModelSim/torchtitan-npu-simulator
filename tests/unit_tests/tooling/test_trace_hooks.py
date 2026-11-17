# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from scripts.numerical_alignment.utils.trace_hooks import TraceHookRegistry, record_trace

pytestmark = pytest.mark.tooling


def test_tuple_output_records_immutable_values():
    records = {}
    pool = torch.nn.MaxPool1d(2, return_indices=True)

    with TraceHookRegistry(records) as hooks:
        hooks.forward(pool, "pooled")
        values, _ = pool(torch.tensor([[[1.0, 4.0, 3.0, 2.0]]], requires_grad=True))
        values.detach().zero_()
    torch.testing.assert_close(records["pooled"], torch.tensor([[[4.0, 3.0]]]), atol=0, rtol=0)
    assert not records["pooled"].requires_grad


def test_non_tensor_trace_warns_and_skips_record():
    records = {}
    with pytest.warns(UserWarning, match="Skipping trace 'unsupported'.*NoneType"):
        record_trace(records, "unsupported", None)
    assert "unsupported" not in records
