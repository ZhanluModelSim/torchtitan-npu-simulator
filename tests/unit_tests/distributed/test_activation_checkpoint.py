# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.distributed import activation_checkpoint as titan_ac

from torchtitan_npu.distributed.activation_checkpoint import (
    extend_selective_ac_save_ops,
)

_GROUPED_MM_ATTR = "_grouped_mm"


def _get_titan_sac_save_ops():
    return vars(titan_ac)["_get_save_ops"]()


def _get_grouped_mm_op():
    return getattr(torch.ops.aten, _GROUPED_MM_ATTR).default


def _native_mm_save_ops():
    return {torch.ops.aten.mm.default}


def test_extend_selective_ac_save_ops_is_scoped(monkeypatch):
    native_op = torch.ops.aten.mm.default
    original_get_save_ops = _native_mm_save_ops
    monkeypatch.setattr(titan_ac, "_get_save_ops", original_get_save_ops)

    grouped_mm_op = _get_grouped_mm_op()
    with extend_selective_ac_save_ops({grouped_mm_op}):
        assert _get_titan_sac_save_ops() == {
            native_op,
            grouped_mm_op,
        }

    assert vars(titan_ac)["_get_save_ops"] is original_get_save_ops


def test_extend_selective_ac_save_ops_restores_policy_after_error(monkeypatch):
    original_get_save_ops = _native_mm_save_ops
    monkeypatch.setattr(titan_ac, "_get_save_ops", original_get_save_ops)

    with (
        pytest.raises(RuntimeError, match="expected failure"),
        extend_selective_ac_save_ops({_get_grouped_mm_op()}),
    ):
        assert _get_grouped_mm_op() in _get_titan_sac_save_ops()
        raise RuntimeError("expected failure")

    assert vars(titan_ac)["_get_save_ops"] is original_get_save_ops
