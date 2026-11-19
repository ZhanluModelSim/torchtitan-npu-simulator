# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Translate explicit per-model dtype selections to native FSDP callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import torch
    from torch import nn


def make_param_dtype_override_fn(
    model: nn.Module,
    parameter_dtypes: Mapping[str, torch.dtype],
) -> Callable[[nn.Parameter], torch.dtype | None]:
    """Resolve exact parameter names before FSDP adds aliases or replaces parameters.

    The native callback receives the parameter itself, so tied aliases and the
    temporary MTP layer names resolve to the same policy. The selected logical
    parameters must be identical on every rank, as required by FSDP.
    """
    parameters = dict(model.named_parameters(remove_duplicate=False))
    unknown = parameter_dtypes.keys() - parameters.keys()
    if unknown:
        raise ValueError(f"Parameter dtype overrides name unknown parameters: {sorted(unknown)}")
    dtypes_by_id = {}
    for name, dtype in parameter_dtypes.items():
        param_id = id(parameters[name])
        if param_id in dtypes_by_id and dtypes_by_id[param_id] != dtype:
            raise ValueError(f"Conflicting dtype overrides for tied parameter {name}")
        dtypes_by_id[param_id] = dtype

    def override(param: nn.Parameter) -> torch.dtype | None:
        return dtypes_by_id.get(id(param))

    return override
