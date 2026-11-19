# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Upstream-PR: https://github.com/pytorch/torchtitan/pull/5774

"""Backport module-owned precision declarations without replacing config schemas.

TorchTitan 0.4.0's slotted config subclasses can already be imported when this
patch loads. Keep their identities and declare parameter names on built modules.
The iterator also honors explicit declarations on foreign ``nn.Module`` children,
which preserves the existing router policy across optional component adaptors.
Remove this patch when the pinned version provides the upstream config API.
"""

from collections.abc import Iterator

import torchtitan.models.base.module
from torch import nn


def _named_preserve_dtype_params(self: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """Yield declared own parameters, resolving their current objects by name."""
    for module_fqn, module in self.named_modules():
        names = getattr(module, "_preserve_dtype_params", ())
        if not names:
            continue
        parameters = {name: param for name, param in module._parameters.items() if param is not None}
        unknown = set(names) - parameters.keys()
        if unknown:
            raise ValueError(
                f"preserve_dtype_params names {sorted(unknown)} are not own parameters of "
                f"{type(module).__name__} at {module_fqn or '<root>'}. "
                f"Its parameters are {sorted(parameters)}."
            )
        for name in names:
            yield f"{module_fqn}.{name}" if module_fqn else name, parameters[name]


def apply() -> None:
    """Extend the existing class so previously imported components see the API."""
    if hasattr(torchtitan.models.base.module.Module, "named_preserve_dtype_params"):
        return
    torchtitan.models.base.module.Module._preserve_dtype_params = ()  # pyrefly: ignore [missing-attribute]
    torchtitan.models.base.module.Module.named_preserve_dtype_params = (  # pyrefly: ignore [missing-attribute]
        _named_preserve_dtype_params
    )


apply()
