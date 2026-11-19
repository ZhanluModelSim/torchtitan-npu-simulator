# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Upstream-PR: https://github.com/pytorch/torchtitan/pull/5774

"""Preserve native DeepSeek component precision on the 0.4.0 config schemas.

These module defaults mirror the upstream config declarations. Native subclasses
and NPU component overrides inherit them without replacing imported config types.
Remove this patch when the dependency includes the upstream declarations.
"""

import torchtitan.models.base.module
import torchtitan.models.deepseek_v4.compressor
import torchtitan.models.deepseek_v4.mhc


def apply() -> None:
    """Keep existing schemas and defer to native declarations after an upgrade."""
    if "preserve_dtype_params" in torchtitan.models.base.module.Module.Config.__dataclass_fields__:
        return
    torchtitan.models.deepseek_v4.mhc.HcPre._preserve_dtype_params = (  # pyrefly: ignore [missing-attribute]
        "hc_fn",
        "hc_base",
        "hc_scale",
    )
    torchtitan.models.deepseek_v4.mhc.HcHead._preserve_dtype_params = (  # pyrefly: ignore [missing-attribute]
        "hc_fn",
        "hc_base",
        "hc_scale",
    )
    torchtitan.models.deepseek_v4.compressor.Compressor._preserve_dtype_params = (  # pyrefly: ignore[missing-attribute]
        "ape",
    )


apply()
