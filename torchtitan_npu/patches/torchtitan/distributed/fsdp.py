# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Upstream-PR: https://github.com/pytorch/torchtitan/pull/5774

"""Apply module-owned precision through TorchTitan 0.4.0's native FSDP path.

Retain the released decoder and MTP sharding implementations. Only policy
construction is backported; the config schemas and other precision consumers
remain on the pinned release. Remove this patch when the dependency includes
the upstream preservation API.
"""

from functools import wraps
from inspect import signature

import torch
import torchtitan.distributed.fsdp
import torchtitan.models.deepseek_v3.mtp
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchtitan.models.base.module import Module

_original_apply_fsdp_to_decoder = torchtitan.distributed.fsdp.apply_fsdp_to_decoder
_decoder_signature = signature(_original_apply_fsdp_to_decoder)


def mixed_precision_policy(
    module: nn.Module,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    cast_forward_inputs: bool = True,
) -> MixedPrecisionPolicy:
    """Preserve declared parameters in the wider of their own and compute dtype."""
    preserved = {
        id(param)
        for _, param in Module.named_preserve_dtype_params(module)  # pyrefly: ignore [missing-attribute]
    }

    def param_dtype_override_fn(param: nn.Parameter) -> torch.dtype | None:
        return torch.promote_types(param.dtype, param_dtype) if id(param) in preserved else None

    # The pinned Torch 2.15 provides this field; older CI images have 2.14 types.
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=cast_forward_inputs,
        param_dtype_override_fn=param_dtype_override_fn if preserved else None,  # pyrefly: ignore [unexpected-keyword]
    )


@wraps(_original_apply_fsdp_to_decoder)
def _apply_fsdp_to_decoder(*args, **kwargs) -> None:
    bound = _decoder_signature.bind(*args, **kwargs)
    if bound.arguments.get("param_dtype_override_fn") is None:
        policy = mixed_precision_policy(
            bound.arguments["model"],
            param_dtype=bound.arguments["param_dtype"],
            reduce_dtype=bound.arguments["reduce_dtype"],
            cast_forward_inputs=False,
        )
        bound.arguments["param_dtype_override_fn"] = (
            policy.param_dtype_override_fn  # pyrefly: ignore [missing-attribute]
        )
    _original_apply_fsdp_to_decoder(*bound.args, **bound.kwargs)


def apply() -> None:
    """Update the canonical function and the release's single captured binding."""
    if hasattr(torchtitan.distributed.fsdp, "mixed_precision_policy"):
        return
    torchtitan.distributed.fsdp.mixed_precision_policy = mixed_precision_policy  # pyrefly: ignore [missing-attribute]
    torchtitan.distributed.fsdp.apply_fsdp_to_decoder = _apply_fsdp_to_decoder

    # Decoder resolves its import at call time; MTP captures this one globally.
    # V4's captured MTP helper keeps the same function and reads this binding.
    torchtitan.models.deepseek_v3.mtp.apply_fsdp_to_decoder = _apply_fsdp_to_decoder


apply()
