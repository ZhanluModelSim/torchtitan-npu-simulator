# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run FlexAttention and block-mask creation eagerly on Ascend NPU.

Importing this module applies the workaround.
"""

import functools
import inspect
from typing import Any, NamedTuple

import torch
import torch.nn.attention.flex_attention as flex_attention
import torch.nn.functional as F
import torchtitan  # noqa: F401
import torchtitan.models.common.attention as common_attention
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_mask,
)
from torch.nn.attention.flex_attention import (
    create_block_mask as eager_create_block_mask,
)
from torch.nn.attention.flex_attention import (
    flex_attention as eager_flex_attention,
)
from torchtitan.distributed import utils as dist_utils

try:
    from torch.nn.attention.flex_attention import AuxOutput as _TorchAuxOutput
    from torch.nn.attention.flex_attention import AuxRequest as _TorchAuxRequest
except ImportError:
    _TorchAuxOutput = None
    _TorchAuxRequest = None

_DENSE_SDPA_OPT_IN = "_torchtitan_npu_dense_sdpa"
_DENSE_SDPA_CACHE = "_torchtitan_npu_dense_mask_cache"
_MAX_DENSE_MASK_ELEMENTS = 16 * 1024 * 1024


class _CompatAuxOutput(NamedTuple):
    """Auxiliary-output shape for older Torch versions."""

    lse: torch.Tensor | None = None
    max_scores: torch.Tensor | None = None


_EAGER_SUPPORTS_RETURN_AUX = "return_aux" in inspect.signature(eager_flex_attention).parameters


def _requests_auxiliary_outputs(return_aux: Any) -> bool:
    if return_aux is None:
        return False
    if isinstance(return_aux, bool):
        return return_aux
    return bool(getattr(return_aux, "lse", False) or getattr(return_aux, "max_scores", False))


def _normalize_aux_request(return_aux: Any) -> Any:
    """Convert legacy bool/None requests to the current AuxRequest protocol."""
    if _TorchAuxRequest is None:
        return return_aux
    if isinstance(return_aux, _TorchAuxRequest):
        return return_aux
    return _TorchAuxRequest(
        lse=bool(getattr(return_aux, "lse", return_aux) if return_aux is not None else False),
        max_scores=bool(getattr(return_aux, "max_scores", False)),
    )


def _make_aux_output(
    *,
    lse: torch.Tensor | None = None,
    max_scores: torch.Tensor | None = None,
) -> Any:
    if _TorchAuxOutput is not None:
        return _TorchAuxOutput(lse=lse, max_scores=max_scores)
    return _CompatAuxOutput(lse=lse, max_scores=max_scores)


def _normalize_flex_result(result: Any) -> tuple[torch.Tensor, Any]:
    """Keep the (output, AuxOutput) contract across Torch API versions."""
    if not isinstance(result, tuple) or len(result) != 2:
        return result, _make_aux_output()
    output, auxiliary = result
    if hasattr(auxiliary, "lse") and hasattr(auxiliary, "max_scores"):
        return output, auxiliary
    if isinstance(auxiliary, torch.Tensor):
        return output, _make_aux_output(lse=auxiliary)
    return output, _make_aux_output()


def mark_dense_sdpa_mask_mod(mask_mod):
    """Opt one mask closure into the bounded NPU dense-SDPA fallback."""
    setattr(mask_mod, _DENSE_SDPA_OPT_IN, True)
    setattr(mask_mod, _DENSE_SDPA_CACHE, {})
    return mask_mod


def _validate_dense_mask_size(
    batch_size: int,
    query_len: int,
    key_len: int,
) -> None:
    elements = batch_size * query_len * key_len
    if elements > _MAX_DENSE_MASK_ELEMENTS:
        raise RuntimeError(
            "NPU dense Flex fallback would materialize "
            f"{elements:,} boolean mask elements; the supported limit is "
            f"{_MAX_DENSE_MASK_ELEMENTS:,}. Reduce image/video resolution "
            "or sequence length."
        )


def _dense_block_mask(
    block_mask: BlockMask,
    *,
    batch_size: int,
    query_len: int,
    key_len: int,
    device: torch.device,
) -> torch.Tensor:
    _validate_dense_mask_size(batch_size, query_len, key_len)
    mask_mod = block_mask.mask_mod
    # Compile-time implicit NPU fallback masks are not marked by the vision
    # patch, so they may not carry the eager cache attribute.
    cache = getattr(mask_mod, _DENSE_SDPA_CACHE, None)
    if cache is None:
        cache = {}
    cache_key = (batch_size, query_len, key_len, device.type, device.index)
    if cache_key in cache:
        return cache[cache_key]

    mask = create_mask(
        mask_mod,
        batch_size,
        1,
        query_len,
        key_len,
        device=device,
    )
    diagonal = torch.eye(query_len, key_len, dtype=torch.bool, device=device)[None, None]
    mask = mask | (~mask.any(dim=-1, keepdim=True) & diagonal)
    cache[cache_key] = mask
    return mask


def _run_eager_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    score_mod,
    block_mask,
    scale,
    enable_gqa,
    return_aux,
    kernel_options,
):
    # During aot_eager, TorchTitan wraps FlexAttention in a regional
    # Inductor subcompile.  The NPU flex lowering then selects CUDA Triton
    # templates; use the bounded dense SDPA path for eligible NPU masks only
    # while tracing.  Eager execution remains explicitly opt-in via the marker.
    if (
        score_mod is None
        and isinstance(block_mask, BlockMask)
        and block_mask.mask_mod is not None
        and (
            getattr(block_mask.mask_mod, _DENSE_SDPA_OPT_IN, False)
            or (torch.compiler.is_compiling() and q.device.type == "npu")
        )
        and block_mask.kv_num_blocks.shape[1] == 1
        and q.shape[2] == k.shape[2]
        and not _requests_auxiliary_outputs(return_aux)
    ):
        del kernel_options
        dense_mask = _dense_block_mask(
            block_mask,
            batch_size=q.shape[0],
            query_len=q.shape[2],
            key_len=k.shape[2],
            device=q.device,
        )
        return _normalize_flex_result(
            (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=dense_mask,
                    scale=scale,
                    enable_gqa=enable_gqa,
                ),
                _make_aux_output(),
            )
        )

    # The pinned torch_npu build advertises return_aux, but its NPU flex
    # lowering sends that form through the CUDA Triton regional path while
    # tracing. Keep the protocol-preserving return_aux path for eager/CPU
    # callers; use the compatible legacy return_lse entry point only for the
    # NPU compile path.
    use_legacy_npu_compile = q.device.type == "npu" and torch.compiler.is_compiling()
    if _EAGER_SUPPORTS_RETURN_AUX and not use_legacy_npu_compile:
        result = eager_flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_aux=_normalize_aux_request(return_aux),
            kernel_options=kernel_options,
        )
    else:
        result = eager_flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=enable_gqa,
            return_lse=_requests_auxiliary_outputs(return_aux),
            kernel_options=kernel_options,
        )
    return _normalize_flex_result(result)


def _install_eager_flex_attention() -> None:
    common_attention.FlexAttention._compiled_flex_attn = staticmethod(_run_eager_flex_attention)


_original_set_determinism = dist_utils.set_determinism


@functools.wraps(_original_set_determinism)
def _set_determinism_preserving_eager_flex(*args, **kwargs):
    result = _original_set_determinism(*args, **kwargs)
    _install_eager_flex_attention()
    return result


def apply() -> None:
    flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True  # pyrefly: ignore [bad-assignment]

    _install_eager_flex_attention()
    dist_utils.set_determinism = _set_determinism_preserving_eager_flex

    def _eager_create_block_mask(*args, **kwargs):
        kwargs.pop("separate_full_blocks", None)
        kwargs["_compile"] = False
        return eager_create_block_mask(*args, **kwargs)

    common_attention._compiled_create_block_mask = _eager_create_block_mask

    if hasattr(flex_attention, "_validate_device"):
        flex_attention._validate_device = (  # pyrefly: ignore [bad-assignment]
            lambda query, key, value: None
        )


apply()
