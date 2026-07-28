"""Run FlexAttention and block-mask creation eagerly on Ascend NPU.

Importing this module applies the workaround.
"""

import torch
import torchtitan

from torch.nn.attention.flex_attention import (
    create_block_mask as eager_create_block_mask,
    flex_attention as eager_flex_attention,
)


def apply() -> None:
    torch.nn.attention.flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True

    torchtitan.models.common.attention.FlexAttention._compiled_flex_attn = staticmethod(
        lambda *args, **kwargs: eager_flex_attention(*args, **kwargs)
    )

    def _eager_create_block_mask(*args, **kwargs):
        kwargs.pop("separate_full_blocks", None)
        kwargs["_compile"] = False
        return eager_create_block_mask(*args, **kwargs)

    torchtitan.models.common.attention._compiled_create_block_mask = (
        _eager_create_block_mask
    )

    if hasattr(torch.nn.attention.flex_attention, "_validate_device"):
        torch.nn.attention.flex_attention._validate_device = lambda q, k, v: None


apply()
