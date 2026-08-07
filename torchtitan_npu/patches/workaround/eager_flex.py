"""Run FlexAttention and block-mask creation eagerly on Ascend NPU.

Importing this module applies the workaround.
"""

import torch
from torch.nn.attention import flex_attention
from torchtitan.models.common import attention as attention_module


def apply() -> None:
    flex_attention._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = (
        True  # pyrefly: ignore [bad-assignment]
    )

    attention_module.FlexAttention._compiled_flex_attn = staticmethod(
        lambda *args, **kwargs: flex_attention.flex_attention(*args, **kwargs)
    )

    def _eager_create_block_mask(*args, **kwargs):
        kwargs.pop("separate_full_blocks", None)
        kwargs["_compile"] = False
        return flex_attention.create_block_mask(*args, **kwargs)

    attention_module._compiled_create_block_mask = _eager_create_block_mask

    if hasattr(flex_attention, "_validate_device"):

        def _validate_device(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> None:
            del query, key, value

        flex_attention._validate_device = _validate_device


apply()
