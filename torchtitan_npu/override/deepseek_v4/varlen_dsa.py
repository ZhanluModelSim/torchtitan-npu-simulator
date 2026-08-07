# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: provide packed varlen metadata to DeepSeek-V4 DSA.

The numerical-reference and fused DSA overrides share this config adapter.
"""

from dataclasses import dataclass, field

from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE
from torchtitan.config import derive, override
from torchtitan.models.common.attention import VarlenAttention

from torchtitan_npu.models.deepseek_v4.attention import (
    DSAFlexAttention,
    DSAIndexerAuxLoss,
)
from torchtitan_npu.models.deepseek_v4.compressor import IndexSelection
from torchtitan_npu.models.deepseek_v4.model import DSV4PackedMetadataHandler
from torchtitan_npu.patches.torchtitan.models.common.mask_handler import (
    BaseMaskHandler,
)


class DSAVarlenAttention(DSAFlexAttention):
    """DSV4 sparse attention with a varlen-typed config."""

    @dataclass(kw_only=True, slots=True)
    class Config(VarlenAttention.Config):  # pyrefly: ignore [bad-override]
        """Varlen-rooted equivalent of ``DSAFlexAttention.Config``.

        ``dsa_window_size`` avoids the tuple-valued varlen ``window_size``.
        """

        dsa_window_size: int
        compress_ratio: int
        softmax_scale: float
        block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE
        kernel_options: dict = field(default_factory=dict)
        index_selection: IndexSelection.Config | None = None
        indexer_aux_loss: DSAIndexerAuxLoss.Config | None = None
        return_lse: bool = False

    def __init__(self, config: Config) -> None:
        super().__init__(config)  # pyrefly: ignore [bad-argument-type]
        # Replace the unused varlen attention window with the DSA window.
        self.window_size = config.dsa_window_size


def derive_varlen_dsa(
    cfg: DSAFlexAttention.Config,
    target: type,
    **extra,
):
    """Derive a varlen DSA config while preserving both window fields."""

    return derive(
        cfg,
        target,
        dsa_window_size=cfg.window_size,
        window_size=(-1, 0),  # unused; the DSA kernels do their own masking
        **extra,
    )


@override(
    target=BaseMaskHandler.Config,
    description=(
        "DSV4 packed compression-plan handler (varlen sparse-attention recipes)"
    ),
)
def npu_dsv4_packed_mask_handler_override(
    cfg: BaseMaskHandler.Config,
) -> DSV4PackedMetadataHandler.Config:
    return DSV4PackedMetadataHandler.Config()
