# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.2 model shell built on the DSV3.2 training implementation.

GLM-5.2 uses the same MLA + DSA + routed-MoE training decomposition as the
DSV3.2 implementation in this repository.  The model-specific differences
are represented in the per-layer config: IndexShare controls whether a layer
owns an indexer, and the MTP iteration can reuse the last main-layer top-k.
"""

from dataclasses import dataclass, field

from torchtitan_npu.models.deepseek_v32.model import DeepSeekV32ModelNpu


class GLM5_2ModelNpu(DeepSeekV32ModelNpu):
    """GLM-5.2 model using the DSV3.2 NPU/meta execution path."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV32ModelNpu.Config):
        # Official GLM-5.2 IndexShare schedule metadata.
        index_topk_freq: int = 4
        index_skip_topk_offset: int = 3
        indexer_types: list[str] = field(default_factory=list)
        indexer_rope_interleave: bool = True
        index_share_for_mtp_iteration: bool = True
        # GLM-5.2 trains one next-token prediction module.
        num_mtp_modules: int = 1


__all__ = ["GLM5_2ModelNpu"]
