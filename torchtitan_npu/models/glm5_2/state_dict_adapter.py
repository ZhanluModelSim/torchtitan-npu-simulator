# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.2 checkpoint adapter.

The HF GLM-MoE-DSA checkpoint names for MLA, indexer, MoE and MTP match the
DSV3.2 names.  Shared IndexShare layers simply omit indexer parameters, so
the DSV3.2 adapter can be reused unchanged for the remaining keys.
"""

from torchtitan_npu.models.deepseek_v32.state_dict_adapter import (
    DeepSeekV32StateDictAdapter,
)


class GLM52StateDictAdapter(DeepSeekV32StateDictAdapter):
    """State adapter for GLM-5.2 HF checkpoints."""


__all__ = ["GLM52StateDictAdapter"]
