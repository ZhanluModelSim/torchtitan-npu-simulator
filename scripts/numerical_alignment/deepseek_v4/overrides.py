# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Named override recipes for DeepSeek-V4 numerical checks."""

GOLDEN_OVERRIDES = (
    "torchtitan_npu.override.common.rope.workaround",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.golden",
)

TEST_OVERRIDES = (
    "torchtitan_npu.override.common.rms_norm.asc",
    "torchtitan_npu.override.common.rope.asc_complex",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
)
