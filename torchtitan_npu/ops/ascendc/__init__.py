# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AscendC operator registrations used by NPU overrides."""

from . import (
    grouped_mm,  # noqa: F401
    moe_re_routing,  # noqa: F401
    moe_token_unpermute,  # noqa: F401
)
