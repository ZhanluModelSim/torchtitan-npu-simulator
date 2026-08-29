# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# ruff: noqa: I001,F401

# `patches` contains code that should considered as upstream,
# and therefore MUST be imported earlier than anything else.
from torchtitan_npu import (
    patches as _patches,
)
from torchtitan_npu import (
    compile as _compile,
)
from torchtitan_npu import (
    config as _config,
)
from torchtitan_npu import (
    extensions as _extensions,
)
from torchtitan_npu import (
    ops as _ops,
)
