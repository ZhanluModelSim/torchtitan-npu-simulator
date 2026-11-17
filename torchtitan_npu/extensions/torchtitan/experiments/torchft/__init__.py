# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in TorchFT integration for Ascend NPU."""

try:
    import torchft  # noqa: F401
except ModuleNotFoundError as error:
    if error.name != "torchft":
        raise
    raise ModuleNotFoundError(
        "The TorchFT experiment requires its optional dependencies: pip install -e '.[torchft]'",
        name="torchft",
    ) from error

from torchtitan_npu.patches import torchft as _torchft_patches  # noqa: F401
from torchtitan_npu.patches.torchtitan.experiments import torchft as _torchtitan_torchft_patches  # noqa: F401
