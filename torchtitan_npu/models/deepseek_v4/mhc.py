# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan's DeepSeek-V4 mHC reference components.

NPU implementations are selected by the corresponding configuration overrides.
The package patch restores DeepSeek's additive epsilon before Sinkhorn balancing
while retaining the upstream components (pytorch/torchtitan#5975).
"""

__all__ = ["HcHead", "HcPost", "HcPre", "HcSplitSinkhorn"]

from torchtitan.models.deepseek_v4.mhc import HcHead, HcPost, HcPre, HcSplitSinkhorn
