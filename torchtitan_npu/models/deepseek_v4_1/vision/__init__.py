# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The DeepSeek-V4.1 vision half: ViT tower, data path, language SFT and adapters.

``encoder`` is the tower and its aligner, ``data`` the image protocol and patch
planner, ``dataloader`` the multimodal data loader, ``language_dataset`` /
``language_encoder`` the vision-language SFT path, ``state_dict_adapter`` the
checkpoint mapping.  The two originally public names are re-exported here so the
model can keep importing them from the package.
"""

from .encoder import DeepSeekV41VisionEncoder, ImageMarkerEmbeddings

__all__ = [
    "DeepSeekV41VisionEncoder",
    "ImageMarkerEmbeddings",
]
