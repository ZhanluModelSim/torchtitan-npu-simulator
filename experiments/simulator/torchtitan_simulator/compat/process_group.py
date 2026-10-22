# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for inspecting torch.distributed process groups on NPU."""

import torch.distributed as dist


def is_fake_process_group(group) -> bool:
    if not dist.is_initialized():
        return False

    try:
        return str(dist.get_backend(group)).lower() == "fake"
    except RuntimeError:
        return False
