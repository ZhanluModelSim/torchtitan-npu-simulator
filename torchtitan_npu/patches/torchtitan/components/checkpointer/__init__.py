# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Upstream-PR: https://github.com/pytorch/torchtitan/pull/5177

"""Bridge the metadata import until TorchTitan includes #5177 (fb45f5e8)."""

from importlib.util import find_spec


def apply() -> None:
    if find_spec("torch_checkpointing") is None:
        return
    import torch_checkpointing.distributed_metadata

    if not hasattr(torch_checkpointing.distributed_metadata, "METADATA_FILE_NAME"):
        from torch_checkpointing.metadata_serialization import METADATA_FILE_NAME

        torch_checkpointing.distributed_metadata.__dict__["METADATA_FILE_NAME"] = METADATA_FILE_NAME


apply()
