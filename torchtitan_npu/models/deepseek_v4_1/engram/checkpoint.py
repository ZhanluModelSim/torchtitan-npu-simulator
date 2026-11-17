# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Expose CPU row shards through PyTorch's CheckpointableTensor protocol."""

import torch.distributed.checkpoint as dcp


def checkpoint_shard(weight, *, key, offset, logical_rows):
    if not hasattr(dcp, "CheckpointableTensor"):
        raise RuntimeError(
            "Engram HF row shards require PyTorch CheckpointableTensor support; "
            "install the PyTorch version pinned in requirements.txt."
        )
    # Metadata belongs to this checkpoint view, not to the model parameter.
    result = weight.detach()
    rows = max(0, min(weight.shape[0], logical_rows - offset))
    result.global_shape = (logical_rows, weight.shape[1])
    result.global_offsets = ((offset, 0),) if rows else ()
    result.local_offsets = ((0, 0),) if rows else ()
    result.local_sizes = ((rows, weight.shape[1]),) if rows else ()
    result._engram_native_key = key
    result._engram_valid_rows = rows
    return result
