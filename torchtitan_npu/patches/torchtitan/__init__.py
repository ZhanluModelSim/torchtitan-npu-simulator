# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .components import checkpointer, metrics, optimizer  # noqa: F401, I001
from .distributed import context_parallel  # noqa: F401
import torchtitan_npu.patches.torchtitan.distributed.checkpoint.filesystem  # noqa: F401
from .experiments.graph_trainer import (  # noqa: F401
    chunked_loss,
    ep_overlap_shape_queries,
)

from . import observability  # noqa: F401

# ``attention_metadata`` carries the backend-owned metadata contract and must
# be applied before ``decoder`` binds the map builder that reads it, and before
# ``mtp`` dispatches that map per layer.
from .models.common import attention_metadata, decoder, moe, rope, token_dispatcher  # noqa: F401
from .models.deepseek_v3 import mtp  # noqa: F401
