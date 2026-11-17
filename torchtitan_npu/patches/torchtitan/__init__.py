# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .components import checkpoint  # noqa: F401, I001
from . import trainer  # noqa: F401
from .components import metrics, optimizer, validate  # noqa: F401
from .distributed import context_parallel, full_dtensor  # noqa: F401
from .distributed.flex_shard import dist_muon  # noqa: F401
from .experiments.graph_trainer import (  # noqa: F401
    chunked_loss,
    ep_chunk_concretization,
    ep_overlap_shape_queries,
    ep_ready_nodes_dedup,
    ep_shape_live_out,
)

# ``attention_metadata`` carries the backend-owned metadata contract and must
# be applied before ``decoder`` binds the map builder that reads it, and before
# ``mtp`` dispatches that map per layer.
from .models.common import attention_metadata, decoder, moe, rope, token_dispatcher  # noqa: F401
from .models.deepseek_v3 import mtp  # noqa: F401
