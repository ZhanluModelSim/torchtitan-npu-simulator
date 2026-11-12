# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# torch 2.14 CI compatibility: upstream torchtitan 0.4.0 imports this newer
# pipelining symbol when ``torchtitan.models.common.dist_moe.runtime`` loads.
# The NPU eager path never runs pipeline parallel, so install a fail-fast stub
# when the running torch predates 2.15.
import torch.distributed.pipelining as _pipelining

if not hasattr(_pipelining, "analyze_pipeline_activation_liveness"):

    def _analyze_pipeline_activation_liveness(*args, **kwargs):
        raise RuntimeError("analyze_pipeline_activation_liveness requires torch >= 2.15")

    _pipelining.analyze_pipeline_activation_liveness = (  # pyrefly: ignore [missing-attribute]
        _analyze_pipeline_activation_liveness
    )

if not hasattr(_pipelining, "PipelineStageInfo"):

    class _PipelineStageInfo:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PipelineStageInfo requires torch >= 2.15")

    _pipelining.PipelineStageInfo = _PipelineStageInfo  # pyrefly: ignore [missing-attribute]


from .components import checkpointer, metrics, optimizer  # noqa: F401, I001
import torchtitan_npu.patches.torchtitan.distributed.checkpoint.filesystem  # noqa: F401

from . import observability  # noqa: F401

from .models.common import moe, rope, token_dispatcher  # noqa: F401
from .models import deepseek_v4  # noqa: F401
