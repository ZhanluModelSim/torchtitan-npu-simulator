# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# The import order below is the patch activation order, not isort's.
# ruff: noqa: I001

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


# Upstream #5744: 0.4.0's ``torchtitan.config`` re-exports ``ParallelismConfig``,
# so a module that imports ``torchtitan.distributed.context_parallel`` first
# re-enters the half-initialized CP module.  Loading the package here makes the
# entry order stop mattering.  Drop it when the pin carries #5744.
import torchtitan.config  # noqa: F401

from .components import metrics, optimizer  # noqa: F401
import torchtitan_npu.patches.torchtitan.distributed.checkpoint.filesystem  # noqa: F401

from . import observability  # noqa: F401

from .models.common import moe, rope, token_dispatcher  # noqa: F401


def _patch_optional_qwen_short_conv() -> None:
    # Apply before a Qwen import resolves the optional CuTeDSL backend.
    from torchtitan.models.common import attn_gym

    resolve = attn_gym.__getattr__
    backend = "attn_gym.linear.short_conv"

    def resolve_optional_short_conv(name: str):
        try:
            return resolve(name)
        except ImportError as error:
            cause = error.__cause__
            if name != "causal_conv1d" or not isinstance(cause, ModuleNotFoundError) or cause.name != "cutlass":
                raise
            # Attention Gym 0.0.17 wraps this missing backend on attribute
            # access. Give TorchTitan the same deferred error it uses when
            # a backend fails during module import.
            vars(attn_gym)["_BACKEND_ERRORS"].setdefault(backend, cause)
            kernel = vars(attn_gym)["_missing_backend_kernel"](backend)
            vars(attn_gym)[name] = kernel
            return kernel

    attn_gym.__getattr__ = resolve_optional_short_conv


_patch_optional_qwen_short_conv()
from .models import deepseek_v4  # noqa: F401
