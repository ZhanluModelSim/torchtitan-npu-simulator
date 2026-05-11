# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "0.2.2.post2"

import sys

_initialized = False


def _apply_patches():
    """Apply all patches for torchtitan-npu"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    # patching optimizer before importing torchtitan.models
    from .patches.optimizer import swap_optimizer  # noqa: F401 # usort:skip

    # patching torchtitan
    from torchtitan_npu.patches.torchtitan import (  # noqa: F401
        activation_checkpoint,
        expert_parallel,
        hf_datasets,
        loss,
    )

    # patching ops
    from . import converters, ops  # noqa: F401  # noqa: F401

    # patching mxfp8/hif8
    from .converters import quant_converter  # noqa: F401

    # patching context_parallel utils
    from .patches.distributed import (  # noqa: F401  # noqa: F401, F811
        cp_input_sharding,
        custom_context_parallel,
        utils,
    )

    # patching step timing
    from .patches.tools import metrics  # noqa: F401

    # async_tp
    # patching torch
    from .patches.torch import clip_grad, micro_pipeline_tp, pipelining  # noqa: F401

    # patching fake process group
    from .patches.torch.testing._internal.distributed import fake_pg  # noqa: F401

    # patching torch_npu
    from .patches.torch_npu import custom_shardings  # noqa: F401

    # patching tools
    from .tools import flight_recorder, profiling  # noqa: F401


def _inject_module(module_path: str, replacement_module):
    """add/replace modules into sys.modules"""
    sys.modules[module_path] = replacement_module


_apply_patches()
