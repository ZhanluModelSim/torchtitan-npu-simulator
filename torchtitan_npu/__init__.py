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

    # patching MTP context-parallel, before importing torchtitan.trainer
    from .patches.distributed import mtp_context_parallel  # noqa: F401

    # async_tp must be patched before importing NPU model modules because
    # their parallelize files import maybe_enable_async_tp by value.
    # ruff: noqa: I001
    from .patches.torch import micro_pipeline_tp  # noqa: F401 # isort: off

    # patching optimizer before importing torchtitan.models
    from .patches.optimizer.optimizer_selector import (
        patch_npu_optimizer_framework,
    )  # usort:skip
    # ruff: noqa: I001

    patch_npu_optimizer_framework()

    import torchtitan.models as titan_models

    # patching torchtitan
    from torchtitan_npu.patches.torchtitan import (  # noqa: F401
        attention,
        attention_varlen_cpu,
        batch_invariance,
        chat_dataset,
        expert_parallel,
        hf_datasets,
        loss,
        optimizer,
        pp_loss_normalize,
        trainer_post_dataloading_process,
    )

    # patching model_converter and ops
    from . import converters, ops  # noqa: F401

    # module injection: register NPU-only model variants
    from .models import deepseek_v4, deepseek_v32, vlm
    from .patches.distributed import cp_shard_mask, utils  # noqa: F401

    # patching step timing
    from .patches.tools import metrics  # noqa: F401

    # async_tp
    # patching torch
    from .patches.torch import (  # noqa: F401
        checkpoint,
        clip_grad,
        functional_collectives_p2p,
        pipelining,
    )

    # patching FSDP2 DTensor dtype sync (backport from PyTorch main)
    from .patches.torch.fsdp import fsdp_dtype_fix  # noqa: F401
    from .patches.torch.fsdp import parameter_precision  # noqa: F401

    # patching fake process group
    from .patches.torch.testing._internal.distributed import fake_pg  # noqa: F401

    # patching torch_npu
    from .patches.torch_npu import determinism  # noqa: F401

    # patching torchao for NPU MXFP8 training support
    from .patches.torchao_npu import (  # noqa: F401
        mx_capability_check,
        mx_linear,
        mxfp8_grouped_mm,
    )

    # Patch Qwen3StateDictAdapter.to_hf to sync fqn_to_index_mapping.
    # Multi-turn chat is handled by the unified chat_dataset patch above.
    from .patches.torchtitan import (
        qwen3_hf_export,  # noqa: F401
    )

    # patching tools
    from .tools import flight_recorder, profiling  # noqa: F401

    new_set = set(titan_models._supported_models)
    new_set.update({"deepseek_v32", "deepseek_v4", "vlm"})
    titan_models._supported_models = frozenset(new_set)

    _inject_module("torchtitan.models.deepseek_v32", deepseek_v32)
    _inject_module("torchtitan.models.deepseek_v4", deepseek_v4)
    _inject_module("torchtitan.models.vlm", vlm)


def _inject_module(module_path: str, replacement_module):
    """add/replace modules into sys.modules"""
    sys.modules[module_path] = replacement_module


_apply_patches()
