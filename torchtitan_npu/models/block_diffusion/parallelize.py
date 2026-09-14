# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed plans for the Block Diffusion model."""

import torchtitan.models.llama4.parallelize as llama4_parallelize

from .model import BlockDiffusionModel


def parallelize_block_diffusion(
    model: BlockDiffusionModel,
    *,
    parallel_dims,
    training,
    model_converters,
    parallelism,
    compile_config,
    ac_config,
    dump_folder: str,
):
    """Apply the maintained sparse-decoder TP/EP/ETP/CP/FSDP plan.

    BlockDiffusion uses the same module contracts as TorchTitan's sparse
    decoder: GQA projections plus common token-choice MoE.  Reusing that plan
    keeps collective ownership and simulator hooks aligned with the training
    branch instead of copying a stale parallel implementation.
    """

    original_cp_applier = llama4_parallelize.apply_cp_to_attention_module
    if parallel_dims.cp_enabled:
        # The upstream SDPA context-parallel dispatcher duplicates the last
        # dimension for this non-causal GQA workload at production head
        # counts.  The maintained NPU Ulysses implementation performs an
        # explicit heads<->sequence all-to-all and preserves the BSND shape.
        from torchtitan_npu.distributed.context_parallel import (
            apply_cp_to_attention_module,
        )

        llama4_parallelize.apply_cp_to_attention_module = (
            apply_cp_to_attention_module
        )

    try:
        return llama4_parallelize.parallelize_llama(
            model,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
    finally:
        llama4_parallelize.apply_cp_to_attention_module = original_cp_applier
