# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 reduced GraphTrainer recipes for A3/A5 text and single-image training."""

import os
from dataclasses import replace

from torchtitan.config.parallelism import ParallelismConfig

from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.models.deepseek_v4_1.flavors import _ENGRAM_8P_FLASH_GEOMETRY
from torchtitan_npu_recipes.tests.models.deepseek_v4_1 import _flash_overrides


def _graph_example(*, a5: bool = False, multimodal: bool = False):
    from torchtitan_npu_recipes.graph_trainer.deepseek_v4_1 import _make_graph_trainer_config

    flavor = None
    if a5:
        flavor = "deepseek_v4_1_flash_40layers_24experts_" + ("multimodal" if multimodal else "text")
    geometry = _ENGRAM_8P_FLASH_GEOMETRY
    table_rows = int(os.environ.get("TORCHTITAN_ENGRAM_TABLE_ROWS", "100000"))
    config = _make_graph_trainer_config(
        multimodal=multimodal,
        flavor=flavor,
        engram=replace(geometry, vocab_size_per_ngram=(table_rows,) * len(geometry.vocab_size_per_ngram)),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            data_parallel_replicate_degree=1,
            expert_parallel_degree=8,
            fsdp_reshard_after_forward="always",
            enable_sequence_parallel=True,
        ),
        seq_len=4096,
        hf_assets_path=os.environ.get("HF_ASSETS_PATH", "/path/to/DeepSeekV41_tokenizer"),
    )
    config.training.steps = 40
    config.optim.lr_scheduler.warmup_steps = 2
    # This hardware recipe selects a complete implementation set, replacing
    # the public recipe's reference RoPE rather than enabling both backends.
    config.override.imports = _flash_overrides(a5=a5, fused_attention=True)
    if multimodal:
        config.override.imports.append("torchtitan_npu.override.deepseek_v4_1.vision.asc_unmasked_sdpa")
    config.debug.print_config = True
    config.profiler = CANNProfiler.Config(enable_profiling=False)
    config.comm.init_timeout_seconds, config.comm.train_timeout_seconds = 7200, 600
    if a5:
        config.extension.quantization.enable_quantized_training = True
        config.extension.quantization.recipe = "all_block_fp8"
    return config


def graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_a3():
    return _graph_example()


def graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_a3_multimodal():
    return _graph_example(multimodal=True)


def graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_a5():
    return _graph_example(a5=True)


def graph_trainer_deepseek_v4_1_flash_8p_cpt_4k_a5_multimodal():
    return _graph_example(a5=True, multimodal=True)
