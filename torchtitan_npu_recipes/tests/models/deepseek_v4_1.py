# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 reduced and mini training recipes."""

import os
from dataclasses import replace

from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4_1 import _flash_engram_geometry
from torchtitan_npu_recipes.models import deepseek_v4_1 as models


def _debug(*, a5: bool, sft: bool = False, qat: bool = False) -> TrainerEx.Config:
    models._reject_legacy_overrides()
    parallelism = models._topology(local_processes=8, nodes=1, shard_degree=8, expert_degree=8)
    geometry = _flash_engram_geometry()
    table_rows = int(os.environ.get("TORCHTITAN_ENGRAM_TABLE_ROWS", "100000"))
    options = dict(
        engram=replace(geometry, vocab_size_per_ngram=(table_rows,) * len(geometry.vocab_size_per_ngram)),
        parallelism=parallelism,
        seq_len=4096,
        hf_assets_path=os.environ.get("HF_ASSETS_PATH", "/path/to/DeepSeekV41_tokenizer"),
    )
    if sft:
        flavor = (
            "deepseek_v4_1_flash_40layers_24experts_multimodal"
            if a5
            else "deepseek_v4_1_flash_40layers_16experts_vision"
        )
        config = models._sft(flavor, checkpoint=False, **options)
    elif a5:
        config = models.deepseek_v4_1_flash_40layers_24experts_multimodal_compiled(**options)
    else:
        config = models.deepseek_v4_1_flash_40layers_16experts_multimodal_compiled(**options)
    models._set_context(config, sequence_length=4096, token_budget=32768, steps=40)
    models._set_muon(config)
    models._set_schedule(config)
    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH", "/path/to/DeepSeekV41_tokenizer")
    config.debug.print_config = True
    config.profiler = CANNProfiler.Config(
        enable_profiling=False, profile_freq=10, profiler_active=10, profiler_warmup=0
    )
    config.comm.init_timeout_seconds, config.comm.train_timeout_seconds = 7200, 600
    config.override.imports = models._overrides(a5=a5, debug=True, qat=qat)
    config.extension.quantization.enable_quantized_training = a5
    config.extension.quantization.recipe = "all_block_fp8"
    if a5:
        config.extension.quantization.enable_fsdp_prequantize = True
        config.extension.quantization.fsdp_prequantize_fqns = list(models._A5_PREQUANTIZE_FQNS)
    if qat:
        config.extension.quantization.enable_mxfp4_qat = True
        config.extension.quantization.li_quantization = "mxfp4"
        config.extension.quantization.enable_sparse_attention_quantization = True
    return config


def deepseek_v4_1_flash_8p_cpt_4k_a3() -> TrainerEx.Config:
    return _debug(a5=False)


def deepseek_v4_1_flash_8p_cpt_4k_a5() -> TrainerEx.Config:
    return _debug(a5=True)


def deepseek_v4_1_flash_8p_qat_4k_a5() -> TrainerEx.Config:
    return _debug(a5=True, qat=True)


def deepseek_v4_1_flash_8p_sft_4k_a3() -> TrainerEx.Config:
    return _debug(a5=False, sft=True)


def deepseek_v4_1_flash_8p_sft_4k_a5() -> TrainerEx.Config:
    return _debug(a5=True, sft=True)


def deepseek_v4_1_mini_1p_cpt_a5() -> TrainerEx.Config:
    models._reject_legacy_overrides()
    config = models.deepseek_v4_1_stripped(
        parallelism=models._topology(local_processes=1, nodes=1, shard_degree=1, expert_degree=1),
        seq_len=2048,
        hf_assets_path=os.environ.get("HF_ASSETS_PATH", "tests/assets/deepseek_v3"),
    )
    models._set_context(config, sequence_length=2048, token_budget=2048, steps=int(os.environ.get("STEPS", "5")))
    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH", "tests/assets/deepseek_v3")
    config.override.imports = [
        "torchtitan_npu.override.common.rms_norm.asc",
        "torchtitan_npu.override.common.rope.asc_complex",
        "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post",
        "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
        "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
    ]
    config.extension.quantization.enable_quantized_training = True
    config.extension.quantization.recipe = "all_block_fp8"
    config.extension.quantization.enable_fsdp_prequantize = True
    config.extension.quantization.fsdp_prequantize_fqns = list(models._A5_PREQUANTIZE_FQNS)
    return config
