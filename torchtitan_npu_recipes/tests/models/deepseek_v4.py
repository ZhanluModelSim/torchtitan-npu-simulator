# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4 debug and reduced-model training configurations."""

from typing import Literal

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import ModelConfigConverter
from torchtitan.distributed.activation_remat import FullActivationRematConfig

from torchtitan_npu.config import OptimizerConfig
from torchtitan_npu.config.configs import KVNormQuantizationConfig
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.deepseek_v4 import build_model_config
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan_npu_recipes.models.deepseek_v4 import _A5_PREQUANTIZE_FQNS, _dsv4_optimizer_config, _trainer_config


def _eager_observability(config: TrainerEx.Config) -> None:
    config.debug.print_config = True
    config.profiler.enable_profiling = False
    config.profiler.save_traces_folder = "profiling_path"
    config.profiler.extension.enable_online_parse = False
    config.profiler.extension.profiler_start = 6
    config.profiler.extension.profiler_end = 7
    config.profiler.extension.profile_ranks = [0]
    config.comm.init_timeout_seconds = 7200
    config.comm.train_timeout_seconds = 600


def _eager_optimizer(model: DeepSeekV4Model.Config, *, name: Literal["AdamW", "Muon"] = "Muon") -> OptimizerConfig:
    if name == "AdamW":
        from torchtitan.components.optim import AdamW

        return OptimizerConfig(
            optimizers=[
                AdamW.Config(
                    pattern=r".*",
                    lr=1e-5,
                    betas=(0.9, 0.95),
                    eps=1e-8,
                    weight_decay=0.1,
                    fused=True,
                    foreach=False,
                )
            ]
        )
    return _dsv4_optimizer_config(model, lr=1e-5)


def deepseek_v4_debugmodel(
    *,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 2048,
    hf_assets_path: str | None = None,
    data_files: str = "tests/assets/c4_test/data.json",
    per_doc_alignment: int | None = None,
) -> TrainerEx.Config:
    parallelism = parallelism or ParallelismConfig(expert_parallel_degree=1, fsdp_reshard_after_forward="always")
    config = _trainer_config(
        build_model_config(
            "debugmodel",
            parallelism=parallelism,
            seq_len=seq_len,
            num_mtp_layers=num_mtp_layers,
            converters=converters,
        ),
        num_mtp_layers=num_mtp_layers,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        data_files=data_files,
        per_doc_alignment=per_doc_alignment,
    )
    config.hf_assets_path = hf_assets_path or "tests/assets/deepseek_v3"
    return config


def deepseek_v4_flash_43layers_16experts_eager_bf16(
    *,
    num_mtp_layers: int = 0,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str = "/path/to/DeepSeekV4_tokenizer",
    data_files: str = "tests/assets/c4_test/data.json",
    per_doc_alignment: int = 128,
) -> TrainerEx.Config:
    """Build 8-rank Flash eager BF16 CPT on A3 with EP8/FSDP8, Muon and fused ops."""
    parallelism = parallelism or ParallelismConfig(
        expert_parallel_degree=8,
        data_parallel_shard_degree=8,
        data_parallel_replicate_degree=1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        context_parallel_degree=1,
        fsdp_reshard_after_forward="always",
    )
    config = _trainer_config(
        build_model_config(
            "deepseek_v4_flash_43layers_16experts",
            parallelism=parallelism,
            seq_len=seq_len,
            num_mtp_layers=num_mtp_layers,
            converters=None,
        ),
        num_mtp_layers=num_mtp_layers,
        parallelism=parallelism,
        hf_assets_path=hf_assets_path,
        data_files=data_files,
        per_doc_alignment=per_doc_alignment,
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = seq_len
    config.training.num_tokens_per_train_step = 64 * seq_len
    config.training.steps = 100
    config.optim.optimizer = _eager_optimizer(config.model)
    config.activation_remat = FullActivationRematConfig()
    _eager_observability(config)
    config.override.imports = [
        "torchtitan_npu.override.common.rms_norm.asc",
        "torchtitan_npu.override.deepseek_v4.lightning_indexer.asc",
        "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
        "torchtitan_npu.override.common.optimizer.swap_optimizer",
        "torchtitan_npu.override.common.rope.asc_complex",
        "torchtitan_npu.override.common.token_dispatcher.asc",
    ]
    return config


def deepseek_v4_flash_43layers_16experts_eager_fp8(
    *,
    num_mtp_layers: int = 0,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    data_files: str = "tests/assets/c4_test/data.json",
    per_doc_alignment: int | None = None,
) -> TrainerEx.Config:
    """Build 8-rank Flash eager block-FP8 configuration on A5."""
    config = deepseek_v4_flash_43layers_16experts_eager_bf16(
        num_mtp_layers=num_mtp_layers,
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        data_files=data_files,
        per_doc_alignment=128 if per_doc_alignment is None else per_doc_alignment,
    )
    # Fused overrides for this variant without swap_optimizer.
    config.override.imports = [
        "torchtitan_npu.override.common.rms_norm.asc",
        "torchtitan_npu.override.deepseek_v4.lightning_indexer.asc",
        "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
        "torchtitan_npu.override.common.rope.asc_complex",
        "torchtitan_npu.override.common.token_dispatcher.asc",
    ]
    config.training.num_tokens_per_microbatch_per_dp_rank = 2 * seq_len
    quantization = config.extension.quantization
    quantization.enable_quantized_training = True
    quantization.recipe = "all_block_fp8"
    quantization.enable_fsdp_prequantize = True
    quantization.fsdp_prequantize_fqns = list(_A5_PREQUANTIZE_FQNS)
    quantization.li_quantization = "fp8"
    quantization.kv_norm_quantization = KVNormQuantizationConfig(
        format="mxfp8",
        fqns=[".attention.kv_norm", ".attention.compressor.norm", ".attention.compressor_128.norm"],
        block_size=64,
    )
    return config


def deepseek_v4_flash_43layers_16experts_eager_fp8_qat(
    *,
    num_mtp_layers: int = 0,
    parallelism: ParallelismConfig | None = None,
    seq_len: int = 4096,
    hf_assets_path: str | None = None,
    data_files: str = "tests/assets/c4_test/data.json",
    per_doc_alignment: int | None = None,
) -> TrainerEx.Config:
    """Build 8-rank Flash eager block-FP8 with MXFP4 QAT on A5."""
    config = deepseek_v4_flash_43layers_16experts_eager_fp8(
        num_mtp_layers=num_mtp_layers,
        parallelism=parallelism,
        seq_len=seq_len,
        hf_assets_path=hf_assets_path,
        data_files=data_files,
        per_doc_alignment=per_doc_alignment,
    )
    config.extension.quantization.enable_mxfp4_qat = True
    config.extension.quantization.li_quantization = "mxfp4"
    return config


def deepseek_v4_flash_43layers_16experts_local_compile_bf16_a5(
    *,
    hf_assets_path: str = "tests/assets/deepseek_v3",
    data_files: str = "tests/assets/c4_test/data.json",
) -> TrainerEx.Config:
    """Build a five-step 8-rank A5 Flash example with BF16, Full AC and local compilation."""
    config = deepseek_v4_flash_43layers_16experts_eager_bf16(
        hf_assets_path=hf_assets_path,
        data_files=data_files,
    )
    config.override.imports.remove("torchtitan_npu.override.common.optimizer.swap_optimizer")
    config.training.num_tokens_per_microbatch_per_dp_rank = 2 * config.training.max_context_length
    config.training.steps = 5
    config.debug.seed = 42
    config.model.local_compile_regions = ["loss", "fp32_to_bf16_split", "rope"]
    return config


def deepseek_v4_flash_1p_2k(
    *,
    hf_assets_path: str = "/path/to/DeepSeekV4_tokenizer",
    data_files: str = "tests/assets/c4_test/data.json",
    per_doc_alignment: int = 128,
    parallelism: ParallelismConfig | None = None,
    steps: int = 100,
) -> TrainerEx.Config:
    parallelism = parallelism or ParallelismConfig(
        fsdp_reshard_after_forward="always",
        expert_parallel_degree=1,
        data_parallel_shard_degree=1,
        data_parallel_replicate_degree=1,
    )
    config = deepseek_v4_debugmodel(
        parallelism=parallelism,
        seq_len=2048,
        hf_assets_path=hf_assets_path,
        data_files=data_files,
        per_doc_alignment=per_doc_alignment,
    )
    config.training.num_tokens_per_microbatch_per_dp_rank = 2048
    config.training.num_tokens_per_train_step = -1
    config.training.steps = steps
    config.optim.optimizer = _eager_optimizer(config.model)
    _eager_observability(config)
    config.override.imports = [
        "torchtitan_npu.override.common.rms_norm.asc",
        "torchtitan_npu.override.common.rope.asc_complex",
        "torchtitan_npu.override.deepseek_v4.lightning_indexer.asc",
        (
            "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
            {"num_heads": 16, "index_num_heads": 8},
        ),
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
        "torchtitan_npu.override.common.token_dispatcher.asc",
    ]
    config.override.imports.append("torchtitan_npu.override.common.optimizer.swap_optimizer")
    return config
