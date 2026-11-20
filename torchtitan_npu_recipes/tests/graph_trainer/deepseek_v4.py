# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reduced DeepSeek-V4 GraphTrainer recipes for debug launchers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import LRSchedulersContainer
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor

from torchtitan_npu.config import OptimConfig, TrainingConfig
from torchtitan_npu.config.configs import KVNormQuantizationConfig
from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.alignment import AlignedTextProcessor
from torchtitan_npu.models.deepseek_v4 import build_model_config
from torchtitan_npu_recipes.graph_trainer.deepseek_v4 import _to_graph_config
from torchtitan_npu_recipes.models.deepseek_v4 import _dsv4_optimizer_config

if TYPE_CHECKING:
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig


def _flash_8p_config() -> TrainerEx.Config:
    """Build the shared 43-layer, 16-expert training configuration before graph conversion."""
    parallelism = ParallelismConfig(
        expert_parallel_degree=8,
        data_parallel_shard_degree=8,
        data_parallel_replicate_degree=1,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        context_parallel_degree=1,
        fsdp_reshard_after_forward="always",
    )
    model = build_model_config("deepseek_v4_flash_43layers_16experts", parallelism=parallelism, seq_len=4096)
    model.local_compile_regions = []
    optimizer = _dsv4_optimizer_config(model, lr=1e-5)
    config = TrainerEx.Config(
        hf_assets_path="/path/to/DeepSeekV4_tokenizer",
        model=model,
        loss=ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model))),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(
                dataset=SingleDatasetConfig(
                    source=HuggingFaceRandomAccessSource.Config(
                        path="json", split="train", load_dataset_kwargs={"data_files": "tests/assets/c4_test/data.json"}
                    ),
                    processor=AlignedTextProcessor.Config(per_doc_alignment=128),
                    post_filters=(lambda sample: sample is not None,),
                ),
            ),
        ),
        optim=OptimConfig(
            optimizer=optimizer,
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=20, decay_ratio=0.8, decay_type="cosine", min_lr_factor=0.01
            ),
        ),
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=4096,
            num_tokens_per_train_step=262144,
            max_context_length=4096,
            steps=100,
        ),
        parallelism=parallelism,
        activation_remat=None,
        compile=None,
        checkpointer=None,
        metrics=MetricsProcessor.Config(log_freq=1),
        profiler=CANNProfiler.Config(
            enable_profiling=False,
            profile_freq=10,
            profiler_active=10,
            profiler_warmup=0,
            save_traces_folder="profiling_path",
        ),
    )
    config.debug.print_config = True
    config.profiler.extension.enable_online_parse = False
    config.profiler.extension.profiler_start = 6
    config.profiler.extension.profiler_end = 7
    config.profiler.extension.profile_ranks = [0]
    config.comm.init_timeout_seconds = 7200
    config.comm.train_timeout_seconds = 600
    return config


def graph_trainer_deepseek_v4_flash_8p_cpt_4k_a3() -> TrainerEx.Config:
    """Build 8-rank Flash graph CPT on A3 with 43 layers, 16 experts and BF16; currently deferred."""
    config = _flash_8p_config()
    config.override.imports = [
        "torchtitan_npu.override.common.rms_norm.asc",
        ("torchtitan_npu.override.deepseek_v4.sparse_attn.asc", {"selector_backend": "asc"}),
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
        "torchtitan_npu.override.common.token_dispatcher.asc",
        "torchtitan_npu.override.common.optimizer.swap_optimizer",
        "torchtitan_npu.override.common.rope.asc_complex",
    ]
    return _to_graph_config(config)


def graph_trainer_deepseek_v4_flash_8p_cpt_4k_a5() -> TrainerEx.Config:
    """Build 8-rank Flash graph CPT on A5 with 43 layers, 16 experts and block-FP8; currently deferred."""
    config = _flash_8p_config()
    config.training.num_tokens_per_microbatch_per_dp_rank = 8192
    config.training.steps = 20
    config.activation_remat = DefaultActivationRematConfig()
    quantization = config.extension.quantization
    quantization.enable_quantized_training = True
    quantization.recipe = "all_block_fp8"
    quantization.enable_fsdp_prequantize = True
    quantization.fsdp_prequantize_fqns = None
    quantization.li_quantization = "fp8"
    quantization.kv_norm_quantization = KVNormQuantizationConfig(
        format="mxfp8",
        fqns=[".attention.kv_norm", ".attention.compressor.norm"],
        block_size=64,
    )
    config.override.imports = [
        "torchtitan_npu.override.deepseek_v4.common.rms_norm",
        "torchtitan_npu.override.deepseek_v4.lightning_indexer.asc",
        ("torchtitan_npu.override.deepseek_v4.sparse_attn.asc", {"selector_backend": "asc"}),
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
        "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
        "torchtitan_npu.override.common.token_dispatcher.asc",
        "torchtitan_npu.override.deepseek_v4.common.rope",
        "torchtitan_npu.override.deepseek_v4.compressor.asc",
        "torchtitan_npu.override.common.swiglu_group.asc",
        "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
    ]
    config = _to_graph_config(config)
    # The NPU memory-policy registry owns this deferred custom policy.
    cast(
        "GraphTrainerCompileConfig", config.compile
    ).memory_policy = "dsv4-mhc-moe-save"  # pyrefly: ignore [bad-assignment]
    return config
