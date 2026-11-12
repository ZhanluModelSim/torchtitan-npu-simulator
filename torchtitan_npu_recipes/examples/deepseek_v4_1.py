# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Complete DeepSeek-V4.1 training configurations selected by example launchers."""

import os
from dataclasses import replace

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.optim import LRSchedulersContainer
from torchtitan.config import derive
from torchtitan.config.override import OverrideImport
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.rope import RoPE

from torchtitan_npu.config import CompileConfig, ModelCompileConfig, OptimizerConfig
from torchtitan_npu.config.configs import ProfilerExtensionConfig
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.extensions.torchtitan.tools.profiler import CANNProfiler
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.alignment import AlignedTextProcessor
from torchtitan_npu.models.deepseek_v4_1 import _attach_engram, _flash_engram_geometry
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model
from torchtitan_npu_recipes.models import deepseek_v4_1 as models


def _reject_legacy_overrides() -> None:
    for name in ("CLI_OVERRIDES", "OPTIMIZER_OVERRIDES", "DATALOADER_OVERRIDES", "MHC_POST_OVERRIDE"):
        if name in os.environ:
            raise ValueError(f"{name} no longer selects training settings; define a Python recipe or use --override")


def _world_size(*, local_processes: int, nodes: int) -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    node_ips = os.environ.get("NODE_IPS", "")
    node_count = len([address for address in node_ips.split(",") if address.strip()]) if node_ips else nodes
    return int(os.environ.get("NGPU", local_processes)) * int(os.environ.get("NNODES", node_count))


def _set_topology(
    config: TrainerEx.Config,
    *,
    local_processes: int,
    nodes: int,
    shard_degree: int,
    expert_degree: int,
    context_degree: int = 1,
    configurable: bool = False,
) -> None:
    if configurable:
        shard_degree = int(os.environ.get("DP_SHARD", shard_degree))
        expert_degree = int(os.environ.get("EP", expert_degree))
        context_degree = int(os.environ.get("CP", context_degree))
    world_size = _world_size(local_processes=local_processes, nodes=nodes)
    assert isinstance(config.model, DeepSeekV41Model.Config)
    num_experts = config.model.layers[0].feed_forward.num_experts
    if (
        min(world_size, shard_degree, expert_degree, context_degree) <= 0
        or world_size % (shard_degree * context_degree)
        or world_size % expert_degree
        or num_experts % expert_degree
    ):
        raise ValueError("WORLD_SIZE must form complete DP/CP/EP groups, and EP must divide the model's expert count")
    config.parallelism = ParallelismConfig(
        data_parallel_shard_degree=shard_degree,
        data_parallel_replicate_degree=world_size // (shard_degree * context_degree),
        expert_parallel_degree=expert_degree,
        context_parallel_degree=context_degree,
        tensor_parallel_degree=1,
        pipeline_parallel_degree=1,
        fsdp_reshard_after_forward="always",
        enable_sequence_parallel=True,
    )


def _set_context(config: TrainerEx.Config, *, sequence_length: int, token_budget: int, steps: int) -> None:
    assert isinstance(config.model, DeepSeekV41Model.Config)
    config.model.max_context_length = sequence_length
    for _, rope, _, _ in config.model.traverse(RoPE.Config):
        rope.max_context_length = sequence_length
    config.training.max_context_length = sequence_length
    config.training.num_tokens_per_microbatch_per_dp_rank = sequence_length
    config.training.num_tokens_per_train_step = token_budget
    config.training.steps = steps
    config.training.disable_cuda_graphs = True
    config.metrics.log_freq = 1


def _set_muon(config: TrainerEx.Config) -> None:
    assert isinstance(config.optim.optimizer, OptimizerConfig)
    optimizer = config.optim.optimizer
    optimizer.name = "Muon"
    optimizer.lr = 1e-5
    optimizer.beta1, optimizer.beta2 = 0.9, 0.95
    optimizer.eps, optimizer.weight_decay = 1e-8, 0.1
    optimizer.muon_momentum = 0.95
    optimizer.muon_enable_nesterov = True
    optimizer.muon_ns_steps = 10
    optimizer.muon_adjust_lr_fn = "match_rms_adamw"
    optimizer.materialize()


def _set_schedule(config: TrainerEx.Config) -> None:
    config.optim.lr_scheduler = LRSchedulersContainer.Config(
        warmup_steps=2,
        total_steps=config.training.steps,
        decay_ratio=1.0,
        decay_type="cosine",
        min_lr_factor=0.01,
    )


def _overrides(*, a5: bool, debug: bool, swap: bool = True, qat: bool = False) -> list[OverrideImport]:
    golden = os.environ.get("USE_GOLDEN", "0") == "1"
    tilelang = a5 and not golden and os.environ.get("USE_TILELANG", "0") == "1"
    result: list[OverrideImport]
    if golden:
        result = ["torchtitan_npu.override.common.rope.workaround"]
    else:
        result = [
            "torchtitan_npu.override.common.rms_norm.asc",
            "torchtitan_npu.override.common.rope.asc_half_rotation",
            (
                "torchtitan_npu.override.deepseek_v4_1.tilelang.tilelang_hc_post"
                if tilelang
                else "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post"
            ),
            "torchtitan_npu.override.common.token_dispatcher.asc",
        ]
    if swap:
        result.append("torchtitan_npu.override.common.optimizer.swap_optimizer")
    # QAT historically selects its fused attention list independently of USE_GOLDEN.
    if qat or not golden:
        result.append(f"torchtitan_npu.override.common.rope.{'asc_partial' if a5 else 'asc_complex'}")
        if tilelang and not qat:
            result.append("torchtitan_npu.override.deepseek_v4_1.tilelang.tilelang_hc_pre")
        if a5:
            result.extend(
                [
                    "torchtitan_npu.override.common.swiglu_group.asc",
                    "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
                ]
            )
        if a5 or debug:
            result.extend(
                [
                    "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
                    "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
                ]
            )
        if a5 and not tilelang and not qat:
            result.append("torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn")
    return result


def _sft(flavor: str, *, checkpoint: bool) -> TrainerEx.Config:
    config = models.deepseek_v4_1_vision_sft(
        flavor=flavor,
        dataset_path=os.environ.get("DSV41_SFT_DATASET") or os.environ.get("DATASET_PATH") or "/path/to/train.jsonl",
        image_root=models._vision_sft_env("IMAGE_ROOT", ""),
        thinking_mode=models._vision_sft_thinking_mode(),
        drop_thinking=models._vision_sft_env_flag("DROP_THINKING", True),
        add_default_bos_token=models._vision_sft_env_flag("ADD_DEFAULT_BOS_TOKEN", True),
        reasoning_effort=models._vision_sft_env("REASONING_EFFORT", "") or None,
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    if checkpoint:
        config.checkpointer = CheckpointManager.Config(load_only=True)
    return config


def _debug(*, a5: bool, sft: bool = False, qat: bool = False) -> TrainerEx.Config:
    _reject_legacy_overrides()
    if sft:
        flavor = (
            "deepseek_v4_1_flash_40layers_24experts_multimodal"
            if a5
            else "deepseek_v4_1_flash_40layers_16experts_vision"
        )
        config = _sft(flavor, checkpoint=False)
    elif a5:
        config = models.deepseek_v4_1_flash_40layers_24experts_multimodal_compiled()
    else:
        config = models.deepseek_v4_1_flash_40layers_16experts_multimodal_compiled()
    assert isinstance(config.model, DeepSeekV41Model.Config)
    geometry = _flash_engram_geometry()
    table_rows = int(os.environ.get("TORCHTITAN_ENGRAM_TABLE_ROWS", "100000"))
    _attach_engram(
        config.model,
        replace(geometry, vocab_size_per_ngram=(table_rows,) * len(geometry.vocab_size_per_ngram)),
    )
    _set_topology(config, local_processes=8, nodes=1, shard_degree=8, expert_degree=8)
    _set_context(config, sequence_length=4096, token_budget=32768, steps=40)
    _set_muon(config)
    _set_schedule(config)
    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH", "/path/to/DeepSeekV41_tokenizer")
    config.debug.print_config = True
    config.profiler = CANNProfiler.Config(
        enable_profiling=False, profile_freq=10, profiler_active=10, profiler_warmup=0
    )
    config.comm.init_timeout_seconds, config.comm.train_timeout_seconds = 7200, 600
    config.override.imports = _overrides(a5=a5, debug=True, qat=qat)
    config.extension.quantization.enable_quantized_training = a5
    config.extension.quantization.recipe = "all_block_fp8"
    if qat:
        config.extension.quantization.enable_mxfp4_qat = True
        config.extension.quantization.li_quantization = "mxfp4"
        config.extension.quantization.enable_sparse_attention_quantization = True
    return config


def _full(*, a5: bool, sft: bool = False, lora: bool = False) -> TrainerEx.Config:
    _reject_legacy_overrides()
    if sft:
        config = _sft("deepseek_v4_1_flash", checkpoint=True)
    elif lora:
        config = models.deepseek_v4_1_flash_lora()
    else:
        config = models.deepseek_v4_1_flash_cpt()
    _set_topology(
        config,
        local_processes=8 if a5 else 16,
        nodes=16 if a5 else 8,
        shard_degree=128,
        expert_degree=128,
        configurable=True,
    )
    _set_context(config, sequence_length=4096, token_budget=4194304, steps=int(os.environ.get("STEPS", "100")))
    _set_muon(config)
    _set_schedule(config)
    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH", "/path/to/DeepSeekV41_tokenizer")
    config.debug.print_config = True
    config.profiler = CANNProfiler.Config(
        enable_profiling=False,
        save_traces_folder="profiling_path",
        extension=ProfilerExtensionConfig(
            enable_online_parse=False, profiler_start=6, profiler_end=7, profile_ranks=[0]
        ),
    )
    config.comm.init_timeout_seconds, config.comm.train_timeout_seconds = 7200, 600
    assert config.checkpointer is not None
    config.checkpointer.folder = os.environ.get("CKPT_SAVE_LOAD_PATH", "/path/to/save_ckpt")
    config.checkpointer.initial_load_path = os.environ.get("CKPT_INIT_LOAD_PATH", "/path/to/init_load_ckpt")
    config.checkpointer.initial_load_in_hf = True
    config.checkpointer.load_only = not lora
    config.override.imports = _overrides(a5=a5, debug=False, swap=not lora)
    config.extension.quantization.enable_quantized_training = a5
    config.extension.quantization.recipe = "all_block_fp8"
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


def deepseek_v4_1_flash_cpt_4k_a3() -> TrainerEx.Config:
    return _full(a5=False)


def deepseek_v4_1_flash_cpt_4k_a5() -> TrainerEx.Config:
    return _full(a5=True)


def deepseek_v4_1_flash_lora_4k_a3() -> TrainerEx.Config:
    return _full(a5=False, lora=True)


def deepseek_v4_1_flash_sft_4k_a3() -> TrainerEx.Config:
    return _full(a5=False, sft=True)


def deepseek_v4_1_flash_sft_4k_a5() -> TrainerEx.Config:
    return _full(a5=True, sft=True)


def deepseek_v4_1_mini_1p_cpt_a5() -> TrainerEx.Config:
    _reject_legacy_overrides()
    config = models.deepseek_v4_1_stripped()
    _set_topology(config, local_processes=1, nodes=1, shard_degree=1, expert_degree=1)
    _set_context(config, sequence_length=2048, token_budget=2048, steps=int(os.environ.get("STEPS", "5")))
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
    return config


def deepseek_v4_1_flash_cpt_1m_a5() -> TrainerEx.Config:
    """Declare the long-context CP configuration; the model still rejects unsupported CP."""
    _reject_legacy_overrides()
    config = models.deepseek_v4_1_flash_text()
    config = derive(config, type(config), engram_enabled=False)
    _set_topology(
        config,
        local_processes=8,
        nodes=16,
        shard_degree=1,
        expert_degree=128,
        context_degree=128,
        configurable=True,
    )
    _set_context(config, sequence_length=1048576, token_budget=1048576, steps=int(os.environ.get("STEPS", "10")))
    _set_muon(config)
    _set_schedule(config)
    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH", "/path/to/DeepSeekV41_tokenizer")
    config.debug.print_config = True
    assert isinstance(config.dataloader, GrainDataLoader.Config)
    packed = config.dataloader.dataset
    assert isinstance(packed, ConcatThenSplitPackingConfig)
    dataset = packed.dataset
    assert isinstance(dataset, SingleDatasetConfig)
    assert isinstance(dataset.source, HuggingFaceRandomAccessSource.Config)
    assert isinstance(dataset.processor, AlignedTextProcessor.Config)
    dataset.source.load_dataset_kwargs["data_files"] = (
        os.environ.get("DATA_FILES") or os.environ.get("DATASET_PATH") or "tests/assets/c4_test/data.json"
    )
    dataset.processor.per_doc_alignment = int(os.environ.get("DSV41_PER_DOC_ALIGNMENT", "256"))
    config.profiler = CANNProfiler.Config(
        enable_profiling=False,
        save_traces_folder="profiling_path",
        extension=ProfilerExtensionConfig(
            enable_online_parse=False, profiler_start=6, profiler_end=7, profile_ranks=[0]
        ),
    )
    config.comm.init_timeout_seconds, config.comm.train_timeout_seconds = 7200, 600
    config.override.imports = [
        "torchtitan_npu.override.common.rms_norm.asc",
        "torchtitan_npu.override.common.rope.asc_partial",
        "torchtitan_npu.override.common.token_dispatcher.asc",
        "torchtitan_npu.override.common.swiglu_group.asc",
        "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
        "torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn",
        "torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post",
        "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
        "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
        "torchtitan_npu.override.common.optimizer.swap_optimizer",
    ]
    return config
