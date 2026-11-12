# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Complete DeepSeek-V4 example configurations selected by the shell launchers.

Compilation, CP, GraphTrainer, SFT and LoRA retain the current implementation's
support limits. Selecting an example does not enable a deferred training path.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig, FullActivationRematConfig
from torchtitan.models.common.rope import RoPE

from torchtitan_npu.config import CompileConfig, ModelCompileConfig, OptimizerConfig
from torchtitan_npu.config.configs import KVNormQuantizationConfig
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu_recipes.models import deepseek_v4 as model_recipes

if TYPE_CHECKING:
    from typing import Any

    from torchtitan.config.override import OverrideImport

    from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
    from torchtitan_npu.models.common.alignment import AlignedTextProcessor
    from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model
    from torchtitan_npu.models.deepseek_v4.peft import DeepSeekV4PEFTCheckpointManager


_SWAP_OPTIMIZER = "torchtitan_npu.override.common.optimizer.swap_optimizer"
_ROPE_REFERENCE = "torchtitan_npu.override.common.rope.workaround"
_A5_OVERRIDES = [
    "torchtitan_npu.override.deepseek_v4.common.rope",
    "torchtitan_npu.override.deepseek_v4.compressor.asc",
    "torchtitan_npu.override.common.swiglu_group.asc",
    "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
]
_A5_PREQUANTIZE_FQNS = [
    ".attention.wq_b",
    ".attention.wo_a",
    ".attention.indexer.wq_b",
    ".feed_forward.routed_experts.w13",
    ".feed_forward.routed_experts.w2",
]


def _check_override_environment() -> None:
    for name in ("CLI_OVERRIDES", "OPTIMIZER_OVERRIDES"):
        if name in os.environ:
            raise ValueError(
                f"{name} is no longer a shell configuration interface; "
                "select a Python recipe or use the registered --override option."
            )


def _world_size(*, local_ranks: int, nodes: int) -> int:
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    local_ranks = int(os.environ.get("NGPU", local_ranks))
    if os.environ.get("NODE_IPS"):
        nodes = len(os.environ["NODE_IPS"].split(","))
    else:
        nodes = int(os.environ.get("NNODES", nodes))
    return local_ranks * nodes


def _is_golden() -> bool:
    return os.environ.get("USE_GOLDEN", "0") == "1"


def _overrides(
    *,
    mini: bool = False,
    a5: bool = False,
    hif8: bool = False,
    pro: bool = False,
    swap: bool = True,
) -> list[OverrideImport]:
    golden = _is_golden()
    if golden:
        targets: list[OverrideImport] = [_ROPE_REFERENCE]
    else:
        rms = (
            "torchtitan_npu.override.deepseek_v4.common.rms_norm"
            if (a5 and not hif8) or pro
            else "torchtitan_npu.override.common.rms_norm.asc"
        )
        sparse_kwargs: dict[str, Any] = {"selector_backend": "asc"}
        if mini:
            sparse_kwargs.update(num_heads=16, index_num_heads=8)
        elif pro:
            sparse_kwargs.update(num_heads=128)
        targets = [rms]
        if mini:
            targets.append("torchtitan_npu.override.common.rope.asc_complex")
        if pro:
            targets.extend(_A5_OVERRIDES[:2])
        targets.extend(
            [
                ("torchtitan_npu.override.deepseek_v4.sparse_attn.asc", sparse_kwargs),
                "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
                "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
                "torchtitan_npu.override.common.token_dispatcher.asc",
            ]
        )
        if pro:
            targets.extend(_A5_OVERRIDES[2:])
    if swap and not (mini and golden):
        targets.append(_SWAP_OPTIMIZER)
    if not golden and not (mini or pro):
        if hif8:
            targets.extend(
                [
                    "torchtitan_npu.override.common.rope.asc_partial",
                    "torchtitan_npu.override.common.swiglu_group.asc",
                    "torchtitan_npu.override.common.swiglu_group.asc_shared_experts",
                ]
            )
        elif a5:
            targets.extend(_A5_OVERRIDES)
        else:
            targets.append("torchtitan_npu.override.common.rope.asc_complex")
    return targets


def _configure(
    config: TrainerEx.Config,
    *,
    local_ranks: int,
    nodes: int = 1,
    ep: int,
    dp_shard: int,
    tokens_per_microbatch: int,
    tokens_per_step: int,
    seq_len: int = 4096,
    steps: int = 100,
    mini: bool = False,
    a5: bool = False,
    hif8: bool = False,
    pro: bool = False,
    swap: bool = True,
    text_data: bool = True,
) -> TrainerEx.Config:
    _check_override_environment()
    parallelism = config.parallelism
    parallelism.tensor_parallel_degree = int(os.environ.get("TP", "1")) if pro else 1
    parallelism.pipeline_parallel_degree = int(os.environ.get("PP", "1")) if pro else 1
    parallelism.context_parallel_degree = int(os.environ.get("CP", "1"))
    parallelism.expert_parallel_degree = int(os.environ.get("EP", ep)) if pro else ep
    parallelism.data_parallel_shard_degree = int(os.environ.get("DP_SHARD", dp_shard)) if pro else dp_shard
    world_size = _world_size(local_ranks=local_ranks, nodes=nodes)
    shard_product = (
        parallelism.data_parallel_shard_degree
        * parallelism.context_parallel_degree
        * parallelism.tensor_parallel_degree
        * parallelism.pipeline_parallel_degree
    )
    if world_size % shard_product or world_size < shard_product:
        raise ValueError(f"WORLD_SIZE={world_size} must be divisible by DP_SHARD*CP*TP*PP={shard_product}")
    parallelism.data_parallel_replicate_degree = world_size // shard_product
    config.hf_assets_path = os.environ.get("HF_ASSETS_PATH") or "/path/to/DeepSeekV4_tokenizer"
    model = cast("DeepSeekV4Model.Config", config.model)
    model.max_context_length = seq_len
    for _, rope, _, _ in model.traverse(RoPE.Config):
        rope.max_context_length = seq_len
    config.training.max_context_length = seq_len
    config.training.num_tokens_per_microbatch_per_dp_rank = tokens_per_microbatch
    config.training.num_tokens_per_train_step = tokens_per_step
    config.training.steps = steps
    config.training.disable_cuda_graphs = True
    packing = cast("GrainDataLoader.Config", config.dataloader).dataset
    assert isinstance(packing, (ConcatThenSplitPackingConfig, FirstFitPackingConfig))
    assert isinstance(packing.dataset, SingleDatasetConfig)
    processor = cast("AlignedTextProcessor.Config", packing.dataset.processor)
    processor.per_doc_alignment = int(
        os.environ.get("DSV4_PER_DOC_ALIGNMENT", 128 if mini else 128 * parallelism.context_parallel_degree)
    )
    if text_data:
        assert isinstance(packing, ConcatThenSplitPackingConfig)
        if "DATA_FILES" in os.environ:
            source = cast("HuggingFaceRandomAccessSource.Config", packing.dataset.source)
            source.load_dataset_kwargs["data_files"] = os.environ["DATA_FILES"]
    config.debug.print_config = True
    config.profiler.enable_profiling = False
    config.profiler.save_traces_folder = "profiling_path"
    config.profiler.extension.enable_online_parse = False
    config.profiler.extension.profiler_start = 6
    config.profiler.extension.profiler_end = 7
    config.profiler.extension.profile_ranks = [0]
    config.comm.init_timeout_seconds = 7200
    config.comm.train_timeout_seconds = 600
    optimizer = config.optim.optimizer
    assert isinstance(optimizer, OptimizerConfig)
    optimizer.name = "AdamW" if mini and _is_golden() else "Muon"
    optimizer.lr = 1e-5
    optimizer.beta1, optimizer.beta2 = 0.9, 0.95
    optimizer.eps = 1e-8
    optimizer.weight_decay = 0.1
    optimizer.muon_momentum = 0.95
    optimizer.muon_enable_nesterov = True
    optimizer.muon_ns_steps = 10
    optimizer.muon_adjust_lr_fn = "match_rms_adamw"
    optimizer.materialize()
    config.override.imports = _overrides(mini=mini, a5=a5, hif8=hif8, pro=pro, swap=swap)
    return config


def _checkpoint(config: TrainerEx.Config, *, load_only: bool = True) -> None:
    if config.checkpointer is None:
        config.checkpointer = CheckpointManager.Config(load_only=load_only)
    checkpoint = config.checkpointer
    checkpoint.load_only = load_only
    checkpoint.folder = os.environ.get("CKPT_SAVE_LOAD_PATH") or "/path/to/save_ckpt"
    checkpoint.initial_load_path = os.environ.get("CKPT_INIT_LOAD_PATH") or "/path/to/init_load_ckpt"
    checkpoint.initial_load_in_hf = True


def _a5_quantization(config: TrainerEx.Config, *, full_model: bool = False) -> None:
    quantization = config.extension.quantization
    quantization.enable_quantized_training = True
    quantization.recipe = "all_block_fp8"
    quantization.enable_fsdp_prequantize = True
    quantization.fsdp_prequantize_fqns = list(_A5_PREQUANTIZE_FQNS) if full_model else None
    quantization.li_quantization = "fp8"
    quantization.kv_norm_quantization = KVNormQuantizationConfig(
        format="mxfp8",
        fqns=[".attention.kv_norm", ".attention.compressor.norm"],
        block_size=64,
    )
    config.activation_remat = DefaultActivationRematConfig()


def _flash_8p(*, a5: bool = False, hif8: bool = False) -> TrainerEx.Config:
    config = _configure(
        model_recipes.deepseek_v4_flash_43layers_16experts_compiled(),
        local_ranks=8,
        ep=8,
        dp_shard=8,
        tokens_per_microbatch=8192 if a5 and not hif8 else 4096,
        tokens_per_step=262144,
        a5=a5,
        hif8=hif8,
        swap=not a5 or hif8,
    )
    if hif8:
        config.extension.quantization.enable_quantized_training = True
        config.extension.quantization.recipe = "all_hif8"
        config.extension.quantization.li_quantization = "hif8"
        config.activation_remat = DefaultActivationRematConfig()
    elif a5:
        _a5_quantization(config)
    return config


def _flash_full(*, a5: bool = False) -> TrainerEx.Config:
    config = _configure(
        model_recipes.deepseek_v4_flash_cpt(),
        local_ranks=8 if a5 else 16,
        nodes=16 if a5 else 8,
        ep=128,
        dp_shard=128,
        tokens_per_microbatch=8192 if a5 else 4096,
        tokens_per_step=4194304,
        a5=a5,
        swap=not a5,
    )
    _checkpoint(config)
    if a5:
        _a5_quantization(config, full_model=True)
    return config


def deepseek_v4_eager_1p() -> TrainerEx.Config:
    _check_override_environment()
    config = model_recipes.deepseek_v4_debugmodel()
    config.override.imports = _overrides(mini=True, swap=False)
    return config


def deepseek_v4_mini_1p_cpt_2k_a3() -> TrainerEx.Config:
    return _configure(
        model_recipes.deepseek_v4_debugmodel(),
        local_ranks=1,
        ep=1,
        dp_shard=1,
        tokens_per_microbatch=2048,
        tokens_per_step=-1,
        seq_len=2048,
        mini=True,
    )


def deepseek_v4_flash_8p_cpt_4k_a3() -> TrainerEx.Config:
    return _flash_8p()


def deepseek_v4_flash_8p_cpt_4k_a3_eager() -> TrainerEx.Config:
    config = _flash_8p()
    config.compile = None
    config.activation_remat = FullActivationRematConfig()
    return config


def deepseek_v4_flash_8p_cpt_4k_a5() -> TrainerEx.Config:
    return _flash_8p(a5=True)


def deepseek_v4_flash_8p_cpt_4k_a5_hif8() -> TrainerEx.Config:
    return _flash_8p(a5=True, hif8=True)


def deepseek_v4_flash_cpt_4k_a3() -> TrainerEx.Config:
    return _flash_full()


def deepseek_v4_flash_cpt_4k_a5() -> TrainerEx.Config:
    return _flash_full(a5=True)


def deepseek_v4_flash_qat_4k_a5() -> TrainerEx.Config:
    config = _flash_full(a5=True)
    config.extension.quantization.enable_mxfp4_qat = True
    config.extension.quantization.li_quantization = "mxfp4"
    return config


def deepseek_v4_flash_cpt_1024k_a5() -> TrainerEx.Config:
    """Retain the explicit CP migration guard from the selected model recipe."""
    config = model_recipes.deepseek_v4_flash_cp_cpt()
    config = _configure(
        config,
        local_ranks=8,
        nodes=16,
        ep=128,
        dp_shard=1,
        tokens_per_microbatch=1048576,
        tokens_per_step=8388608,
        seq_len=1048576,
        a5=True,
        swap=False,
    )
    config.parallelism.context_parallel_degree = 128
    config.parallelism.data_parallel_replicate_degree = 1
    packing = cast("GrainDataLoader.Config", config.dataloader).dataset
    assert isinstance(packing, ConcatThenSplitPackingConfig)
    assert isinstance(packing.dataset, SingleDatasetConfig)
    cast("AlignedTextProcessor.Config", packing.dataset.processor).per_doc_alignment = int(
        os.environ.get("DSV4_PER_DOC_ALIGNMENT", "16384")
    )
    _checkpoint(config)
    _a5_quantization(config, full_model=True)
    return config


def deepseek_v4_pro_32p_cpt_4k_a5() -> TrainerEx.Config:
    config = _configure(
        model_recipes.deepseek_v4_pro_61layers_32experts_cpt(),
        local_ranks=8,
        nodes=4,
        ep=32,
        dp_shard=32,
        tokens_per_microbatch=4096,
        tokens_per_step=1048576,
        steps=int(os.environ.get("STEPS") or "100"),
        pro=True,
    )
    _checkpoint(config)
    config.extension.quantization.enable_quantized_training = True
    config.extension.quantization.recipe = "all_block_fp8"
    return config


def deepseek_v4_pro_8p_cpt_4k_a5_cpuoffload() -> TrainerEx.Config:
    config = _configure(
        model_recipes.deepseek_v4_pro_61layers_12experts_cpt(),
        local_ranks=8,
        ep=4,
        dp_shard=8,
        tokens_per_microbatch=4096,
        tokens_per_step=262144,
        steps=int(os.environ.get("STEPS") or "100"),
        pro=True,
    )
    config.training.enable_cpu_offload = True
    cast("OptimizerConfig", config.optim.optimizer)._cpu_offload = True
    _checkpoint(config)
    return config


def _sft(*, reduced: bool) -> TrainerEx.Config:
    config = model_recipes.deepseek_v4_flash_sft(
        dataset_path=os.environ.get("DATASET_PATH", os.environ.get("DSV4_SFT_DATASET", "/path/to/train.jsonl")),
        flavor="deepseek_v4_flash_43layers_16experts" if reduced else "deepseek_v4_flash",
    )
    config = _configure(
        config,
        local_ranks=8 if reduced else 16,
        nodes=1 if reduced else 8,
        ep=8 if reduced else 128,
        dp_shard=8 if reduced else 128,
        tokens_per_microbatch=4096,
        tokens_per_step=262144 if reduced else 4194304,
        text_data=False,
    )
    config.compile = CompileConfig(components=[], model=ModelCompileConfig())
    if reduced:
        config.checkpointer = CheckpointManager.Config(load_only=False)
    else:
        _checkpoint(config, load_only=False)
    return config


def deepseek_v4_flash_8p_sft_4k_a3() -> TrainerEx.Config:
    return _sft(reduced=True)


def deepseek_v4_flash_sft_4k_a3() -> TrainerEx.Config:
    return _sft(reduced=False)


def deepseek_v4_flash_8node_sft_4k_a3() -> TrainerEx.Config:
    return _sft(reduced=False)


def deepseek_v4_flash_lora_4k_a3() -> TrainerEx.Config:
    config = _configure(
        model_recipes.deepseek_v4_flash_lora_compiled(),
        local_ranks=16,
        nodes=8,
        ep=128,
        dp_shard=128,
        tokens_per_microbatch=4096,
        tokens_per_step=4194304,
    )
    _checkpoint(config, load_only=False)
    cast("DeepSeekV4PEFTCheckpointManager.Config", config.checkpointer).save_training_state = True
    return config


def _graph_example(*, a5: bool, full_model: bool = False) -> TrainerEx.Config:
    """Keep the graph recipe's policy while retaining the existing NPU guard."""
    from torchtitan.config import derive
    from torchtitan.experiments.graph_trainer.configs import EpOverlapConfig, GraphTrainerCompileConfig

    try:
        from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import GraphTrainerEx
    except TypeError as error:
        if "multiple bases have instance lay-out conflict" not in str(error):
            raise
        raise NotImplementedError(
            "NPU GraphTrainer remains deferred: its existing slotted configuration "
            "bases have incompatible instance layouts in TorchTitan 0.4.0."
        ) from error
    from torchtitan_npu.models.deepseek_v4.graph_trainer import to_graph_trainer_config
    from torchtitan_npu.models.deepseek_v4.model import GraphTrainerDeepSeekV4Model
    from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
        functionalize_recompute_mutations as _functionalize_mutations,
    )

    base = _flash_full(a5=a5) if full_model else _flash_8p(a5=a5)
    base.compile = None
    graph = to_graph_trainer_config(base, GraphTrainerDeepSeekV4Model.Config)
    graph.compile = GraphTrainerCompileConfig(
        memory_policy="full",
        pass_pipeline=f"{_functionalize_mutations.PASS_PIPELINE_NAME}+npu_auto_overlap",
        disable_passes=["cudagraph_pass"],
        ep_overlap=EpOverlapConfig(enabled=True, chunk_dim="seq", module_fqn="layers.*.feed_forward"),
    )
    if a5:
        graph.training.steps = int(os.environ.get("STEPS") or "20")
        if full_model:
            graph.training.num_tokens_per_train_step = 2097152
        else:
            # The NPU memory-policy registry owns this deferred custom policy.
            graph.compile.memory_policy = "dsv4-mhc-moe-save"  # pyrefly: ignore [bad-assignment]
    # The existing NPU graph schema owns extension/quantization and its build
    # implementation. Its current compilation guard must not be bypassed.
    return derive(graph, GraphTrainerEx.Config, extension=base.extension)


def deepseek_v4_flash_8p_cpt_4k_a3_graphtrainer() -> TrainerEx.Config:
    return _graph_example(a5=False)


def deepseek_v4_flash_8p_cpt_4k_a5_graphtrainer() -> TrainerEx.Config:
    return _graph_example(a5=True)


def deepseek_v4_flash_cpt_4k_a5_graphtrainer() -> TrainerEx.Config:
    return _graph_example(a5=True, full_model=True)


def deepseek_v4_flash_8p_cpt_4k_a5_cv_parallel() -> TrainerEx.Config:
    """The old CV launcher requires a GraphTrainer backend absent in v0.4.0."""
    raise NotImplementedError(
        "The CV-parallel example requires compile.backend='aot_eager', which "
        "TorchTitan 0.4.0 GraphTrainerCompileConfig no longer exposes. "
        "Its A5 block-FP8, selective-remat and batch-overlap recipe remains deferred."
    )
