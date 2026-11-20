# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from torch.utils.checkpoint import checkpoint
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceStreamingSource,
)
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS, VisionProcessor
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer
from torchtitan.config import OverrideConfig, apply_overrides, derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import ContextParallelTransform
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.attention import VarlenInnerAttention
from torchtitan.models.common.attention.gdn import InnerGDN
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.models.common.moe import MoE
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models import qwen3_5 as _upstream_config_registry

from torchtitan_npu.config import OptimConfig, OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.torchtitan.components.checkpoint import CheckpointManager
from torchtitan_npu.models.qwen3_5 import NPUQwen35Model, Qwen35TextModel, Qwen35TextStateDictAdapter, model_registry
from torchtitan_npu.models.qwen3_5.chat_processor import QwenSFTTokenizer
from torchtitan_npu.models.qwen3_5.dataloader import SFTGrainDataLoader, raw_sft_dataset
from torchtitan_npu.override.qwen3_5.gated_delta import AscContextParallelInnerGDN
from torchtitan_npu.override.qwen3_5.varlen_attention import AscCPVarlenInnerAttention

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn
    from torchtitan.config.override import OverrideImport
    from torchtitan.models.base.module import Module

    _SavedTensorHooks = tuple[Callable[[Tensor], object], Callable[[object], Tensor]]


class QwenSFTCheckpointManager(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        target_steps: int | None = None

    def __init__(self, config: Config, *, sd_adapter, extra_states, **kwargs):
        if not isinstance(sd_adapter, Qwen35TextStateDictAdapter):
            raise TypeError("Qwen SFT checkpointing requires the text model's state dict adapter")
        self.text_adapter = sd_adapter
        self.target_steps = config.target_steps
        super().__init__(
            config, sd_adapter=sd_adapter, extra_states={**extra_states, "qwen_hf_source": sd_adapter}, **kwargs
        )

    def load(self, step: int = -1) -> bool:
        loaded = super().load(step)
        if self.target_steps is not None:
            completed = self.states["train_state"].num_completed_steps
            if completed >= self.target_steps:
                raise ValueError(f"Target step {self.target_steps} must exceed restored step {completed}.")
        return loaded

    def _load_checkpoint(self, states, checkpoint_id, *, from_hf: bool, from_quantized: bool) -> None:
        if from_hf:
            self.text_adapter.prepare_hf_load(checkpoint_id)
        super()._load_checkpoint(states, checkpoint_id, from_hf=from_hf, from_quantized=from_quantized)

    def _save_last_step(self, curr_step: int) -> None:
        if self.last_save_in_hf:
            with self.text_adapter.export_hf():
                super()._save_last_step(curr_step)
        else:
            super()._save_last_step(curr_step)


class QwenFullActivationRematConfig(FullActivationRematConfig):
    """Recompute Qwen blocks without torch_remat's NPU storage placeholders."""

    def apply(
        self,
        model: Module,
        *,
        saved_tensors_hooks: Callable[[nn.Module, str], _SavedTensorHooks | None] | None = None,
    ) -> None:
        if saved_tensors_hooks is not None:
            raise ValueError("Qwen full activation recomputation does not support custom saved tensor hooks.")
        # Wrap forward before FSDP, keeping the model's checkpoint keys intact.
        for block in model.get_submodule("layers").children():
            feed_forward = getattr(block, "feed_forward", None)
            regions: list[nn.Module] = [block]
            if isinstance(feed_forward, MoE):
                # Checkpoint stateless projections so routing counters update once.
                regions = [
                    block.get_submodule("attention"),
                    feed_forward.routed_experts.get_submodule("w13"),
                    feed_forward.routed_experts.get_submodule("w2"),
                ]
                if feed_forward.shared_experts is not None:
                    regions.append(feed_forward.shared_experts)
            for region in regions:
                region.forward = partial(
                    checkpoint, region.forward, use_reentrant=False, determinism_check=self.determinism_check
                )


def _multimodal_dataloader() -> GrainDataLoader.Config:
    """Build the multimodal loader for DATASET, honouring DATASET_PATH.

    The dataset, collator and shuffle policy are the upstream multimodal ones;
    this constructs the loader directly instead of patching a text loader.
    """
    dataset_name = os.environ.get("DATASET", "cc12m-test")
    if dataset_name not in MM_DATASETS:
        raise ValueError(f"Unknown DATASET {dataset_name!r}; expected one of {sorted(MM_DATASETS)}")
    dataset = copy.deepcopy(MM_DATASETS[dataset_name])
    dataset_path = os.environ.get("QWEN_DATASET_PATH") or os.environ.get("DATASET_PATH")
    if dataset_path is not None and not dataset_path.strip():
        raise ValueError("QWEN_DATASET_PATH / DATASET_PATH must be non-empty when set")
    if dataset_path:
        source = dataset.source
        assert isinstance(source, HuggingFaceStreamingSource.Config)
        source.path = dataset_path
    processor = dataset.processor
    assert isinstance(processor, VisionProcessor.Config)
    return GrainDataLoader.Config(
        dataset=dataset,
        collator=MultiModalCollator.Config(
            build_mrope_positions=True,
            patch_size=processor.patch_size,
            temporal_patch_size=processor.temporal_patch_size,
            spatial_merge_size=processor.spatial_merge_size,
        ),
        streaming_shuffle_buffer_size=128,
    )


def qwen35_debugmodel(seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN) -> Trainer.Config:
    config = _upstream_config_registry.qwen35_debugmodel(seq_len=seq_len)
    config.model = derive(config.model, NPUQwen35Model.Config)
    config.dataloader = _multimodal_dataloader()
    return config


def qwen35_debugmodel_varlen_attn(seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN) -> Trainer.Config:
    config = qwen35_debugmodel(seq_len=seq_len)
    config.model = model_registry("debugmodel", parallelism=config.parallelism, seq_len=seq_len, attn_backend="varlen")
    config.training.disable_cuda_graphs = True
    return config


def qwen35_debugmodel_moe(
    seq_len: int | None = DEFAULT_DEBUG_MODEL_SEQ_LEN,
) -> Trainer.Config:
    config = _upstream_config_registry.qwen35_debugmodel_moe(seq_len=seq_len)
    config.model = derive(config.model, NPUQwen35Model.Config)
    config.dataloader = _multimodal_dataloader()
    return config


def qwen35_0_8b(seq_len: int | None = None) -> Trainer.Config:
    raise NotImplementedError("Qwen3.5 0.8B recipe requires migration to TorchTitan 0.4")


def qwen35_2b(seq_len: int | None = None) -> Trainer.Config:
    raise NotImplementedError("Qwen3.5 2B recipe requires migration to TorchTitan 0.4")


def qwen35_4b(seq_len: int | None = None) -> Trainer.Config:
    raise NotImplementedError("Qwen3.5 4B recipe requires migration to TorchTitan 0.4")


def qwen35_9b(seq_len: int | None = None) -> Trainer.Config:
    raise NotImplementedError("Qwen3.5 9B recipe requires migration to TorchTitan 0.4")


def qwen35_35b_a3b(seq_len: int | None = None) -> Trainer.Config:
    raise NotImplementedError("Qwen3.5 35B-A3B recipe requires migration to TorchTitan 0.4")


def qwen35_27b(seq_len: int | None = None) -> Trainer.Config:
    raise NotImplementedError("Qwen3.5 27B recipe requires migration to TorchTitan 0.4")


def qwen35_122b_a10b(seq_len: int | None = None) -> Trainer.Config:
    raise NotImplementedError("Qwen3.5 122B-A10B recipe requires migration to TorchTitan 0.4")


def qwen35_397b_a17b(seq_len: int | None = None) -> Trainer.Config:
    raise NotImplementedError("Qwen3.5 397B-A17B recipe requires migration to TorchTitan 0.4")


def qwen35_debugmodel_moe_float8_lora(seq_len: int | None = 512) -> Trainer.Config:
    """LoRA on NPU block-FP8 weights with block-32 activation quantization."""
    from torchtitan.config.transform import (
        LoRATransform,
        apply_transforms,
    )
    from torchtitan.models.common.linear import GroupedLinear, Linear

    from torchtitan_npu.extensions.adaptor.torchao.param_swaps import block_fp8_param_swap
    from torchtitan_npu.extensions.adaptor.torchao.recipes import NpuQuantizeConverter

    config = _upstream_config_registry.qwen35_debugmodel_moe(seq_len=seq_len)
    config.model = model_registry(
        "debugmodel_moe",
        parallelism=config.parallelism,
        seq_len=seq_len,
        converters=[
            NpuQuantizeConverter.Config(
                base_config=block_fp8_param_swap(),
                unified_npu_quant_modules=(Linear.Config, GroupedLinear.Config),
            )
        ],
    )
    return apply_transforms(
        config,
        [
            LoRATransform(
                rank=8,
                alpha=16.0,
                target_modules=["w13", "w2"],
            )
        ],
    )


def _text_sft_config(
    *,
    flavor: str,
    length: int,
    cp: int,
    dp_shard: int,
    steps: int,
    load_checkpoint: bool,
    long_text: bool = True,
    dataloader: GrainDataLoader.Config | None = None,
) -> Trainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=dp_shard,
        tensor_parallel_degree=1,
        context_parallel_degree=cp,
        context_parallel_load_balancer=None,
    )
    model = derive(
        model_registry(flavor, parallelism=parallelism, seq_len=length, attn_backend="varlen"),
        Qwen35TextModel.Config,
        text_only=True,
        vision_encoder=None,
        local_compile_regions=[],
    )
    overrides: list[OverrideImport] = []
    if cp > 1:
        ContextParallelTransform(
            inner_attention_map={VarlenInnerAttention: AscCPVarlenInnerAttention, InnerGDN: AscContextParallelInnerGDN}
        ).transform(model)
        native_gdn = "torchtitan_npu.override.qwen3_5.gated_delta.context_parallel"
    else:
        native_gdn = "torchtitan_npu.override.qwen3_5.gated_delta.npu"
        overrides.append("torchtitan_npu.override.qwen3_5.varlen_attention.asc")
    overrides.append("torchtitan_npu.override.qwen3_5.gated_delta.qwen_sft")
    # Chunked CE accumulates several LM-head gradients within one optimizer step.
    # Keep each contribution's previous BF16 rounding before accumulation.
    model.lm_head.inplace_wgrad_accum = False

    assets = os.environ.get("HF_ASSETS_PATH", "./assets/hf/Qwen3.5-27B")
    checkpoint = QwenSFTCheckpointManager.Config(
        interval=1000 if long_text else 256,
        last_save_model_only=long_text,
        last_save_in_hf=long_text,
        export_dtype="bfloat16" if long_text else "float32",
    )
    if load_checkpoint:
        checkpoint.initial_load_path = str(Path(os.environ.get("CHECKPOINT_INITIAL_LOAD_PATH", assets)).resolve())
        checkpoint.initial_load_model_only = True
        checkpoint.initial_load_in_hf = True
    if dataloader is None:
        files = os.environ.get("DATA_FILES", "").strip()
        if files and Path(files).is_file():
            dataset, input_contract = raw_sft_dataset(dataset_path=files)
        else:
            patterns = tuple(path.strip() for path in files.split(",") if path.strip())
            dataset, input_contract = raw_sft_dataset(patterns=patterns)
        dataloader = SFTGrainDataLoader.Config(
            dataset=FirstFitPackingConfig(dataset=dataset, num_packing_bins=1),
            input_contract=input_contract,
            seed=42,
        )
    if length >= 32768:
        dataloader.max_num_documents = 512
    config = Trainer.Config(
        model=model,
        hf_assets_path=assets,
        tokenizer=QwenSFTTokenizer.Config(),
        dataloader=dataloader,
        training=TrainingConfig(
            max_context_length=length,
            num_tokens_per_microbatch_per_dp_rank=length,
            steps=steps,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        activation_remat=QwenFullActivationRematConfig(),
        loss=ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.Config(global_vocab_size=model.vocab_size)),
        optim=OptimConfig(
            max_norm=1.0,
            optimizer=OptimizerConfig(optimizers=[AdamW.Config(pattern=r".*", lr=1e-5, weight_decay=0.1)]),
            lr_scheduler=LRSchedulersContainer.Config(warmup_steps=16, decay_ratio=1.0, min_lr_factor=0.1),
        ),
        checkpointer=checkpoint,
        metrics=MetricsProcessor.Config(log_freq=1),
        override=OverrideConfig(imports=overrides),
    )
    # Apply the native default before user overrides can select another GDN.
    apply_overrides(OverrideConfig(imports=[native_gdn]), config)
    return config


def qwen35_27b_4k_sft(*, seq_len: int | None = None) -> Trainer.Config:
    length = seq_len if seq_len is not None else int(os.environ.get("SEQ_LEN", "4096"))
    if not 64 <= length <= 4096 or length % 64:
        raise ValueError("seq_len must be a multiple of 64 in [64, 4096].")
    return _text_sft_config(
        flavor="27B",
        length=length,
        cp=1,
        dp_shard=-1,
        steps=int(os.environ.get("QWEN_STEPS", "256")),
        load_checkpoint=True,
        long_text=False,
    )


def qwen35_27b_long_text_sft() -> Trainer.Config:
    return _text_sft_config(
        flavor="27B",
        length=32768,
        cp=4,
        dp_shard=-1,
        steps=int(os.environ.get("QWEN_STEPS", "256")),
        load_checkpoint=True,
    )


def qwen35_27b_sft_64k_cp8() -> Trainer.Config:
    return _text_sft_config(
        flavor="27B",
        length=65536,
        cp=8,
        dp_shard=2,
        steps=int(os.environ.get("QWEN_STEPS", "256")),
        load_checkpoint=True,
    )


def run_qwen3_5_4k(*, seq_len: int | None = None) -> Trainer.Config:
    if not os.environ.get("HF_ASSETS_PATH"):
        raise ValueError("Set HF_ASSETS_PATH to the Qwen3.5-series Hugging Face model directory.")
    config = qwen35_27b_4k_sft(seq_len=seq_len)
    config.dump_folder = os.environ.get("DUMP_FOLDER", "./outputs/qwen3_5_4k_sft")
    return config


run_qwen_sft = run_qwen3_5_4k


def run_qwen3_5_32k_cp8() -> Trainer.Config:
    return _cp8_sft(32768, 16, "./outputs/qwen3_5_32k_sft")


def run_qwen3_5_32k_cp8_fla() -> Trainer.Config:
    from torchtitan_npu.override.qwen3_5.fla import use_master_fla

    config = _cp8_sft(32768, 16, "./outputs/qwen3_5_32k_fla_sft")
    use_master_fla(config)
    training = config.training
    assert isinstance(training, TrainingConfig)
    training.extension.allow_hf32 = False
    training.num_tokens_per_train_step = 2 * training.num_tokens_per_microbatch_per_dp_rank
    config.debug.seed = 42
    config.debug.deterministic = True
    return config


def run_qwen3_5_64k_cp8() -> Trainer.Config:
    return _cp8_sft(65536, 32, "./outputs/qwen3_5_64k_sft")


def run_qwen3_5_128k_cp8() -> Trainer.Config:
    return _cp8_sft(131072, 64, "./outputs/qwen3_5_128k_sft")


def _cp8_sft(length: int, loss_chunks: int, dump_folder: str) -> Trainer.Config:
    if not os.environ.get("HF_ASSETS_PATH"):
        raise ValueError("Set HF_ASSETS_PATH to the Qwen3.5-series Hugging Face model directory.")
    config = _text_sft_config(
        flavor="27B",
        length=length,
        cp=8,
        dp_shard=2,
        steps=int(os.environ.get("QWEN_STEPS", "256")),
        load_checkpoint=True,
        long_text=False,
    )
    config.parallelism.fsdp_reshard_after_forward = "always"
    assert isinstance(config.loss, ChunkedLossWrapper.Config)
    config.loss.num_chunks = loss_chunks
    config.dump_folder = os.environ.get("DUMP_FOLDER", dump_folder)
    return config


def qwen3_5_27b_sft_64k_cp8() -> Trainer.Config:
    config = qwen35_27b_sft_64k_cp8()
    config.dump_folder = os.environ.get("DUMP_FOLDER", "./outputs/qwen_v040_27b_sft_64k_cp8")
    return config
