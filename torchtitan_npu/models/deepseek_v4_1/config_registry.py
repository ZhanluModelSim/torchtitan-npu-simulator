# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from dataclasses import dataclass
from typing import cast

import torch
from torch.distributed.tensor import Shard
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import ParamGroupConfig, default_adamw
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import derive
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
)
from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan_npu.config import MuonOptimizerProfile, OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.components.gradient_clipping import GradientClippingTrainer
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.models.common.muon import make_owned_layout

from . import model_registry
from .indexer import Selector
from .model import DeepSeekV41Model, DeepSeekV41MultimodalModel, compression_alignment


def _per_doc_alignment(model_spec: ModelSpec) -> int:
    """The pooling granularity a packed document must be padded to.

    Derived from the model's compression ratios so the loader and the compressor
    cannot disagree: a document whose length is not a multiple of every ratio would
    get a group straddling its edge.
    """
    return compression_alignment(model_spec.model.compress_ratios)  # pyrefly: ignore [missing-attribute]


@dataclass(kw_only=True)
class EngramTableParamGroupConfig(ParamGroupConfig):
    """The recipe-owned table group, distinct from user-defined groups."""


class DeepSeekV41Trainer(TrainerEx, GradientClippingTrainer):
    """V4.1 training with instance-owned Host sparse gradient clipping."""

    @dataclass(kw_only=True, slots=True)
    class Config(TrainerEx.Config):
        """Expose the model switch while ModelSpec is suppressed from the CLI."""

        engram_enabled: bool = True

        def __post_init__(self) -> None:
            assert self.model_spec is not None and isinstance(self.model_spec.model, DeepSeekV41Model.Config)
            if not self.engram_enabled:
                model_spec = copy.deepcopy(self.model_spec)
                self.model_spec = model_spec
                model = model_spec.model
                assert isinstance(model, DeepSeekV41Model.Config)
                # Remove the HostSparse capability together with Engram.
                self.optimizer = derive(copy.deepcopy(self.optimizer), OptimizerConfig)
                for layer in model.layers:
                    layer.engram = None
                self.optimizer.param_groups = [
                    group for group in self.optimizer.param_groups if not isinstance(group, EngramTableParamGroupConfig)
                ]
            TrainerEx.Config.__post_init__(self)

    def __init__(self, config):
        super().__init__(config)
        # The step denominator is for the selectors that consume the fused teacher.  A port
        # subclasses :class:`Selector`, so the reference selector -- the only module whose
        # exact class is ``Selector`` -- is the one that does not.
        self._indexer_selectors = [
            module
            for part in self.model_parts
            for module in part.modules()
            if isinstance(module, Selector) and type(module) is not Selector
        ]

    def step_batches(self, batches):
        batches = super().step_batches(batches)
        if not self._indexer_selectors:
            return batches
        self._num_step_queries = 0
        self._indexer_denominator_ready = False

        def counted_batches():
            for inputs, labels in batches:
                # Count before CP sharding; LI also trains unsupervised input positions.
                self._num_step_queries += inputs["input"].numel()
                yield inputs, labels

        return counted_batches()

    def forward_backward_step(self, *args, **kwargs):
        if self._indexer_selectors and not self._indexer_denominator_ready:
            # The parent prefetches every accumulation microbatch before the first forward.
            denominator = torch.tensor(self._num_step_queries, device=self.device, dtype=torch.float32)
            if self.parallel_dims.dp_enabled:
                denominator = dist_utils.dist_sum_tensor(denominator, self.parallel_dims.get_mesh("batch"))
            for selector in self._indexer_selectors:
                selector.num_global_queries = denominator
            self._indexer_denominator_ready = True
        return super().forward_backward_step(*args, **kwargs)

    def clip_grad_norm(self, parameters, max_norm, **kwargs):
        if isinstance(self.optimizers, HostSparseOptimizersContainer):
            return self.optimizers.clip_grad_norm(parameters, max_norm, **kwargs)
        return super().clip_grad_norm(parameters, max_norm, **kwargs)


def _v41_muon_profile(model_spec: ModelSpec) -> MuonOptimizerProfile:
    """Build the V4.1-owned parameter and FlexShard policy for Muon.

    Same sharding policy as the DSV4 profile where the parameter spaces
    overlap (per-head ``wq_b``/``wo_a`` BlockShard, expert-sharded routed
    GEMMs, Owned for the rest).  V4.1 differences: no MTP depths, no
    ``hc_head``, no compressor APE.  The indexer parameters do receive LI
    gradients on the new baseline, but they stay on the AdamW fallback group
    together with the vision tower and every 1-D parameter — this profile
    does not change the parameter policy the verified stacks ran with.
    """
    model_config = cast("DeepSeekV41Model.Config", model_spec.model)
    # FSDP folds the (dp_shard, cp) storage mesh into the single
    # ``dp_shard_cp`` axis when context parallelism is enabled, and the
    # partial_dtensor backend stores dense parameters on the plain ``fsdp``
    # axis (see parallelize.py).  Keep every axis name so the layouts match
    # whichever storage mesh the run actually uses.
    dense_dp_axes = (
        MeshAxisName.DP_SHARD.value,
        f"{MeshAxisName.DP_SHARD.value}_{MeshAxisName.CP.value}",
        MeshAxisName.FSDP.value,
    )
    owned = make_owned_layout(dense_dp_axes)
    owned_attention_projections = {"wq_a": owned, "wkv": owned, "wo_b": owned}
    attention_projections = (*owned_attention_projections, "wq_b", "wo_a")
    expert_projections = ("w1", "w2", "w3")
    routed_expert_projections = ("w1_EFD", "w2_EDF", "w3_EFD")
    compressor_projections = ("wkv", "wgate")
    hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")
    expert_sharding = ComputeLayout(
        shardings_by_mesh_axis={
            **{axis: Shard(0) for axis in dense_dp_axes},
            MeshAxisName.EFSDP.value: Shard(0),
            MeshAxisName.EP.value: Shard(0),
        },
        shard_order_by_tensor_dim={  # pyrefly: ignore [unexpected-keyword]
            0: (MeshAxisName.EP.value, MeshAxisName.EFSDP.value),
        },
    )

    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    bucket_configs: list[BucketConfig] = []
    for layer_id, layer_config in enumerate(model_config.layers):
        attention = layer_config.attention
        attention_shardings = {
            f"layers.{layer_id}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in owned_attention_projections.items()
        }
        # ``wq_b`` stores one [head_dim, q_lora_rank] matrix per head and
        # ``wo_a`` one [o_lora_rank, per_group_in] matrix per group, both
        # flattened on dim 0; BlockShard computes Muon per matrix.
        attention_shardings[f"layers.{layer_id}.attention.wq_b.weight"] = ComputeLayout(
            shardings_by_mesh_axis={axis: BlockShard(dim=0, block_size=attention.head_dim) for axis in dense_dp_axes},
        )
        attention_shardings[f"layers.{layer_id}.attention.wo_a.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_size=attention.wo_a.out_features) for axis in dense_dp_axes
            },
        )
        # Every layer carries a compressor; only the sources own weights.
        for projection in compressor_projections:
            if getattr(attention.compressor, projection, None) is not None:
                attention_shardings[f"layers.{layer_id}.attention.compressor.{projection}.weight"] = owned
        attention_shardings[f"layers.{layer_id}.hc_attn_pre.hc_fn"] = owned
        dense_shardings = {
            f"layers.{layer_id}.moe.shared_experts.{projection}.weight": owned for projection in expert_projections
        }
        dense_shardings[f"layers.{layer_id}.moe.router.gate.weight"] = owned
        dense_shardings[f"layers.{layer_id}.hc_ffn_pre.hc_fn"] = owned
        if layer_config.engram is not None:
            # q_weight/k_weight are normalization scales, not projections.
            dense_shardings[f"layers.{layer_id}.engram.gate.wkv"] = owned
        routed_shardings = {
            f"layers.{layer_id}.moe.routed_experts.inner_experts.{projection}": expert_sharding
            for projection in routed_expert_projections
        }
        if getattr(model_config, "lora", None) is not None:
            from torchtitan_npu.models.deepseek_v4.lora import DEEPSEEK_V4_LORA_TARGETS

            for shardings in (attention_shardings, dense_shardings):
                for base_key, layout in tuple(shardings.items()):
                    owner = base_key.removesuffix(".weight")
                    if owner.removeprefix(f"layers.{layer_id}.") in DEEPSEEK_V4_LORA_TARGETS:
                        shardings[f"{owner}.lora_a.weight"] = owned
                        shardings[f"{owner}.lora_b.weight"] = layout
            for projection in ("w13_lora_a", "w1_lora_b", "w3_lora_b", "w2_lora_a", "w2_lora_b"):
                routed_shardings[f"layers.{layer_id}.moe.routed_experts.inner_experts.{projection}"] = expert_sharding
        # Split each layer into three buckets: the routed-expert GEMMs are
        # by far the largest and get a bucket of their own so the
        # Newton-Schulz all-gather peak stays bounded on the 40-layer
        # FSDP8 shape (a single per-layer bucket OOMs at ~89% HBM).
        compute_sharding_by_fqn.update(attention_shardings)
        compute_sharding_by_fqn.update(dense_shardings)
        compute_sharding_by_fqn.update(routed_shardings)
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.attn", patterns=tuple(attention_shardings)))
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.dense", patterns=tuple(dense_shardings)))
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.routed", patterns=tuple(routed_shardings)))

    lora_pattern = ""
    if getattr(model_config, "lora", None) is not None:
        from torchtitan_npu.models.deepseek_v4.config_registry import _lora_muon_pattern

        lora_pattern = _lora_muon_pattern(compute_sharding_by_fqn)
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_projections)})\.weight|"
        rf"attention\.compressor\.(?:{'|'.join(compressor_projections)})\.weight|"
        rf"moe\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"moe\.routed_experts\.inner_experts\.(?:{'|'.join(routed_expert_projections)})|"
        r"moe\.router\.gate\.weight|"
        r"engram\.gate\.wkv|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn" + (f"|{lora_pattern}" if lora_pattern else "") + r")$"
    )
    return MuonOptimizerProfile(
        muon_pattern=muon_pattern,
        optimizer_factory_kwargs={
            "DistMuon": {
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
                "bucket_configs": tuple(bucket_configs),
            }
        },
    )


def _v41_optimizer_config(model_spec: ModelSpec) -> OptimizerConfig:
    adamw = default_adamw(lr=1e-5, eps=1e-6)
    if getattr(model_spec.model, "lora", None) is not None:
        return derive(adamw, OptimizerConfig, name="AdamW", _muon_profile=_v41_muon_profile(model_spec))
    has_engram = any(
        getattr(layer, "engram", None) is not None for layer in cast("DeepSeekV41Model.Config", model_spec.model).layers
    )
    optimizer_type = HostSparseOptimizersContainer.Config if has_engram else OptimizerConfig
    name = "AdamW" if has_engram else "Muon"
    groups = adamw.param_groups
    if has_engram:
        groups = [
            EngramTableParamGroupConfig(
                pattern=r".*\.engram\.table\.weight$",
                optimizer_name="SparseAdam",
                optimizer_kwargs={"lr": 5e-5, "betas": (0.9, 0.95), "eps": 1e-6},
            ),
            *groups,
        ]
    return optimizer_type(
        name=name,
        param_groups=groups,
        implementation=adamw.implementation,
        optimizer_factory_kwargs_by_name=adamw.optimizer_factory_kwargs_by_name,
        _muon_profile=_v41_muon_profile(model_spec),
    )


def _vision_dataloader_config(model_spec: ModelSpec):
    """The image-conditioned caption loader, for a multimodal model only.

    Imported here rather than at module scope: the multimodal dataset stack pulls in the
    image-preprocessing dependencies, and a text-only run must not need them.
    """
    from .vision.dataloader import DeepSeekV41DataLoader

    return DeepSeekV41DataLoader.Config(per_doc_alignment=_per_doc_alignment(model_spec))


def _text_dataloader_config(model_spec: ModelSpec):
    """The packed text loader with the same per-document pooling alignment."""
    from torchtitan_npu.patches.torchtitan.hf_datasets.text_datasets import AlignedHuggingfaceDataloader

    return AlignedHuggingfaceDataloader.Config(per_doc_alignment=_per_doc_alignment(model_spec))


def _make_trainer_config(flavor: str, *, converters=None) -> TrainerEx.Config:
    model_spec = model_registry(flavor, converters=converters)
    if model_spec.model.n_layers != len(model_spec.model.layers):  # pyrefly: ignore [missing-attribute]
        raise ValueError("registered V4.1 model does not describe every configured layer")
    vision = isinstance(model_spec.model, DeepSeekV41MultimodalModel.Config)

    return DeepSeekV41Trainer.Config(
        # One packed row per rank, which is what the model asserts and the loaders
        # produce: documents are packed into a single row and the row is cut at
        # ``seq_len``.  Upstream's default is 8, so the recipe states it rather than
        # inheriting it -- a launcher that wants a different value still overrides the
        # flag, and the model rejects it.
        training=TrainingConfig(local_batch_size=1),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        model_spec=model_spec,
        tokenizer=HuggingFaceTokenizer.Config(),
        dataloader=(_vision_dataloader_config(model_spec) if vision else _text_dataloader_config(model_spec)),
        optimizer=_v41_optimizer_config(model_spec),
        activation_checkpoint=FullAC.Config(),
    )


def deepseek_v4_1_flash() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash")


def deepseek_v4_1_flash_text() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_text")


def deepseek_v4_1_flash_40layers_16experts_multimodal() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_40layers_16experts_vision")


def deepseek_v4_1_flash_40layers_16experts_text() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_flash_40layers_16experts_text")


def deepseek_v4_1_debugmodel_multimodal() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_debugmodel")


def deepseek_v4_1_debugmodel_text() -> TrainerEx.Config:
    return _make_trainer_config("deepseek_v4_1_debugmodel_text")


def deepseek_v4_1_stripped() -> TrainerEx.Config:
    """The single-card flavor: debug widths, released indexer geometry, ten layers.

    See :func:`torchtitan_npu.models.deepseek_v4_1._stripped` for what is and is not
    shrunk; the point of the flavor is to reach all three branches of the fused selector
    on one card.
    """
    return _make_trainer_config("deepseek_v4_1_stripped")


def deepseek_v4_1_flash_lora() -> TrainerEx.Config:
    return _make_lora_trainer_config("deepseek_v4_1_flash")


def deepseek_v4_1_flash_text_lora() -> TrainerEx.Config:
    return _make_lora_trainer_config("deepseek_v4_1_flash_text")


def deepseek_v4_1_debugmodel_lora() -> TrainerEx.Config:
    return _make_lora_trainer_config("deepseek_v4_1_debugmodel_text")


def _make_lora_trainer_config(flavor: str) -> TrainerEx.Config:
    from .checkpoint import DeepSeekV41LoRACheckpointManager
    from .lora import DeepSeekV41LoRAConverter

    config = _make_trainer_config(
        flavor, converters=[DeepSeekV41LoRAConverter.Config(rank=16, alpha=32, rank_experts=16)]
    )
    config.checkpoint = derive(config.checkpoint, DeepSeekV41LoRACheckpointManager.Config)
    config.checkpoint.enable = True
    config.checkpoint.last_save_model_only = False
    config.checkpoint.last_save_in_hf = False
    return config
