# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native training configuration factories for the Qwen3 dense decoder."""

from dataclasses import replace

from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.models.qwen3 import model_registry
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from .model import as_npu_config

__all__ = ["qwen3_32b", "qwen3_32b_smoketest"]


def _trainer_config(
    flavor: str,
    *,
    num_layers: int | None = None,
    num_mtp_layers: int = 0,
) -> Trainer.Config:
    if num_mtp_layers:
        raise ValueError("Qwen3 dense models have no MTP modules")
    # The varlen backend matches the NPU training path (packed-document
    # metadata); flex attention has no Ascend kernel boundary.
    spec = model_registry(flavor, attn_backend="varlen")
    model = as_npu_config(spec.model)
    if num_layers is not None:
        if not 0 < num_layers < len(model.layers):
            raise ValueError(f"reduced depth must be within (0, {len(model.layers)}), got {num_layers}")
        model.layers = model.layers[:num_layers]
    model_spec = replace(spec, model=model)
    return Trainer.Config(
        model_spec=model_spec,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=model.vocab_size),
        ),
        optimizer=default_adamw(lr=8e-4),
        # Dense causal attention has no CP load balancer; the runner passes
        # this value through to build_attention_masks unchanged.
        parallelism=ParallelismConfig(context_parallel_load_balancer=None),
        # FP32 raw parameters satisfy the unified dtype constraint; the
        # simulator and FSDP keep BF16 compute projections.
        training=TrainingConfig(local_batch_size=1, seq_len=128, steps=1, dtype="float32"),
    )


def qwen3_32b(*, num_mtp_layers: int = 0) -> Trainer.Config:
    """Full Qwen3-32B: 64 layers, GQA 64:8, head_dim 128, SwiGLU 25600."""
    return _trainer_config("32B", num_mtp_layers=num_mtp_layers)


def qwen3_32b_smoketest(*, num_mtp_layers: int = 0) -> Trainer.Config:
    """Four-layer slice of the real 32B geometry.

    Only depth is reduced: GQA 64:8, QK-norm, fused QKV, SwiGLU 25600, the
    untied head, vocabulary 151936 and rope theta 1e6 all stay on the real
    path so the smoke exercises every distinctive branch.
    """
    return _trainer_config("32B", num_layers=4, num_mtp_layers=num_mtp_layers)
