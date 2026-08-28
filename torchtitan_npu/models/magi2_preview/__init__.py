# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MAGI-2-preview model registration and config builders.

Provides model_registry() for torchtitan's ModelSpec discovery, and flavor
factories that construct the joint video/audio diffusion MoE transformer
configs (debug bring-up flavor and the official full-size architecture).
"""

from torchtitan.components.loss import build_mse_loss
from torchtitan.protocols.model_spec import ModelSpec

from .model import Magi2PreviewModel
from .parallelize import parallelize_magi2_preview
from .state_dict_adapter import Magi2PreviewStateDictAdapter


def _debug_model() -> Magi2PreviewModel.Config:
    """Minimal debug config: 4 layers (2 mm + 2 MoE), 8 experts, small dims.

    ``text_in_channels`` is shrunk to 64 so the text embedder stays cheap;
    the synthetic dataloader pads text tokens to the full channel width and
    the PreAdapter slices only the first ``text_in_channels`` columns.
    """
    return Magi2PreviewModel.Config(
        num_layers=4,
        hidden_size=512,
        head_dim=128,
        num_stream=4,
        video_in_channels=48,
        audio_in_channels=64,
        text_in_channels=64,
        time_channel_dim=64,
        dense_intermediate_size=512,
        mm_layers=[0, 3],
        moe_layers=[1, 2],
        moe_num_heads=2,
        num_experts=8,
        moe_top_k=2,
        expert_intermediate_size=64,
        shared_expert_intermediate_size=64,
    )


def _full_model() -> Magi2PreviewModel.Config:
    """Full MAGI-2-preview 114B model: 40 layers, 256 experts/head, top-6."""
    return Magi2PreviewModel.Config(
        num_layers=40,
        hidden_size=3072,
        head_dim=128,
        num_stream=4,
        video_in_channels=48,
        audio_in_channels=64,
        text_in_channels=5120,
        time_channel_dim=64,
        dense_intermediate_size=8192,
        mm_layers=[0, 1, 38, 39],
        moe_layers=list(range(2, 38)),
        moe_num_heads=12,
        num_experts=256,
        moe_top_k=6,
        expert_intermediate_size=1280,
        shared_expert_intermediate_size=1280,
    )


magi2_preview_configs = {
    "debug": _debug_model,
    "full": _full_model,
}


def model_registry(flavor: str) -> ModelSpec:
    model_config = magi2_preview_configs[flavor]()
    return ModelSpec(
        name="magi2_preview",
        flavor=flavor,
        model=model_config,
        parallelize_fn=parallelize_magi2_preview,
        pipelining_fn=None,
        build_loss_fn=build_mse_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Magi2PreviewStateDictAdapter,
    )
