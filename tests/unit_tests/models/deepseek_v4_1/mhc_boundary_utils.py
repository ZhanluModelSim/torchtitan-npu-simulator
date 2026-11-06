# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Registered mHC backbone with fixed boundaries for the not-yet-folded attention/MoE."""

from dataclasses import dataclass

import torch
from torchtitan.protocols.module import Module

from torchtitan_npu.models.deepseek_v4_1 import model_registry


class AttentionBoundaryStub(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    attention_metadata_key = "mhc_boundary"

    def __init__(self, config):
        super().__init__()

    def forward(self, x, positions, attention_metadata, **state):
        return (
            x.tanh() * 0.75,
            *(state[key] for key in ("cmp_k", "idx_k", "topk_indices", "topk_scores", "candidates")),
        )


class MoEBoundaryStub(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config):
        super().__init__()

    def forward(self, x, *, image_mask=None, padding_mask_T=None):
        return torch.nn.functional.silu(x) * 0.5 - 0.125


def build_mhc_boundary_model(*, dim=4, vocab_size=11):
    config = model_registry("deepseek_v4_1_debugmodel_text")
    config.dim = config.tok_embeddings.embedding_dim = config.lm_head.in_features = dim
    config.vocab_size = config.tok_embeddings.num_embeddings = config.lm_head.out_features = vocab_size
    config.norm.normalized_shape = dim
    config.n_layers = 2
    config.layers = config.layers[:2]
    config.compress_ratios = (0, 0)
    for layer in config.layers:
        layer.engram = None
        layer.attention = AttentionBoundaryStub.Config()
        layer.moe = MoEBoundaryStub.Config()
        layer.attention_norm.normalized_shape = layer.ffn_norm.normalized_shape = dim
        layer.hc_attn_pre.dim = layer.hc_ffn_pre.dim = dim
        layer.hc_attn_pre.sinkhorn_iters = layer.hc_ffn_pre.sinkhorn_iters = 4
    with torch.random.fork_rng(devices=[]):
        model = config.build()
    generator = torch.Generator().manual_seed(68)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.15)
            if "norm.weight" in name or name.endswith("hc_scale"):
                parameter.add_(0.8)
    return model
