# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.2 training model registration."""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan_npu.models.deepseek_v32 import _make_dsv32_model_config
from torchtitan_npu.models.deepseek_v32.parallelize import parallelize_deepseekv32

from .config_overrides import GLM52ModelOverrides, make_indexer_types
from .model import GLM5_2ModelNpu
from .state_dict_adapter import GLM52StateDictAdapter


def _as_glm_config(base_config, *, index_topk_freq: int, index_skip_topk_offset: int):
    import dataclasses

    values = {field.name: getattr(base_config, field.name) for field in dataclasses.fields(base_config)}
    n_main_layers = len(base_config.layers) - base_config.num_mtp_modules
    schedule = make_indexer_types(
        n_main_layers,
        freq=index_topk_freq,
        offset=index_skip_topk_offset,
    )
    return GLM5_2ModelNpu.Config(
        **values,
        index_topk_freq=index_topk_freq,
        index_skip_topk_offset=index_skip_topk_offset,
        indexer_types=schedule,
        indexer_rope_interleave=True,
        index_share_for_mtp_iteration=True,
    )


def _smoketest_model() -> GLM5_2ModelNpu.Config:
    # The smoke model keeps the official topology (three dense layers,
    # IndexShare and one MTP module) while shrinking matrix sizes.
    base = _make_dsv32_model_config(
        vocab_size=320,
        dim=128,
        inter_dim=256,
        moe_inter_dim=64,
        n_layers=6,
        n_dense_layers=3,
        n_heads=4,
        num_experts=8,
        num_shared_experts=1,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        norm_eps=1e-5,
        num_mtp_modules=1,
        router_top_k=2,
        router_route_scale=2.5,
        router_route_norm=True,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=16,
        indexer_types=make_indexer_types(6),
        indexer_rope_interleave=True,
        index_share_for_mtp_iteration=True,
        rope_max_seq_len=128,
        rope_scaling="none",
        rope_theta=8000000.0,
        rope_factor=1.0,
        rope_original_seq_len=128,
    )
    return _as_glm_config(base, index_topk_freq=4, index_skip_topk_offset=3)


def _78layers_1mtp_model() -> GLM5_2ModelNpu.Config:
    base = _make_dsv32_model_config(
        vocab_size=154880,
        dim=6144,
        inter_dim=12288,
        moe_inter_dim=2048,
        n_layers=78,
        n_dense_layers=3,
        n_heads=64,
        num_experts=256,
        num_shared_experts=1,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        norm_eps=1e-5,
        num_mtp_modules=1,
        router_top_k=8,
        router_score_func="sigmoid",
        router_route_scale=2.5,
        router_route_norm=True,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        indexer_types=make_indexer_types(78),
        indexer_rope_interleave=True,
        index_share_for_mtp_iteration=True,
        rope_max_seq_len=1048576,
        rope_scaling="none",
        rope_theta=8000000.0,
        rope_factor=1.0,
        rope_original_seq_len=1048576,
    )
    return _as_glm_config(base, index_topk_freq=4, index_skip_topk_offset=3)


glm5_2_configs = {
    "smoketest": _smoketest_model,
    "78layers_1mtp": _78layers_1mtp_model,
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="glm5_2",
        flavor=flavor,
        model=glm5_2_configs[flavor](),
        parallelize_fn=parallelize_deepseekv32,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=GLM52StateDictAdapter,
    )


__all__ = [
    "GLM52ModelOverrides",
    "GLM5_2ModelNpu",
    "glm5_2_configs",
    "model_registry",
]
