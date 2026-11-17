# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Retain the legacy Pro 20T geometry using native config-tree construction."""

from dataclasses import replace


def deepseek_v4_pro_20t(*, num_mtp_layers=0):
    from torchtitan_npu.models.deepseek_v4 import _make_v4_config
    from torchtitan_npu.models.deepseek_v4.config_registry import _dsv4_optimizer_config, deepseek_v4_pro

    trainer = deepseek_v4_pro(num_mtp_layers=num_mtp_layers)
    model = _make_v4_config(
        dim=12288,
        n_layers=96,
        vocab_size=129280,
        n_heads=192,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=2304,
        o_lora_rank=1536,
        n_groups=16,
        # Legacy ratio 0 means no compressed stream; native layout uses 1.
        compress_ratios=tuple([128, 4] * 47 + [128, 1]),
        window_size=128,
        norm_eps=1e-6,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=1024,
        moe_inter_dim=2752,
        num_experts=2048,
        num_shared_experts=1,
        top_k=23,
        n_hash_layers=3,
        route_norm=True,
        route_scale=1.5,
        load_balance_coeff=1e-3,
        hc_mult=4,
        sinkhorn_iters=24,
        hc_eps=1e-6,
        max_seq_len=4096,
        compress_rope_theta=160000.0,
        original_seq_len=65536,
        rope_theta=10000.0,
        rope_factor=16.0,
        num_mtp_layers=num_mtp_layers,
    )
    assert trainer.model_spec is not None
    model_spec = replace(trainer.model_spec, flavor="legacy_pro_20t_geometry", model=model)
    trainer.model_spec = model_spec
    trainer.optimizer = _dsv4_optimizer_config(model_spec, lr=1e-5)
    return trainer


def _glm5_next_trainer_config(factory_name: str, *, num_mtp_layers: int = 0):
    """Build a native GLM-5.3-Flash TrainerEx.Config through the main package.

    GLM-5.3-Flash has no MTP modules, so the shared factory argument is
    rejected explicitly. The capture flavors run text-only geometry: the
    multimodal flavor keeps its vision tower and consumes the mm inputs the
    loader provides, while the text flavors detach it (native text-only SFT
    recipe) via the flavor itself.
    """
    if num_mtp_layers:
        raise ValueError("GLM-5.3-Flash has no MTP modules; num_mtp_layers must be 0")
    from torchtitan_npu.models.glm5_next import config_registry as glm_registry

    return getattr(glm_registry, factory_name)()


def glm5_next_debugmodel_text(*, num_mtp_layers: int = 0):
    return _glm5_next_trainer_config("glm5_next_debugmodel_text", num_mtp_layers=num_mtp_layers)


def glm5_next_debugmodel_multimodal(*, num_mtp_layers: int = 0):
    """Multimodal model flavor under a text-only capture.

    The packed text loader feeds no pixel_values, so the vision tower would
    run zero steps and its parameters would fail the gradient manifest. The
    native text-only SFT recipe detaches the tower instead of freezing an
    unused tower's parameters — mirror it (qwen3_5 pattern). The tower stays
    in the model config; wiring a real mm loader into the capture is the
    remaining multimodal boundary.
    """
    trainer = _glm5_next_trainer_config("glm5_next_debugmodel_multimodal", num_mtp_layers=num_mtp_layers)
    _original_parallelize = trainer.model_spec.parallelize_fn

    def _parallelize_text_capture(model, **kwargs):
        model.vision_encoder = None
        return _original_parallelize(model, **kwargs)

    assert trainer.model_spec is not None
    trainer.model_spec = replace(trainer.model_spec, parallelize_fn=_parallelize_text_capture)
    return trainer


def glm5_next_reduced_2l(*, num_mtp_layers: int = 0):
    return _glm5_next_trainer_config("glm5_next_reduced_2l", num_mtp_layers=num_mtp_layers)


def glm5_next(*, num_mtp_layers: int = 0):
    return _glm5_next_trainer_config("glm5_next", num_mtp_layers=num_mtp_layers)


def glm5_next_910b_test(*, num_mtp_layers: int = 0):
    return _glm5_next_trainer_config("glm5_next_910b_test", num_mtp_layers=num_mtp_layers)
