# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Retain the legacy Pro 20T geometry using native config-tree construction."""

from dataclasses import replace


def _qwen3_5_trainer_config(factory_name: str, *, num_mtp_layers: int = 0):
    """Build a native Qwen3.5 Trainer.Config through the upstream factory.

    Importing ``torchtitan_npu.models.qwen3_5`` activates the NPU model
    package: the FLA import shim, the multimodal patches and the NPU
    parallelizer swap on the upstream config registry. Qwen3.5 has no MTP
    modules, so the shared factory argument is rejected explicitly.
    """
    if num_mtp_layers:
        raise ValueError("Qwen3.5 has no MTP modules; num_mtp_layers must be 0")
    from torchtitan_npu.models.qwen3_5._fla_compat import ensure_qwen3_5_importable

    # Import order matters: the FLA placeholders must cover the upstream
    # qwen3_5 import, which binds fla symbols at module level.
    ensure_qwen3_5_importable()
    from torchtitan.models.qwen3_5 import config_registry as upstream_config_registry

    import torchtitan_npu.models.qwen3_5  # noqa: F401
    from torchtitan_npu.override.qwen3_5.parallelize import parallelize_qwen3_5_npu

    def _parallelize_text_capture(model, **kwargs):
        # Text-only synthetic capture never forwards the vision tower; the
        # native text-only SFT recipe detaches it, so mirror that instead of
        # freezing an unused tower's parameters.
        model.vision_encoder = None
        return parallelize_qwen3_5_npu(model, **kwargs)

    factory = getattr(upstream_config_registry, factory_name)
    trainer = factory()
    assert trainer.model_spec is not None
    trainer.model_spec = replace(trainer.model_spec, parallelize_fn=_parallelize_text_capture)
    return trainer


def qwen3_5_debugmodel(*, num_mtp_layers: int = 0):
    return _qwen3_5_trainer_config("qwen35_debugmodel", num_mtp_layers=num_mtp_layers)


def qwen3_5_debugmodel_moe(*, num_mtp_layers: int = 0):
    return _qwen3_5_trainer_config("qwen35_debugmodel_moe", num_mtp_layers=num_mtp_layers)


def multi_head_moe_debugmodel(*, num_mtp_layers: int = 0):
    """Build the SLA2 + Multi-Head-MoE debug trainer through the main package.

    The family has no MTP modules, so the shared factory argument is rejected
    explicitly.
    """
    if num_mtp_layers:
        raise ValueError("multi_head_moe has no MTP modules; num_mtp_layers must be 0")
    from torchtitan_npu.models.multi_head_moe.config_registry import multi_head_moe_debugmodel as _factory

    return _factory()


def _deepseek_v4_1_trainer_config(factory_name: str, *, num_mtp_layers: int = 0):
    """Build a native DeepSeek-V4.1 TrainerEx.Config through the main package.

    DeepSeek-V4.1 has no MTP modules, so the shared factory argument is
    rejected explicitly. The text flavor keeps the vision tower out of the
    capture, matching the text-only dataset the flavor itself selects.
    """
    if num_mtp_layers:
        raise ValueError("DeepSeek-V4.1 has no MTP modules; num_mtp_layers must be 0")
    from torchtitan_npu.models.deepseek_v4_1 import config_registry as v41_registry

    return getattr(v41_registry, factory_name)()


def deepseek_v4_1_debugmodel_text(*, num_mtp_layers: int = 0):
    return _deepseek_v4_1_trainer_config("deepseek_v4_1_debugmodel_text", num_mtp_layers=num_mtp_layers)


def deepseek_v4_1_debugmodel_multimodal(*, num_mtp_layers: int = 0):
    return _deepseek_v4_1_trainer_config("deepseek_v4_1_debugmodel_multimodal", num_mtp_layers=num_mtp_layers)


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
