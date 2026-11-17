# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSV4 per-parameter FSDP mixed-precision policy selection on CPU.

``_dsv4_fp32_overrides`` maps exact parameter FQNs to an FP32 policy for the
SmoE hyper-connection modules, the MoE routers, the global ``hc_head``, the
compressor/indexer APE score biases and the attention sinks.  ``policy_overrides``
(patches/torch/distributed/fsdp/__init__.py) attaches the mapping to every group
policy built inside the block, and ``init_dtype_attrs`` resolves each parameter
by its exact FQN.

``FP32_MODULES`` lists the parameters that must be FP32 and is compared with the
model and with the override map in both directions; ``HF_F32`` lists the
families the released checkpoint stores as F32, so a family missing from the
list fails here instead of silently running in BF16.
"""

import fnmatch

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.experiments.graph_trainer.common_utils import matches_module_fqn_pattern

from tests.unit_tests.models.mtp_test_utils import SEQ_LEN, VOCAB_SIZE, build_cpu_model
from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.models.deepseek_v4.parallelize import _dsv4_fp32_overrides

def _build_policy_model():
    """A small MTP model covering every FP32 family: hc pre-heads and heads,
    MoE routers, compressor/indexer APE biases, in both layer scopes."""
    config = _make_v4_config(
        dim=32,
        n_layers=3,
        vocab_size=VOCAB_SIZE,
        n_heads=4,
        head_dim=8,
        rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=8,
        n_groups=2,
        compress_ratios=(128, 128, 4),
        window_size=SEQ_LEN,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        moe_inter_dim=16,
        num_experts=2,
        num_shared_experts=1,
        top_k=1,
        n_hash_layers=0,
        route_norm=False,
        route_scale=1.0,
        load_balance_coeff=1e-3,
        hc_mult=2,
        sinkhorn_iters=2,
        hc_eps=1e-6,
        max_seq_len=SEQ_LEN,
        compress_rope_theta=10000.0,
        original_seq_len=65536,
        num_mtp_layers=1,
    )
    return build_cpu_model(config)


# Parameters that must keep FP32 compute.  Key = pattern over the internal
# parameter FQNs of this repo; value = the same tensors in the released
# checkpoint (deepseek-ai/DeepSeek-V4-Flash-0731, revision 7872f01b), where
# hc_*, ape and attn_sink are F32 while the router gate is BF16 -- training
# keeps the gate FP32 on purpose.
FP32_MODULES = {
    "layers.*.hc_attn_pre.*": "layers.*.hc_attn_*",
    "layers.*.hc_ffn_pre.*": "layers.*.hc_ffn_*",
    "layers.*.moe.router.gate.*": "layers.*.ffn.gate.weight",
    "mtp_layers.*.hc_attn_pre.*": "mtp.*.hc_attn_*",
    "mtp_layers.*.hc_ffn_pre.*": "mtp.*.hc_ffn_*",
    "mtp_layers.*.moe.router.gate.*": "mtp.*.ffn.gate.weight",
    "mtp_layers.*.hc_head.*": "mtp.*.hc_head_*",
    "hc_head.*": "hc_head_*",
    "layers.*.attention.compressor.ape": "layers.*.attn.compressor.ape",
    "layers.*.attention.indexer.compressor.ape": "layers.*.attn.indexer.compressor.ape",
    "layers.*.attention.attn_sink": "layers.*.attn.attn_sink",
    "mtp_layers.*.attention.attn_sink": "mtp.*.attn.attn_sink",
}
# The families the release stores as F32, from a dtype audit of all 48 shards of
# that revision (72,317 tensors, 433 F32).  Each one must be covered above or be
# a buffer listed below.
HF_F32 = (
    "layers.*.attn.attn_sink",
    "layers.*.attn.compressor.ape",
    "layers.*.attn.indexer.compressor.ape",
    "layers.*.ffn.gate.bias",
    "layers.*.hc_attn_*",
    "layers.*.hc_ffn_*",
    "mtp.*.attn.attn_sink",
    "mtp.*.ffn.gate.bias",
    "mtp.*.hc_attn_*",
    "mtp.*.hc_ffn_*",
    "mtp.*.hc_head_*",
    "hc_head_*",
)
# F32 in the release but buffers rather than parameters, so no FP32 override:
# ffn.gate.bias is moe.expert_bias_E, a float32 load-balancing buffer.
HF_F32_BUFFERS = ("layers.*.ffn.gate.bias", "mtp.*.ffn.gate.bias")


def test_fp32_modules_are_all_overridden():
    model = _build_policy_model()
    names = [name for name, _ in model.named_parameters()]
    overrides = _dsv4_fp32_overrides(model, TrainingConfig())

    # unused: a key of FP32_MODULES matches no parameter of the model (one fqn of
    # the list is not found in the model), i.e. the entry is wrong -- typo,
    # renamed module or a stale entry.
    unused = [pattern for pattern in FP32_MODULES if not any(matches_module_fqn_pattern(pattern, name) for name in names)]
    assert not unused, f"FP32 patterns match no parameter of the model: {unused}"

    # missing: a parameter of the model matches a key of FP32_MODULES but is not
    # in the override map (one fqn of the model is not found in
    # _dsv4_fp32_overrides), so it runs in BF16.
    missing = [
        name
        for name in names
        if any(matches_module_fqn_pattern(pattern, name) for pattern in FP32_MODULES) and name not in overrides
    ]
    assert not missing, f"these parameters match an FP32 pattern but have no override: {missing}"

    # extra: _dsv4_fp32_overrides keeps a parameter FP32 that the list does not
    # mention (one fqn of the override map is not found in the list), i.e. the
    # list is incomplete.
    extra = [
        name for name in overrides if not any(matches_module_fqn_pattern(pattern, name) for pattern in FP32_MODULES)
    ]
    assert not extra, f"these parameters have an FP32 override but are not in the list: {extra}"


def test_every_f32_tensor_of_the_release_is_covered():
    # in the release as F32, but neither a key of FP32_MODULES nor a known buffer
    # => the list, and with it _dsv4_fp32_overrides, misses that parameter
    missing = [
        family
        for family in HF_F32
        if not any(fnmatch.fnmatchcase(family, hf) for hf in FP32_MODULES.values())
        and not any(fnmatch.fnmatchcase(family, buffer) for buffer in HF_F32_BUFFERS)
    ]
    assert not missing, f"the release stores these as F32 but the list does not cover them: {missing}"

def test_dsv4_fp32_policy_fields():
    training = TrainingConfig()
    overrides = _dsv4_fp32_overrides(_build_policy_model(), training)

    # The default policy comes from the training configuration; the overrides
    # only change the compute dtype, so they keep its reduce/output settings.
    assert overrides, "the model must contain FP32-family parameters"
    default_reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    for fqn, fp32_policy in overrides.items():
        assert fp32_policy.param_dtype == torch.float32, fqn
        assert fp32_policy.reduce_dtype == default_reduce_dtype, fqn
        assert fp32_policy.output_dtype is None, fqn
        assert fp32_policy.cast_forward_inputs is False, fqn


def test_dsv4_fp32_overrides_match_through_checkpoint_wrappers():
    plain_overrides = set(_dsv4_fp32_overrides(_build_policy_model(), TrainingConfig()))

    wrapped_model = _build_policy_model()
    wrapped_model.layers["0"].attention = checkpoint_wrapper(wrapped_model.layers["0"].attention)
    wrapped_overrides = set(_dsv4_fp32_overrides(wrapped_model, TrainingConfig()))

    assert any("_checkpoint_wrapped_module" in fqn for fqn in wrapped_overrides), (
        "checkpoint-wrapped parameters must still be selected"
    )
    stripped = {".".join(p for p in fqn.split(".") if p != "_checkpoint_wrapped_module") for fqn in wrapped_overrides}
    assert stripped == plain_overrides


def test_matches_any_fqn_pattern_through_checkpoint_wrappers():
    def strip_checkpoint_wrapper(fqn: str) -> str:
        return ".".join(part for part in fqn.split(".") if part != "_checkpoint_wrapped_module")

    patterns = ("layers.*.hc_attn_pre", "mtp_layers.*.moe.router", "hc_head")
    assert any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("layers.0.hc_attn_pre")) for p in patterns)
    assert any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("mtp_layers.0.moe.router")) for p in patterns)
    assert any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("hc_head")) for p in patterns)
    assert any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("layers.0._checkpoint_wrapped_module.hc_attn_pre")) for p in patterns)
    assert not any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("layers.0.attention")) for p in patterns)
    assert not any(matches_module_fqn_pattern(p, strip_checkpoint_wrapper("layers.0.moe.router")) for p in patterns)
