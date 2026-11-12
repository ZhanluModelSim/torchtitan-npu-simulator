# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4 FP32 parameter families passed to native FSDP precision callbacks.

The expected families below come from the released checkpoint's dtype audit
and the deliberate FP32 training-router policy, independently of the selector.
The native V4 layout stores HCA APE under ``compressor_128`` and attention sinks
inside ``Linear.weight``; both must retain the same checkpoint precision.
"""

import fnmatch

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torchtitan.config import TrainingConfig

from tests.unit_tests.models.mtp_test_utils import SEQ_LEN, VOCAB_SIZE, build_cpu_model
from torchtitan_npu.models.common.fsdp import make_param_dtype_override_fn
from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.models.deepseek_v4.parallelize import _dsv4_param_dtypes


def _build_policy_model():
    """Construct main HCA/CSA blocks and a dense MTP block through the factory."""
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
    with torch.random.fork_rng(devices=[]):
        return build_cpu_model(config)


# Internal parameter family -> corresponding released-checkpoint family.
# Checkpoint: deepseek-ai/DeepSeek-V4-Flash-0731, revision 7872f01b.
# hc_*, ape and attn_sink are F32; the checkpoint gate weight is BF16,
# while training deliberately keeps the router gate in FP32.
FP32_PARAMETERS = {
    "layers.*.hc_attn_pre.*": "layers.*.hc_attn_*",
    "layers.*.hc_ffn_pre.*": "layers.*.hc_ffn_*",
    "layers.*.feed_forward.router.gate.weight": "layers.*.ffn.gate.weight",
    "mtp_layers.*.hc_attn_pre.*": "mtp.*.hc_attn_*",
    "mtp_layers.*.hc_ffn_pre.*": "mtp.*.hc_ffn_*",
    "mtp_layers.*.feed_forward.router.gate.weight": "mtp.*.ffn.gate.weight",
    "mtp_layers.*.hc_head.*": "mtp.*.hc_head_*",
    "hc_head.*": "hc_head_*",
    "layers.*.attention.compressor.ape": "layers.*.attn.compressor.ape",
    "layers.*.attention.compressor_128.ape": "layers.*.attn.compressor.ape",
    "layers.*.attention.indexer.compressor.ape": "layers.*.attn.indexer.compressor.ape",
    "layers.*.attention.attn_sink.weight": "layers.*.attn.attn_sink",
    "mtp_layers.*.attention.attn_sink.weight": "mtp.*.attn.attn_sink",
}
# Independent audit of all 48 release shards: 72,317 tensors, 433 F32.
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
# Checkpoint gate.bias is the FP32 feed_forward.expert_bias_E buffer.
HF_F32_BUFFERS = ("layers.*.ffn.gate.bias", "mtp.*.ffn.gate.bias")


def test_native_v4_fp32_parameter_families_are_selected_exactly():
    model = _build_policy_model()
    names = tuple(name for name, _ in model.named_parameters())
    unused = [pattern for pattern in FP32_PARAMETERS if not any(fnmatch.fnmatchcase(name, pattern) for name in names)]
    assert not unused, f"Expected FP32 families absent from the model: {unused}"
    expected = {name for name in names if any(fnmatch.fnmatchcase(name, pattern) for pattern in FP32_PARAMETERS)}

    dtypes = _dsv4_param_dtypes(model, TrainingConfig())

    assert set(dtypes) == expected
    assert set(dtypes.values()) == {torch.float32}
    assert "layers.0.attention.compressor_128.ape" in dtypes
    assert "layers.2.attention.compressor.ape" in dtypes
    assert "layers.2.attention.indexer.compressor.ape" in dtypes
    assert "mtp_layers.0.attention.attn_sink.weight" in dtypes
    for name in ("tok_embeddings.weight", "lm_head.weight", "norm.weight", "layers.0.attention.wq_a.weight"):
        assert name in names and name not in dtypes


def test_every_f32_tensor_of_the_release_is_covered():
    missing = [
        family
        for family in HF_F32
        if not any(fnmatch.fnmatchcase(family, hf) for hf in FP32_PARAMETERS.values())
        and not any(fnmatch.fnmatchcase(family, buffer) for buffer in HF_F32_BUFFERS)
    ]
    assert not missing, f"Released F32 families missing from the training contract: {missing}"


@pytest.mark.parametrize("reduce_dtype", ("float32", "bfloat16"))
def test_native_callback_only_changes_selected_parameter_compute_dtype(reduce_dtype):
    model = _build_policy_model()
    training = TrainingConfig(mixed_precision_param="bfloat16", mixed_precision_reduce=reduce_dtype)
    dtypes = _dsv4_param_dtypes(model, training)
    callback = make_param_dtype_override_fn(model, dtypes)

    for name, parameter in model.named_parameters():
        expected = torch.float32 if any(fnmatch.fnmatchcase(name, pattern) for pattern in FP32_PARAMETERS) else None
        assert callback(parameter) is expected, name


@pytest.mark.parametrize("scope", ("main-attention", "main-block", "mtp-block"))
def test_native_dtype_selection_survives_checkpoint_wrapping(scope):
    plain_dtypes = _dsv4_param_dtypes(_build_policy_model(), TrainingConfig())
    model = _build_policy_model()
    if scope == "main-attention":
        model.layers["0"].attention = checkpoint_wrapper(model.layers["0"].attention)
    elif scope == "main-block":
        model.layers["0"] = checkpoint_wrapper(model.layers["0"])
    else:
        model.mtp_layers[0] = checkpoint_wrapper(model.mtp_layers[0])

    dtypes = _dsv4_param_dtypes(model, TrainingConfig())
    assert any("_checkpoint_wrapped_module" in name for name in dtypes)
    canonical = {
        ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module"): dtype
        for name, dtype in dtypes.items()
    }
    assert canonical == plain_dtypes
    callback = make_param_dtype_override_fn(model, dtypes)
    for name, parameter in model.named_parameters():
        assert callback(parameter) is dtypes.get(name), name


def test_native_callback_preserves_policy_through_temporary_mtp_layer_aliases():
    model = _build_policy_model()
    original_layers = tuple(model.layers)
    dtypes = _dsv4_param_dtypes(model, TrainingConfig())
    callback = make_param_dtype_override_fn(model, dtypes)
    key = str(len(model.layers))
    model.layers[key] = model.mtp_layers[0]
    try:
        aliased_parameters = dict(model.named_parameters(remove_duplicate=False))
        for suffix, parameter in model.mtp_layers[0].named_parameters():
            assert aliased_parameters[f"layers.{key}.{suffix}"] is parameter
            assert callback(parameter) is dtypes.get(f"mtp_layers.0.{suffix}"), suffix
    finally:
        del model.layers[key]
    assert tuple(model.layers) == original_layers
