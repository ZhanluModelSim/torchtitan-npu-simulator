# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4 parameter declarations consumed by the shared FSDP precision policy.

The expected families below come from the released checkpoint's dtype audit
and the deliberate FP32 training-router policy, independently of the declarations.
The native V4 layout stores HCA APE under ``compressor_128`` and attention sinks
inside ``Linear.weight``; both must retain the same checkpoint precision.
"""

import fnmatch
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torchtitan.distributed.fsdp as fsdp
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torchtitan.config import TrainingConfig, derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4.moe import DeepSeekV4Router
from torchtitan.tools.utils import set_default_dtype

from tests.unit_tests.models.mtp_test_utils import SEQ_LEN, VOCAB_SIZE, build_cpu_model
from torchtitan_npu.models.deepseek_v4 import _make_v4_config


class _RouterWithAdapter(DeepSeekV4Router):
    """Router owner containing a foreign adapter before model construction ends."""

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV4Router.Config):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.adapter = nn.Linear(config.gate.in_features, config.num_experts, bias=True)


def _build_policy_model(*, router_adapter=False, master_dtype=torch.float32):
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
    if router_adapter:
        for layer in (*config.layers, *config.mtp_layers):
            layer.feed_forward.router = derive(layer.feed_forward.router, _RouterWithAdapter.Config)
    with torch.random.fork_rng(devices=[]), set_default_dtype(master_dtype):
        return build_cpu_model(config)


@pytest.fixture
def cpu_mesh(tmp_path):
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))
    finally:
        dist.destroy_process_group()


# Protected parameter families from the released checkpoint and training-router policy.
# Checkpoint: deepseek-ai/DeepSeek-V4-Flash-0731, revision 7872f01b.
# hc_*, ape and attn_sink are F32; the checkpoint gate weight is BF16,
# while training deliberately keeps the router gate in FP32.
# The 48-shard audit found 433 F32 tensors. Gate bias is the expert_bias_E
# buffer, so it is outside the parameter preservation policy.
FP32_PARAMETERS = (
    "layers.*.hc_attn_pre.*",
    "layers.*.hc_ffn_pre.*",
    "layers.*.feed_forward.router.gate.weight",
    "mtp_layers.*.hc_attn_pre.*",
    "mtp_layers.*.hc_ffn_pre.*",
    "mtp_layers.*.feed_forward.router.gate.weight",
    "mtp_layers.*.hc_head.*",
    "hc_head.*",
    "layers.*.attention.compressor.ape",
    "layers.*.attention.compressor_128.ape",
    "layers.*.attention.indexer.compressor.ape",
    "layers.*.attention.attn_sink.weight",
    "mtp_layers.*.attention.attn_sink.weight",
)


def test_v4_declares_the_expected_fp32_parameter_families():
    model = _build_policy_model()
    patterns = FP32_PARAMETERS
    names = tuple(name for name, _ in model.named_parameters())
    unused = [pattern for pattern in patterns if not any(fnmatch.fnmatchcase(name, pattern) for name in names)]
    assert not unused, f"Expected FP32 families absent from the model: {unused}"
    expected = {name for name in names if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)}

    declared = dict(model.named_preserve_dtype_params())

    assert set(declared) == expected
    assert {parameter.dtype for parameter in declared.values()} == {torch.float32}
    assert "layers.0.attention.compressor_128.ape" in declared
    assert "layers.2.attention.compressor.ape" in declared
    assert "layers.2.attention.indexer.compressor.ape" in declared
    assert "mtp_layers.0.attention.attn_sink.weight" in declared
    for name in ("tok_embeddings.weight", "lm_head.weight", "norm.weight", "layers.0.attention.wq_a.weight"):
        assert name in names and name not in declared


def test_model_declares_foreign_router_adapter_parameters_before_fsdp():
    model = _build_policy_model(router_adapter=True)
    declared = dict(model.named_preserve_dtype_params())
    callback = fsdp.mixed_precision_policy(
        model, param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    ).param_dtype_override_fn
    assert callback is not None

    for prefix, layer in (
        *((f"layers.{index}", layer) for index, layer in model.layers.items()),
        *((f"mtp_layers.{index}", layer) for index, layer in enumerate(model.mtp_layers)),
    ):
        adapter = layer.feed_forward.router.adapter
        assert type(adapter) is nn.Linear
        for suffix in ("weight", "bias"):
            parameter = getattr(adapter, suffix)
            assert declared[f"{prefix}.feed_forward.router.adapter.{suffix}"] is parameter
            assert callback(parameter) is torch.float32
        assert callback(layer.feed_forward.routed_experts.w13.weight) is None


@pytest.mark.parametrize("compute_dtype", ("bfloat16", "float32"), ids=("bf16-compute", "fp32-compute"))
@pytest.mark.parametrize("master_dtype", (torch.float32, torch.bfloat16), ids=("fp32-masters", "bf16-masters"))
def test_v4_fsdp_keeps_released_precision_support_boundary_and_parameter_storage(cpu_mesh, master_dtype, compute_dtype):
    model = _build_policy_model(master_dtype=master_dtype)
    reduce_dtype = "float32" if compute_dtype == "bfloat16" else "bfloat16"
    original_layers = tuple(model.layers)
    original_parameters = dict(model.named_parameters())
    original_values = {name: parameter.detach().clone() for name, parameter in original_parameters.items()}
    protected = {
        id(parameter)
        for name, parameter in original_parameters.items()
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in FP32_PARAMETERS)
    }
    # Independent oracle: the released V4 policy selected FP32 unconditionally.
    released_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, compute_dtype),
        reduce_dtype=getattr(torch, reduce_dtype),
        cast_forward_inputs=False,
        param_dtype_override_fn=lambda parameter: torch.float32 if id(parameter) in protected else None,
    )
    context = SimpleNamespace(
        get_activated_mesh=lambda _axes: cpu_mesh,
        ep_enabled=False,
        pp_enabled=False,
        ep=1,
    )
    training = TrainingConfig(mixed_precision_param=compute_dtype, mixed_precision_reduce=reduce_dtype)
    parallelism = ParallelismConfig(fsdp_reshard_after_forward="never")

    if master_dtype is torch.bfloat16 and compute_dtype == "bfloat16":
        parameter = next(p for p in original_parameters.values() if id(p) in protected and p.dtype is torch.bfloat16)
        with pytest.raises(ValueError, match="original dtype"):
            released_policy._resolve_for_param(parameter)
        with pytest.raises(ValueError, match="requires FP32 storage for protected parameter"):
            model._apply_fsdp(parallelism_context=context, training=training, parallelism=parallelism)
        assert not any(isinstance(module, FSDPModule) for module in model.modules())
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, original_values[name], rtol=0, atol=0)
        return

    expected = {name: released_policy._resolve_for_param(parameter) for name, parameter in original_parameters.items()}
    model._apply_fsdp(parallelism_context=context, training=training, parallelism=parallelism)

    assert tuple(model.layers) == original_layers
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    observed = {}
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        group = module._get_fsdp_state()._fsdp_param_group
        if group is None:
            continue
        for parameter in group.fsdp_params:
            name = names_by_id[id(parameter.sharded_param)]
            actual = parameter.mp_policy
            observed[name] = actual.param_dtype
            assert actual.param_dtype is expected[name].param_dtype, name
            assert actual.reduce_dtype is getattr(torch, reduce_dtype)
            assert actual.cast_forward_inputs is False
            torch.testing.assert_close(parameter.sharded_param.to_local(), original_values[name], rtol=0, atol=0)
    assert observed.keys() == original_parameters.keys()


def test_declared_dtype_selection_survives_checkpoint_wrapping():
    plain_names = set(dict(_build_policy_model().named_preserve_dtype_params()))
    model = _build_policy_model()
    model.layers["0"] = checkpoint_wrapper(model.layers["0"])
    model.mtp_layers[0] = checkpoint_wrapper(model.mtp_layers[0])

    declared = dict(model.named_preserve_dtype_params())
    assert any("_checkpoint_wrapped_module" in name for name in declared)
    canonical = {
        ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module")
        for name in declared
    }
    assert canonical == plain_names
    callback = fsdp.mixed_precision_policy(
        model, param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    ).param_dtype_override_fn
    assert callback is not None
    for name, parameter in model.named_parameters():
        expected = torch.float32 if name in declared else None
        assert callback(parameter) is expected, name
