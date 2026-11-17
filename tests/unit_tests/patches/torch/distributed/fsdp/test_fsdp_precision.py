# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contract tests for the per-parameter FSDP mixed-precision patch.

The shim classes run against fake FSDP internals: no process group, no FSDP
group and no NPU are needed.  The native functions are monkeypatched so the
assertions observe exactly what each shim passes through.

``TestUnshardPrecision`` instead drives real composable FSDP on a single-rank
gloo process group with a tiny DSV4-style model (an FP32 hyper-connection
module next to a default-BF16 projection): the FP32 override must make
``init_dtype_attrs`` clamp ``param_dtype`` to ``None`` so the unshard copies
the FP32 parameter out verbatim, while the BF16 default still casts.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

import torchtitan_npu.patches.torch.distributed.fsdp as fsdp_patch


def _policy(
    *,
    param_dtype,
    reduce_dtype=torch.bfloat16,
    output_dtype=None,
    cast_forward_inputs=False,
):
    return MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        output_dtype=output_dtype,
        cast_forward_inputs=cast_forward_inputs,
    )


def _carrier(overrides, **kwargs):
    """Build a policy inside ``policy_overrides``, as FSDP's caller does."""
    with fsdp_patch.policy_overrides(overrides):
        return _policy(**kwargs)


def _fake_param(fqn: str) -> SimpleNamespace:
    return SimpleNamespace(
        _param_fqn=fqn,
        _module_info=SimpleNamespace(param_name=fqn.rsplit(".", 1)[-1]),
    )


def _pooled_param(reduce_dtype):
    """A stand-in for the FSDPParam objects ``post_backward`` pools with grads."""
    return SimpleNamespace(reduce_dtype=reduce_dtype)


class TestPolicyOverrides:
    """Policy created inside ``policy_overrides`` carries ``.overrides``."""

    def test_policy_built_inside_the_block_carries_the_overrides(self):
        fp32 = _policy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        overrides = {"layers.0.hc_attn_pre.hc_fn": fp32}

        inside = _carrier(overrides, param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        outside = _policy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

        assert inside.overrides == overrides
        # The carrier is still a policy, which is what FSDP's module-level hooks read.
        assert inside.param_dtype == torch.bfloat16
        assert inside.cast_forward_inputs is False
        assert not hasattr(outside, "overrides")

    def test_nested_blocks_restore_the_outer_overrides(self):
        outer = _policy(param_dtype=torch.float32)
        inner = _policy(param_dtype=torch.bfloat16)

        with fsdp_patch.policy_overrides({"outer.weight": outer}):
            with fsdp_patch.policy_overrides({"inner.weight": inner}):
                assert _policy(param_dtype=torch.bfloat16).overrides == {"inner.weight": inner}
            assert _policy(param_dtype=torch.bfloat16).overrides == {"outer.weight": outer}

        assert not hasattr(_policy(param_dtype=torch.bfloat16), "overrides")

    def test_policy_overrides_owns_the_constructor_only_inside_the_block(self):
        assert MixedPrecisionPolicy.__init__ is fsdp_patch._ORIGINAL_POLICY_INIT

        with fsdp_patch.policy_overrides({"layers.0.norm.weight": _policy(param_dtype=torch.float32)}):
            assert MixedPrecisionPolicy.__init__ is not fsdp_patch._ORIGINAL_POLICY_INIT

        assert MixedPrecisionPolicy.__init__ is fsdp_patch._ORIGINAL_POLICY_INIT


class TestInitDtypeAttrs:
    """``_patched_init_dtype_attrs`` resolves per-parameter policies."""

    def test_resolves_exact_parameter_fqns(self, monkeypatch):
        fp32 = _policy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        applied = []

        def native_init_dtype_attrs(self, mp_policy):
            applied.append((id(self), mp_policy))
            self.param_dtype = mp_policy.param_dtype
            # Mirrors the native clamp: no casting is needed when the dtypes match.
            self.reduce_dtype = None if mp_policy.reduce_dtype == mp_policy.param_dtype else mp_policy.reduce_dtype

        monkeypatch.setattr(fsdp_patch, "_ORIGINAL_INIT_DTYPE_ATTRS", native_init_dtype_attrs)
        mp_policy = _carrier(
            {"layers.0.hc_attn_pre.hc_fn": fp32},
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        default_param = _fake_param("layers.0.norm.weight")
        fp32_param = _fake_param("layers.0.hc_attn_pre.hc_fn")
        fsdp_patch._patched_init_dtype_attrs(default_param, mp_policy)
        fsdp_patch._patched_init_dtype_attrs(fp32_param, mp_policy)

        assert applied == [(id(default_param), mp_policy), (id(fp32_param), fp32)]
        assert default_param.mp_policy is mp_policy
        assert fp32_param.mp_policy is fp32
        assert default_param.param_dtype == torch.bfloat16
        assert fp32_param.param_dtype == torch.float32
        # The FP32 override clamps to None natively; the group still needs one
        # explicit reduce dtype for its reduce-scatter.
        assert default_param.reduce_dtype == torch.float32
        assert fp32_param.reduce_dtype == torch.float32

    def test_matches_through_checkpoint_wrapper(self, monkeypatch):
        fp32 = _policy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        applied = []

        def native_init_dtype_attrs(self, mp_policy):
            applied.append(mp_policy)

        monkeypatch.setattr(fsdp_patch, "_ORIGINAL_INIT_DTYPE_ATTRS", native_init_dtype_attrs)
        mp_policy = _carrier(
            {"layers.0.hc_attn_pre.hc_fn": fp32},
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

        # Activation checkpointing nests the target module one level deeper.
        wrapped_param = _fake_param("layers.0._checkpoint_wrapped_module.hc_attn_pre.hc_fn")
        fsdp_patch._patched_init_dtype_attrs(wrapped_param, mp_policy)

        assert applied == [fp32]
        assert wrapped_param.mp_policy is fp32

    def test_passes_plain_policy_through(self, monkeypatch):
        policy = _policy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        applied = []

        def native_init_dtype_attrs(self, mp_policy):
            applied.append(mp_policy)

        monkeypatch.setattr(fsdp_patch, "_ORIGINAL_INIT_DTYPE_ATTRS", native_init_dtype_attrs)

        param = _fake_param("layers.0.norm.weight")
        fsdp_patch._patched_init_dtype_attrs(param, policy)

        assert applied == [policy]
        # A policy without overrides keeps native behavior untouched.
        assert not hasattr(param, "mp_policy")
        assert not hasattr(param, "reduce_dtype")


class TestForeachReduce:
    """``_patched_foreach_reduce`` unifies mixed-dtype gradients."""

    def test_unifies_mixed_gradient_dtypes(self, monkeypatch):
        captured = {}

        def native_foreach_reduce(fsdp_params, unsharded_grads, *args, **kwargs):
            captured["dtypes"] = [grad.dtype for grad in unsharded_grads]
            captured["ids"] = [id(grad) for grad in unsharded_grads]

        monkeypatch.setattr(fsdp_patch, "_ORIGINAL_FOREACH_REDUCE", native_foreach_reduce)

        bf16_grad = torch.zeros(4, dtype=torch.bfloat16)
        fp32_grad = torch.zeros(4, dtype=torch.float32)
        group = [_pooled_param(torch.float32), _pooled_param(torch.float32)]

        fsdp_patch._patched_foreach_reduce(group, [bf16_grad, fp32_grad])

        assert captured["dtypes"] == [torch.float32, torch.float32]
        # The gradient that already matched the reduce dtype must not be copied.
        assert captured["ids"][1] == id(fp32_grad)

    def test_keeps_uniform_gradients(self, monkeypatch):
        captured = {}

        def native_foreach_reduce(fsdp_params, unsharded_grads, *args, **kwargs):
            captured["dtypes"] = [grad.dtype for grad in unsharded_grads]
            captured["ids"] = [id(grad) for grad in unsharded_grads]

        monkeypatch.setattr(fsdp_patch, "_ORIGINAL_FOREACH_REDUCE", native_foreach_reduce)

        grad = torch.zeros(4, dtype=torch.bfloat16)

        fsdp_patch._patched_foreach_reduce([_pooled_param(torch.float32)], [grad])

        # Uniform grads: the list must be passed through untouched.
        assert captured["dtypes"] == [torch.bfloat16]
        assert captured["ids"] == [id(grad)]


# ---------------------------------------------------------------------------
# A tiny DSV4-flavoured model for the real-FSDP unshard precision tests.
# ---------------------------------------------------------------------------

_DIM, _MIX = 16, 4
_CAST_OPS = {"aten::to", "aten::_to_copy", "aten::_foreach_copy_"}


def _hc_math(hc_fn, hc_base, x):
    """The hyper-connection mixing formula in a caller-chosen dtype.

    Mirrors ``mhc.HcPre``: gates the input with a sigmoid mixture and sums
    the gated copies.  Low parameter bits land in ``mixes``, so a BF16-rounded
    ``hc_fn`` shifts the gate values measurably.
    """
    mixes = F.linear(x, hc_fn) + hc_base
    pre = torch.sigmoid(mixes)
    return (pre.unsqueeze(-1) * x.unsqueeze(1)).sum(dim=1)


class _HcModule(nn.Module):
    """The FP32 family; uses the production ``.float()`` defensive upcast.

    ``_fp32_out`` records the value before the output is quantized back to the
    input dtype: it is exactly what the unshard precision protects, and it lets
    the tests observe the difference between an FP32 and a BF16-rounded
    parameter without the output quantization swamping the signal.
    """

    def __init__(self, dim, mix):
        super().__init__()
        self.hc_fn = nn.Parameter(torch.randn(mix, dim))
        self.hc_base = nn.Parameter(torch.randn(mix))
        self._fp32_out = None

    def forward(self, x):
        dtype = x.dtype
        y = _hc_math(self.hc_fn.float(), self.hc_base.float(), x.float())
        self._fp32_out = y
        return y.to(dtype)


class _TinyModel(nn.Module):
    """One decoder layer: an FP32 ``hc_attn_pre`` next to a BF16 ``proj``."""

    def __init__(self, dim, mix):
        super().__init__()
        self.layers = nn.ModuleList([_Block(dim, mix)])

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


class _Block(nn.Module):
    """A single transformer-like block: HC mixing followed by a projection."""

    def __init__(self, dim, mix):
        super().__init__()
        self.hc_attn_pre = _HcModule(dim, mix)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(self.hc_attn_pre(x))


def _hc_params(model):
    """Yield ``(FSDPParam, fp32 master tensor)`` for the override family."""
    for block in model.layers:
        group = block.hc_attn_pre._get_fsdp_state()._fsdp_param_group
        yield from group.fsdp_params


def _unsharded(param):
    """The materialized unsharded parameter as a plain tensor."""
    tensor = param.unsharded_param
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


@pytest.fixture(scope="module")
def _mesh():
    """A single-rank CPU gloo mesh for real ``fully_shard`` calls."""
    if not dist.is_initialized():
        import tempfile

        store = f"file://{tempfile.mkdtemp()}/store"
        dist.init_process_group("gloo", init_method=store, rank=0, world_size=1)
    yield init_device_mesh("cpu", (1,))
    dist.destroy_process_group()


class TestUnshardPrecision:
    """The native clamp turns an FP32 param policy into ``param_dtype=None``,
    so the unshard step copies the FP32 parameter out verbatim instead of
    casting it to the training dtype; the BF16 default still casts.

    These tests run the real composable-FSDP unshard on a tiny model (one
    FP32 hyper-connection module and one default-BF16 projection) and assert
    the outcome three ways: the unsharded tensors are bitwise-identical to the
    FP32 master (no cast touched the data), the forward math is bitwise-
    identical to a plain FP32 forward of the same master weights (no round-
    trip through a narrower dtype), and the profile shows the override removed
    exactly one dtype cast per overridden parameter.
    """

    @pytest.fixture(scope="class")
    def sharded(self, _mesh):
        torch.manual_seed(1234)
        master = _TinyModel(_DIM, _MIX)
        master_state = {name: value.clone() for name, value in master.state_dict().items()}

        def shard(model, use_overrides):
            # Mirror ``parallelize_deepseek_v4``: the FP32 override map covers
            # the ``layers.*.hc_attn_pre`` family, and the group policy built
            # inside ``policy_overrides`` carries the map to ``fully_shard``.
            fp32 = _policy(param_dtype=torch.float32, reduce_dtype=torch.float32)
            overrides = (
                {name: fp32 for name, _ in model.named_parameters() if ".hc_attn_pre." in name} if use_overrides else {}
            )
            with fsdp_patch.policy_overrides(overrides):
                group_policy = _policy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
                for block in model.layers:
                    fully_shard(
                        block.hc_attn_pre,
                        mesh=_mesh,
                        mp_policy=group_policy,
                        reshard_after_forward=False,
                    )
                    fully_shard(block, mesh=_mesh, mp_policy=group_policy, reshard_after_forward=False)
                fully_shard(model, mesh=_mesh, mp_policy=group_policy, reshard_after_forward=False)
            return model

        torch.manual_seed(1234)
        overridden = shard(_TinyModel(_DIM, _MIX), use_overrides=True)
        torch.manual_seed(1234)
        default = shard(_TinyModel(_DIM, _MIX), use_overrides=False)

        x = torch.randn(3, _DIM, dtype=torch.bfloat16)
        with torch.no_grad():
            overridden(x)
            default(x)

        # The pure-torch FP32 reference: the same hc module, no FSDP, no
        # overrides, so the master weights reach the math verbatim.
        reference = _TinyModel(_DIM, _MIX)
        reference.load_state_dict(master_state)
        with torch.no_grad():
            reference.layers[0].hc_attn_pre(x)

        return SimpleNamespace(
            overridden=overridden,
            default=default,
            reference=reference,
            master_state=master_state,
            x=x,
        )

    def test_fp32_override_unshards_the_master_bits_verbatim(self, sharded):
        for param in _hc_params(sharded.overridden):
            unsharded = _unsharded(param)
            master = sharded.master_state[param._param_fqn]
            assert param.param_dtype is None, param._param_fqn
            assert unsharded.dtype == torch.float32
            assert torch.equal(unsharded, master), (
                f"{param._param_fqn}: FP32 override must copy the master bits, not cast"
            )

    def test_default_bf16_policy_still_casts(self, sharded):
        for param in _hc_params(sharded.default):
            unsharded = _unsharded(param)
            assert param.param_dtype == torch.bfloat16, param._param_fqn
            assert unsharded.dtype == torch.bfloat16
            assert unsharded.dtype != sharded.master_state[param._param_fqn].dtype

    def test_fp32_override_forward_is_bitwise_equal_to_plain_fp32(self, sharded):
        """The FP32 override keeps the weight verbatim, so the module's fp32
        math is bitwise identical to a plain (non-FSDP) forward with the same
        FP32 master weights — no round-trip through a narrower dtype touches
        the data.  The BF16 default loses low bits, so its result differs."""
        ref = sharded.reference.layers[0].hc_attn_pre._fp32_out
        over = sharded.overridden.layers[0].hc_attn_pre._fp32_out
        default = sharded.default.layers[0].hc_attn_pre._fp32_out

        assert torch.equal(over, ref), "FP32 override must reproduce a plain FP32 forward bit-for-bit"
        assert not torch.equal(default, ref), "BF16 default must deviate from the plain FP32 forward"
        diff = (default - ref).abs().max().item()
        assert diff > 1e-3, f"BF16 default must lose measurable precision, got max diff {diff}"

    def test_profile_shows_one_fewer_cast_per_overridden_param(self, sharded):
        """The FP32 override removes exactly one ``aten::to``-family op per
        overridden parameter: those parameters skip the BF16 rounding that
        ``unshard`` applies under the default policy.

        On the pinned torch (2.14.0.dev20260719) a full forward emits 6
        dtype-cast ops for the override model and 8 for the default model;
        the 2-op difference is the two overridden hc parameters (``hc_fn``,
        ``hc_base``) that no longer get rounded.
        """

        def count_casts(model):
            with torch.no_grad():
                model.reshard()
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
                    model(sharded.x)
            return sum(1 for event in prof.events() if event.name in _CAST_OPS)

        assert count_casts(sharded.overridden) == 6, "FP32 override: 6 dtype casts expected"
        assert count_casts(sharded.default) == 8, "BF16 default: 8 dtype casts expected"
