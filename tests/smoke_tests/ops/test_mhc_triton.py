# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override-parity and compile-safety tests for the mHC Triton custom ops.

Override-parity tests (``test_hc_*_override_matches_eager``) compare the
``torchtitan_npu.override.deepseek_v4.mhc`` Triton modules against their eager
``torchtitan_npu.models.deepseek_v4.mhc`` counterparts for forward output and
backward gradients.

Compile-safety tests (``test_mhc_*_compiled_backward``) verify that each
forward op's backward, registered through ``register_autograd``, is itself a
compile-safe custom op (with ``register_fake``). Otherwise AOTAutograd traces
the backward with Fake/Proxy tensors during ``torch.compile`` and the raw
Triton launcher fails with "Cannot access data pointer of Tensor". These tests
run every op through ``torch.compile(backend="aot_eager", fullgraph=True)``
forward + backward and compare the gradients against eager.

Opcheck tests (``test_mhc_opcheck``) run ``torch.library.opcheck`` (schema
conformance, faketensor metadata vs real outputs, AOT dispatch dynamic
forward+backward) on every forward op. ``test_autograd_registration`` is
excluded because it rejects non CPU/CUDA/XPU devices in this torch version;
gradient numerics are already covered by the tests above. The ``post_bmm2``
bf16 case is skipped: its eager Triton kernel intermittently aborts the NPU
with error 507035 (vector core exception).
"""

import pytest
import torch

from tests.conftest import assert_tensor_finite, stable_randn
from torchtitan_npu.models.deepseek_v4.mhc import HcHead, HcPost, HcPre
from torchtitan_npu.ops.triton.mhc.post_bmm1 import mhc_post_bmm1_op
from torchtitan_npu.ops.triton.mhc.post_bmm2 import mhc_post_bmm2_op
from torchtitan_npu.ops.triton.mhc.pre_bmm import mhc_pre_bmm_op
from torchtitan_npu.ops.triton.mhc.prepost_sinkhorn import (
    mhc_pre_only_sinkhorn_op,
    mhc_pre_sinkhorn_op,
)
from torchtitan_npu.override.deepseek_v4.mhc import (
    triton_hc_head,
    triton_hc_post,
    triton_hc_pre,
)
from torchtitan_npu.override.deepseek_v4.mhc.triton import (
    TritonHcHead,
    TritonHcPost,
    TritonHcPre,
)

pytestmark = pytest.mark.smoke

_BATCH = 1
_SEQ_LEN = 96
_HC_MULT = 4
_DIM = 256
_EPS = 1e-6
_DTYPES = (torch.float32, torch.bfloat16)


def _tolerances(dtype):
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    return 1e-2, 1e-2


def _assert_tensors_close(eager, triton, *, dtype, name):
    assert_tensor_finite(eager, f"eager {name} should be finite")
    assert_tensor_finite(triton, f"Triton {name} should be finite")
    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(eager, triton, rtol=rtol, atol=atol)


def _initialize_parameters(module):
    torch.manual_seed(42)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.uniform_(-0.1, 0.1)


def _build_override_pair(config, override_factory, triton_type, device):
    eager = config.build()
    _initialize_parameters(eager)

    triton_config = override_factory(config)
    triton = triton_config.build()
    assert isinstance(triton, triton_type)
    triton.load_state_dict(eager.state_dict())

    return eager.to(device), triton.to(device)


def _clone_leaf_pair(value):
    eager = value.detach().clone().requires_grad_(True)
    triton = value.detach().clone().requires_grad_(True)
    return eager, triton


def _make_leaf_pair(*shape, device, dtype):
    return _clone_leaf_pair(stable_randn(*shape, device=device, dtype=dtype, scale=0.1))


def _assert_leaf_grads_close(pairs, *, dtype):
    for name, eager, triton in pairs:
        assert eager.grad is not None, f"eager {name}.grad should not be None"
        assert triton.grad is not None, f"Triton {name}.grad should not be None"
        _assert_tensors_close(eager.grad, triton.grad, dtype=dtype, name=f"{name}.grad")


def _assert_parameter_grads_close(eager_module, triton_module, *, dtype):
    eager_parameters = dict(eager_module.named_parameters())
    triton_parameters = dict(triton_module.named_parameters())
    assert eager_parameters.keys() == triton_parameters.keys()

    for name in eager_parameters:
        eager_grad = eager_parameters[name].grad
        triton_grad = triton_parameters[name].grad
        assert eager_grad is not None, f"eager parameter {name}.grad should not be None"
        assert triton_grad is not None, f"Triton parameter {name}.grad should not be None"
        _assert_tensors_close(eager_grad, triton_grad, dtype=dtype, name=f"parameter {name}.grad")


def _leaf(tensor):
    return tensor.detach().clone().requires_grad_(True)


def _assert_eager_matches_compiled(base_leaves, loss_fn, *, dtype, names):
    eager_leaves = [_leaf(t) for t in base_leaves]
    compiled_leaves = [_leaf(t) for t in base_leaves]

    eager_loss = loss_fn(*eager_leaves)
    eager_loss.backward()

    compiled = torch.compile(loss_fn, backend="aot_eager", fullgraph=True)
    compiled_loss = compiled(*compiled_leaves)
    compiled_loss.backward()

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(
        eager_loss.detach().float().cpu(),
        compiled_loss.detach().float().cpu(),
        rtol=rtol,
        atol=atol,
    )
    for name, eager_leaf, compiled_leaf in zip(names, eager_leaves, compiled_leaves, strict=True):
        assert eager_leaf.grad is not None, f"eager {name}.grad should not be None"
        assert compiled_leaf.grad is not None, f"compiled {name}.grad should not be None"
        torch.testing.assert_close(
            eager_leaf.grad.detach().float().cpu(),
            compiled_leaf.grad.detach().float().cpu(),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("dtype", _DTYPES, ids=("fp32", "bf16"))
def test_hc_head_override_matches_eager(npu_device, dtype):
    config = HcHead.Config(
        hc_mult=_HC_MULT,
        dim=_DIM,
        norm_eps=_EPS,
        eps=_EPS,
    )
    eager_module, triton_module = _build_override_pair(
        config,
        triton_hc_head,
        TritonHcHead,
        npu_device,
    )

    x_eager, x_triton = _make_leaf_pair(
        _BATCH,
        _SEQ_LEN,
        _HC_MULT,
        _DIM,
        device=npu_device,
        dtype=dtype,
    )

    y_eager = eager_module(x_eager)
    y_triton = triton_module(x_triton)
    _assert_tensors_close(y_eager, y_triton, dtype=dtype, name="HcHead output")

    grad_y = stable_randn(*y_eager.shape, device=npu_device, dtype=y_eager.dtype, scale=0.1)
    y_eager.backward(grad_y)
    y_triton.backward(grad_y)

    _assert_leaf_grads_close([("x", x_eager, x_triton)], dtype=dtype)
    _assert_parameter_grads_close(eager_module, triton_module, dtype=dtype)


@pytest.mark.parametrize("dtype", _DTYPES, ids=("fp32", "bf16"))
def test_hc_post_override_matches_eager(npu_device, dtype):
    eager_module, triton_module = _build_override_pair(
        HcPost.Config(),
        triton_hc_post,
        TritonHcPost,
        npu_device,
    )

    x_eager, x_triton = _make_leaf_pair(
        _BATCH,
        _SEQ_LEN,
        _DIM,
        device=npu_device,
        dtype=dtype,
    )
    residual_eager, residual_triton = _make_leaf_pair(
        _BATCH,
        _SEQ_LEN,
        _HC_MULT,
        _DIM,
        device=npu_device,
        dtype=dtype,
    )
    post_eager, post_triton = _make_leaf_pair(
        _BATCH,
        _SEQ_LEN,
        _HC_MULT,
        device=npu_device,
        dtype=torch.float32,
    )
    comb_eager, comb_triton = _make_leaf_pair(
        _BATCH,
        _SEQ_LEN,
        _HC_MULT,
        _HC_MULT,
        device=npu_device,
        dtype=torch.float32,
    )

    y_eager = eager_module(x_eager, residual_eager, post_eager, comb_eager)
    y_triton = triton_module(x_triton, residual_triton, post_triton, comb_triton)
    _assert_tensors_close(y_eager, y_triton, dtype=dtype, name="HcPost output")

    grad_y = stable_randn(*y_eager.shape, device=npu_device, dtype=y_eager.dtype, scale=0.1)
    y_eager.backward(grad_y)
    y_triton.backward(grad_y)

    _assert_leaf_grads_close(
        [
            ("x", x_eager, x_triton),
            ("residual", residual_eager, residual_triton),
            ("post", post_eager, post_triton),
            ("comb", comb_eager, comb_triton),
        ],
        dtype=dtype,
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=("fp32", "bf16"))
def test_hc_pre_override_matches_eager(npu_device, dtype):
    config = HcPre.Config(
        hc_mult=_HC_MULT,
        dim=_DIM,
        sinkhorn_iters=20,
        eps=_EPS,
        norm_eps=_EPS,
    )
    eager_module, triton_module = _build_override_pair(
        config,
        triton_hc_pre,
        TritonHcPre,
        npu_device,
    )

    x_eager, x_triton = _make_leaf_pair(
        _BATCH,
        _SEQ_LEN,
        _HC_MULT,
        _DIM,
        device=npu_device,
        dtype=dtype,
    )

    outputs_eager = eager_module(x_eager)
    outputs_triton = triton_module(x_triton)
    for name, eager, triton in zip(
        ("HcPre output", "HcPre post", "HcPre comb"),
        outputs_eager,
        outputs_triton,
        strict=True,
    ):
        _assert_tensors_close(eager, triton, dtype=dtype, name=name)

    output_grads = tuple(
        stable_randn(*output.shape, device=npu_device, dtype=output.dtype, scale=0.1) for output in outputs_eager
    )
    torch.autograd.backward(outputs_eager, output_grads)
    torch.autograd.backward(outputs_triton, output_grads)

    _assert_leaf_grads_close([("x", x_eager, x_triton)], dtype=dtype)
    _assert_parameter_grads_close(eager_module, triton_module, dtype=dtype)


def _op_args(op, device, dtype):
    total_dim = (2 + _HC_MULT) * _HC_MULT
    if op is mhc_pre_bmm_op:
        return (
            stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, device=device, dtype=dtype, scale=0.1),
            stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, _DIM, device=device, dtype=dtype, scale=0.1),
        )
    if op is mhc_post_bmm1_op:
        return (
            stable_randn(_BATCH, _SEQ_LEN, _DIM, device=device, dtype=dtype, scale=0.1),
            stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, device=device, scale=0.1),
        )
    if op is mhc_post_bmm2_op:
        return (
            stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, _HC_MULT, device=device, scale=0.1),
            stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, _DIM, device=device, dtype=dtype, scale=0.1),
        )
    if op is mhc_pre_sinkhorn_op:
        return (
            stable_randn(_BATCH, _SEQ_LEN, total_dim, device=device, dtype=dtype, scale=0.1),
            stable_randn(3, device=device, scale=0.1),
            stable_randn(total_dim, device=device, scale=0.1),
        )
    if op is mhc_pre_only_sinkhorn_op:
        return (
            stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, device=device, dtype=dtype, scale=0.1),
            stable_randn(3, device=device, scale=0.1),
            stable_randn(_HC_MULT, device=device, scale=0.1),
        )
    raise ValueError(f"unsupported op: {op}")


@pytest.mark.parametrize("dtype", _DTYPES, ids=("fp32", "bf16"))
def test_mhc_pre_bmm_compiled_backward(npu_device, dtype):
    base = _op_args(mhc_pre_bmm_op, npu_device, dtype)
    seed = stable_randn(_BATCH, _SEQ_LEN, _DIM, device=npu_device, scale=0.1)

    def loss_fn(h_pre, x):
        return (mhc_pre_bmm_op(h_pre, x) * seed).sum()

    _assert_eager_matches_compiled(base, loss_fn, dtype=dtype, names=("h_pre", "x"))


@pytest.mark.parametrize("dtype", _DTYPES, ids=("fp32", "bf16"))
def test_mhc_post_bmm1_compiled_backward(npu_device, dtype):
    base = _op_args(mhc_post_bmm1_op, npu_device, dtype)
    seed = stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, _DIM, device=npu_device, scale=0.1)

    def loss_fn(h_out, h_post):
        return (mhc_post_bmm1_op(h_out, h_post) * seed).sum()

    _assert_eager_matches_compiled(base, loss_fn, dtype=dtype, names=("h_out", "h_post"))


@pytest.mark.parametrize("dtype", _DTYPES, ids=("fp32", "bf16"))
def test_mhc_post_bmm2_compiled_backward(npu_device, dtype):
    base = _op_args(mhc_post_bmm2_op, npu_device, dtype)
    seed = stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, _DIM, device=npu_device, scale=0.1)

    def loss_fn(h_res, x):
        return (mhc_post_bmm2_op(h_res, x) * seed).sum()

    _assert_eager_matches_compiled(base, loss_fn, dtype=dtype, names=("h_res", "x"))


@pytest.mark.parametrize("dtype", _DTYPES, ids=("fp32", "bf16"))
def test_mhc_pre_sinkhorn_compiled_backward(npu_device, dtype):
    base = _op_args(mhc_pre_sinkhorn_op, npu_device, dtype)
    seed_pre = stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, device=npu_device, scale=0.1)
    seed_post = stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, device=npu_device, scale=0.1)
    seed_comb = stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, _HC_MULT, device=npu_device, scale=0.1)

    def loss_fn(mixes, hc_scale, hc_base):
        pre, post, comb = mhc_pre_sinkhorn_op(mixes, hc_scale, hc_base)
        return (pre * seed_pre).sum() + (post * seed_post).sum() + (comb * seed_comb).sum()

    _assert_eager_matches_compiled(base, loss_fn, dtype=dtype, names=("mixes", "hc_scale", "hc_base"))


@pytest.mark.parametrize("dtype", _DTYPES, ids=("fp32", "bf16"))
def test_mhc_pre_only_sinkhorn_compiled_backward(npu_device, dtype):
    base = _op_args(mhc_pre_only_sinkhorn_op, npu_device, dtype)
    seed = stable_randn(_BATCH, _SEQ_LEN, _HC_MULT, device=npu_device, scale=0.1)

    def loss_fn(mixes, hc_scale, hc_base):
        return (mhc_pre_only_sinkhorn_op(mixes, hc_scale, hc_base) * seed).sum()

    _assert_eager_matches_compiled(base, loss_fn, dtype=dtype, names=("mixes", "hc_scale", "hc_base"))


_OPCHECK_UTILS = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")

_OP_DTYPE_CASES = (
    pytest.param(mhc_pre_bmm_op, torch.float32, id="pre_bmm-fp32"),
    pytest.param(mhc_pre_bmm_op, torch.bfloat16, id="pre_bmm-bf16"),
    pytest.param(mhc_post_bmm1_op, torch.float32, id="post_bmm1-fp32"),
    pytest.param(mhc_post_bmm1_op, torch.bfloat16, id="post_bmm1-bf16"),
    pytest.param(mhc_post_bmm2_op, torch.float32, id="post_bmm2-fp32"),
    pytest.param(mhc_post_bmm2_op, torch.bfloat16, id="post_bmm2-bf16"),
    pytest.param(mhc_pre_sinkhorn_op, torch.float32, id="pre_sinkhorn-fp32"),
    pytest.param(mhc_pre_sinkhorn_op, torch.bfloat16, id="pre_sinkhorn-bf16"),
    pytest.param(mhc_pre_only_sinkhorn_op, torch.float32, id="pre_only_sinkhorn-fp32"),
    pytest.param(mhc_pre_only_sinkhorn_op, torch.bfloat16, id="pre_only_sinkhorn-bf16"),
)


@pytest.mark.parametrize(("op", "dtype"), _OP_DTYPE_CASES)
def test_mhc_opcheck(npu_device, op, dtype):
    args = tuple(_leaf(t) for t in _op_args(op, npu_device, dtype))
    torch.library.opcheck(op, args, test_utils=_OPCHECK_UTILS)
