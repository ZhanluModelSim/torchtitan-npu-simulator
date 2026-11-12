# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native FSDP precision retains model and MTP weights before computation."""

import copy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.models.common.linear import Linear
from torchtitan.models.deepseek_v4.mhc import HcPre

from torchtitan_npu.models.common.fsdp import apply_fsdp_to_mtp_decoder, make_param_dtype_override_fn


class _ProjectionBlock(nn.Module):
    """Use actual V4 mixing and Linear modules in the decoder sharding layout."""

    def __init__(self):
        super().__init__()
        self.hc_attn_pre = HcPre.Config(dim=8, hc_mult=2, sinkhorn_iters=2).build()
        self.feed_forward = Linear.Config(in_features=8, out_features=8, inplace_wgrad_accum=False).build()

    def forward(self, x):
        reduced, post, comb = self.hc_attn_pre(x)
        return self.feed_forward(reduced), post, comb


class _MTPDecoderLayout(nn.Module):
    """The layer/MTP containers consumed by the production sharding bridge."""

    def __init__(self):
        super().__init__()
        self.enable_weight_tying = False
        self.tok_embeddings = self.norm = self.lm_head = None
        self.layers = nn.ModuleDict({"0": _ProjectionBlock()})
        self.mtp_layers = nn.ModuleList([_ProjectionBlock()])

    def forward(self, x):
        return [block(x) for block in (*self.layers.values(), *self.mtp_layers)]


@pytest.fixture
def cpu_mesh(tmp_path):
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))
    finally:
        dist.destroy_process_group()


def test_native_callback_preserves_fp32_model_and_mtp_weights_and_gradients(cpu_mesh):
    with torch.random.fork_rng(devices=[]):
        model = _MTPDecoderLayout()
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            values = torch.linspace(-0.35, 0.45, parameter.numel()).reshape(parameter.shape)
            parameter.copy_(values + (index + 1) * 0.00137)
    reference = copy.deepcopy(model)
    masters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for block in (reference.layers["0"], reference.mtp_layers[0]):
        block.feed_forward.to(dtype=torch.bfloat16)
    overrides = {name: torch.float32 for name in masters if ".hc_attn_pre." in name}
    callback = make_param_dtype_override_fn(model, overrides)
    original_layers = tuple(model.layers)

    apply_fsdp_to_mtp_decoder(
        model,
        cpu_mesh,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        pp_enabled=False,
        reshard_after_forward_policy="never",
        param_dtype_override_fn=callback,
    )

    assert tuple(model.layers) == original_layers
    observed = {}
    handles = []
    for prefix, block in (("layers.0", model.layers["0"]), ("mtp_layers.0", model.mtp_layers[0])):
        for suffix, module in (("hc_attn_pre", block.hc_attn_pre), ("feed_forward", block.feed_forward)):

            def capture(module, _inputs, *, prefix=prefix, suffix=suffix):
                for name, parameter in module.named_parameters():
                    observed[f"{prefix}.{suffix}.{name}"] = parameter.detach().clone()

            handles.append(module.register_forward_pre_hook(capture))
    x = torch.linspace(-1.7, 1.3, 3 * 2 * 8).reshape(3, 2, 8).bfloat16()
    actual_x, expected_x = x.clone().requires_grad_(), x.clone().requires_grad_()
    try:
        actual = model(actual_x)
        expected = reference(expected_x)
        actual_loss = sum(
            output.float().square().sum() * 0.25 + post.square().sum() + comb.square().sum()
            for output, post, comb in actual
        )
        expected_loss = sum(
            output.float().square().sum() * 0.25 + post.square().sum() + comb.square().sum()
            for output, post, comb in expected
        )
        actual_loss.backward()
        expected_loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    for name, original in masters.items():
        if name in overrides:
            assert observed[name].dtype is torch.float32
            torch.testing.assert_close(observed[name], original, rtol=0, atol=0)
        else:
            assert observed[name].dtype is torch.bfloat16
            torch.testing.assert_close(observed[name], original.bfloat16(), rtol=0, atol=0)
    for (output, post, comb), (expected_output, expected_post, expected_comb) in zip(actual, expected, strict=True):
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(post, expected_post, rtol=0, atol=0)
        torch.testing.assert_close(comb, expected_comb, rtol=0, atol=0)
    torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=0, atol=0)
    reference_params = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        grad = parameter.grad.to_local()
        assert grad.dtype is torch.float32
        torch.testing.assert_close(grad, reference_params[name].grad.float(), rtol=0, atol=0)
