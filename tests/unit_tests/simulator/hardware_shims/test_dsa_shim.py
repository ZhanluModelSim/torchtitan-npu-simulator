# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan_npu.converters.kernels.dsa import DSAModelConfig
from torchtitan_npu.models.deepseek_v32 import deepseekv32_configs
from torchtitan_npu.models.deepseek_v32.model import DSV32_SDPA
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.dsa_converter import (
    apply_dsa_shims,
    unapply_dsa_shims,
)


class _FakeModelSpec:
    name = "deepseek_v32"


def _build_module() -> DSV32_SDPA:
    config = deepseekv32_configs["smoketest"]().layers[0].attention
    return DSV32_SDPA(config).to("meta")


def _bind_simulator_forward(module: DSV32_SDPA) -> None:
    model = nn.Sequential(module)
    apply_dsa_shims()
    converter = DSAModelConfig.model_converter(_FakeModelSpec())
    converter.convert(model)


def test_apply_dsa_shims_replaces_and_restores_converter():
    original = DSAModelConfig.model_converter
    try:
        apply_dsa_shims()
        assert DSAModelConfig.model_converter is not original
    finally:
        unapply_dsa_shims()
        assert DSAModelConfig.model_converter is original


def test_dsa_binding_preserves_module_identity_and_hooks():
    module = _build_module()
    hook = module.register_forward_hook(lambda target, args, output: None)
    module_id = id(module)
    hook_ids = set(module._forward_hooks)
    try:
        _bind_simulator_forward(module)
        _bind_simulator_forward(module)
        assert id(module) == module_id
        assert set(module._forward_hooks) == hook_ids
        assert module._simulator_dsa_shim_installed is True
    finally:
        hook.remove()
        unapply_dsa_shims()


def test_dsa_records_fused_forward_and_backward_contract():
    module = _build_module()
    _bind_simulator_forward(module)
    batch, heads, sequence = 1, 4, 8
    kv_rank = module.config.kv_lora_rank
    rope_dim = module.config.qk_rope_head_dim
    index_heads = module.config.index_n_heads
    index_dim = module.config.index_head_dim
    q = torch.empty(
        (batch, heads, sequence, kv_rank + rope_dim),
        device="meta",
        requires_grad=True,
    )
    k = torch.empty(
        (batch, 1, sequence, kv_rank + rope_dim),
        device="meta",
        requires_grad=True,
    )
    v = torch.empty(
        (batch, 1, sequence, kv_rank),
        device="meta",
        requires_grad=True,
    )
    q_indexer = torch.empty(
        (batch, sequence, index_heads, index_dim),
        device="meta",
        requires_grad=True,
    )
    k_indexer = torch.empty(
        (batch, sequence, 1, index_dim),
        device="meta",
        requires_grad=True,
    )
    weights = torch.empty(
        (batch, sequence, index_heads),
        device="meta",
        requires_grad=True,
    )
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    try:
        with capture:
            loss, output = module(
                q,
                k,
                v,
                scale=0.125,
                q_indexer=q_indexer,
                k_indexer=k_indexer,
                weights=weights,
                end_pos=sequence,
                index_topk=module.config.index_topk,
            )
            phase["value"] = "backward"
            (output.sum() + loss).backward()
    finally:
        unapply_dsa_shims()

    nodes = list(capture.build_nodes().values())
    raw_names = [node.annotations["raw_op_type"] for node in nodes]
    assert raw_names.count("npu_lightning_indexer") == 1
    assert raw_names.count("npu_sparse_flash_attention") == 1
    assert raw_names.count("npu_sparse_flash_attention_grad") == 1
    assert (
        raw_names.count("npu_sparse_lightning_indexer_grad_kl_loss") == 1
    )
    assert output.shape == (batch, heads, sequence, kv_rank)
    assert loss.shape == ()
    for tensor in (q, k, v, q_indexer, k_indexer, weights):
        assert tensor.grad is not None


def test_unapply_dsa_shims_is_idempotent():
    unapply_dsa_shims()
    unapply_dsa_shims()
