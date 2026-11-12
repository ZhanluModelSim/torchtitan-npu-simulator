# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 recipe precision reaches native FSDP with and without Host Engram."""

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torchtitan.config import derive
from torchtitan.distributed import fsdp as upstream_fsdp

from torchtitan_npu.extensions.torchtitan.distributed import fsdp as npu_fsdp
from torchtitan_npu.models.deepseek_v4_1 import parallelize
from torchtitan_npu_recipes.models.deepseek_v4_1 import deepseek_v4_1_debugmodel_text
from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable


@pytest.mark.parametrize("engram_enabled", (False, True), ids=("decoder", "host-engram"))
def test_recipe_fp32_parameters_reach_every_shard_with_checkpoint_wrapping(
    monkeypatch, single_rank_spmd, engram_enabled
):
    config = deepseek_v4_1_debugmodel_text()
    if not engram_enabled:
        config = derive(config, type(config), engram_enabled=False)
    config.hf_assets_path = "tests/assets/deepseek_v3"
    config.model.update_from_config(config=config)
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        model = config.model.build()
    # One wrapped block and the remaining plain blocks exercise the same logical policy.
    model.layers["0"] = checkpoint_wrapper(model.layers["0"])
    expected_ignored = {module.weight for module in model.modules() if isinstance(module, HostEngramTable)}
    assert bool(expected_ignored) is engram_enabled
    shard_calls = []

    def record_shard(module, **kwargs):
        shard_calls.append((module, kwargs))

    # Resolve meshes, select the decoder helper, exclude actual Host tables, and
    # construct real MixedPrecisionPolicy objects. Only device sharding is recorded.
    monkeypatch.setattr(upstream_fsdp, "fully_shard", record_shard)
    monkeypatch.setattr(npu_fsdp, "fully_shard", record_shard)

    parallelize.apply_deepseek_v4_1_fsdp(
        model,
        parallelism_context=single_rank_spmd,
        training=config.training,
        parallelism=config.parallelism,
    )

    assert shard_calls and shard_calls[-1][0] is model
    policy = shard_calls[-1][1]["mp_policy"]
    assert policy.param_dtype_override_fn is not None
    assert policy.reduce_dtype is torch.float32
    assert all(kwargs["mp_policy"] is policy for _, kwargs in shard_calls)
    for _, kwargs in shard_calls:
        if engram_enabled:
            assert kwargs["ignored_params"] == expected_ignored
        else:
            assert "ignored_params" not in kwargs

    # Independent parameter inventory: two mHC coefficient sets and one sink per
    # decoder block. Router gates and ordinary projections retain the BF16 default.
    expected_fp32 = {
        f"layers.{layer_id}.{module}.{parameter}"
        for layer_id in range(len(model.layers))
        for module in ("hc_attn_pre", "hc_ffn_pre")
        for parameter in ("hc_fn", "hc_base", "hc_scale")
    }
    expected_fp32.update(f"layers.{layer_id}.attention.attn_sink" for layer_id in range(len(model.layers)))
    parameters = dict(model.named_parameters(remove_duplicate=False))
    assert any("_checkpoint_wrapped_module" in name for name in parameters)
    actual_fp32 = set()
    for name, parameter in parameters.items():
        if parameter in expected_ignored:
            continue
        canonical = ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module")
        expected_dtype = torch.float32 if canonical in expected_fp32 else torch.bfloat16
        resolved_dtype = policy._resolve_for_param(parameter).param_dtype
        assert resolved_dtype is expected_dtype, name
        if resolved_dtype is torch.float32:
            actual_fp32.add(canonical)
    assert actual_fp32 == expected_fp32
