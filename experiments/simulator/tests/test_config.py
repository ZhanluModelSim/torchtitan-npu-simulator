# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torchtitan_simulator.cli import main
from torchtitan_simulator.config import SimulationConfig


@pytest.mark.parametrize("mtp", [0, 1], ids=["main-loss", "main-and-mtp"])
def test_cli_builds_native_dsv4_with_eager_meta_parameters(mtp, capsys):
    report = main(["--build-only", "--training.num-mtp-modules", str(mtp)])

    assert report["scope"] == "model_build_only"
    assert report["parameter_devices"] == ["meta"]
    assert report["mtp_layers"] == mtp
    assert report["num_layers"] == 4
    assert report["token_embedding_shape"] == [129280, 256]
    assert report["compile"] is False
    assert report["cuda_graphs"] is False
    assert report["spmd_backend"] == "spmd_types"
    assert '"model_build_only"' in capsys.readouterr().out


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"pipeline_parallel_degree": 2}, "PP is deferred"),
        ({"enable_ep_dispatch_fp8": True}, "FP8 communication"),
        ({"enable_fsdp_allgather_fp8": True}, "FP8 communication"),
    ],
    ids=["pp", "fp8-dispatch", "fp8-ag"],
)
def test_deferred_options_fail_before_model_construction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SimulationConfig(**kwargs).validate()


def test_plain_imports_leave_native_training_bindings_unchanged():
    import os
    import subprocess
    import sys

    script = """
import sys
import torch
import torch.distributed as dist
import torchtitan_npu
assert "torchtitan_simulator" not in sys.modules
bindings = (torch.Tensor.backward, torch.autocast, dist.all_reduce)
import torchtitan_simulator
assert bindings == (torch.Tensor.backward, torch.autocast, dist.all_reduce)
assert "torchtitan_simulator.runtime" not in sys.modules
assert not any(name.startswith("torchtitan_simulator.compat.") for name in sys.modules)
"""
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True, text=True)


def test_unverified_target_is_explicitly_rejected_by_cli():
    with pytest.raises(ValueError, match="A5 backend acceptance is pending"):
        main(["--build-only", "--simulation.target-npu-device-type", "a5"])


def test_logical_device_selection_does_not_inherit_host_a5_flags(monkeypatch):
    from importlib import import_module

    from torchtitan_simulator.config import build_meta_model, prepare_config
    from torchtitan_simulator.runtime import meta_runtime

    prepared = prepare_config(SimulationConfig(num_mtp_modules=1))
    owners = [
        import_module(name)
        for name in (
            "torchtitan_npu.override",
            "torchtitan_npu.override.deepseek_v4.mhc.ascendc",
            "torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc",
        )
    ]
    for owner in owners:
        monkeypatch.setattr(owner, "_IS_A5", True)
    with meta_runtime() as runtime:
        runtime.configure_model(prepared)
        assert all(owner._IS_A5 is False for owner in owners)
        model = build_meta_model(prepared)
        fused_pre = [module for module in model.modules() if hasattr(module, "_use_split_backend")]
        assert fused_pre and all(not module._use_split_backend for module in fused_pre)
        assert runtime.capabilities["target_npu_device_type"] == "non_a5"
    assert all(owner._IS_A5 is True for owner in owners)


@pytest.mark.parametrize("preset", ["flash", "pro", "pro_20t"])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("quantized", [False, True])
def test_precision_specific_presets_are_rejected_before_factory_import(preset, precision, quantized, monkeypatch):
    from torchtitan_simulator import config

    def unexpected_import(name):
        pytest.fail(f"Removed preset must fail before importing a model factory: {name}")

    monkeypatch.setattr(config, "import_module", unexpected_import)
    with pytest.raises(ValueError, match=f"was removed; use config='deepseek_v4_{preset}'"):
        config.prepare_config(
            SimulationConfig(config=f"deepseek_v4_{preset}_baseline_{precision}", enable_quantized_training=quantized)
        )


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_cli_rejects_removed_precision_specific_presets_with_migration_guidance(precision):
    with pytest.raises(ValueError, match="Control quantization explicitly"):
        main(["--build-only", "--config", f"deepseek_v4_flash_baseline_{precision}"])
