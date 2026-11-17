# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU-resident optimizer state under ``--training.enable-cpu-offload``.

``swap_optimizer`` stays the single toggle: listed -> CPU-canonical optimizer
state (staged per step); absent -> the moments live and update on the compute
device with no per-step round trip.
"""

import pytest
import torch
import torch_npu

from torchtitan_npu.config.configs import MuonOptimizerProfile, OptimizerConfig, TrainingConfig
from torchtitan_npu.extensions.cpu_offload.cpu_offload_adamw import CpuOffloadAdamW
from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx

requires_npu = pytest.mark.skipif(
    not torch_npu.npu.is_available(), reason="NPU not available"
)


def _muon_factory_kwargs(*, enable_hsdp_replica_dedup: bool) -> dict[str, object]:
    config = OptimizerConfig(
        name="Muon",
        muon_enable_hsdp_replica_dedup=enable_hsdp_replica_dedup,
        _muon_profile=MuonOptimizerProfile(
            muon_pattern=r".*",
            optimizer_factory_kwargs={
                "DistMuon": {
                    "compute_sharding_by_fqn": {},
                    "bucket_configs": (),
                }
            },
        ),
    )
    config.materialize()
    return config.optimizer_factory_kwargs_by_name["DistMuon"]


def test_cpu_offload_muon_factory_consumes_disabled_replica_dedup(monkeypatch) -> None:
    from torchtitan_npu.extensions.cpu_offload import cpu_offload_muon

    captured: dict[str, object] = {}

    class FakeMuon:
        def __init__(self, _params, **kwargs) -> None:
            captured.update(kwargs)
            self._redistribution_runtime = object()
            self._bucket_plans = ()
            self._local_tensor_spec = object()

    class FakeLocalPrefetchRuntime:
        @staticmethod
        def from_runtime(_runtime, *, local_stream):
            captured["local_stream"] = local_stream
            return type(
                "FakeRuntime",
                (),
                {"reserve_buffers": lambda self, _plans, *, local_tensor_spec: None},
            )()

    monkeypatch.setattr(cpu_offload_muon, "CpuOffloadDistributedMuon", FakeMuon)
    monkeypatch.setattr(cpu_offload_muon, "LocalPrefetchRuntime", FakeLocalPrefetchRuntime)

    staging = type("FakeStaging", (), {"stream": object()})()
    optimizer = cpu_offload_muon.build_cpu_offload_distributed_muon(
        [],
        staging=staging,
        offload_states=False,
        **_muon_factory_kwargs(enable_hsdp_replica_dedup=False),
    )

    assert isinstance(optimizer, FakeMuon)
    assert "enable_hsdp_replica_dedup" not in captured


def test_cpu_offload_muon_factory_rejects_enabled_replica_dedup() -> None:
    from torchtitan_npu.extensions.cpu_offload.cpu_offload_muon import (
        build_cpu_offload_distributed_muon,
    )

    with pytest.raises(ValueError, match="not supported with CPU-offload Muon"):
        build_cpu_offload_distributed_muon(
            [],
            staging=None,  # type: ignore[arg-type]
            offload_states=False,
            **_muon_factory_kwargs(enable_hsdp_replica_dedup=True),
        )


def _step_sequence(
    offload_states: bool, grads: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, set[int], str]:
    torch.manual_seed(0)
    parameter = torch.nn.Parameter(torch.randn(4, 6))
    device = torch.device("npu", torch_npu.npu.current_device())
    staging = CpuStaging(device, owner="test-no-opt-state")
    optimizer = CpuOffloadAdamW(
        [parameter],
        lr=1e-2,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        staging=staging,
        offload_states=offload_states,
    )
    for gradient in grads:
        parameter.grad = gradient.clone()
        optimizer.step()
    optimizer._staging.wait()
    staged_kinds = {key[1] for key in optimizer._work_buffers}
    exp_avg = optimizer.state[parameter]["exp_avg"].detach()
    placement = exp_avg.device.type
    exp_avg_value = exp_avg.cpu().clone()
    staging.close()
    return parameter.detach().clone(), exp_avg_value, staged_kinds, placement


@requires_npu
def test_npu_resident_state_matches_cpu_canonical_updates() -> None:
    torch.manual_seed(1)
    grads = [torch.randn(4, 6) for _ in range(3)]
    cpu_parameter, cpu_exp_avg, _, _ = _step_sequence(True, grads)
    npu_parameter, npu_exp_avg, _, _ = _step_sequence(False, grads)
    torch.testing.assert_close(npu_parameter, cpu_parameter, rtol=0, atol=0)
    torch.testing.assert_close(npu_exp_avg, cpu_exp_avg, rtol=0, atol=0)


@requires_npu
def test_npu_resident_state_placement_and_staging_planes() -> None:
    torch.manual_seed(2)
    grads = [torch.randn(4, 6) for _ in range(2)]

    _, _, cpu_kinds, cpu_placement = _step_sequence(True, grads)
    assert cpu_placement == "cpu"
    assert cpu_kinds == {0, 1, 2, 3}

    _, _, npu_kinds, npu_placement = _step_sequence(False, grads)
    assert npu_placement == "npu"
    # Only the parameter (0) and gradient (1) planes are staged; the moments
    # update in place instead of round-tripping through work buffers.
    assert npu_kinds == {0, 1}


def test_enable_cpu_offload_defaults_to_npu_resident_state(monkeypatch) -> None:
    from dataclasses import replace

    from torchtitan_npu.extensions import trainer as trainer_module
    from torchtitan_npu.override.common.optimizer import (
        CpuOffloadNpuStateOptimizersContainer,
        CpuOffloadOptimizersContainer,
        swap_optimizer,
    )
    from torchtitan_npu.patches.torch_npu import cpu_dtensor_init

    class ReachedUpstreamError(Exception):
        pass

    captured: dict[str, object] = {}

    def capture_runtime(self, runtime) -> None:
        captured["optimizer"] = runtime.optimizer
        raise ReachedUpstreamError

    monkeypatch.setattr(trainer_module, "set_allow_hf32", lambda _: None)
    monkeypatch.setattr(cpu_dtensor_init, "install", lambda: None)
    monkeypatch.setattr(trainer_module.Trainer, "__init__", capture_runtime)

    config = TrainerEx.Config(training=TrainingConfig(enable_cpu_offload=True))
    # Config rebuilds keep the plain schema: the derive happens exactly once,
    # at trainer construction (regression: config.build() does replace(self),
    # which re-runs __post_init__ on the already-derived node).
    for cfg in (config, replace(config)):
        assert type(cfg.optimizer) is OptimizerConfig
        assert isinstance(cfg.optimizer.materialize(), type(None))

    with pytest.raises(ReachedUpstreamError):
        TrainerEx(config)
    runtime_optimizer = captured["optimizer"]
    assert isinstance(runtime_optimizer, CpuOffloadNpuStateOptimizersContainer.Config)
    assert runtime_optimizer._cpu_offload is True

    # Listing swap_optimizer re-derives to the CPU-canonical-state container.
    derived = swap_optimizer(runtime_optimizer)
    assert isinstance(derived, CpuOffloadOptimizersContainer.Config)
    assert not isinstance(derived, CpuOffloadNpuStateOptimizersContainer.Config)

    # Without the offload switch the optimizer schema stays untouched.
    plain = TrainerEx.Config()
    assert type(plain.optimizer) is OptimizerConfig


@requires_npu
def test_adamw_restore_continues_on_compute_device() -> None:
    device = torch.device("npu", torch_npu.npu.current_device())

    def build() -> tuple[CpuOffloadAdamW, torch.nn.Parameter]:
        torch.manual_seed(7)
        parameter = torch.nn.Parameter(torch.randn(4, 6))
        staging = CpuStaging(device, owner="test-restore")
        optimizer = CpuOffloadAdamW(
            [parameter], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
            staging=staging, offload_states=False,
        )
        return optimizer, parameter

    torch.manual_seed(11)
    grads = [torch.randn(4, 6) for _ in range(3)]

    continuous_optimizer, continuous_param = build()
    for gradient in grads:
        continuous_param.grad = gradient.clone()
        continuous_optimizer.step()
    continuous_optimizer._staging.wait()

    saved_at_two, restored_param, restored_optimizer = None, None, None
    restored_optimizer, restored_param = build()
    for gradient in grads[:2]:
        restored_param.grad = gradient.clone()
        restored_optimizer.step()
    restored_optimizer._staging.wait()
    saved = restored_optimizer.state_dict()

    fresh_optimizer, fresh_param = build()
    fresh_optimizer.load_state_dict(saved)
    # PyTorch's load cast moved the moments to the CPU parameter storage.
    fresh_state = fresh_optimizer.state[fresh_param]
    assert fresh_state["exp_avg"].device.type == "cpu"
    # The checkpoint also carries the stepped parameters.
    with torch.no_grad():
        fresh_param.copy_(restored_param)
    fresh_param.grad = grads[2].clone()
    fresh_optimizer.step()
    fresh_optimizer._staging.wait()

    torch.testing.assert_close(fresh_param.detach(), continuous_param.detach(), rtol=0, atol=0)
    state = fresh_optimizer.state[fresh_param]
    assert state["exp_avg"].device.type == "npu"
    assert state["exp_avg_sq"].device.type == "npu"


@requires_npu
def test_muon_checkpoint_roundtrip_steps_on_compute_device() -> None:
    """Save -> fresh optimizer -> real load_state_dict -> step again.

    Exercises the production restore chain: PyTorch's load cast sends the
    momentum to the CPU parameter storage, upstream's load post-hook
    recomputes plans and resets ``_first_step_validated``, and the next
    ``step()`` re-validates the momentum storage layout before the bucket
    pipeline runs.
    """
    import os
    import socket

    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, distribute_tensor
    from torchtitan.distributed.flex_shard.optimizer_reshard import BucketConfig, ComputeLayout, Owned

    from torchtitan_npu.extensions.cpu_offload.cpu_offload_muon import build_cpu_offload_distributed_muon

    if not dist.is_initialized():
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    mesh = init_device_mesh("cpu", mesh_shape=(1,), mesh_dim_names=("dp_shard",))
    device = torch.device("npu", torch_npu.npu.current_device())

    def make_parameter():
        torch.manual_seed(5)
        return torch.nn.Parameter(distribute_tensor(torch.randn(8, 4), mesh, [Replicate()]))

    def make_gradient():
        torch.manual_seed(9)
        return distribute_tensor(torch.randn(8, 4), mesh, [Replicate()])

    staging = CpuStaging(device, owner="test-muon-roundtrip")

    def build(parameter):
        return build_cpu_offload_distributed_muon(
            [{"params": [parameter], "param_names": ["weight"]}],
            staging=staging,
            compute_sharding_by_fqn={
                "weight": ComputeLayout(shardings_by_mesh_axis={"dp_shard": Owned()})
            },
            bucket_configs=[BucketConfig(patterns=("weight",), name="test")],
            offload_states=False,
            lr=1e-2,
            momentum=0.9,
        )

    parameter = make_parameter()
    optimizer = build(parameter)
    parameter.grad = make_gradient()
    optimizer.step()
    optimizer._staging.wait()
    momentum = optimizer.state[parameter]["momentum_buffer"]
    assert momentum.to_local().device.type == "npu"

    saved = optimizer.state_dict()

    restored_parameter = make_parameter()
    restored_optimizer = build(restored_parameter)
    with torch.no_grad():
        restored_parameter.copy_(parameter)
    restored_optimizer.load_state_dict(saved)
    # The loader cast the momentum to the CPU parameter storage and our
    # override moved it back; the post-hook reset first-step validation.
    momentum = restored_optimizer.state[restored_parameter]["momentum_buffer"]
    assert momentum.to_local().device.type == "npu"
    assert restored_optimizer._first_step_validated is False

    restored_parameter.grad = make_gradient()
    restored_optimizer.step()
    restored_optimizer._staging.wait()
    restored_parameter.grad = make_gradient()
    restored_optimizer.step()
    restored_optimizer._staging.wait()
    momentum = restored_optimizer.state[restored_parameter]["momentum_buffer"]
    assert momentum.to_local().device.type == "npu"
