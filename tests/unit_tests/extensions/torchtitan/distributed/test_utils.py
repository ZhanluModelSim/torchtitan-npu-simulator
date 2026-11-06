# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Distributed-init compatibility without creating real process groups."""

import functools
import importlib.util
import inspect
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.config as dist_config
import torchtitan.distributed.utils as upstream_utils
from torchtitan.config import CommConfig
from torchtitan.distributed.parallelism_context import DistributedTopology

_HAS_PER_EDGE_P2P = hasattr(dist_config, "pipeline_per_edge_p2p")
requires_legacy_runtime = pytest.mark.skipif(_HAS_PER_EDGE_P2P, reason="runtime does not need the PP=1 fallback")


@pytest.fixture
def init_extension(monkeypatch):
    # Preserve the live binding even when the complete plugin was imported by
    # another test. Loading this definition installs the real apply() hook.
    monkeypatch.setattr(upstream_utils, "init_distributed", upstream_utils.init_distributed)
    path = Path(__file__).resolve().parents[5] / "torchtitan_npu/extensions/torchtitan/distributed/utils.py"
    spec = importlib.util.spec_from_file_location("_npu_distributed_init_definition", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    multithreading = torch.autograd.is_multithreading_enabled()
    spec.loader.exec_module(module)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(upstream_utils, "enable_fp32_matmul_emulation_with_bf16x9", lambda: None)
    try:
        yield module
    finally:
        torch.autograd.set_multithreading_enabled(multithreading)


@pytest.fixture
def pg_calls(monkeypatch):
    calls = []
    original = dist.init_process_group

    @functools.wraps(original)
    def record_pg(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(dist, "init_process_group", record_pg)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    return calls


@requires_legacy_runtime
@pytest.mark.parametrize("cpu_backend,trace_size", [(False, 0), (True, 64)], ids=["hccl", "hccl-and-gloo-traces"])
def test_pp1_fallback_preserves_backend_timeout_and_trace_environment(
    init_extension, pg_calls, monkeypatch, tmp_path, cpu_backend, trace_size
):
    monkeypatch.setattr(init_extension, "device_type", "npu")
    monkeypatch.setitem(dist.Backend.default_device_backend_map, "npu", "hccl")
    names = (
        "TORCH_NCCL_ASYNC_ERROR_HANDLING",
        "TORCH_FR_BUFFER_SIZE",
        "TORCH_NCCL_DUMP_ON_TIMEOUT",
        "TORCH_FR_DUMP_TEMP_FILE",
    )
    for name in names:
        monkeypatch.setenv(name, "preexisting")
    config = CommConfig(
        init_timeout_seconds=37,
        trace_buf_size=trace_size,
        save_traces_folder="trace-data",
        save_traces_file_prefix="worker_",
    )

    topology = upstream_utils.init_distributed(config, cpu_backend, str(tmp_path))

    assert upstream_utils.init_distributed is init_extension.init_distributed
    expected = {"backend": "npu:hccl,cpu:gloo" if cpu_backend else "hccl", "timeout": timedelta(seconds=37)}
    if "_ranks" in inspect.signature(dist.init_process_group).parameters:
        expected["_ranks"] = []
    assert pg_calls == [((), expected)]
    assert topology == DistributedTopology(world_size=4)
    assert torch.autograd.is_multithreading_enabled() is False
    assert os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] == "3"
    assert os.environ["TORCH_FR_BUFFER_SIZE"] == str(trace_size)
    if trace_size:
        assert os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
        assert os.environ["TORCH_FR_DUMP_TEMP_FILE"] == str(tmp_path / "trace-data/worker_")
        assert (tmp_path / "trace-data").is_dir()
    else:
        assert os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "preexisting"
        assert os.environ["TORCH_FR_DUMP_TEMP_FILE"] == "preexisting"
        assert not (tmp_path / "trace-data").exists()


def test_engine_consumer_delegates_already_initialized_runtime(init_extension, pg_calls, monkeypatch):
    from torchtitan.training_engine import dist_utils as engine_utils

    delegated = []
    original = init_extension._original_init_distributed

    def record_upstream(*args, **kwargs):
        delegated.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(init_extension, "_original_init_distributed", record_upstream)
    config = CommConfig()
    before = torch.autograd.is_multithreading_enabled()

    topology = engine_utils.init_distributed(config, True, "output", None, pipeline_parallel_degree=1)

    assert engine_utils.init_distributed is init_extension.init_distributed
    assert delegated == [((config, True, "output", None), {"pipeline_parallel_degree": 1})]
    assert topology.world_size == 4
    assert pg_calls == []
    assert torch.autograd.is_multithreading_enabled() is before


@pytest.mark.skipif(not _HAS_PER_EDGE_P2P, reason="runtime has no native pipeline_per_edge_p2p capability")
def test_supported_runtime_delegates_all_arguments(init_extension, pg_calls, monkeypatch):
    delegated = []
    expected = DistributedTopology(world_size=8)

    def record_upstream(*args, **kwargs):
        delegated.append((args, kwargs))
        return expected

    monkeypatch.setattr(init_extension, "_original_init_distributed", record_upstream)
    config = CommConfig()
    ranks = [0, 2]

    result = upstream_utils.init_distributed(config, True, "output", ranks, pipeline_parallel_degree=2)

    assert result is expected
    assert delegated == [((config, True, "output", ranks), {"pipeline_parallel_degree": 2})]
    assert pg_calls == []


@requires_legacy_runtime
@pytest.mark.parametrize("backend,pp", [("default", 2), ("real_pp_fake_spmd", 1)], ids=["pp2", "real-pp-fake-spmd"])
def test_legacy_runtime_rejects_pipeline_before_process_group_creation(init_extension, pg_calls, backend, pp):
    before = torch.autograd.is_multithreading_enabled()

    with pytest.raises(NotImplementedError, match="pipeline_per_edge_p2p"):
        upstream_utils.init_distributed(CommConfig(backend=backend), pipeline_parallel_degree=pp)

    assert pg_calls == []
    assert torch.autograd.is_multithreading_enabled() is before


@requires_legacy_runtime
def test_fake_pp1_initializes_logical_world_without_real_pg(init_extension, pg_calls, monkeypatch):
    calls = []
    monkeypatch.setenv("NGPU", "4")
    monkeypatch.delenv("FAKE_PP_RANK", raising=False)
    monkeypatch.setattr(upstream_utils, "init_fake_mode", lambda world_size, *, rank: calls.append((world_size, rank)))

    topology = upstream_utils.init_distributed(CommConfig(backend="fake"))

    assert calls == [(4, 0)]
    assert topology.world_size == 4
    assert pg_calls == []
