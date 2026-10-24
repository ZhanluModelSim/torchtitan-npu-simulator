# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare the candidate queue against native full scans at every boundary."""

import pytest
from torchtitan_simulator.capture import storage_lifetimes
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate


@pytest.mark.parametrize(
    "options",
    [
        {"world_size": 4, "data_parallel_replicate_degree": 2},
        {"world_size": 4, "data_parallel_replicate_degree": 2, "activation_checkpoint_mode": "full"},
        {
            "world_size": 4,
            "data_parallel_replicate_degree": 2,
            "activation_checkpoint_mode": "selective",
            "enable_quantized_training": True,
        },
        {
            "world_size": 16,
            "logical_rank": 15,
            "data_parallel_replicate_degree": 2,
            "tensor_parallel_degree": 2,
            "context_parallel_degree": 2,
            "expert_parallel_degree": 2,
            "seq_len": 512,
        },
    ],
    ids=["hsdp", "hsdp-full-ac", "hsdp-selective-mxfp8", "tp-cp-ep-hsdp"],
)
def test_native_capture_releases_match_full_scan_at_every_boundary(monkeypatch, options):
    checks = []
    original = storage_lifetimes.StorageLifetimeTracker

    class CheckedTracker(original):
        def poll(self):
            active_before = set(self._active)
            expected = {key for key, ref in self._active.items() if ref.expired()}
            super().poll()
            actual = active_before.difference(self._active)
            assert actual == expected, "Candidate queue differs from the native storage release set"
            assert all(not ref.expired() for ref in self._active.values()), "Unobserved expired storage remains"
            checks.append(len(expected))

    monkeypatch.setattr(storage_lifetimes, "StorageLifetimeTracker", CheckedTracker)
    workload = simulate(SimulationConfig(num_mtp_modules=1, **options))
    validation = workload.iteration.schedule.annotations["capture_validation"]
    assert validation["peak_storage_unique"] and validation["graph_acyclic"]
    assert validation["gradient_parameters"] > 0
    assert len(checks) > 1000 and sum(checks) > 100
