# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

from torchtitan_simulator.memory.estimator import _build_timeline, _phase_peak_active_bytes
from torchtitan_simulator.memory.records import TensorLifetime


def test_phase_peaks_follow_each_chunk_instead_of_enclosing_other_phases():
    phases = [(0, "forward"), (2, "backward"), (4, "forward"), (6, "backward"), (8, "optimizer")]
    events = [SimpleNamespace(seq_idx=seq, phase=phase) for seq, phase in phases]
    points = [
        (0, "forward", 10),
        (2, "backward", 30),
        (2, "backward", 20),
        (4, "forward", 25),
        (4, "forward", 5),
        (6, "backward", 20),
        (6, "backward", 15),
        (8, "optimizer", 115),
        (8, "optimizer", 100),
        (9, "optimizer", 0),
    ]
    timeline = [SimpleNamespace(seq_idx=seq, phase=phase, active_bytes_after=active) for seq, phase, active in points]

    peaks = _phase_peak_active_bytes(timeline, events)

    assert peaks == {"forward": 25, "backward": 30, "optimizer": 115}


def test_end_of_step_release_uses_execution_phase_instead_of_producer_phase():
    lifetime = TensorLifetime(
        tensor_id="saved",
        kind="activation",
        num_bytes=16,
        birth_seq=0,
        death_seq=9,
        producer_op=1,
        producer_phase="forward",
        consumer_phases=["forward"],
    )

    timeline = _build_timeline([lifetime], {0: "forward", 8: "optimizer"})

    assert timeline[-1].phase == "optimizer"
