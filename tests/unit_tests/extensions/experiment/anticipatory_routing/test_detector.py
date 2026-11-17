# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch

from torchtitan_npu.extensions.experiment.anticipatory_routing.cache import resolve_store_dtype
from torchtitan_npu.extensions.experiment.anticipatory_routing.config import (
    AnticipatoryRoutingConfig,
    validate_anticipatory_config,
)
from torchtitan_npu.extensions.experiment.anticipatory_routing.detector import LossSpikeDetector


def observe(detector, step, loss):
    detector.record_optimizer_update()
    return detector.observe(step, loss)


def test_sustained_level_drift_updates_after_freeze_window():
    detector = LossSpikeDetector(LossSpikeDetector.Config(warmup_steps=4, onset_lookback=4))
    for step, loss in enumerate((2.4, 2.6, 2.4, 2.6), 1):
        assert observe(detector, step, loss) is None
    # Jump to a fixed higher plateau, initially four sigma above the baseline.
    plateau = detector.predicted_loss + 4 * detector.sigma
    for step in range(5, 17):
        count = detector._count
        assert observe(detector, step, plateau) is None
        assert detector._count == count + int(step > 8)
    assert observe(detector, 17, detector.predicted_loss) is None
    assert detector._frozen_steps == 0


def test_constant_loss_roundoff_does_not_trigger():
    detector = LossSpikeDetector(LossSpikeDetector.Config())
    for step in range(1, 101):
        assert observe(detector, step, 2.5) is None
    assert detector.sigma == 0
    assert observe(detector, 101, 2.5 + 1e-9) is None
    assert detector._recent[-1][1] == pytest.approx(4e-4, rel=1e-6)
    assert observe(detector, 102, 3.0) is not None


def test_cooldown_counts_actual_updates_across_rollback_and_recovery():
    detector = LossSpikeDetector(LossSpikeDetector.Config(warmup_steps=2, cooldown_steps=5))
    observe(detector, 998, 2.5)
    observe(detector, 999, 2.5)
    assert observe(detector, 1000, 3.0) == 1000
    detector.reset()
    # ACTIVE/DRAIN updates count even though they are not detector observations.
    for _ in range(2):
        detector.record_optimizer_update()
    detector.reset()
    observe(detector, 903, 2.5)
    observe(detector, 904, 2.5)
    assert observe(detector, 905, 3.0) is None  # Exactly five updates: still cooling down.
    assert observe(detector, 906, 3.0) == 905


@pytest.mark.parametrize("field", ["z_threshold", "onset_z_threshold"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_thresholds_must_be_finite(field, value):
    with pytest.raises(ValueError, match=field):
        LossSpikeDetector.Config(**{field: value})


def test_detection_only_skips_checkpoint_requirements_but_rejects_pp():
    config = SimpleNamespace(
        anticipatory=AnticipatoryRoutingConfig(enable=True, max_rollbacks=0),
        parallelism=SimpleNamespace(pipeline_parallel_degree=1),
        debug=SimpleNamespace(moe_force_load_balance=False),
        checkpoint=SimpleNamespace(enable=False, load_only=True, exclude_from_loading=["optimizer"]),
    )
    validate_anticipatory_config(config)
    config.anticipatory.max_rollbacks = 1
    with pytest.raises(ValueError, match="checkpoint saving"):
        validate_anticipatory_config(config)
    config.anticipatory.max_rollbacks = 0
    config.parallelism.pipeline_parallel_degree = 2
    with pytest.raises(ValueError, match="pipeline_parallel_degree"):
        validate_anticipatory_config(config)


@pytest.mark.parametrize("num_experts,dtype", [(32768, torch.int16), (32769, torch.int32), (2**31 + 1, torch.int64)])
def test_auto_index_dtype_boundaries(num_experts, dtype):
    assert resolve_store_dtype("auto", num_experts) is dtype
