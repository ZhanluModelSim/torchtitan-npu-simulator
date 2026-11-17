# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise checkpoint metric comparison using real TensorBoard event files."""

from itertools import product

import pytest
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from tests.integration_tests.loss_compare import TB_LOSS_TAG, compare_checkpoint_metrics

pytestmark = pytest.mark.tooling
EXPECTED_STEPS = ((1, 2, 3, 4), (3, 4))


def _write_phase(root, phase, *, changed_tag=None, changed_value=None, steps=None):
    writer = EventFileWriter(str(root / f"tb_phase_{phase}" / "rank_0"))
    phase_steps = EXPECTED_STEPS[phase] if steps is None else steps
    try:
        for step, tag in product(phase_steps, (TB_LOSS_TAG, "grad_norm")):
            value = float(step) if tag == TB_LOSS_TAG else step / 8
            if tag == changed_tag and step == 3:
                value = changed_value
            summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
            writer.add_event(Event(wall_time=float(step), step=step, summary=summary))
    finally:
        writer.close()


def test_checkpoint_metrics_match_uninterrupted_steps(tmp_path):
    _write_phase(tmp_path, 0)
    _write_phase(tmp_path, 1)

    compare_checkpoint_metrics(tmp_path, EXPECTED_STEPS)


@pytest.mark.parametrize("tag", [TB_LOSS_TAG, "grad_norm"], ids=["loss", "grad_norm"])
def test_checkpoint_metrics_reject_single_scalar_difference(tmp_path, tag):
    _write_phase(tmp_path, 0)
    original = 3.0 if tag == TB_LOSS_TAG else 3 / 8
    _write_phase(tmp_path, 1, changed_tag=tag, changed_value=original + 2**-20)

    with pytest.raises(AssertionError, match="mismatch at step 3"):
        compare_checkpoint_metrics(tmp_path, EXPECTED_STEPS)


@pytest.mark.parametrize("steps", [(3,), (1, 2, 3, 4)], ids=["incomplete_resume", "restarted_training"])
def test_checkpoint_metrics_reject_wrong_resume_steps(tmp_path, steps):
    _write_phase(tmp_path, 0)
    _write_phase(tmp_path, 1, steps=steps)

    with pytest.raises(ValueError, match="expected steps"):
        compare_checkpoint_metrics(tmp_path, EXPECTED_STEPS)


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "infinity"])
def test_checkpoint_metrics_reject_nonfinite_grad_norm(tmp_path, value):
    _write_phase(tmp_path, 0)
    _write_phase(tmp_path, 1, changed_tag="grad_norm", changed_value=value)

    with pytest.raises(ValueError, match="non-finite value"):
        compare_checkpoint_metrics(tmp_path, EXPECTED_STEPS)


def test_checkpoint_metrics_reject_duplicate_steps(tmp_path):
    _write_phase(tmp_path, 0)
    _write_phase(tmp_path, 1, steps=(3, 3, 4))

    with pytest.raises(ValueError, match="duplicate step"):
        compare_checkpoint_metrics(tmp_path, EXPECTED_STEPS)
