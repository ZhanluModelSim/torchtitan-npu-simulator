# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from scripts.parse_profiling_data import _find_ascend_pt_dirs

pytestmark = pytest.mark.tooling


def test_find_ascend_pt_dirs_accepts_single_directory(tmp_path):
    trace_dir = tmp_path / "worker_0_ascend_pt"
    trace_dir.mkdir()

    assert _find_ascend_pt_dirs(str(trace_dir)) == [str(trace_dir)]


def test_find_ascend_pt_dirs_scans_flat_and_nested_layouts(tmp_path):
    flat = tmp_path / "worker_0_ascend_pt"
    nested = tmp_path / "profiling_data" / "worker_1_ascend_pt"
    flat.mkdir()
    nested.mkdir(parents=True)

    assert _find_ascend_pt_dirs(str(tmp_path)) == sorted((str(flat), str(nested)))


def test_find_ascend_pt_dirs_returns_empty_for_missing_traces(tmp_path):
    assert _find_ascend_pt_dirs(str(tmp_path)) == []
