# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 0.6B save→load smoke test — local assets only."""

import pytest

from tests.smoke_tests.save_load.base_save_load import run_save_load_test


@pytest.mark.smoke
def test_qwen3_save_load(tmp_path):
    run_save_load_test(
        tmp_path,
        module="torchtitan_npu.models.qwen3",
        config="debug_qwen3_06b_single_node",
        tokenizer_path="./tests/assets/tokenizer/qwen3-tokenizer",
        steps=1,
        seq_len=128,
        extra_args="--checkpoint.no-initial_load_in_hf --checkpoint.enable --checkpoint.last_save_model_only",
    )
