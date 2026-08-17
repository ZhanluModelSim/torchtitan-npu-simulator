# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.integration_tests import OverrideDefinitions


def build_deepseek_v4_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            override_args=[
                (
                    "--training.steps=100",
                    "--hf-assets-path", "tests/assets/deepseek_v3",
                    "--training.global-batch-size", "2",
                )
            ],
            test_descr="DeepSeek-V4 golden 1rank",
            test_name="dsv4_golden_1rank",
            ngpu=1,
            env_vars={"USE_GOLDEN": "1"},
        ),
        OverrideDefinitions(
            override_args=[
                (
                    "--training.steps=100",
                    "--parallelism.expert-parallel-degree", "2",
                    "--hf-assets-path", "tests/assets/deepseek_v3",
                    "--training.global-batch-size", "2",
                )
            ],
            test_descr="DeepSeek-V4 golden ep2 fsdp2",
            test_name="dsv4_golden_ep2_fsdp2",
            ngpu=2,
            env_vars={"USE_GOLDEN": "1"},
        ),
    ]
