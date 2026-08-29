# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Minimal DeepSeek-V3.2 smoke definition.

The cases keep the NPU-owned DSV3.2 model separate from the upstream DSV3
definition.  The RoPE workaround avoids the unsupported complex64 cache gather,
while the CANN DSA overrides avoid upstream FlexAttention compilation on NPU.
"""

from tests.integration_tests import OverrideDefinitions

_DSA_OVERRIDES = (
    "torchtitan_npu.override.common.rope.workaround,"
    "torchtitan_npu.override.deepseek_v3_2.sparse_attn.asc_metadata,"
    "torchtitan_npu.override.deepseek_v3_2.sparse_attn.asc"
)


def build_deepseek_v3_2_test_list() -> list[OverrideDefinitions]:
    """Return deterministic DSV3.2 DSA smoke cases."""

    return [
        OverrideDefinitions(
            override_args=[
                (
                    "--training.steps=100",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                    "--training.global-batch-size=4",
                    "--training.seq-len=128",
                    "--optimizer.param-groups.0.optimizer-kwargs.lr=1e-5",
                    "--override.imports",
                    _DSA_OVERRIDES,
                )
            ],
            test_descr="DeepSeek-V3.2 DSA smoke 1rank",
            test_name="dsv3_2_dsa_1rank",
            ngpu=1,
            env_vars={
                "MODULE": "torchtitan_npu.models.deepseek_v3_2",
                "CONFIG": "deepseek_v3_2_debugmodel",
            },
        ),
        OverrideDefinitions(
            override_args=[
                (
                    "--training.steps=100",
                    "--parallelism.expert-parallel-degree=2",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                    "--training.local-batch-size=2",
                    "--training.global-batch-size=4",
                    "--training.seq-len=128",
                    "--optimizer.param-groups.0.optimizer-kwargs.lr=1e-5",
                    "--override.imports",
                    _DSA_OVERRIDES,
                )
            ],
            test_descr="DeepSeek-V3.2 DSA EP2/FSDP2 smoke",
            test_name="dsv3_2_dsa_ep2_fsdp2",
            ngpu=2,
            env_vars={
                "MODULE": "torchtitan_npu.models.deepseek_v3_2",
                "CONFIG": "deepseek_v3_2_debugmodel",
            },
        ),
        OverrideDefinitions(
            override_args=[
                (
                    "--training.steps=1",
                    "--parallelism.context-parallel-degree=2",
                    "--parallelism.context-parallel-load-balancer=headtail",
                    "--parallelism.spmd-backend=spmd_types",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                    "--training.local-batch-size=2",
                    "--training.global-batch-size=4",
                    "--training.seq-len=128",
                    "--optimizer.param-groups.0.optimizer-kwargs.lr=1e-5",
                    "--override.imports",
                    _DSA_OVERRIDES,
                )
            ],
            test_descr="DeepSeek-V3.2 DSA CP2 smoke",
            test_name="dsv3_2_dsa_cp2",
            ngpu=2,
            use_golden=False,
            check_loss=False,
            env_vars={
                "MODULE": "torchtitan_npu.models.deepseek_v3_2",
                "CONFIG": "deepseek_v3_2_debugmodel",
            },
        ),
    ]
