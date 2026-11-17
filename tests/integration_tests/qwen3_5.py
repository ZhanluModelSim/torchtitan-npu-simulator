# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.integration_tests import OverrideDefinitions

# The Qwen3.5 launcher supplies the tokenizer and cc12m-test defaults.  Keep
# only the common metrics switches here so the suite exercises that launcher
# exactly as users do.
QWEN3_5_TRAIN_ARGS = (
    "--metrics.enable_tensorboard",
    "--metrics.log_freq=1",
    "--metrics.save_tb_folder=tb",
    "--training.disable-cuda-graphs",
)


def _build_case(
    *,
    test_name: str,
    test_descr: str,
    ngpu: int,
    extra_args: tuple[str, ...],
    disabled: bool = False,
) -> OverrideDefinitions:
    return OverrideDefinitions(
        override_args=[extra_args],
        test_descr=test_descr,
        test_name=test_name,
        ngpu=ngpu,
        disabled=disabled,
        env_vars={
            "MODULE": "torchtitan_npu.models.qwen3_5",
            "CONFIG": "qwen35_debugmodel",
        },
        use_golden=False,
        check_loss=False,
        train_script="examples/qwen3_5/qwen3_5_debugmodel_1p_cpt_512.sh",
        train_args=QWEN3_5_TRAIN_ARGS,
    )


def build_qwen3_5_test_list() -> list[OverrideDefinitions]:
    """Return the runnable Qwen3.5 NPU smoke case.

    The multimodal stack (torchvision and the torchtitan tokenizer and
    cc12m-test assets) is installed by .ci/smoke_test.sh in the smoke
    runner environment.
    """
    common = (
        "--training.steps=1",
        "--training.local-batch-size=1",
        "--training.global-batch-size=1",
        "--training.seq-len=512",
    )
    return [
        _build_case(
            test_name="qwen3_5_debugmodel_1rank",
            test_descr="Qwen3.5 debug model 1rank",
            ngpu=1,
            extra_args=common,
        ),
    ]
