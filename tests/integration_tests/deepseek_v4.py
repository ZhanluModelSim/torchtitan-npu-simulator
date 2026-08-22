# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from tests.integration_tests import OverrideDefinitions

# DeepSeek-V4 NPU recipe from the example scripts.  Ascend attention and MHC
# overrides are only used on the non-golden path.
NPU_OVERRIDES = (
    "--override.imports",
    "torchtitan_npu.override.common.rms_norm.asc",
    "torchtitan_npu.override.common.rope.asc_complex",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
    "torchtitan_npu.override.common.token_dispatcher.npu_all_to_all_token_dispatcher",
)

# DeepSeek-V4 golden reference recipe. Golden uses the reference MoE token
# dispatcher path so the checked-in loss baselines stay independent of the NPU
# fused token-dispatcher implementation.
GOLDEN_OVERRIDES = (
    "--override.imports",
    "torchtitan_npu.override.common.rope.workaround",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.golden",
)


def _build_case(
    *,
    test_name: str,
    test_descr: str,
    ngpu: int,
    extra_args: tuple[str, ...],
    use_golden: bool = True,
    check_loss: bool = True,
) -> OverrideDefinitions:
    """Build one DeepSeek-V4 integration case.

    ``use_golden`` selects the operator recipe and ``check_loss`` controls
    whether deterministic loss comparison is enabled. The switches stay
    independent so eager SMLA paths can be covered without deterministic loss
    checking.
    """
    recipe = GOLDEN_OVERRIDES if use_golden else NPU_OVERRIDES
    return OverrideDefinitions(
        override_args=[recipe + extra_args],
        test_descr=test_descr,
        test_name=test_name,
        ngpu=ngpu,
        use_golden=use_golden,
        check_loss=check_loss,
    )


def build_deepseek_v4_test_list() -> list[OverrideDefinitions]:
    return [
        _build_case(
            test_name="dsv4_golden_1rank",
            test_descr="DeepSeek-V4 golden 1rank",
            ngpu=1,
            extra_args=(
                "--training.steps=100",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.global-batch-size=2",
            ),
            use_golden=True,
            check_loss=True,
        ),
        _build_case(
            test_name="dsv4_golden_ep2_fsdp2",
            test_descr="DeepSeek-V4 golden ep2 fsdp2",
            ngpu=2,
            extra_args=(
                "--training.steps=100",
                "--parallelism.expert-parallel-degree=2",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.global-batch-size=2",
            ),
            use_golden=True,
            check_loss=True,
        ),
        _build_case(
            test_name="dsv4_smla_1rank_aot_eager",
            test_descr="DeepSeek-V4 SMLA 1rank aot_eager",
            ngpu=1,
            extra_args=(
                "--training.steps=1",
                "--compile.enable",
                "--compile.backend=aot_eager",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.global-batch-size=2",
            ),
            use_golden=False,
            check_loss=False,
        ),
        _build_case(
            test_name="dsv4_smla_ep2_fsdp2",
            test_descr="DeepSeek-V4 SMLA ep2 fsdp2",
            ngpu=2,
            extra_args=(
                "--training.steps=1",
                "--parallelism.expert-parallel-degree=2",
                "--compile.enable",
                "--compile.backend=aot_eager",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.global-batch-size=2",
            ),
            use_golden=False,
            check_loss=False,
        ),
        _build_case(
            test_name="dsv4_smla_cp2_ep2_fsdp2",
            test_descr="DeepSeek-V4 SMLA cp2 ep2 fsdp2",
            ngpu=4,
            extra_args=(
                "--training.steps=1",
                "--parallelism.expert-parallel-degree=2",
                "--parallelism.context-parallel-degree=2",
                "--parallelism.spmd-backend=spmd_types",
                "--compile.enable",
                "--compile.backend=aot_eager",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.global-batch-size=2",
            ),
            use_golden=False,
            check_loss=False,
        ),
        _build_case(
            test_name="dsv4_smla_cp2",
            test_descr="DeepSeek-V4 SMLA cp2",
            ngpu=2,
            extra_args=(
                "--training.steps=1",
                "--parallelism.context-parallel-degree=2",
                "--parallelism.spmd-backend=spmd_types",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.global-batch-size=2",
            ),
            use_golden=False,
            check_loss=False,
        ),
    ]
