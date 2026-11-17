# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

from tests.integration_tests import OverrideDefinitions

# DeepSeek-V4 NPU recipe from the example scripts.  Ascend attention and MHC
# overrides are only used on the non-golden path.
NPU_OVERRIDES = (
    "--override.imports",
    "torchtitan_npu.override.common.rms_norm.asc",
    "torchtitan_npu.override.common.rope.asc_complex",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata",
    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
    "torchtitan_npu.override.common.token_dispatcher.asc",
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
    extra_override_imports: tuple[str, ...] = (),
) -> OverrideDefinitions:
    """Build one DeepSeek-V4 integration case.

    ``use_golden`` selects the operator recipe and ``check_loss`` controls
    whether deterministic loss comparison is enabled. The switches stay
    independent so eager SMLA paths can be covered without deterministic loss
    checking.
    """
    recipe = GOLDEN_OVERRIDES if use_golden else NPU_OVERRIDES
    return OverrideDefinitions(
        override_args=[recipe + extra_override_imports + extra_args],
        test_descr=test_descr,
        test_name=test_name,
        ngpu=ngpu,
        use_golden=use_golden,
        check_loss=check_loss,
    )


def build_deepseek_v4_checkpoint_resume_test_list() -> list[OverrideDefinitions]:
    """Compare two resumed steps with uninterrupted FSDP2/EP2 training."""
    checkpoint_args = (
        "--checkpoint.enable",
        "--checkpoint.interval=2",
        "--checkpoint.no-last-save-model-only",
        "--checkpoint.keep-latest-k=0",
        "--hf-assets-path=tests/assets/deepseek_v3",
        "--training.global-batch-size=2",
        "--training.steps=4",
        "--parallelism.data-parallel-shard-degree=2",
        "--parallelism.expert-parallel-degree=2",
    )
    adamw_swap_args = (*checkpoint_args, "--optimizer.name=AdamW")
    adamw_swap_overrides = (
        *GOLDEN_OVERRIDES,
        "torchtitan_npu.override.common.optimizer.swap_optimizer",
    )
    checkpoint_case = OverrideDefinitions(
        override_args=[
            adamw_swap_overrides + adamw_swap_args,
            adamw_swap_overrides + adamw_swap_args + ("--checkpoint.load-step=2",),
        ],
        test_descr="DeepSeek-V4 AdamW NovaSwap checkpoint FSDP2 EP2 exact resume",
        test_name="dsv4_checkpoint_resume_ep2_fsdp2",
        expected_steps=((1, 2, 3, 4), (3, 4)),
        ngpu=2,
        use_golden=True,
        check_loss=False,
        check_resume=True,
    )
    lora_args = GOLDEN_OVERRIDES + checkpoint_args + ("--checkpoint.save-training-state", "--optimizer.name=Muon")
    lora_checkpoint_case = replace(
        checkpoint_case,
        test_name="dsv4_lora_resume_ep2_fsdp2",
        test_descr="DeepSeek-V4 LoRA checkpoint resume EP2/FSDP2",
        env_vars={"MODULE": "tests.integration_tests.lora_config", "CONFIG": "deepseek_v4_lora_training"},
        override_args=[lora_args, (*lora_args, "--checkpoint.load-step=2")],
    )
    return [checkpoint_case, lora_checkpoint_case]


def build_deepseek_v4_quantized_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            test_name="dsv4_mxfp4_communication_fsdp2",
            test_descr="MXFP4 QAT communication, two forward/backward/optimizer steps on A5",
            env_vars={"MODULE": "torchtitan_npu.models.deepseek_v4", "CONFIG": "deepseek_v4_debugmodel"},
            override_args=[
                (
                    *GOLDEN_OVERRIDES,
                    "--training.steps=2",
                    "--training.global-batch-size=2",
                    "--training.seq-len=256",
                    "--parallelism.data-parallel-shard-degree=2",
                    "--parallelism.expert-parallel-degree=1",
                    "--parallelism.fsdp-reshard-after-forward=always",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                    "--extension.quantization.enable-quantized-training",
                    "--extension.quantization.recipe=all_block_fp8",
                    "--extension.quantization.enable-mxfp4-qat",
                    "--extension.quantization.enable-fsdp-prequantize",
                    "--extension.quantization.fsdp-prequantize-dtype=mxfp4",
                )
            ],
            ngpu=2,
            use_golden=True,
            check_loss=False,
            expected_steps=((1, 2),),
        ),
        replace(
            build_deepseek_v4_checkpoint_resume_test_list()[1],
            test_name="dsv4_lora_block_fp8_ep2_fsdp2",
            test_descr="DeepSeek-V4 block-FP8 base and floating LoRA training EP2/FSDP2",
            env_vars={"MODULE": "tests.integration_tests.lora_config", "CONFIG": "deepseek_v4_lora_quantized_training"},
            override_args=[
                (
                    *GOLDEN_OVERRIDES,
                    "--training.steps=2",
                    "--training.global-batch-size=2",
                    "--optimizer.name=AdamW",
                    "--parallelism.data-parallel-shard-degree=2",
                    "--parallelism.expert-parallel-degree=2",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                )
            ],
            expected_steps=((1, 2),),
            check_resume=False,
        ),
    ]


def build_deepseek_v4_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            test_name="dsv4_anticipatory_recovery",
            test_descr="DeepSeek-V4 CLI loss spike, native checkpoint rollback and route replay",
            env_vars={
                "MODULE": "torchtitan_npu.models.deepseek_v4",
                "CONFIG": "deepseek_v4_debugmodel",
                "TRAIN_FILE": "tests.integration_tests.anticipatory_routing",
            },
            override_args=[(*GOLDEN_OVERRIDES,
                "--training.steps=10", "--training.seq-len=256", "--training.global-batch-size=1",
                "--parallelism.data-parallel-shard-degree=1", "--parallelism.expert-parallel-degree=1",
                "--hf-assets-path=tests/assets/deepseek_v3", "--debug.seed=42",
                "--checkpoint.enable", "--checkpoint.no-load-only", "--checkpoint.interval=2",
                "--checkpoint.keep-latest-k=0", "--checkpoint.no-last-save-model-only",
                "--anticipatory.enable", "--anticipatory.delay-steps=1", "--anticipatory.active-steps=2",
                "--anticipatory.max-rollbacks=1", "--anticipatory.detector.warmup-steps=4",
            )],
            ngpu=1,
            timeout=600,
            use_golden=True,
            check_loss=False,
        ),
        OverrideDefinitions(
            test_name="dsv4_lora_ep2_fsdp2",
            test_descr="DeepSeek-V4 LoRA A/B training, frozen base and PEFT export EP2/FSDP2",
            env_vars={"MODULE": "tests.integration_tests.lora_config", "CONFIG": "deepseek_v4_lora_training"},
            override_args=[(*GOLDEN_OVERRIDES,
                "--training.steps=2", "--training.global-batch-size=2", "--optimizer.name=Muon",
                "--parallelism.data-parallel-shard-degree=2", "--parallelism.expert-parallel-degree=2",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--checkpoint.enable", "--checkpoint.interval=10000",
            )],
            ngpu=2,
            use_golden=True,
            check_loss=False,
            check_resume=False,
            expected_steps=((1, 2),),
        ),
        _build_case(
            test_name="dsv4_golden_1rank",
            test_descr="DeepSeek-V4 golden 1rank",
            ngpu=1,
            extra_args=(
                "--training.steps=30",
                "--lr-scheduler.total-steps=100",
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
                "--training.steps=30",
                "--lr-scheduler.total-steps=100",
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
            test_descr="DeepSeek-V4 SMLA Muon cp2 ep2 fsdp2",
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
                "--optimizer.name=Muon",
            ),
            use_golden=False,
            check_loss=False,
        ),
        _build_case(
            test_name="dsv4_mtp_smla_cp2_headtail",
            test_descr="DeepSeek-V4 MTP SMLA Muon cp2 headtail",
            ngpu=2,
            extra_args=(
                "--training.steps=1",
                "--parallelism.context-parallel-degree=2",
                "--parallelism.context-parallel-load-balancer=headtail",
                "--parallelism.spmd-backend=spmd_types",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.global-batch-size=2",
                "--optimizer.name=Muon",
            ),
            use_golden=False,
            check_loss=False,
        ),
    ]
