# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import replace

from tests.integration_tests import OverrideDefinitions

# DeepSeek-V4 NPU recipe from the example scripts.  Ascend attention and MHC
# overrides are only used on the fused path.
NPU_OVERRIDES = (
    "--override.imports",
    "torchtitan_npu.override.common.rms_norm.asc",
    "torchtitan_npu.override.common.rope.asc_complex",
    "torchtitan_npu.override.deepseek_v4.lightning_indexer.asc",
    'torchtitan_npu.override.deepseek_v4.sparse_attn.asc={"num_heads":16,"index_num_heads":8}',
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre",
    "torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post",
    "torchtitan_npu.override.common.token_dispatcher.asc",
)

# DeepSeek-V4 reference recipe. It runs the model-dir default (attn-gym's
# reference attention and the reference MoE token dispatcher) so the checked-in
# loss baselines stay independent of the NPU fused implementations.
GOLDEN_OVERRIDES = (
    "--override.imports",
    "torchtitan_npu.override.common.rope.workaround",
)

_LORA_ROPE_OVERRIDES = ("--override.imports", "tests.integration_tests.lora_config.lora_rope_workaround")

# The V4 Grain recipe already points at tests/assets/c4_test/data.json.
V4_TRAIN_ARGS = (
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
    use_golden: bool = True,
    check_loss: bool = True,
    extra_override_imports: tuple[str, ...] = (),
    env_vars: dict[str, str] | None = None,
    disabled: bool = False,
    use_cp_attention: bool = False,
) -> OverrideDefinitions:
    """Build one DeepSeek-V4 integration case.

    ``use_golden`` selects the operator recipe and ``check_loss`` controls
    whether deterministic loss comparison is enabled. The switches stay
    independent so eager SMLA paths can be covered without deterministic loss
    checking.
    """
    recipe = GOLDEN_OVERRIDES if use_golden else NPU_OVERRIDES
    if use_cp_attention:
        # The CP recipe selects the fused CP backend before Trainer config validation.
        recipe = tuple(
            arg for arg in recipe if not arg.startswith("torchtitan_npu.override.deepseek_v4.sparse_attn.asc=")
        )
    return OverrideDefinitions(
        override_args=[recipe + extra_override_imports + extra_args],
        train_args=V4_TRAIN_ARGS,
        test_descr=test_descr,
        test_name=test_name,
        ngpu=ngpu,
        use_golden=use_golden,
        check_loss=check_loss,
        env_vars=env_vars,
        disabled=disabled,
    )


def build_deepseek_v4_checkpoint_resume_test_list() -> list[OverrideDefinitions]:
    """Compare two resumed steps with uninterrupted FSDP2/EP2 training."""
    checkpoint_args = (
        "--checkpointer.interval=2",
        "--checkpointer.no-last-save-model-only",
        "--checkpointer.keep-latest-k=0",
        "--hf-assets-path=tests/assets/deepseek_v3",
        "--training.num-tokens-per-train-step=4096",
        "--training.steps=4",
        "--parallelism.data-parallel-shard-degree=2",
        "--parallelism.expert-parallel-degree=2",
    )
    adamw_swap_args = (*checkpoint_args, "--optim.optimizer.name=AdamW")
    adamw_swap_overrides = (
        *GOLDEN_OVERRIDES,
        "torchtitan_npu.override.common.optimizer.swap_optimizer",
    )
    checkpoint_case = OverrideDefinitions(
        train_args=V4_TRAIN_ARGS,
        env_vars={"MODULE": "tests.integration_tests.optimizer_config", "CONFIG": "deepseek_v4_checkpoint"},
        override_args=[
            adamw_swap_overrides + adamw_swap_args,
            adamw_swap_overrides + adamw_swap_args + ("--checkpointer.load-step=2",),
        ],
        test_descr="DeepSeek-V4 AdamW NovaSwap checkpoint FSDP2 EP2 exact resume",
        test_name="dsv4_checkpoint_resume_ep2_fsdp2",
        expected_steps=((1, 2, 3, 4), (3, 4)),
        ngpu=2,
        use_golden=True,
        check_loss=False,
        check_resume=True,
    )
    lora_args = (
        *_LORA_ROPE_OVERRIDES,
        *checkpoint_args,
        "--checkpointer.save-training-state",
        "--optim.optimizer.name=Muon",
    )
    lora_checkpoint_case = replace(
        checkpoint_case,
        test_name="dsv4_lora_resume_ep2_fsdp2",
        test_descr="DeepSeek-V4 LoRA checkpoint resume EP2/FSDP2",
        env_vars={"MODULE": "tests.integration_tests.lora_config", "CONFIG": "deepseek_v4_lora_training"},
        override_args=[lora_args, (*lora_args, "--checkpointer.load-step=2")],
    )
    return [checkpoint_case, lora_checkpoint_case]


def build_deepseek_v4_quantized_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            train_args=V4_TRAIN_ARGS,
            test_name="dsv4_mxfp4_communication_fsdp2",
            test_descr="MXFP4 QAT communication, two forward/backward/optimizer steps on A5",
            env_vars={"MODULE": "torchtitan_npu.models.deepseek_v4", "CONFIG": "deepseek_v4_debugmodel"},
            override_args=[
                (
                    *GOLDEN_OVERRIDES,
                    "--training.steps=2",
                    "--training.num-tokens-per-microbatch-per-dp-rank=256",
                    "--training.num-tokens-per-train-step=512",
                    "--training.max-context-length=256",
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
                    *_LORA_ROPE_OVERRIDES,
                    "--training.steps=2",
                    "--training.num-tokens-per-train-step=4096",
                    "--optim.optimizer.name=AdamW",
                    "--parallelism.data-parallel-shard-degree=2",
                    "--parallelism.expert-parallel-degree=2",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                )
            ],
            expected_steps=((1, 2),),
            check_resume=False,
        ),
    ]


def _build_grain_sft_case() -> OverrideDefinitions:
    """Opt in with official V4 assets and a short, supervised JSONL corpus."""
    assets = os.environ.get("DSV4_SFT_ASSETS", "")
    dataset = os.environ.get("DSV4_SFT_DATASET", "")
    return OverrideDefinitions(
        test_name="dsv4_grain_sft",
        test_descr="DeepSeek-V4 native Grain SFT: two eager training steps",
        disabled=not (assets and dataset),
        ngpu=1,
        timeout=600,
        env_vars={
            "MODULE": "torchtitan_npu.models.deepseek_v4",
            "CONFIG": "deepseek_v4_debugmodel_sft",
            "DSV4_SFT_DATASET": dataset,
        },
        train_args=("--metrics.enable_tensorboard", "--metrics.log_freq=1", "--training.disable-cuda-graphs"),
        override_args=[
            (
                *GOLDEN_OVERRIDES,
                f"--hf-assets-path={assets}",
                "--training.steps=2",
                "--training.max-context-length=512",
                "--training.num-tokens-per-microbatch-per-dp-rank=512",
                "--training.num-tokens-per-train-step=512",
                "--parallelism.data-parallel-shard-degree=1",
                "--parallelism.expert-parallel-degree=1",
            )
        ],
        expected_steps=((1, 2),),
        use_golden=True,
        # External assets/corpus have no fixed golden loss. This checks completion only.
        check_loss=False,
    )


def build_deepseek_v4_test_list() -> list[OverrideDefinitions]:
    return [
        _build_grain_sft_case(),
        OverrideDefinitions(
            train_args=V4_TRAIN_ARGS,
            test_name="dsv4_anticipatory_recovery",
            test_descr="DeepSeek-V4 CLI loss spike, native checkpoint rollback and route replay",
            env_vars={
                "MODULE": "tests.integration_tests.optimizer_config",
                "CONFIG": "deepseek_v4_checkpoint",
                "TRAIN_FILE": "tests.integration_tests.anticipatory_routing",
            },
            override_args=[
                (
                    *GOLDEN_OVERRIDES,
                    "--training.steps=10",
                    "--training.num-tokens-per-microbatch-per-dp-rank=256",
                    "--training.num-tokens-per-train-step=256",
                    "--training.max-context-length=256",
                    "--parallelism.data-parallel-shard-degree=1",
                    "--parallelism.expert-parallel-degree=1",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                    "--debug.seed=42",
                    "--checkpointer.no-load-only",
                    "--checkpointer.interval=2",
                    "--checkpointer.keep-latest-k=0",
                    "--checkpointer.no-last-save-model-only",
                    "--anticipatory.enable",
                    "--anticipatory.delay-steps=1",
                    "--anticipatory.active-steps=2",
                    "--anticipatory.max-rollbacks=1",
                    "--anticipatory.detector.warmup-steps=4",
                )
            ],
            ngpu=1,
            timeout=600,
            use_golden=True,
            check_loss=False,
        ),
        OverrideDefinitions(
            train_args=V4_TRAIN_ARGS,
            test_name="dsv4_lora_ep2_fsdp2",
            test_descr="DeepSeek-V4 LoRA A/B training, frozen base and PEFT export EP2/FSDP2",
            env_vars={"MODULE": "tests.integration_tests.lora_config", "CONFIG": "deepseek_v4_lora_training"},
            override_args=[
                (
                    *_LORA_ROPE_OVERRIDES,
                    "--training.steps=2",
                    "--training.num-tokens-per-train-step=4096",
                    "--optim.optimizer.name=Muon",
                    "--parallelism.data-parallel-shard-degree=2",
                    "--parallelism.expert-parallel-degree=2",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                    "--checkpointer.interval=10000",
                )
            ],
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
                "--optim.lr-scheduler.total-steps=100",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.num-tokens-per-train-step=4096",
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
                "--optim.lr-scheduler.total-steps=100",
                "--parallelism.expert-parallel-degree=2",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.num-tokens-per-train-step=4096",
            ),
            use_golden=True,
            check_loss=True,
        ),
        _build_case(
            test_name="dsv4_smla_1rank_aot_eager",
            test_descr="DeepSeek-V4 SMLA 1rank aot_eager",
            ngpu=1,
            env_vars={"MODULE": "tests.integration_tests.optimizer_config", "CONFIG": "deepseek_v4_aot_eager"},
            extra_args=(
                "--training.steps=1",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.num-tokens-per-train-step=4096",
            ),
            use_golden=False,
            check_loss=False,
        ),
        _build_case(
            test_name="dsv4_smla_ep2_fsdp2",
            test_descr="DeepSeek-V4 SMLA ep2 fsdp2",
            ngpu=2,
            env_vars={"MODULE": "tests.integration_tests.optimizer_config", "CONFIG": "deepseek_v4_aot_eager"},
            extra_args=(
                "--training.steps=1",
                "--parallelism.expert-parallel-degree=2",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.num-tokens-per-train-step=4096",
            ),
            use_golden=False,
            check_loss=False,
        ),
        _build_case(
            test_name="dsv4_smla_cp2_ep2_fsdp2",
            test_descr="DeepSeek-V4 SMLA Muon cp2 ep2 fsdp2",
            ngpu=4,
            env_vars={
                "MODULE": "tests.integration_tests.mtp_config",
                "CONFIG": "deepseek_v4_debugmodel_cp",
                "DSV4_PER_DOC_ALIGNMENT": "256",
            },
            use_cp_attention=True,
            extra_args=(
                "--training.steps=1",
                "--parallelism.expert-parallel-degree=2",
                "--parallelism.context-parallel-degree=2",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.num-tokens-per-train-step=4096",
                "--optim.optimizer.name=Muon",
            ),
            use_golden=False,
            check_loss=False,
            # The fused CP2 path now hands the compressed and sliding-window kernels their
            # rank-local prefix pools (see ``shard_asc_v4_metadata``), so the slab-width
            # token indices that used to trip the AscendC IndexCheck stay inside each
            # kernel's own key frame.  Re-enabled with that contract; the numeric CP gate
            # (same weights via a seed checkpoint) is the 910 run.
        ),
        _build_case(
            test_name="dsv4_mtp_smla_cp2",
            test_descr="DeepSeek-V4 MTP SMLA AdamW cp2",
            ngpu=2,
            env_vars={
                "MODULE": "tests.integration_tests.mtp_config",
                "CONFIG": "deepseek_v4_debugmodel_mtp",
                "DSV4_PER_DOC_ALIGNMENT": "256",
            },
            use_cp_attention=True,
            extra_args=(
                "--training.steps=1",
                "--parallelism.context-parallel-degree=2",
                "--hf-assets-path=tests/assets/deepseek_v3",
                "--training.num-tokens-per-train-step=4096",
                "--optim.optimizer.name=Muon",
            ),
            use_golden=False,
            check_loss=False,
            # Re-enabled with the rank-local prefix-pool contract: the selector, SLIG,
            # SMLA and SMLAG share one key frame per rank, so the MTP CP2 case no longer
            # hits the document-tail window.  Numeric gate: 910 (seed-checkpoint aligned).
        ),
    ]
