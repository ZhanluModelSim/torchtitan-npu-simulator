# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4.1 integration cases.

``build_deepseek_v4_1_test_list`` and its checkpoint/quantized variants cover the
LoRA recipes (text and image-conditioned) that run in the default ``models``
suite and in the dedicated ``deepseek_v4_1*`` suites.

The golden case pins the deterministic loss trajectory of the reduced-width V4.1
debug model on 2 cards (FSDP2/EP2) so the default CI pool gates the V4.1 recipe
again; it was dropped from the default list while the V4.1 stack was refactored
and is restored here against the current model module.  The multimodal fused
smoke stays outside the default pool: it needs the vision fixture data and is
selected explicitly.
"""

from dataclasses import replace

from tests.integration_tests import OverrideDefinitions


def build_deepseek_v4_1_test_list() -> list[OverrideDefinitions]:
    args = (
        "--override.imports", "torchtitan_npu.override.common.rope.workaround",
        "--hf-assets-path=tests/assets/deepseek_v3", "--training.steps=2",
        "--training.seq-len=128", "--training.global-batch-size=2",
        "--parallelism.data-parallel-shard-degree=2", "--parallelism.expert-parallel-degree=2",
        "--checkpoint.enable", "--checkpoint.no-last-save-model-only",
        "--checkpoint.keep-latest-k=0", "--debug.seed=42", "--debug.deterministic",
        "--optimizer.name=AdamW",
    )
    text = OverrideDefinitions(
        test_name="dsv41_lora_engram_ep2_fsdp2",
        test_descr="V4.1 LoRA with frozen Engram, adapter updates and PEFT export",
        env_vars={"MODULE": "tests.integration_tests.v41_lora_config", "CONFIG": "deepseek_v4_1_lora_training"},
        override_args=[args], expected_steps=((1, 2),), ngpu=2,
        use_golden=True, check_loss=False,
    )
    multimodal = replace(
        text, test_name="dsv41_lora_multimodal_ep2_fsdp2",
        test_descr="V4.1 image-conditioned LoRA with frozen vision and Engram",
        env_vars={
            "MODULE": "tests.integration_tests.v41_lora_config",
            "CONFIG": "deepseek_v4_1_lora_multimodal_training",
        },
    )
    return [text, multimodal]


def build_deepseek_v4_1_checkpoint_test_list(
    *, optimizers: tuple[str, ...] = ("AdamW", "Muon"), include_adapter_only: bool = True,
) -> list[OverrideDefinitions]:
    case = build_deepseek_v4_1_test_list()[0]
    common = tuple(arg for arg in case.override_args[0] if not arg.startswith(("--training.steps=", "--optimizer.name=")))
    result = []
    for optimizer in optimizers:
        args = (*common, "--training.steps=4", "--checkpoint.interval=2", f"--optimizer.name={optimizer}")
        result.append(replace(
            case, test_name=f"dsv41_lora_{optimizer.lower()}_resume_ep2_fsdp2",
            test_descr=f"V4.1 LoRA {optimizer} exact native checkpoint resume",
            override_args=[args, (*args, "--checkpoint.load-step=2")],
            expected_steps=((1, 2, 3, 4), (3, 4)), check_resume=True,
            env_vars={
                "MODULE": "tests.integration_tests.v41_lora_config",
                "CONFIG": ("deepseek_v4_1_lora_adapter_only_training" if optimizer == "AdamW"
                           else "deepseek_v4_1_lora_training"),
            },
        ))
    if include_adapter_only and "Muon" in optimizers:
        args = (*common, "--training.steps=10", "--checkpoint.interval=5", "--optimizer.name=Muon")
        result.append(replace(
            case, test_name="dsv41_lora_muon_adapter_only_resume_ep2_fsdp2",
            test_descr="V4.1 LoRA Muon adapter-only resume with five exact post-resume steps",
            override_args=[args, (*args, "--checkpoint.load-step=5")],
            expected_steps=(tuple(range(1, 11)), tuple(range(6, 11))), check_resume=True,
            env_vars={
                "MODULE": "tests.integration_tests.v41_lora_config",
                "CONFIG": "deepseek_v4_1_lora_adapter_only_training",
            },
        ))
    return result


def build_deepseek_v4_1_quantized_test_list() -> list[OverrideDefinitions]:
    return [replace(
        case, test_name=case.test_name.replace("_resume_", "_block_fp8_resume_"),
        env_vars={"MODULE": "tests.integration_tests.v41_lora_config", "CONFIG": "deepseek_v4_1_lora_quantized_training"},
    ) for case in build_deepseek_v4_1_checkpoint_test_list(include_adapter_only=False)]


# DeepSeek-V4.1 golden reference recipe: the same operator overrides the
# real-width baseline was verified with (forward verified boundary-by-boundary
# against the ds-code inference implementation).  The case runs the
# reduced-width full-structure debug model (real 40-layer compression/source
# layout, debug widths) and pins its deterministic loss trajectory in
# tests/assets/losses/dsv41_golden_2p_ep2_fsdp2.txt (2 cards, so the CI smoke
# pool can schedule it).  The 8-card shape stays available as a manual A/B
# regression via run_train.sh; its anchor is intentionally not committed.
GOLDEN_OVERRIDES = (
    "--override.imports",
    "torchtitan_npu.override.common.rope.workaround",
    "torchtitan_npu.override.common.optimizer.virtual",
)

# Environment of the frozen baseline.  MODULE/CONFIG route the trainer to the
# V4.1 text debug model regardless of the runner's CLI defaults (per-case env
# vars take precedence); the dataloader tokenizer is the committed
# tests/assets/deepseek_v3 mini tokenizer, the same one the DeepSeek-V4 golden
# cases use.
GOLDEN_ENV = {
    "MODULE": "torchtitan_npu.models.deepseek_v4_1",
    "CONFIG": "deepseek_v4_1_debugmodel_text",
}


def build_deepseek_v4_1_golden_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            override_args=[
                GOLDEN_OVERRIDES
                + (
                    "--training.steps=30",
                    "--training.local-batch-size=1",
                    "--training.global-batch-size=2",
                    "--training.seq-len=512",
                    "--parallelism.spmd-backend=partial_dtensor",
                    "--parallelism.data-parallel-shard-degree=2",
                    "--parallelism.data-parallel-replicate-degree=1",
                    "--parallelism.expert-parallel-degree=2",
                    "--parallelism.tensor-parallel-degree=1",
                    "--parallelism.context-parallel-degree=1",
                    "--parallelism.pipeline-parallel-degree=1",
                    "--parallelism.context-parallel-load-balancer=None",
                    "--debug.no-moe-force-load-balance",
                    "--hf-assets-path=tests/assets/deepseek_v3",
                    # The LR recipe (warmup 25 / total 40) is part of the frozen
                    # anchor: the trajectory is produced with the 40-step
                    # baseline schedule. Keep it unchanged when shortening the
                    # smoke run so its steps remain an exact prefix of that
                    # baseline.
                    "--lr-scheduler.warmup-steps=25",
                    "--lr-scheduler.total-steps=40",
                    "--lr-scheduler.decay-type=cosine",
                    "--lr-scheduler.decay-ratio=1.0",
                    "--lr-scheduler.min-lr-factor=0.01",
                    "--checkpoint.no-enable",
                    "--metrics.disable-color-printing",
                )
            ],
            test_descr="DeepSeek-V4.1 golden debugmodel 2p fsdp2 ep2 exact loss",
            test_name="dsv41_golden_2p_ep2_fsdp2",
            ngpu=2,
            env_vars=GOLDEN_ENV,
            use_golden=True,
            check_loss=True,
            timeout=7200,
        ),
    ]


def build_deepseek_v4_1_fused_test_list() -> list[OverrideDefinitions]:
    """Explicit A3 multimodal packing/fusion/Muon smoke; outside the default CI pool.

    Restored from the case that was removed with the V4.1 refactor, with the two
    arguments the current config registry needs (``deepseek_v4_1_debugmodel_multimodal``
    and ``--training.seq-len=512``, the debug model's RoPE limit).  It stays opt-in
    because it needs the vision fixture data and because the multimodal forward
    currently dies in ``aclnnIndex`` (DT_COMPLEX64 input) on the 2026-09-20 PTA
    build, which is independent of this case list.
    """
    ngpu = 2
    args = (
        "--dataloader.dataset-path=tests/assets/cc12m_test",
        "--hf-assets-path=tests/assets/deepseek_v3",
        "--training.steps=2",
        "--training.seq-len=512",
        "--dataloader.packing-buffer-size=4",
        "--debug.seed=42",
        "--debug.deterministic",
        f"--training.global-batch-size={ngpu}",
        f"--parallelism.data-parallel-shard-degree={ngpu}",
        "--parallelism.data-parallel-replicate-degree=1",
        f"--parallelism.expert-parallel-degree={ngpu}",
        "--parallelism.tensor-parallel-degree=1",
        "--parallelism.context-parallel-degree=1",
        "--parallelism.pipeline-parallel-degree=1",
        "--parallelism.context-parallel-load-balancer=None",
        "--checkpoint.no-enable",
    )
    args += ("--optimizer.name=Muon",)
    return [
        OverrideDefinitions(
            override_args=[args],
            test_descr="V4.1 multimodal A3 fused/Muon",
            test_name="dsv41_multimodal_muon_2p",
            ngpu=ngpu,
            env_vars={
                "MODULE": "torchtitan_npu.models.deepseek_v4_1",
                "CONFIG": "deepseek_v4_1_debugmodel_multimodal",
            },
            use_golden=False,
            check_loss=False,
            expected_steps=((1, 2),),
            timeout=3600,
        )
    ]
