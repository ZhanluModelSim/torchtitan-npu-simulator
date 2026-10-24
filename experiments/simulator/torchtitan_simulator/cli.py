# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit simulator entry point with a separate build-only diagnostic."""

import argparse
import json

from torchtitan_simulator.config import SimulationConfig, build_meta_model, prepare_config


def main(argv=None):
    from torchtitan_simulator.model_overrides import add_cli_arguments, take_cli_overrides

    parser = argparse.ArgumentParser(description=__doc__)
    add_cli_arguments(parser)
    parser.add_argument("--config", default="deepseek_v4_smoketest")
    parser.add_argument("--simulation.world-size", dest="world_size", type=int, default=1)
    parser.add_argument("--simulation.logical-rank", dest="logical_rank", type=int, default=0)
    parser.add_argument(
        "--simulation.target-npu-device-type", dest="target_npu_device_type", choices=["non_a5", "a5"], default="non_a5"
    )
    for option, destination, choices in (
        ("dtype", "training_dtype", ["float32", "bfloat16"]),
        ("mixed-precision-param", "mixed_precision_param", ["float32", "bfloat16"]),
        ("mixed-precision-reduce", "mixed_precision_reduce", ["float32"]),
    ):
        parser.add_argument(f"--training.{option}", dest=destination, choices=choices)
    parser.add_argument("--training.seq-len", dest="seq_len", type=int, default=128)
    parser.add_argument("--training.num-mtp-modules", dest="num_mtp_modules", type=int, default=0)
    parser.add_argument("--training.local-batch-size", dest="local_batch_size", type=int)
    parser.add_argument("--training.global-batch-size", dest="global_batch_size", type=int)
    parser.add_argument(
        "--extension.quantization.enable-quantized-training", dest="enable_quantized_training", action="store_true"
    )
    parser.add_argument(
        "--extension.quantization.recipe", dest="quantization_recipe", choices=["all_mxfp8"], default="all_mxfp8"
    )
    parser.add_argument("--simulation.selective-ac-save-ops", dest="selective_ac_save_ops", nargs="+")
    parser.add_argument(
        "--simulation.memory-parameter-storage-dtype", dest="memory_parameter_storage_dtype", default=""
    )
    parser.add_argument(
        "--simulation.memory-offload-ac-saved-tensors", dest="memory_offload_ac_saved_tensors", action="store_true"
    )
    parser.add_argument(
        "--activation-checkpoint.mode",
        dest="activation_checkpoint_mode",
        choices=["none", "full", "selective"],
        default="none",
    )
    parser.add_argument("--parallelism.pipeline-parallel-degree", dest="pipeline_parallel_degree", type=int, default=1)
    parser.add_argument("--simulation.enable-fsdp-allgather-fp8", dest="enable_fsdp_allgather_fp8", action="store_true")
    parser.add_argument("--simulation.enable-ep-dispatch-fp8", dest="enable_ep_dispatch_fp8", action="store_true")
    parser.add_argument("--simulation.output-dir", dest="output_dir", default="./simulator_output")
    parser.add_argument("--simulation.output-formats", dest="output_formats", nargs="*", choices=["mem"], default=[])
    for name, default in (
        ("data-parallel-replicate-degree", 1),
        ("data-parallel-shard-degree", -1),
        ("tensor-parallel-degree", 1),
        ("context-parallel-degree", 1),
        ("expert-parallel-degree", 1),
    ):
        parser.add_argument(f"--parallelism.{name}", dest=name.replace("-", "_"), type=int, default=default)
    parser.add_argument(
        "--build-only", action="store_true", help="Build model only; does not certify graph or gradients"
    )
    args = vars(parser.parse_args(argv))
    args["model_overrides"] = take_cli_overrides(args)
    build_only = args.pop("build_only")
    options = SimulationConfig(**args)
    options.validate()
    if not build_only:
        from torchtitan_simulator.runner import export_result, simulate

        workload = simulate(options)
        export_result(workload, options)
        return workload
    from torchtitan_simulator.runtime import meta_runtime

    prepared = prepare_config(options)
    with meta_runtime() as runtime:
        runtime.configure_model(prepared)
        model = build_meta_model(prepared)
        report = {
            "scope": "model_build_only",
            "config_factory": prepared.config_factory,
            "parameter_numel": sum(p.numel() for p in model.parameters()),
            "parameter_devices": sorted({p.device.type for p in model.parameters()}),
            "num_layers": len(model.layers),
            "token_embedding_shape": list(model.tok_embeddings.weight.shape),
            "mtp_layers": 0 if model.mtp_layers is None else len(model.mtp_layers),
            "compile": prepared.trainer.compile.enable,
            "cuda_graphs": not prepared.trainer.training.disable_cuda_graphs,
            "spmd_backend": prepared.trainer.parallelism.spmd_backend,
            "adaptations": prepared.adaptations,
            "capabilities": runtime.capabilities,
        }
    print(json.dumps(report, indent=2))
    return report
