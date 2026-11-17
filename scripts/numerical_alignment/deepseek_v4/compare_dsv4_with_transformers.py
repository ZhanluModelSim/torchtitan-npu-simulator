# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare TorchTitan DSV4 logits with the Transformers inference reference.

The entry point only orchestrates input generation, the two model adapters and
the metric gate. Transformers checkpoint handling and TorchTitan model details
remain isolated in ``hf_model`` and ``tt_model`` respectively.
"""

from __future__ import annotations

import argparse
import gc
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

import torch

from ..utils import metrics
from . import config as dsv4_config
from . import hf_model
from . import tt_model as dsv4_tt_model
from .overrides import GOLDEN_OVERRIDES, TEST_OVERRIDES

DEFAULT_SEQ_LEN = 4096


def _disable_hf32() -> None:
    """Disable HF32 on every NPU math backend for reproducible alignment."""

    from torchtitan_npu.extensions.torchtitan.trainer import set_allow_hf32

    set_allow_hf32(False)


@torch.no_grad()
def _forward_torchtitan(
    checkpoint: Path,
    inputs: torch.Tensor,
    n_layers: int,
    attention: str,
    device: torch.device,
    trace: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    print(f"[candidate] loading TorchTitan DCP on NPU ({attention})", flush=True)
    torch.set_default_device(device)
    overrides = GOLDEN_OVERRIDES if attention == "golden" else TEST_OVERRIDES
    model = dsv4_tt_model.load_model(
        checkpoint,
        n_layers,
        overrides,
        device,
        inputs.shape[0],
        seq_len=inputs.shape[1],
        model_flavor=dsv4_config.DEFAULT_MODEL_FLAVOR,
        num_experts=dsv4_config.NUM_EXPERTS,
        top_k=dsv4_config.TOP_K,
    )
    model.eval()
    trace_adapter = import_module(f"{__package__}.trace") if trace is not None else None
    hook_registry = trace_adapter.register_torchtitan(model, trace) if trace_adapter is not None else None
    try:
        logits = dsv4_tt_model.forward_logits(model, inputs.to(device))
        result = logits.cpu()
        if trace is not None:
            trace["logits"] = result.detach().clone()
    finally:
        if hook_registry is not None:
            hook_registry.remove()
    del model, logits
    _empty_cache(device)
    return result


def _empty_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-checkpoint-path", type=Path, required=True)
    parser.add_argument("--tt-checkpoint-path", type=Path, required=True)
    parser.add_argument(
        "--n-layers",
        type=int,
        default=dsv4_config.DEFAULT_N_LAYERS,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=DEFAULT_SEQ_LEN,
        help="sequence length (Golden requires a multiple of 128)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="input batch size (Golden supports only 1)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--mode",
        choices=("bitwise", "close"),
        default="bitwise",
        help="logits gate: exact equality or numerical approximation",
    )
    parser.add_argument(
        "--trace-modules",
        action="store_true",
        default=False,
        help="compare intermediate Transformers and TorchTitan module outputs",
    )
    parser.add_argument(
        "--attention",
        choices=("golden", "test"),
        default="golden",
    )
    parser.add_argument(
        "--apply-transformers-patch",
        action=argparse.BooleanOptionalAction,
        default=find_spec(f"{__package__}.transformers_dsv4_patch") is not None,
        help=(
            "apply the numerical-alignment patch (enabled by default when installed); use "
            "--no-apply-transformers-patch for the unmodified TF model"
        ),
    )
    args = parser.parse_args()
    if not args.hf_checkpoint_path.is_dir():
        parser.error(f"HF checkpoint not found: {args.hf_checkpoint_path}")
    if not args.tt_checkpoint_path.is_dir():
        parser.error(f"DCP checkpoint not found: {args.tt_checkpoint_path}")
    if args.n_layers <= 0 or args.seq_len <= 0 or args.batch_size <= 0:
        parser.error("--n-layers, --seq-len and --batch-size must be positive")
    if args.n_layers > dsv4_config.DEFAULT_N_LAYERS:
        parser.error("--n-layers must be in [1, 4] for the inference reference")
    if args.attention == "golden":
        if args.batch_size != 1:
            parser.error("Golden attention requires --batch-size=1 for the packed metadata layout")
        if args.seq_len % 128 != 0:
            parser.error("Golden attention requires --seq-len to be divisible by 128")
    if args.device.split(":", 1)[0] != "npu":
        parser.error("this alignment recipe requires an NPU device")
    for requested, module in (
        (args.apply_transformers_patch, "transformers_dsv4_patch"),
        (args.trace_modules, "trace"),
    ):
        if requested and find_spec(f"{__package__}.{module}") is None:
            parser.error(f"optional alignment module {module!r} is not installed in this checkout")
    return args


def main() -> int:
    args = _parse_args()
    if args.device.split(":", 1)[0] == "npu":
        import torch_npu  # noqa: F401

        _disable_hf32()
        print("HF32        : disabled (matmul/conv/aclnn)", flush=True)

    transformers_version = hf_model.ensure_transformers_version()
    print(f"Transformers : {transformers_version}", flush=True)

    original_deterministic = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    original_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    original_device = torch.get_default_device()
    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randint(
        0,
        dsv4_config.VOCAB_SIZE,
        (args.batch_size, args.seq_len),
        generator=generator,
        device="cpu",
    )
    golden_trace = {} if args.trace_modules else None
    candidate_trace = {} if args.trace_modules else None
    try:
        golden = hf_model.forward_logits(
            args.hf_checkpoint_path,
            inputs,
            args.n_layers,
            device,
            apply_patch=args.apply_transformers_patch,
            trace=golden_trace,
        )
        candidate = _forward_torchtitan(
            args.tt_checkpoint_path,
            inputs,
            args.n_layers,
            args.attention,
            device,
            candidate_trace,
        )
    finally:
        torch.set_default_device(original_device)
        torch.set_default_dtype(original_dtype)
        torch.use_deterministic_algorithms(original_deterministic, warn_only=original_warn_only)

    result = metrics.compare_tensors(candidate, golden)
    if golden_trace is not None and candidate_trace is not None:
        metrics.print_trace_metrics(
            candidate_trace,
            golden_trace,
            candidate_label="TorchTitan",
            reference_label="Transformers",
        )
    print("\n=== Transformers vs TorchTitan ===")
    print(f"model flavor: {dsv4_config.DEFAULT_MODEL_FLAVOR}")
    print(f"MoE config  : {dsv4_config.NUM_EXPERTS} experts, top-{dsv4_config.TOP_K}")
    print(f"attention   : {args.attention}")
    print(f"TF patch    : {'enabled' if args.apply_transformers_patch else 'disabled'}")
    print(f"logits      : {tuple(candidate.shape)}")
    print(f"mean_ae           : {result['mean_ae']:.8e}")
    print(f"max_re            : {result['max_re']:.8e}")
    print(f"mean_re           : {result['mean_re']:.8e}")
    print(f"cos               : {result['cos']:.10f}")
    if args.mode == "bitwise":
        failed = () if torch.equal(candidate, golden) else ("bitwise",)
    else:
        failed = metrics.failed_logits_metrics(result)
    print(f"gate        : {args.mode}")
    print(f"status      : {'PASS' if not failed else 'FAIL'}")
    if failed:
        print(f"failed      : {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
