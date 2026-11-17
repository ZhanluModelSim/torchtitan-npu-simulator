# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Example of comparing a local module before and after a change.

This example uses RMSNorm: the Torch implementation is the pre-override
reference and AscRMSNorm is the post-override candidate. Both paths use the
same input, weight, dtype, and upstream gradient. The script compares the
forward output, input gradient, and weight gradient with the repository's
numerical-alignment metrics.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Mapping

from .utils import metrics

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shape", type=int, nargs="+", default=[2, 17, 4096])
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="npu")
    args = parser.parse_args()
    if any(dim <= 0 for dim in args.shape):
        parser.error("all --shape dimensions must be positive")
    if args.eps <= 0:
        parser.error("--eps must be positive")
    if args.device.split(":", 1)[0] != "npu":
        parser.error("the CANN fused RMSNorm comparison requires an NPU device")
    return args


def _run(
    module: torch.nn.Module,
    inputs: torch.Tensor,
    grad_output: torch.Tensor,
) -> dict[str, torch.Tensor]:
    module.zero_grad(set_to_none=True)
    module_inputs = inputs.detach().clone().requires_grad_(True)
    output = module(module_inputs)
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("RMSNorm forward did not return a tensor")
    output.backward(grad_output)
    if module_inputs.grad is None:
        raise RuntimeError("RMSNorm input gradient was not produced")
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.grad is None:
        raise RuntimeError("RMSNorm weight gradient was not produced")
    return {
        "forward": output.detach().cpu(),
        "input_grad": module_inputs.grad.detach().cpu(),
        "weight_grad": weight.grad.detach().cpu(),
    }


def _print_metrics(result: Mapping[str, float | int]) -> None:
    print(f"mean_ae           : {result['mean_ae']:.8e}")
    print(f"max_re            : {result['max_re']:.8e}")
    print(f"mean_re           : {result['mean_re']:.8e}")
    print(f"cos               : {result['cos']:.10f}")


def main() -> int:
    args = _parse_args()
    import torch_npu  # noqa: F401
    from torchtitan.models.common.nn_modules import RMSNorm

    from torchtitan_npu.override.common.rms_norm import AscRMSNorm

    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    torch.use_deterministic_algorithms(True, warn_only=True)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    inputs = torch.randn(args.shape, generator=generator).to(device, dtype)
    grad_output = torch.randn(args.shape, generator=generator).to(device, dtype)
    weight = (1.0 + 0.1 * torch.randn(args.shape[-1], generator=generator)).to(device, dtype)

    baseline = (
        RMSNorm.Config(
            normalized_shape=args.shape[-1],
            eps=args.eps,
        )
        .build()
        .to(device, dtype)
    )
    candidate = (
        AscRMSNorm.Config(
            normalized_shape=args.shape[-1],
            eps=args.eps,
        )
        .build()
        .to(device, dtype)
    )
    with torch.no_grad():
        baseline.weight.copy_(weight)
        candidate.weight.copy_(weight)

    baseline_result = _run(baseline, inputs, grad_output)
    candidate_result = _run(candidate, inputs, grad_output)

    print("=== RMSNorm AscendC precision comparison ===")
    print("baseline          : torch.nn.RMSNorm")
    print("candidate         : torch_npu.npu_rms_norm")
    print(f"shape             : {tuple(args.shape)}")
    print(f"input dtype       : {inputs.dtype}")
    print(f"baseline weight   : {baseline.weight.dtype}")
    print(f"candidate weight  : {candidate.weight.dtype}")
    print(f"eps               : {args.eps}")

    failed_sections: list[str] = []
    for name in ("forward", "input_grad", "weight_grad"):
        result = metrics.compare_tensors(candidate_result[name], baseline_result[name])
        failed = metrics.failed_logits_metrics(result)
        print(f"\n--- {name} ---")
        _print_metrics(result)
        print(f"status            : {'PASS' if not failed else 'FAIL'}")
        if failed:
            print(f"failed metrics    : {', '.join(failed)}")
            failed_sections.append(name)

    print(f"\noverall status    : {'PASS' if not failed_sections else 'FAIL'}")
    if failed_sections:
        print(f"failed sections   : {', '.join(failed_sections)}")
    return 0 if not failed_sections else 1


if __name__ == "__main__":
    raise SystemExit(main())
