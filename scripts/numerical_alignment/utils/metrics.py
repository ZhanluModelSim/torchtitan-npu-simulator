# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reusable tensor comparison metrics for numerical checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TextIO

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "LOGITS_CLOSE_THRESHOLDS",
    "RELATIVE_ERROR_FLOOR",
    "LogitsThresholds",
    "compare_tensors",
    "failed_logits_metrics",
    "print_trace_metrics",
]


RELATIVE_ERROR_FLOOR = 1e-4


@dataclass(frozen=True)
class LogitsThresholds:
    """Numerical-approximation thresholds for end-to-end logits."""

    mean_ae: float
    mean_re: float
    cos: float


LOGITS_CLOSE_THRESHOLDS = LogitsThresholds(
    mean_ae=0.02,
    mean_re=0.02,
    cos=0.9999,
)


def _prepare_tensors(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: candidate={candidate.shape}, reference={reference.shape}")
    if candidate.numel() == 0:
        raise ValueError("cannot compare empty tensors")

    candidate_values = candidate.double().flatten()
    reference_values = reference.double().flatten()
    if not torch.isfinite(candidate_values).all() or not torch.isfinite(reference_values).all():
        raise ValueError("tensors contain NaN or Inf")
    return candidate_values, reference_values


def compare_tensors(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float | int]:
    """Compare two same-shaped tensors with scale-aware error metrics."""

    reference_dtype = reference.dtype
    if not reference_dtype.is_floating_point:
        if candidate.shape != reference.shape:
            raise ValueError(f"shape mismatch: candidate={candidate.shape}, reference={reference.shape}")
        if candidate.numel() == 0:
            raise ValueError("cannot compare empty tensors")
        mismatch_count = int((candidate != reference).sum().item())
        return {
            "exact_match": int(mismatch_count == 0),
            "mismatch_count": mismatch_count,
            "mismatch_rate": mismatch_count / reference.numel(),
        }

    candidate, reference = _prepare_tensors(candidate, reference)

    diff = candidate - reference
    abs_diff = diff.abs()
    # Raw logits may legitimately be close to zero. A meaningful floor keeps
    # isolated near-zero reference values from dominating relative metrics.
    scale = reference.abs().clamp_min(RELATIVE_ERROR_FLOOR)
    relative_error = abs_diff / scale
    candidate_norm = torch.linalg.vector_norm(candidate).item()
    reference_norm = torch.linalg.vector_norm(reference).item()
    if candidate_norm == 0 and reference_norm == 0:
        cos = 1.0
    elif candidate_norm == 0 or reference_norm == 0:
        cos = 0.0
    else:
        cos = F.cosine_similarity(candidate, reference, dim=0).item()
    result: dict[str, float | int] = {
        "mean_ae": abs_diff.mean().item(),
        "max_re": relative_error.max().item(),
        "mean_re": relative_error.mean().item(),
        "cos": cos,
    }
    return result


def failed_logits_metrics(
    result: Mapping[str, float | int],
    thresholds: LogitsThresholds = LOGITS_CLOSE_THRESHOLDS,
) -> tuple[str, ...]:
    """Return logits metrics that exceed numerical-approximation thresholds.

    ``max_re`` remains a diagnostic metric for end-to-end logits.
    """

    upper_bounds = {
        "mean_ae": thresholds.mean_ae,
        "mean_re": thresholds.mean_re,
    }
    failed = [name for name, limit in upper_bounds.items() if result[name] > limit]
    if result["cos"] < thresholds.cos:
        failed.append("cos")
    return tuple(failed)


def print_trace_metrics(
    candidate: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    *,
    candidate_label: str = "candidate",
    reference_label: str = "reference",
    file: TextIO | None = None,
) -> None:
    """Print per-key metrics for two intermediate tensor traces."""

    print(
        f"\n=== Module trace: {reference_label} vs {candidate_label} ===",
        file=file,
    )
    keys = list(dict.fromkeys([*reference, *candidate]))
    key_width = max(32, *(len(key) for key in keys)) if keys else 32
    for key in keys:
        if key not in reference or key not in candidate:
            missing = reference_label if key not in reference else candidate_label
            print(f"{key:<{key_width}} missing={missing}", file=file)
            continue
        if reference[key].shape != candidate[key].shape:
            print(
                f"{key:<{key_width}} "
                f"shape {reference_label}={tuple(reference[key].shape)} "
                f"{candidate_label}={tuple(candidate[key].shape)}",
                file=file,
            )
            continue
        result = compare_tensors(candidate[key], reference[key])
        if "exact_match" in result:
            print(
                f"{key:<{key_width}} exact_match={bool(result['exact_match'])} "
                f"mismatch_count={result['mismatch_count']} "
                f"mismatch_rate={result['mismatch_rate']:.8e}",
                file=file,
            )
        else:
            print(
                f"{key:<{key_width}} mean_ae={result['mean_ae']:.8e} "
                f"max_re={result['max_re']:.8e} "
                f"mean_re={result['mean_re']:.8e} "
                f"cos={result['cos']:.10f}",
                file=file,
            )
