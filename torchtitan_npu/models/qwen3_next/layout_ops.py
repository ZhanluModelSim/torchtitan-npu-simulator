"""Synthetic records for NPU aclnn kernels that view-class dispatcher ops
materialize into on device.

``torch.split``/``torch.chunk`` are metadata-only views in the dispatcher,
so the L0 capture annotates them ``metadata_view`` and ``fold_metadata_views``
removes them from the exported graph. On Ascend NPU the same calls launch a
real ``SplitV`` kernel — the measured profiler shows it as a data-movement
op. The model files call :func:`record_split_v` right after the real view op
to add one synthetic node with the same input/output tensors, so op counts,
dependency edges and liveness all stay consistent. No-op outside capture.
"""

from __future__ import annotations

import torch


def record_split_v(inputs: list[torch.Tensor], chunks: list[torch.Tensor]) -> None:
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture
    from torchtitan_npu.simulator.synthetic_ac import run_synthetic_op

    capture = get_active_capture()
    if capture is None:
        return
    module_path = (
        capture.module_path_tracker.current_path()
        if capture.module_path_tracker is not None
        else ""
    )
    run_synthetic_op(
        "npu.aclnn_split_v.default",
        inputs=inputs,
        output_factory=lambda: chunks,
        module_path=module_path,
    )
