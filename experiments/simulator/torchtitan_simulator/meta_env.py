# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Capture context shared with the retained IR implementation.

Runtime activation lives in scoped compatibility adapters. This module does
not modify Trainer, global device discovery, or distributed initialization.
"""

from contextlib import contextmanager

_pp_context = {"mb_idx": 0, "phase": "forward", "stage": -1, "comp_type": "F", "fsdp_state": "NA"}
_comm_layer = ""
_in_metadata_inference = False
_is_meta_simulation = False
_fsdp_state = {}


@contextmanager
def compute_block(comp_type):
    """Own an outer training block, including nested chunked-loss backward."""
    previous = {key: _pp_context.get(key) for key in ("compute_owner", "comp_type")}
    _pp_context.update(compute_owner=comp_type, comp_type=comp_type)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                _pp_context.pop(key, None)
            else:
                _pp_context[key] = value


def patch_device_type_to_meta():
    if not _is_meta_simulation:
        raise RuntimeError("Run capture inside meta_runtime(); implicit global activation is disabled")


def _consume_p2p_op_context(tensor, direction):
    raise ValueError("Pipeline parallel capture is not enabled in this migration stage")


@contextmanager
def _use_p2p_op_context(context):
    if context is not None:
        raise ValueError("Pipeline parallel capture is not enabled in this migration stage")
    yield
