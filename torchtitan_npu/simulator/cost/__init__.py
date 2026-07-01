# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extensible per-op FLOPs/memory/communication-byte cost model, used to
annotate L0 OpNodes for observability. Never raises: an op_type with no
registered handler returns a zeroed, explicitly-flagged CostEstimate (see
docs/superpowers/specs/2026-07-01-npu-simulator-design.md §5.8/§9)."""
