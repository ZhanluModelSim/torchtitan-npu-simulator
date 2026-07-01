# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator-only replacements for model submodules whose real production
implementation requires actual NPU hardware (raw Triton kernels, JIT-
compiled aclnn extensions) and therefore cannot execute under meta-device
simulation. Each shim preserves the *real* op name that would run in
production (via OpDispatchCapture.record_synthetic_op) and the real output
shape (computed analytically), without invoking any real kernel. See
docs/superpowers/specs/2026-07-01-mhc-real-op-name-capture-design.md."""
