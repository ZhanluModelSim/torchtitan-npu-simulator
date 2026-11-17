# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU-side FSDP and gradient-clip integration for CPU offload.

Importing this package has no side effects: the ``grad_accum`` and
``grad_clip`` submodules are imported and installed explicitly by the
CPU-offload optimizer container when the user selects the ``cpu_offload``
override, so runs without CPU offload keep pristine upstream behavior.
"""
