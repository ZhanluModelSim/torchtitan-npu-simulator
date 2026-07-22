# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Integration glue between ``torchao_npu`` and its host frameworks.

Each module in this package hosts functions and classes that exist
specifically to plug ``torchao_npu`` into a particular host framework's
machinery.  A new framework integration is added by creating a new
``interfaces/<framework>.py`` module.

Supported frameworks:

* **torchtitan / torchtitan-npu** (``torchtitan.py``): ``NpuQuantizeConverter``
  and related helpers that wire ``torchao_npu`` into torchtitan's
  ``QuantizationConverter`` machinery.
"""
