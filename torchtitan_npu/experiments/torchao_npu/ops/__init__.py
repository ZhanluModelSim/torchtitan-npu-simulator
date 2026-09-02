# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Low-precision NPU operators.

Import ops from their concrete submodules; each submodule declares its
public surface via ``__all__``:

- :mod:`torchao_npu.ops.block_ops` — Block FP8 matmul ops
- :mod:`torchao_npu.ops.float8_ops` — FP8 row-wise fake quantization
- :mod:`torchao_npu.ops.mx_ops` — MX (FP8/FP4) matmul and (de)quantization ops
"""
