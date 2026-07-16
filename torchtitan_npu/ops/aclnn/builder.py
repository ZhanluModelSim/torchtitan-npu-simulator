# Adapted from
# https://gitcode.com/Ascend/MindSpeed/blob/9f9eebfa28faf06ee97474a3c996f0fcd0897ea6/mindspeed/op_builder/builder.py
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch_npu
from torch.utils.cpp_extension import load

_loaded_ops = {}


def build_op(name, sources, verbose=False):
    if name in _loaded_ops:
        return _loaded_ops[name]

    cwd = os.path.dirname(os.path.abspath(__file__))
    ascend_home = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann")
    torch_path = os.path.dirname(os.path.abspath(torch.__file__))
    tnpu_path = os.path.dirname(os.path.abspath(torch_npu.__file__))

    cflags = [
        "-D_FORTIFY_SOURCE=2",
        "-O2",
        f"-I{cwd}",
        f"-I{ascend_home}/include",
        f"-I{torch_path}/include",
        f"-I{torch_path}/include/torch/csrc/api/include",
        f"-I{tnpu_path}/include",
        f"-I{tnpu_path}/include/torch_npu/csrc/framework/utils",
        f"-I{tnpu_path}/include/torch_npu/csrc/aten",
        f"-I{tnpu_path}/include/third_party/hccl/inc",
        f"-I{tnpu_path}/include/third_party/acl/inc",
    ]

    ldflags = [
        f"-L{ascend_home}/lib64",
        f"-L{tnpu_path}/lib",
        "-ltorch_npu",
        "-lascendcl",
    ]

    op_module = load(
        name=name,
        sources=[os.path.join(cwd, s) for s in sources],
        extra_cflags=cflags,
        extra_ldflags=ldflags,
        verbose=verbose,
    )
    _loaded_ops[name] = op_module
    return op_module
