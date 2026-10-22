# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Read logical parameter names through native checkpoint-wrapper APIs."""

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper


def logical_named_parameters(model):
    seen = set()

    def visit(module, prefix):
        if isinstance(module, CheckpointWrapper):
            # The public wrapper method removes its implementation prefix.
            parameters = module.named_parameters()
        else:
            parameters = module.named_parameters(recurse=False)
        for name, parameter in parameters:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield prefix + name, parameter
        if not isinstance(module, CheckpointWrapper):
            for name, child in module.named_children():
                yield from visit(child, prefix + name + ".")

    yield from visit(model, "")
