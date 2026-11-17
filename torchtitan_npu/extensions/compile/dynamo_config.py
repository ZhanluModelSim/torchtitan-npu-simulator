# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime configuration for torch._dynamo guard evaluation on the NPU stack.

``use_recursive_dict_tags_for_guards`` speeds up guard execution of nested
``nn.Module`` trees: the guard manager marks tag safe roots while it builds the
guard tree and then, on every later call, skips the full guard body of a
subtree whose recursive dict tags still match.  For the deep, repeatedly
structured blocks this repository compiles, that shortens the per-call guard
check, without changing what is captured.

The knob only has to be set before the first graph is captured, which is well
after package import: ``extensions`` is loaded by the ``torchtitan_npu``
package entry, and this module is listed explicitly in
``extensions/__init__.py``.

``specialize_float`` is deliberately *not* set here.  It pushes float inputs
into ``CONSTANT_MATCH`` and re-specializes on every new value, so any Python
float that changes across steps inside a compiled region (a wrapped parameter,
a scalar weight passed to the loss, a schedule value) recompiles once per step
and reintroduces exactly the host-side stall this module is meant to remove.
Leaving it at the torch default keeps such floats dynamic instead.
"""

import torch._dynamo.config
from torchtitan.tools.logging import logger

_GUARD_KNOB = "use_recursive_dict_tags_for_guards"


def apply() -> None:
    # The flag is a plain ``bool`` at runtime: upstream initializes it to
    # ``False`` and leaves it unannotated.  Pyrefly narrows that initializer to
    # ``Literal[False]`` and rejects a direct assignment of ``True`` as
    # bad-assignment, so the runtime override goes through setattr.
    setattr(torch._dynamo.config, _GUARD_KNOB, True)
    logger.info("Dynamo guard knob enabled: %s", _GUARD_KNOB)


apply()
