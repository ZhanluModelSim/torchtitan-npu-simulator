# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek-V4 MHC overrides (registry-facing module).

The CANN fused implementations live in ``ascendc.py`` and the hand-written
Triton kernels in ``triton.py``; the registrations are defined here so the
override paths stay ``override.deepseek_v4.mhc.{asc_hc_pre, asc_hc_post,
triton_hc_pre, triton_hc_post, triton_hc_head}``.  Both entry families
target the same ``HcPre.Config`` / ``HcPost.Config`` / ``HcHead.Config``
nodes, so only one of each pair can be enabled at a time
(``--override.imports`` factory conflict).
"""

from torchtitan.config import derive, override

from torchtitan_npu.models.deepseek_v4.mhc import HcHead, HcPost, HcPre

from .ascendc import AscHcPost, AscHcPre
from .triton import TritonHcHead, TritonHcPost, TritonHcPre


@override(
    target=HcPre.Config,
    exact=True,
    description="NPU DeepSeek-V4 HcPre with A3 fused and A5 split operators",
)
def asc_hc_pre(cfg: HcPre.Config) -> AscHcPre.Config:
    return derive(cfg, AscHcPre.Config)


@override(
    target=HcPost.Config,
    exact=True,
    description=("NPU fused DeepSeek-V4 HcPost via cann_ops_transformer.ops.mhc_post"),
)
def asc_hc_post(cfg: HcPost.Config) -> AscHcPost.Config:
    return derive(cfg, AscHcPost.Config)


@override(
    target=HcPre.Config,
    exact=True,
    description="Triton DeepSeek-V4 HcPre kernels",
)
def triton_hc_pre(cfg: HcPre.Config) -> TritonHcPre.Config:
    return derive(cfg, TritonHcPre.Config)


@override(
    target=HcPost.Config,
    exact=True,
    description="Triton DeepSeek-V4 HcPost kernels",
)
def triton_hc_post(cfg: HcPost.Config) -> TritonHcPost.Config:
    return derive(cfg, TritonHcPost.Config)


@override(
    target=HcHead.Config,
    exact=True,
    description="Triton DeepSeek-V4 HcHead kernels",
)
def triton_hc_head(cfg: HcHead.Config) -> TritonHcHead.Config:
    return derive(cfg, TritonHcHead.Config)
