from dataclasses import dataclass

import torch
import torch_npu

from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.config import derive, override


class NPURMSNorm(RMSNorm):
    """NPU fused RMSNorm using ``torch_npu.npu_rms_norm``."""

    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]


@override(
    target=RMSNorm.Config,
    description="NPU fused RMSNorm via torch_npu.npu_rms_norm",
)
def fused(cfg: RMSNorm.Config) -> NPURMSNorm.Config:
    return derive(cfg, NPURMSNorm.Config)
