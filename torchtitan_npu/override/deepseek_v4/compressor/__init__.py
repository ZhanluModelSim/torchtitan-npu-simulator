"""DeepSeek-V4 compressor overrides."""

from dataclasses import replace
from typing import TYPE_CHECKING

from torchtitan.config import derive, override
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.deepseek_v4.compressor import Compressor

from torchtitan_npu.models.deepseek_v4.compressor import CompressorImplementation

if TYPE_CHECKING:
    from torchtitan.models.base.module import Module

    from .ascendc import AscCompressor, AscNativeCompressor


def _fuse_compressor_norm(norm: "Module.Config") -> "Module.Config":
    from torchtitan_npu.override.common.rms_norm import asc as asc_norm

    if isinstance(norm, RMSNorm.Config):
        return asc_norm(norm)

    # Quantization wraps the norm before overrides run. Fuse the wrapped
    # RMSNorm so the wrapper still applies fake quantization to its output.
    from torchtitan_npu.extensions.adaptor.torchao.components.deepseek_v4 import FakeQuantizedNormFactory

    if isinstance(norm, FakeQuantizedNormFactory.Config) and isinstance(norm.norm, RMSNorm.Config):
        return replace(norm, norm=asc_norm(norm.norm))

    raise TypeError(f"Fused compressor norm expected RMSNorm, got {type(norm).__qualname__}")


@override(
    target=Compressor.Config,
    exact=True,
    description="Use CANN compression with the upstream DeepSeek-V4 component contract",
)
def asc(
    cfg: Compressor.Config,
    *,
    fused_norm: bool = True,
    fused_rope: bool = True,
    fused_pooling: bool = True,
) -> "AscNativeCompressor.Config":
    from .ascendc import AscNativeCompressor

    norm, rope = cfg.norm, cfg.rope
    if rope.dim != cfg.rope_head_dim or getattr(rope, "split", 0):
        raise ValueError("The native compressor RoPE must rotate its already-sliced tail")
    if fused_norm:
        norm = _fuse_compressor_norm(norm)
    if fused_rope:
        from torchtitan.models.common.rope import ComplexRoPE

        from torchtitan_npu.override.common.rope import asc_complex

        if not isinstance(rope, ComplexRoPE.Config):
            raise ValueError("The fused compressor RoPE requires ComplexRoPE")
        rope = asc_complex(rope)
    return derive(cfg, AscNativeCompressor.Config, norm=norm, rope=rope, fused_pooling=fused_pooling)


@override(
    target=CompressorImplementation.Config,
    exact=True,
    description="Use the CANN compressor in the legacy NPU geometry backend",
)
def legacy_asc(cfg: CompressorImplementation.Config) -> "AscCompressor.Config":
    from .ascendc import AscCompressor

    return derive(cfg, AscCompressor.Config)
