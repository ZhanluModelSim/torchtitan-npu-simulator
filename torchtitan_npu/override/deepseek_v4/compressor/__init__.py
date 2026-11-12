"""DeepSeek-V4 compressor overrides."""

from typing import TYPE_CHECKING

from torchtitan.config import derive, override
from torchtitan.models.deepseek_v4.compressor import Compressor

from torchtitan_npu.models.deepseek_v4.compressor import CompressorImplementation

if TYPE_CHECKING:
    from .ascendc import AscCompressor, AscNativeCompressor


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
) -> "AscNativeCompressor.Config":
    from .ascendc import AscNativeCompressor

    norm, rope = cfg.norm, cfg.rope
    if rope.dim != cfg.rope_head_dim or getattr(rope, "split", 0):
        raise ValueError("The native compressor RoPE must rotate its already-sliced tail")
    if fused_norm:
        from torchtitan_npu.override.common.rms_norm import asc as asc_norm

        norm = asc_norm(norm)
    if fused_rope:
        from torchtitan.models.common.rope import ComplexRoPE

        from torchtitan_npu.override.common.rope import asc_complex

        if not isinstance(rope, ComplexRoPE.Config):
            raise ValueError("The fused compressor RoPE requires ComplexRoPE")
        rope = asc_complex(rope)
    return derive(cfg, AscNativeCompressor.Config, norm=norm, rope=rope)


@override(
    target=CompressorImplementation.Config,
    exact=True,
    description="Use the CANN compressor in the legacy NPU geometry backend",
)
def legacy_asc(cfg: CompressorImplementation.Config) -> "AscCompressor.Config":
    from .ascendc import AscCompressor

    return derive(cfg, AscCompressor.Config)
