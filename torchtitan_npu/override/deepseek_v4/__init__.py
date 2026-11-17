"""DeepSeek-V4 AscendC overrides.

Importing this package registers the ``compressor`` and ``sparse_attn``
override factories. Fused implementations are loaded only when selected.
"""

from . import compressor, sparse_attn  # noqa: F401
