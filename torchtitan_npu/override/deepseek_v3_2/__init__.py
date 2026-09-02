"""DeepSeek-V3.2 AscendC overrides.

Importing this package imports ``sparse_attn`` (whose ``__init__`` pulls in
``ascendc.py``), firing all override registrations.
"""

from . import sparse_attn  # noqa: F401
