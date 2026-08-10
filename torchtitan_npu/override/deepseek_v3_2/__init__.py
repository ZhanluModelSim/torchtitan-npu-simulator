"""DeepSeek-V3.2 CANN overrides.

Importing this package imports ``sparse_attn`` (whose ``__init__`` pulls in
``cann.py``), firing all override registrations.
"""

from . import sparse_attn  # noqa: F401
