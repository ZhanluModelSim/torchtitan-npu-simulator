"""DeepSeek-V4 CANN overrides.

Importing this package imports ``sparse_attn`` (whose ``__init__`` pulls in
``cann.py`` and ``golden.py``), firing all override registrations.
"""

from . import sparse_attn  # noqa: F401
