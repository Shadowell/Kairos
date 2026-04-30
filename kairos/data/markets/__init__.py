"""Market-specific data adapters.

Importing this package registers the built-in adapters into the registry
exposed by :mod:`kairos.data.markets.base`. New markets live in their own
module (for example ``crypto.py``) and call :func:`register_adapter` at import
time.
"""

from __future__ import annotations

from .base import (
    STD_COLS,
    FetchTask,
    FeatureContext,
    MarketAdapter,
    available_adapters,
    get_adapter,
    register_adapter,
    sanitize_symbol,
)

# Trigger adapter registration by importing the concrete modules. We swallow
# ImportError from optional dependencies so that e.g. ``import kairos.data``
# still works when a venue backend dependency is missing.
for _mod in ("crypto",):
    try:
        __import__(f"{__name__}.{_mod}")
    except ImportError:  # pragma: no cover
        pass


__all__ = [
    "STD_COLS",
    "FetchTask",
    "FeatureContext",
    "MarketAdapter",
    "available_adapters",
    "get_adapter",
    "register_adapter",
    "sanitize_symbol",
]
