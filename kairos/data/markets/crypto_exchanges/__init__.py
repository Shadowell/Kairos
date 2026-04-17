"""Per-venue crypto exchange backends.

Importing this package registers the built-in exchanges (OKX today,
Binance / Bybit / ... in the future) into the registry exposed by
:mod:`kairos.data.markets.crypto_exchanges.base`. Optional dependencies
are imported lazily so that users who skip ``[crypto]`` extras don't pay
any import cost.
"""

from __future__ import annotations

from .base import (
    CryptoExchange,
    ExchangeConfig,
    available_exchanges,
    get_exchange,
    register_exchange,
)


def _safe_import(name: str) -> None:
    try:
        __import__(f"{__name__}.{name}")
    except ImportError:
        # ccxt (or another optional dep) not installed; skip this venue.
        pass


for _venue in ("okx", "binance_vision"):
    _safe_import(_venue)


__all__ = [
    "CryptoExchange",
    "ExchangeConfig",
    "available_exchanges",
    "get_exchange",
    "register_exchange",
]
