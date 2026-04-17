"""Crypto exchange abstraction.

A :class:`CryptoExchange` is a thin wrapper around one venue (OKX, Binance,
Bybit, ...). Exchanges differ in:

* Symbol conventions (``BTC/USDT:USDT`` on OKX perps vs ``BTCUSDT`` on Binance
  perps over the raw REST API).
* Pagination limits (OKX caps history bars at 100-300 per request depending on
  endpoint; Binance at 1000).
* Rate limits and weight accounting.
* Contract semantics (inverse vs linear, USD-margined vs coin-margined).

To keep the rest of Kairos market-agnostic we hide all of that behind this
ABC. Every exchange returns the same canonical OHLCV schema defined in
:data:`kairos.data.markets.base.STD_COLS`.

New exchanges are registered via :func:`register_exchange` at module import
time; :func:`get_exchange` instantiates them lazily so that users who only
care about, say, OKX never pay the startup cost of loading Binance.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from ..base import STD_COLS


log = logging.getLogger("kairos.crypto.exchange")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MARKET_TYPE = "swap"
"""ccxt market type for Kairos' primary use case (USDT-margined perpetuals)."""


@dataclass
class ExchangeConfig:
    """Runtime configuration for a :class:`CryptoExchange` instance.

    Secrets are never passed as positional arguments; callers either set the
    appropriate ``*_API_KEY`` env vars (preferred) or plug them in explicitly
    via the :attr:`credentials` dict. Public market-data endpoints do not
    require any credentials at all.
    """

    market_type: str = DEFAULT_MARKET_TYPE  # spot / swap / future
    proxy: Optional[str] = None             # e.g. "http://127.0.0.1:7890"
    timeout_ms: int = 30_000
    enable_rate_limit: bool = True
    credentials: Dict[str, str] = field(default_factory=dict)
    extra_options: Dict[str, object] = field(default_factory=dict)

    def resolve_proxy(self) -> Optional[str]:
        """Return the effective proxy URL, honouring env vars if unset."""
        if self.proxy:
            return self.proxy
        for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
            value = os.environ.get(var)
            if value:
                return value
        return None


class CryptoExchange(ABC):
    """Abstract crypto exchange wrapper.

    Concrete subclasses typically sit on top of `ccxt`_ but can use a native
    SDK instead if that's more convenient.

    .. _ccxt: https://github.com/ccxt/ccxt
    """

    name: str = ""
    """Short identifier used by the registry (e.g. ``"okx"``)."""

    supported_freqs: tuple[str, ...] = ()
    """Canonical Kairos frequency strings this venue can serve."""

    def __init__(self, config: Optional[ExchangeConfig] = None) -> None:
        self.config = config or ExchangeConfig()

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------
    @abstractmethod
    def list_markets(self) -> List[Dict]:
        """Return the raw market metadata list from the venue.

        Each entry is expected to be the dict-like structure ccxt hands back
        from ``exchange.load_markets()``. Concrete subclasses may post-process
        this (e.g. filter to USDT-margined perps) but should not strip fields
        that universe resolvers downstream rely on (``symbol``, ``active``,
        ``base``, ``quote``, ``contract``, ``linear``, ``info``).
        """

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        freq: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch one symbol's OHLCV between two unix-ms timestamps.

        Returns a DataFrame with the columns in :data:`STD_COLS`. Columns the
        venue cannot provide (e.g. ``turnover`` for a venue that doesn't
        report free-float) should be filled with ``NaN``.
        """

    # ------------------------------------------------------------------
    # Helpers shared by subclasses
    # ------------------------------------------------------------------
    @staticmethod
    def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
        """Enforce the canonical schema, filling missing columns with NaN."""
        for col in STD_COLS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[STD_COLS].copy()
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
        df = df.sort_values("datetime").drop_duplicates("datetime")
        numeric = [c for c in STD_COLS if c != "datetime"]
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[[ExchangeConfig], CryptoExchange]] = {}


def register_exchange(
    name: str,
    factory: Callable[[ExchangeConfig], CryptoExchange],
    *,
    overwrite: bool = False,
) -> None:
    """Register an exchange factory keyed by ``name``.

    Pass ``overwrite=True`` to replace an existing registration; this is
    mostly useful in tests that monkey-patch the ccxt client and reload the
    backend module. Without it, re-registration is an error to prevent
    accidental silent overrides in production.
    """
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"exchange already registered: {name}")
    _REGISTRY[name] = factory


def get_exchange(
    name: str,
    config: Optional[ExchangeConfig] = None,
) -> CryptoExchange:
    """Instantiate the exchange registered under ``name``."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown crypto exchange: {name!r}; available: {available}"
        )
    return _REGISTRY[name](config or ExchangeConfig())


def available_exchanges() -> List[str]:
    return sorted(_REGISTRY)


__all__ = [
    "CryptoExchange",
    "ExchangeConfig",
    "DEFAULT_MARKET_TYPE",
    "register_exchange",
    "get_exchange",
    "available_exchanges",
]
