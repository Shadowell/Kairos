"""Market adapter abstractions.

A *market adapter* encapsulates all market-specific concerns so that the rest
of the pipeline (factors, dataset packing, training, backtest) can stay
market-agnostic.

Adapters are responsible for:

1. Resolving a named *universe* (e.g. ``csi300`` / ``top10``) into a list of
   raw instrument identifiers.
2. Fetching raw OHLCV bars and standardising them into the shared
   :data:`STD_COLS` schema.
3. Exposing a trading calendar so that dataset packing knows which timestamps
   are legitimate bars.
4. (Optionally) computing market-specific exogenous features that the common
   feature builder cannot produce on its own (e.g. ``turnover`` for A-shares,
   ``funding_rate`` for crypto perps).

Concrete adapters register themselves through :func:`register_adapter` so that
CLI callers can simply pass ``--market ashare`` / ``--market crypto``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd


STD_COLS: List[str] = [
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "pct_chg",
    "vwap",
]
"""Canonical column order produced by every adapter.

Columns that a given market cannot provide (e.g. ``turnover`` for crypto)
should be present with ``NaN`` values so that downstream code can rely on a
stable schema.
"""


@dataclass
class FetchTask:
    """Describes a single-symbol fetch request."""

    symbol: str
    freq: str
    start: str
    end: str
    out_dir: Path
    adjust: str = ""
    extra: Optional[dict] = None

    @property
    def out_path(self) -> Path:
        return self.out_dir / f"{self.symbol}.parquet"


class MarketAdapter(ABC):
    """Base class for all market-specific data adapters."""

    name: str = ""
    supported_freqs: tuple[str, ...] = ()

    @abstractmethod
    def list_symbols(self, universe: str) -> List[str]:
        """Resolve a named universe to a list of raw instrument identifiers."""

    @abstractmethod
    def fetch_ohlcv(self, task: FetchTask) -> pd.DataFrame:
        """Fetch one symbol's OHLCV data.

        Must return a DataFrame containing the columns in :data:`STD_COLS`;
        missing optional columns should be filled with ``NaN``.
        """

    @abstractmethod
    def trading_calendar(
        self, start: datetime, end: datetime, freq: str
    ) -> pd.DatetimeIndex:
        """Return the timestamps that should contain a bar in ``[start, end]``.

        For 24/7 markets this is simply a contiguous range; for session-based
        markets (A-shares) it should exclude non-trading days and lunch breaks.
        """

    def market_features(self, df: pd.DataFrame) -> pd.DataFrame:  # noqa: D401
        """Return market-specific exogenous features as additional columns.

        The default implementation returns an empty DataFrame (no extra
        factors). Concrete adapters override this to plug in e.g. ``turnover``
        stats for A-shares or ``funding_rate`` for crypto perps.
        """

        return pd.DataFrame(index=df.index)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[[], MarketAdapter]] = {}


def register_adapter(name: str, factory: Callable[[], MarketAdapter]) -> None:
    """Register an adapter factory under ``name``.

    Factories are used (instead of instances) so that optional heavy
    dependencies, such as ``ccxt``, are only imported when the user actually
    selects that market.
    """

    if name in _REGISTRY:
        raise ValueError(f"adapter already registered: {name}")
    _REGISTRY[name] = factory


def get_adapter(name: str) -> MarketAdapter:
    """Instantiate the adapter registered under ``name``.

    Unknown names raise :class:`ValueError` and list the available adapters so
    users don't have to grep the codebase.
    """

    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown market adapter: {name!r}; available: {available}"
        )
    return _REGISTRY[name]()


def available_adapters() -> List[str]:
    return sorted(_REGISTRY)


__all__ = [
    "STD_COLS",
    "FetchTask",
    "MarketAdapter",
    "register_adapter",
    "get_adapter",
    "available_adapters",
]
