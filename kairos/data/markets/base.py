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
from dataclasses import dataclass, field
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
class FeatureContext:
    """Auxiliary data the feature pipeline may hand to adapters.

    This is a grab-bag on purpose: different markets need different side
    channels (A-shares want an index K-line for relative returns, crypto
    wants funding/OI history keyed by ccxt symbol, ...), and forcing every
    adapter to accept a fixed argument list would either explode into 20
    keyword arguments or pollute the base class.
    """

    #: Adapter-native symbol currently being processed, e.g. ``"600000"`` or
    #: ``"BTC/USDT:USDT"``. Adapters use this to look up per-symbol auxiliary
    #: series inside ``extras``.
    symbol: str | None = None
    #: Optional reference index K-line (used by A-shares for excess return).
    index_df: pd.DataFrame | None = None
    #: Free-form adapter-specific payload (crypto funding history, OI,
    #: spot-vs-perp basis, ...). Adapters are expected to know their own
    #: keys; unknown keys are ignored.
    extras: dict = field(default_factory=dict)


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
        return self.out_dir / f"{sanitize_symbol(self.symbol)}.parquet"


def sanitize_symbol(symbol: str) -> str:
    """Map a venue-native symbol to a filesystem-safe stem.

    A-share codes (``600000``) pass through unchanged. ccxt symbols such as
    ``BTC/USDT:USDT`` become ``BTC_USDT-USDT`` — the mapping is reversible by
    convention (first ``/`` → ``_``, first ``:`` → ``-``) and keeps the stem
    legible when you ``ls raw/crypto/1min``.
    """
    return symbol.replace("/", "_").replace(":", "-")


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

    #: Canonical list of market-specific feature names the adapter produces.
    #: Concrete adapters override this. ``build_features`` relies on the
    #: length being exactly ``n_exog - len(COMMON_EXOG_COLS)`` so the final
    #: exog vector has a stable dimension (32 by default) across markets.
    MARKET_EXOG_COLS: tuple[str, ...] = ()

    def market_features(  # noqa: D401
        self,
        df: pd.DataFrame,
        *,
        context: "FeatureContext | None" = None,
    ) -> pd.DataFrame:
        """Return market-specific exogenous features as additional columns.

        The default implementation fills ``MARKET_EXOG_COLS`` with zeros so
        that every adapter contributes the same vector width even if the
        venue doesn't expose (or hasn't wired up) certain factors yet.
        Concrete adapters override this to plug in e.g. ``turnover`` for
        A-shares or ``funding_rate`` for crypto perps.

        Parameters
        ----------
        df : DataFrame
            Input OHLCV window. Adapters should not mutate it; return a
            *new* frame with the same row index and only the extra columns
            in :attr:`MARKET_EXOG_COLS`.
        context : FeatureContext, optional
            Auxiliary data the pipeline may pass in (e.g. the index K-line
            for A-share excess returns, or funding-rate history for crypto).
            Adapters that don't need it can ignore it.
        """

        out = pd.DataFrame(index=df.index)
        for col in self.MARKET_EXOG_COLS:
            out[col] = 0.0
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[..., MarketAdapter]] = {}


def register_adapter(
    name: str,
    factory: Callable[..., MarketAdapter],
    *,
    overwrite: bool = False,
) -> None:
    """Register an adapter factory under ``name``.

    Factories are callables that *may* accept keyword arguments so that
    dispatchers can forward market-specific configuration (e.g. ``proxy``
    for crypto venues) without the registry knowing the details.

    Factories are used instead of instances so that optional heavy
    dependencies, such as ``ccxt``, are only imported when the user actually
    selects that market.

    Re-registering the same name is an error by default; pass
    ``overwrite=True`` if you intentionally want to swap in a different
    implementation (used mostly by tests that reload the module).
    """

    if name in _REGISTRY and not overwrite:
        raise ValueError(f"adapter already registered: {name}")
    _REGISTRY[name] = factory


def get_adapter(name: str, **kwargs) -> MarketAdapter:
    """Instantiate the adapter registered under ``name``.

    Extra keyword arguments are forwarded to the factory; factories that do
    not accept them will raise ``TypeError``, which callers can catch to fall
    back to defaults.

    Unknown names raise :class:`ValueError` and list the available adapters so
    users don't have to grep the codebase.
    """

    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise ValueError(
            f"unknown market adapter: {name!r}; available: {available}"
        )
    factory = _REGISTRY[name]
    try:
        return factory(**kwargs)
    except TypeError:
        if kwargs:
            return factory()
        raise


def available_adapters() -> List[str]:
    return sorted(_REGISTRY)


__all__ = [
    "STD_COLS",
    "FetchTask",
    "FeatureContext",
    "MarketAdapter",
    "register_adapter",
    "get_adapter",
    "available_adapters",
    "sanitize_symbol",
]
