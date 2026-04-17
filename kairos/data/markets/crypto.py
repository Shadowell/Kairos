"""Crypto market adapter.

Plugs the cross-venue :mod:`crypto_exchanges` layer into Kairos'
:class:`~kairos.data.markets.base.MarketAdapter` contract so that
``kairos-collect --market crypto`` is a first-class alternative to
``--market ashare``.

Design notes
------------
* The adapter is **venue-pluggable**: by default it uses OKX (registered in
  :mod:`crypto_exchanges.okx`), but you can swap in Binance, Bybit, etc. by
  registering a new :class:`CryptoExchange` and selecting it through the
  ``extra["exchange"]`` field of :class:`FetchTask` or the environment
  variable ``KAIROS_CRYPTO_EXCHANGE``.
* Universe resolution supports three call patterns:
    1. Named presets (``top10`` / ``top50``) — resolved against the venue's
       live market list, ranked by 24h quote volume.
    2. A comma-separated list of ccxt symbols (``BTC/USDT:USDT,ETH/USDT:USDT``).
    3. A single ccxt symbol (same format).
* The trading calendar for 24/7 crypto is simply a dense range at the chosen
  frequency.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .base import FetchTask, MarketAdapter, register_adapter
from .crypto_exchanges import (
    CryptoExchange,
    ExchangeConfig,
    available_exchanges,
    get_exchange,
)


log = logging.getLogger("kairos.market.crypto")


DEFAULT_EXCHANGE = "okx"
"""Default venue when ``KAIROS_CRYPTO_EXCHANGE`` is not set."""


# ---------------------------------------------------------------------------
# Freq conversion for the trading calendar
# ---------------------------------------------------------------------------
_FREQ_TO_PANDAS: dict[str, str] = {
    "1min": "1min",
    "3min": "3min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "60min": "1h",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1D",
    "daily": "1D",
}


class CryptoAdapter(MarketAdapter):
    """Crypto market adapter (USDT-margined perpetuals by default)."""

    name = "crypto"
    supported_freqs = tuple(_FREQ_TO_PANDAS.keys())

    def __init__(
        self,
        exchange: Optional[str] = None,
        config: Optional[ExchangeConfig] = None,
    ) -> None:
        self._exchange_name = (
            exchange
            or os.environ.get("KAIROS_CRYPTO_EXCHANGE")
            or DEFAULT_EXCHANGE
        )
        if self._exchange_name not in available_exchanges():
            raise ValueError(
                f"crypto exchange {self._exchange_name!r} not available; "
                f"installed: {available_exchanges()}. "
                "Make sure `ccxt` is installed via `pip install "
                "'kairos-kronos[crypto]'`."
            )
        self._config = config or ExchangeConfig()
        self._exchange: Optional[CryptoExchange] = None  # lazy-instantiate

    # ------------------------------------------------------------------
    # Lazy venue instantiation
    # ------------------------------------------------------------------
    @property
    def exchange(self) -> CryptoExchange:
        if self._exchange is None:
            self._exchange = get_exchange(self._exchange_name, self._config)
        return self._exchange

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------
    def list_symbols(self, universe: str) -> List[str]:
        universe = universe.strip()

        # 1) Ad-hoc comma-separated list, or a single ccxt symbol.
        if "," in universe or "/" in universe:
            items = [s.strip() for s in universe.split(",") if s.strip()]
            return items

        # 2) Named preset.
        if universe.lower().startswith("top"):
            try:
                top_n = int(universe[3:])
            except ValueError as e:
                raise ValueError(
                    f"universe {universe!r} must look like 'top10', 'top50', ..."
                ) from e
            return self._top_by_volume(top_n)

        raise ValueError(
            f"unknown crypto universe: {universe!r}; expected 'topN' or a "
            "comma-separated list of ccxt symbols (e.g. 'BTC/USDT:USDT,ETH/USDT:USDT')"
        )

    def _top_by_volume(self, top_n: int) -> List[str]:
        """Return the top-N USDT-quoted perpetuals ranked by 24h quote volume."""
        markets = self.exchange.list_markets()
        # Keep linear USDT-margined perpetuals only; this is the contract type
        # that matches DEFAULT_MARKET_TYPE="swap" and the EXOG factors Kairos
        # will add in Phase 2 (funding rate, open interest).
        candidates = [
            m
            for m in markets
            if m.get("active", True)
            and m.get("swap", False)
            and m.get("linear", False)
            and m.get("quote") == "USDT"
        ]

        tickers = self.exchange._ccxt.fetch_tickers(  # ccxt: intentional
            [m["symbol"] for m in candidates]
        )

        def key(sym: str) -> float:
            t = tickers.get(sym) or {}
            return float(t.get("quoteVolume") or 0.0)

        ranked = sorted(
            (m["symbol"] for m in candidates),
            key=key,
            reverse=True,
        )
        return ranked[:top_n]

    # ------------------------------------------------------------------
    # OHLCV
    # ------------------------------------------------------------------
    def fetch_ohlcv(self, task: FetchTask) -> pd.DataFrame:
        if task.freq not in self.supported_freqs:
            raise ValueError(
                f"freq {task.freq!r} not supported for crypto; "
                f"available: {list(self.supported_freqs)}"
            )
        start_ms = _to_unix_ms(task.start)
        end_ms = _to_unix_ms(task.end, end_of_day=True)
        return self.exchange.fetch_ohlcv(
            symbol=task.symbol,
            freq=task.freq,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------
    def trading_calendar(
        self,
        start: datetime,
        end: datetime,
        freq: str,
    ) -> pd.DatetimeIndex:
        if freq not in _FREQ_TO_PANDAS:
            raise ValueError(f"freq {freq!r} not supported for crypto calendar")
        return pd.date_range(start, end, freq=_FREQ_TO_PANDAS[freq])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_unix_ms(value: str | datetime, end_of_day: bool = False) -> int:
    """Parse ``YYYY-MM-DD`` / datetime to millisecond unix timestamp (UTC)."""
    if isinstance(value, datetime):
        ts = value
    else:
        ts = datetime.fromisoformat(value.replace("/", "-"))
    if end_of_day and ts.hour == 0 and ts.minute == 0 and ts.second == 0:
        # Treat a bare date as inclusive of the whole day.
        ts = ts.replace(hour=23, minute=59, second=59, microsecond=999_000)
    return int(ts.timestamp() * 1000)


# ---------------------------------------------------------------------------
# Registry entry — lazy so that missing ccxt doesn't blow up pure A-share users.
# ---------------------------------------------------------------------------
def _factory(**kwargs) -> MarketAdapter:
    """Adapter factory used by :func:`kairos.data.markets.get_adapter`.

    Accepts optional keyword arguments so the dispatcher can forward
    configuration such as ``proxy`` without coupling the registry to
    crypto-specific fields.
    """
    proxy = kwargs.pop("proxy", None)
    exchange = kwargs.pop("exchange", None)
    config = ExchangeConfig(proxy=proxy) if proxy else None
    return CryptoAdapter(exchange=exchange, config=config)


register_adapter("crypto", _factory, overwrite=True)


__all__ = ["CryptoAdapter", "DEFAULT_EXCHANGE"]
