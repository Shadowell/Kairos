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
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from ..common_features import rolling_z
from .base import FeatureContext, FetchTask, MarketAdapter, register_adapter
from .crypto_exchanges import (
    CryptoExchange,
    ExchangeConfig,
    available_exchanges,
    get_exchange,
)


log = logging.getLogger("kairos.market.crypto")


DEFAULT_EXCHANGE = "okx"
"""Default venue when ``KAIROS_CRYPTO_EXCHANGE`` is not set."""


def _align_series(source, dt: pd.Series) -> pd.Series:
    """Align an optional external series to the feature window timestamps.

    Returns a Series with NaN for every bar when ``source`` is missing or
    not alignable. The caller decides the fill policy (zero, forward-fill,
    clip) so this helper stays pure.
    """

    if source is None:
        return pd.Series(np.nan, index=range(len(dt)))
    if isinstance(source, pd.Series):
        s = source
    elif isinstance(source, pd.DataFrame):
        if source.shape[1] == 0:
            return pd.Series(np.nan, index=range(len(dt)))
        s = source.iloc[:, 0]
    else:
        arr = np.asarray(source).reshape(-1)
        if len(arr) != len(dt):
            return pd.Series(np.nan, index=range(len(dt)))
        return pd.Series(arr, index=range(len(dt)))

    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s = s.copy()
            s.index = pd.to_datetime(s.index)
        except Exception:  # noqa: BLE001
            return pd.Series(np.nan, index=range(len(dt)))
    reindexed = s.reindex(dt.values, method="ffill")
    reindexed.index = range(len(dt))
    return reindexed


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

    #: 8 market-specific columns. The first 5 get populated once
    #: funding/OI/basis/dominance series are supplied via ``FeatureContext``;
    #: until then they stay at 0 so the exog vector keeps a stable shape.
    #: ``hour_sin`` / ``hour_cos`` encode the 24h intraday cycle (crypto has
    #: no session breaks, so this is the intraday-seasonality hook).
    MARKET_EXOG_COLS = (
        "funding_rate",
        "funding_rate_z",
        "oi_change",
        "basis",
        "btc_dominance",
        "hour_sin",
        "hour_cos",
        "pad_crypto_0",
    )

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
        """Return the top-N USDT-quoted symbols ranked by 24h quote volume.

        The ranking logic is venue-specific (ccxt ``fetch_tickers`` for
        some, raw REST for others), so we prefer an exchange-side hook
        when one is available and fall back to the ccxt path otherwise.
        This keeps :class:`CryptoAdapter` decoupled from ccxt — venues
        like :class:`BinanceVisionExchange`, which serve a public data
        mirror without ccxt, plug in cleanly.
        """

        ex = self.exchange
        hook = getattr(ex, "list_symbols_by_volume", None)
        if callable(hook):
            return hook(top_n)

        markets = ex.list_markets()
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

        ccxt_client = getattr(ex, "_ccxt", None)
        if ccxt_client is None:
            # No ccxt and no volume hook: best-effort, preserve exchange order.
            log.warning(
                f"exchange {ex.name!r} has no list_symbols_by_volume hook "
                "and no ccxt client; returning markets in their reported "
                "order, which is NOT a liquidity ranking."
            )
            return [m["symbol"] for m in candidates[:top_n]]

        tickers = ccxt_client.fetch_tickers([m["symbol"] for m in candidates])

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
    # Auxiliary channels (funding / open interest / spot basis)
    # ------------------------------------------------------------------
    def fetch_extras(
        self,
        task: FetchTask,
        kinds: "Iterable[str]",
        *,
        oi_freq: str = "5min",
    ) -> "dict[str, pd.DataFrame]":
        """Fetch the auxiliary channels listed in ``kinds`` for one symbol.

        Parameters
        ----------
        task : FetchTask
            Same fetch window / symbol as :meth:`fetch_ohlcv`; only ``freq``
            is used to pick a sensible OI timeframe and to drive spot
            collection cadence.
        kinds : iterable of str
            Subset of :data:`kairos.data.crypto_extras.ALL_KINDS`. Unknown
            kinds are ignored so callers can just pass the raw CLI flag
            through.
        oi_freq : str
            OKX only exposes OI history at ``1m/5m/15m/1h/...`` granularity
            but the 1m endpoint returns empty for most symbols, so default
            to ``5min`` which is the densest grain that reliably has data.

        Returns
        -------
        dict
            Keys are the extras kinds (``"funding"`` / ``"open_interest"`` /
            ``"spot"``), values are DataFrames with ``datetime`` +
            single payload column ready to be handed to
            :mod:`kairos.data.crypto_extras` writers.

        Notes
        -----
        * This method is **not** on the :class:`MarketAdapter` ABC because
          it is crypto-specific. ``kairos.data.collect`` imports the
          adapter directly and only calls this when ``market == "crypto"``
          and ``--crypto-extras`` is non-empty.
        * Exchanges that don't implement a particular endpoint surface as
          ``AttributeError`` / ``RuntimeError``; we log and skip rather
          than aborting the whole task.
        """

        from .. import crypto_extras as _ce

        kinds = {k for k in kinds if k in _ce.ALL_KINDS}
        if not kinds:
            return {}

        start_ms = _to_unix_ms(task.start)
        end_ms = _to_unix_ms(task.end, end_of_day=True)
        out: dict[str, pd.DataFrame] = {}
        ex = self.exchange

        if _ce.KIND_FUNDING in kinds:
            hook = getattr(ex, "fetch_funding_rate_history", None)
            if hook is None:
                log.warning(
                    f"exchange {ex.name!r} has no fetch_funding_rate_history; "
                    "skipping funding"
                )
            else:
                try:
                    df = hook(task.symbol, start_ms=start_ms, end_ms=end_ms)
                except Exception as e:  # noqa: BLE001
                    log.warning(f"[{task.symbol}] funding fetch failed: {e}")
                else:
                    if df is not None and len(df) > 0:
                        out[_ce.KIND_FUNDING] = _reset_datetime_index(df)

        if _ce.KIND_OI in kinds:
            hook = getattr(ex, "fetch_open_interest_history", None)
            if hook is None:
                log.warning(
                    f"exchange {ex.name!r} has no fetch_open_interest_history; "
                    "skipping OI"
                )
            else:
                try:
                    df = hook(task.symbol, freq=oi_freq, start_ms=start_ms, end_ms=end_ms)
                except Exception as e:  # noqa: BLE001
                    log.warning(f"[{task.symbol}] OI fetch failed: {e}")
                else:
                    if df is not None and len(df) > 0:
                        out[_ce.KIND_OI] = _reset_datetime_index(df)

        if _ce.KIND_SPOT in kinds:
            hook = getattr(ex, "fetch_spot_ohlcv", None)
            if hook is None:
                log.warning(
                    f"exchange {ex.name!r} has no fetch_spot_ohlcv; "
                    "skipping spot basis"
                )
            else:
                spot_symbol = _perp_to_spot_symbol(task.symbol)
                try:
                    df = hook(
                        spot_symbol,
                        freq=task.freq,
                        start_ms=start_ms,
                        end_ms=end_ms,
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning(f"[{task.symbol}] spot {spot_symbol} fetch failed: {e}")
                else:
                    if df is not None and len(df) > 0:
                        # fetch_spot_ohlcv returns the canonical STD_COLS; we
                        # only keep datetime + close for basis computation.
                        spot = df[["datetime", "close"]].copy()
                        out[_ce.KIND_SPOT] = spot

        return out

    # ------------------------------------------------------------------
    # Market-specific features
    # ------------------------------------------------------------------
    def market_features(
        self,
        df: pd.DataFrame,
        *,
        context: FeatureContext | None = None,
    ) -> pd.DataFrame:
        """Crypto exogenous features.

        Five data-driven factors (``funding_rate``, ``funding_rate_z``,
        ``oi_change``, ``basis``, ``btc_dominance``) expect to receive
        pre-aligned series via ``context.extras`` keyed as follows:

        * ``"funding_rate"`` — DataFrame/Series indexed by datetime, rate
          per 8h settlement, usually sourced from
          :meth:`OkxExchange.fetch_funding_rate_history`.
        * ``"open_interest"`` — Series of OI snapshots; we compute
          log-change internally.
        * ``"spot_close"`` — spot close price aligned to the perp bars;
          we compute basis as ``(perp / spot - 1)``.
        * ``"btc_dominance"`` — pre-computed BTC market-cap dominance
          (0-1) at each bar.

        When a series is missing, the column is filled with 0.0 so the
        resulting exog vector has the expected width.

        The last three columns (``hour_sin``, ``hour_cos``, ``pad_crypto_0``)
        are computed locally from the timestamp and never rely on external
        series, so they always carry real signal.
        """

        out = pd.DataFrame(index=df.index)
        n = len(df)
        dt = (
            pd.to_datetime(df["datetime"])
            if "datetime" in df.columns
            else pd.to_datetime(df.index)
        )
        extras = context.extras if context is not None else {}

        # --- funding rate ---
        funding = _align_series(extras.get("funding_rate"), dt)
        out["funding_rate"] = funding.fillna(0.0).values
        out["funding_rate_z"] = rolling_z(funding.fillna(0.0), 60).fillna(0.0).values

        # --- open-interest change (log-delta) ---
        oi = _align_series(extras.get("open_interest"), dt)
        oi_change = np.log(oi / oi.shift(1))
        out["oi_change"] = oi_change.replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0).values

        # --- basis = perp_close / spot_close - 1 ---
        spot = _align_series(extras.get("spot_close"), dt)
        perp_close = df["close"].astype(float).values
        with np.errstate(divide="ignore", invalid="ignore"):
            basis = perp_close / spot.values - 1.0
        basis = np.where(np.isfinite(basis), basis, 0.0)
        out["basis"] = basis

        # --- btc dominance (0..1) ---
        dominance = _align_series(extras.get("btc_dominance"), dt)
        out["btc_dominance"] = dominance.fillna(0.0).values

        # --- 24h intraday cycle (always computable) ---
        hour_of_day = dt.dt.hour.astype(float) + dt.dt.minute.astype(float) / 60.0
        radians = 2.0 * np.pi * (hour_of_day.values / 24.0)
        out["hour_sin"] = np.sin(radians)
        out["hour_cos"] = np.cos(radians)

        # --- local pad ---
        out["pad_crypto_0"] = 0.0

        if len(out) != n:
            raise RuntimeError(
                f"market_features length mismatch: expected {n}, got {len(out)}"
            )
        return out

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


def _reset_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Lift a DatetimeIndex into a column so the extras writer can handle it.

    The exchange helpers (``fetch_funding_rate_history`` etc.) usually
    return DataFrames indexed by datetime; :mod:`crypto_extras` writers
    expect a ``datetime`` *column* instead.
    """

    if isinstance(df.index, pd.DatetimeIndex) and "datetime" not in df.columns:
        out = df.reset_index().rename(columns={df.index.name or "index": "datetime"})
        return out
    return df


def _perp_to_spot_symbol(perp_symbol: str) -> str:
    """Map ``BTC/USDT:USDT`` → ``BTC/USDT`` for the spot basis channel.

    ccxt encodes a USDT-margined perpetual as ``BASE/QUOTE:SETTLE``; the
    spot market for basis computation is the plain ``BASE/QUOTE`` form.
    If the symbol already looks like a spot one we pass it through.
    """

    if ":" in perp_symbol:
        return perp_symbol.split(":", 1)[0]
    return perp_symbol


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
