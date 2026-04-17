"""OKX crypto exchange adapter (via ccxt).

Focuses on the public market-data endpoints that Kairos' training pipeline
needs (candles, funding history, open interest); private endpoints are not
wired up here — if you later want to place orders, subclass or extend
:class:`OkxExchange` and plug in credentials via :class:`ExchangeConfig`.

OKX specifics worth knowing
---------------------------
* USDT-margined perpetual contracts use the ccxt symbol form
  ``BASE/USDT:USDT`` (e.g. ``BTC/USDT:USDT``).
* The OHLCV REST endpoint returns at most 300 bars per request — we paginate
  forwards from ``start_ms`` until we reach ``end_ms``.
* Rate limits are generous for public endpoints (20 req/2s per IP); ccxt's
  built-in throttle is on by default so we don't hand-roll one.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from .base import CryptoExchange, ExchangeConfig, register_exchange


log = logging.getLogger("kairos.crypto.okx")


# Kairos freq → OKX/ccxt timeframe string
_FREQ_TO_TIMEFRAME: Dict[str, str] = {
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "60min": "1h",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
    "daily": "1d",
}


_TIMEFRAME_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class OkxExchange(CryptoExchange):
    """OKX public market-data adapter built on ccxt."""

    name = "okx"
    supported_freqs = tuple(_FREQ_TO_TIMEFRAME.keys())

    _PAGE_LIMIT = 300  # OKX REST cap for /api/v5/market/candles

    def __init__(self, config: Optional[ExchangeConfig] = None) -> None:
        super().__init__(config)
        try:
            import ccxt  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "ccxt is required for the OKX adapter. "
                "Install it with `pip install 'kairos-kronos[crypto]'` "
                "or `pip install ccxt`."
            ) from e

        opts: Dict[str, object] = {
            "enableRateLimit": self.config.enable_rate_limit,
            "timeout": self.config.timeout_ms,
            "options": {
                "defaultType": self.config.market_type,
                **self.config.extra_options,
            },
        }
        creds = self.config.credentials or {}
        if {"apiKey", "secret", "password"}.issubset(creds.keys()):
            opts.update({
                "apiKey": creds["apiKey"],
                "secret": creds["secret"],
                "password": creds["password"],
            })

        self._ccxt = ccxt.okx(opts)

        proxy = self.config.resolve_proxy()
        if proxy:
            # ccxt accepts a per-instance proxy via the `proxies` mapping on
            # the underlying session; using the documented `https_proxy`
            # attribute is the supported cross-venue knob.
            self._ccxt.https_proxy = proxy
            self._ccxt.http_proxy = proxy
            log.info(f"OKX adapter using proxy: {proxy}")

    # ------------------------------------------------------------------
    # Market metadata
    # ------------------------------------------------------------------
    def list_markets(self) -> List[Dict]:
        markets = self._ccxt.load_markets()
        # ccxt returns a dict keyed by symbol; we want the values as a list.
        return list(markets.values())

    # ------------------------------------------------------------------
    # OHLCV fetch with pagination
    # ------------------------------------------------------------------
    def fetch_ohlcv(
        self,
        symbol: str,
        freq: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        if freq not in _FREQ_TO_TIMEFRAME:
            raise ValueError(
                f"freq {freq!r} not supported by OKX adapter; "
                f"available: {list(_FREQ_TO_TIMEFRAME)}"
            )
        timeframe = _FREQ_TO_TIMEFRAME[freq]
        step_ms = _TIMEFRAME_MS[timeframe]

        rows: list[list] = []
        cursor = start_ms
        empty_in_a_row = 0
        while cursor < end_ms:
            batch = self._fetch_ohlcv_with_retry(
                symbol=symbol,
                timeframe=timeframe,
                since=cursor,
                limit=self._PAGE_LIMIT,
            )
            if not batch:
                empty_in_a_row += 1
                # OKX occasionally returns empty when the requested window
                # lands in an outage; skip forward one page and keep trying,
                # but don't loop forever.
                cursor += step_ms * self._PAGE_LIMIT
                if empty_in_a_row >= 3:
                    log.debug(f"[{symbol}] 3 empty pages in a row, stopping")
                    break
                continue
            empty_in_a_row = 0
            rows.extend(batch)
            last_ts = batch[-1][0]
            # Advance past the last timestamp to avoid re-requesting the same
            # page. ccxt may return up to `limit` bars including `since`.
            next_cursor = last_ts + step_ms
            if next_cursor <= cursor:
                break  # defensive: never move backwards
            cursor = next_cursor

        if not rows:
            return self.normalise_frame(pd.DataFrame(columns=["datetime"]))

        df = pd.DataFrame(
            rows,
            columns=["ts", "open", "high", "low", "close", "volume"],
        )
        df = df.drop_duplicates("ts").sort_values("ts")
        df = df[df["ts"] < end_ms]
        df["datetime"] = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_convert(None)
        # OKX doesn't expose a USDT "amount" in the vanilla candles endpoint;
        # approximate with close * volume so downstream code that divides by
        # volume for VWAP still works. Mark turnover/pct_chg as NaN.
        df["amount"] = (df["close"] * df["volume"]).astype(float)
        df["vwap"] = df["close"]  # per-bar midpoint approximation
        return self.normalise_frame(df)

    # ------------------------------------------------------------------
    # Optional exogenous channels (funding / OI / spot)
    # ------------------------------------------------------------------
    # These helpers are *not* part of the core MarketAdapter contract — the
    # crypto adapter calls them opportunistically to enrich the exog vector.
    # They're intentionally thin wrappers around ccxt so the heavy lifting
    # (pagination, schema normalisation) lives in one place.
    def fetch_funding_rate_history(
        self,
        symbol: str,
        start_ms: int,
        end_ms: int,
        page_limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch historical funding rates for a perpetual swap.

        Returns a DataFrame indexed by datetime with a single ``funding_rate``
        column. OKX settles funding every 8h, so expect ~3 rows per day.

        Notes
        -----
        * Uses ``ccxt.fetchFundingRateHistory`` which forwards to
          ``GET /api/v5/public/funding-rate-history`` on OKX.
        * ccxt returns a list of dicts with ``timestamp`` + ``fundingRate``.
        * Paginates forward from ``start_ms`` until ``end_ms``; OKX caps
          at 100 records per call.
        """

        if not self._ccxt.has.get("fetchFundingRateHistory"):
            raise RuntimeError(
                "This ccxt build doesn't expose fetchFundingRateHistory; "
                "upgrade ccxt or implement the raw /public/funding-rate-history "
                "call directly."
            )

        rows: list[dict] = []
        cursor = start_ms
        while cursor < end_ms:
            batch = self._ccxt.fetch_funding_rate_history(
                symbol=symbol, since=cursor, limit=page_limit
            )
            if not batch:
                break
            rows.extend(batch)
            last_ts = batch[-1]["timestamp"]
            if last_ts <= cursor:
                break
            cursor = last_ts + 1

        if not rows:
            return pd.DataFrame(columns=["funding_rate"])
        df = pd.DataFrame(
            [
                {
                    "timestamp": r["timestamp"],
                    "funding_rate": float(r.get("fundingRate", 0.0)),
                }
                for r in rows
                if start_ms <= r["timestamp"] < end_ms
            ]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df = df.drop_duplicates("datetime").sort_values("datetime")
        return df.set_index("datetime")[["funding_rate"]]

    def fetch_open_interest_history(
        self,
        symbol: str,
        freq: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch historical open-interest snapshots for a perpetual swap.

        Returns a DataFrame indexed by datetime with a single
        ``open_interest`` column (contract count, as OKX reports).

        OKX exposes ``GET /api/v5/rubik/stat/contracts/open-interest-history``;
        ccxt wraps this as ``fetchOpenInterestHistory``.
        """

        if not self._ccxt.has.get("fetchOpenInterestHistory"):
            raise RuntimeError(
                "This ccxt build doesn't expose fetchOpenInterestHistory; "
                "upgrade ccxt (≥ 4.3) or call the OKX endpoint directly."
            )
        if freq not in _FREQ_TO_TIMEFRAME:
            raise ValueError(f"freq {freq!r} not supported; see {list(_FREQ_TO_TIMEFRAME)}")

        tf = _FREQ_TO_TIMEFRAME[freq]
        rows = self._ccxt.fetch_open_interest_history(
            symbol=symbol,
            timeframe=tf,
            since=start_ms,
            limit=min(500, (end_ms - start_ms) // _TIMEFRAME_MS[tf] + 1),
        )
        if not rows:
            return pd.DataFrame(columns=["open_interest"])
        df = pd.DataFrame(
            [
                {
                    "timestamp": r["timestamp"],
                    "open_interest": float(
                        r.get("openInterestAmount")
                        or r.get("openInterestValue")
                        or r.get("openInterest")
                        or 0.0
                    ),
                }
                for r in rows
                if start_ms <= r["timestamp"] < end_ms
            ]
        )
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df = df.drop_duplicates("datetime").sort_values("datetime")
        return df.set_index("datetime")[["open_interest"]]

    def fetch_spot_ohlcv(
        self,
        symbol: str,
        freq: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        """Fetch spot-market OHLCV for basis computation.

        ``symbol`` should be the *spot* ccxt symbol (e.g. ``"BTC/USDT"``),
        not the perp form. We temporarily flip ``options.defaultType`` to
        ``"spot"`` so ccxt routes to the right OKX endpoint, then restore
        the prior setting.
        """

        prev = self._ccxt.options.get("defaultType")
        try:
            self._ccxt.options["defaultType"] = "spot"
            return self.fetch_ohlcv(symbol, freq, start_ms, end_ms)
        finally:
            if prev is None:
                self._ccxt.options.pop("defaultType", None)
            else:
                self._ccxt.options["defaultType"] = prev

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fetch_ohlcv_with_retry(
        self,
        symbol: str,
        timeframe: str,
        since: int,
        limit: int,
        retries: int = 3,
        pause: float = 0.5,
    ) -> list:
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return self._ccxt.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                )
            except Exception as e:  # ccxt raises a wide taxonomy; log+retry
                last_err = e
                log.debug(
                    f"[{symbol}] fetch_ohlcv attempt {attempt + 1}/{retries} "
                    f"failed: {e}"
                )
                time.sleep(pause * (2**attempt))
        assert last_err is not None
        raise last_err


def _factory(config: ExchangeConfig) -> CryptoExchange:
    return OkxExchange(config)


register_exchange("okx", _factory, overwrite=True)


__all__ = ["OkxExchange"]


# ---------------------------------------------------------------------------
# Convenience: parse human-friendly date → unix-ms (used by the adapter and
# tests). Kept here so OKX-specific callers don't have to reinvent it.
# ---------------------------------------------------------------------------
def to_unix_ms(value: str | datetime) -> int:
    """Convert ``YYYY-MM-DD`` / datetime to millisecond unix timestamp (UTC)."""
    if isinstance(value, datetime):
        ts = value
    else:
        ts = datetime.fromisoformat(value)
    return int(ts.timestamp() * 1000)
