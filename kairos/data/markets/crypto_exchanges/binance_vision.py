"""Binance public data mirror adapter (``data-api.binance.vision``).

This backend exists for **one reason**: the main Binance endpoints
(``api.binance.com``, ``fapi.binance.com``) are frequently blocked from
corporate / office networks in mainland China, while
``data-api.binance.vision`` — Binance's public historical-data CDN — is
normally reachable from the same network (same TLS cert, but served out of
a different edge). It lets Kairos run end-to-end data collection and
training smoke tests from an office desk while waiting for a "real" venue
(OKX perp, Binance perp) to be reachable.

Capabilities
------------
* **Spot K-line only.** The mirror exposes the Spot REST API
  (``/api/v3/klines``, ``/api/v3/exchangeInfo``, ``/api/v3/ticker/24hr``);
  the Futures endpoints (``/fapi/*``) return 404 on this host.
* **No credentials, no trading.** Only public market data. Kairos never
  issues authenticated calls through this backend.
* **No funding / open-interest / basis.** Those live on the perp endpoints
  that this mirror does not proxy; :meth:`fetch_funding_rate_history` and
  friends raise :class:`NotImplementedError` so callers hit an explicit
  error rather than silent zeros.

Because of those limitations this backend is tagged as a *degraded*
channel in the documentation: use it to validate plumbing end-to-end,
then switch to OKX (or native Binance) for real training runs.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import pandas as pd

from .base import CryptoExchange, ExchangeConfig, register_exchange


log = logging.getLogger("kairos.crypto.binance_vision")


BASE_URL = "https://data-api.binance.vision"
"""Binance's public data mirror. Do not swap for ``api.binance.com``
without re-verifying reachability — that host is blocked from many
corporate networks this backend is specifically designed to serve."""


# Kairos freq → Binance interval string
_FREQ_TO_INTERVAL: Dict[str, str] = {
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


_INTERVAL_MS: Dict[str, int] = {
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


class BinanceVisionExchange(CryptoExchange):
    """Read-only Binance public-mirror backend.

    Symbols follow the Binance spot convention internally (``BTCUSDT``),
    but :meth:`fetch_ohlcv` and :meth:`list_markets` accept both the
    Binance-native form (``BTCUSDT``) and the ccxt-style unified form
    (``BTC/USDT``). The ccxt-style "perp" form ``BTC/USDT:USDT`` is also
    accepted — we strip the ``:USDT`` suffix with a warning, since no
    perpetual data is actually available on this venue.
    """

    name = "binance_vision"
    supported_freqs = tuple(_FREQ_TO_INTERVAL.keys())

    #: Binance Spot REST caps /api/v3/klines at 1000 bars per request.
    #: We use 1000 to minimise the number of round trips at 1-minute
    #: granularity (2 years × 525k bars × 2 symbols / 1000 ≈ 1050 calls).
    _PAGE_LIMIT = 1000

    def __init__(self, config: Optional[ExchangeConfig] = None) -> None:
        super().__init__(config)
        try:
            import requests  # noqa: F401  stdlib-adjacent; always available in our env
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The binance_vision backend requires `requests`; install "
                "it with `pip install requests` (normally already present)."
            ) from e

        import requests

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "kairos-kronos/binance_vision",
            "Accept": "application/json",
        })

        proxy = self.config.resolve_proxy()
        if proxy:
            self._session.proxies = {"http": proxy, "https": proxy}
            log.info(f"binance_vision using proxy: {proxy}")

        self._timeout = max(1, int(self.config.timeout_ms / 1000))

    # ------------------------------------------------------------------
    # Symbol normalisation
    # ------------------------------------------------------------------
    @staticmethod
    def to_native_symbol(symbol: str) -> str:
        """Convert various symbol formats to Binance's ``BTCUSDT`` form.

        Accepts:
        * ``"BTCUSDT"`` — returned as-is.
        * ``"BTC/USDT"`` — slash stripped.
        * ``"BTC/USDT:USDT"`` — ccxt perpetual form; the ``:USDT`` is
          dropped (this venue has no perpetual data), and a warning is
          emitted so users aren't surprised when basis / funding columns
          stay at zero.
        """

        s = symbol.strip()
        if not s:
            raise ValueError(f"empty symbol: {symbol!r}")
        if ":" in s:
            head, _, _settle = s.partition(":")
            log.warning(
                f"binance_vision has no perp endpoint; treating {symbol!r} "
                f"as the equivalent spot pair {head!r}. funding / OI / basis "
                f"features will remain unset."
            )
            s = head
        return s.replace("/", "").upper()

    @staticmethod
    def to_unified_symbol(native: str) -> str:
        """Format a native Binance spot symbol back into ``BASE/QUOTE``."""
        native = native.upper()
        # Prefer the obvious "USDT", "USDC", "BUSD" suffixes; fall back to 3-char
        # quote for anything else (conservative heuristic — Binance spot has
        # ~20 quotes but the long tail is negligible for our use).
        for q in ("USDT", "USDC", "BUSD", "FDUSD", "TUSD"):
            if native.endswith(q) and len(native) > len(q):
                return f"{native[:-len(q)]}/{q}"
        return f"{native[:-3]}/{native[-3:]}"

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------
    def list_markets(self) -> List[Dict]:
        """Return all active USDT-quoted spot markets.

        The payload shape is a subset of ccxt's ``load_markets()`` output —
        we only populate the fields ``CryptoAdapter._top_by_volume`` reads
        (``symbol``, ``active``, ``base``, ``quote``, ``swap``, ``linear``,
        ``info``). ``swap`` is always ``False`` here: this is a spot
        mirror.
        """

        payload = self._get_json(f"{BASE_URL}/api/v3/exchangeInfo")
        raw = payload.get("symbols", []) or []
        out: List[Dict] = []
        for s in raw:
            if s.get("status") != "TRADING":
                continue
            quote = s.get("quoteAsset", "")
            base = s.get("baseAsset", "")
            native = s.get("symbol", "")
            if not native or not base or not quote:
                continue
            out.append({
                "symbol": f"{base}/{quote}",
                "native_symbol": native,
                "active": True,
                "swap": False,
                "linear": True,
                "spot": True,
                "base": base,
                "quote": quote,
                "info": s,
            })
        return out

    def list_symbols_by_volume(self, top_n: int, quote: str = "USDT") -> List[str]:
        """Return the top-N spot symbols ranked by 24h quote volume.

        This is the hook ``CryptoAdapter`` calls in place of the ccxt-only
        ``fetch_tickers``. Keeping the ranking logic inside the backend
        means the adapter doesn't need to know whether the venue speaks
        ccxt or raw REST.
        """

        tickers = self._get_json(f"{BASE_URL}/api/v3/ticker/24hr")
        filtered = [
            t for t in tickers
            if t.get("symbol", "").endswith(quote)
        ]
        ranked = sorted(
            filtered,
            key=lambda t: float(t.get("quoteVolume", 0) or 0),
            reverse=True,
        )
        return [self.to_unified_symbol(t["symbol"]) for t in ranked[:top_n]]

    def fetch_ohlcv(
        self,
        symbol: str,
        freq: str,
        start_ms: int,
        end_ms: int,
    ) -> pd.DataFrame:
        if freq not in _FREQ_TO_INTERVAL:
            raise ValueError(
                f"freq {freq!r} not supported by binance_vision; "
                f"available: {list(_FREQ_TO_INTERVAL)}"
            )
        native = self.to_native_symbol(symbol)
        interval = _FREQ_TO_INTERVAL[freq]
        step_ms = _INTERVAL_MS[interval]

        rows: list[list] = []
        cursor = start_ms
        empty_in_a_row = 0
        while cursor < end_ms:
            batch = self._fetch_klines_with_retry(
                symbol=native,
                interval=interval,
                start_ms=cursor,
                end_ms=end_ms,
                limit=self._PAGE_LIMIT,
            )
            if not batch:
                empty_in_a_row += 1
                # Skip a whole page forward and retry; bail after 3 empties.
                cursor += step_ms * self._PAGE_LIMIT
                if empty_in_a_row >= 3:
                    log.debug(f"[{native}] 3 empty pages in a row; stopping")
                    break
                continue
            empty_in_a_row = 0
            rows.extend(batch)
            last_open_ms = int(batch[-1][0])
            next_cursor = last_open_ms + step_ms
            if next_cursor <= cursor:
                break  # defensive; never walk backwards
            cursor = next_cursor

        if not rows:
            return self.normalise_frame(pd.DataFrame(columns=["datetime"]))

        # Binance kline row layout:
        # [open_ms, open, high, low, close, volume, close_ms, quote_volume,
        #  trades, taker_base_vol, taker_quote_vol, ignore]
        df = pd.DataFrame(
            rows,
            columns=[
                "open_ms", "open", "high", "low", "close", "volume",
                "close_ms", "amount", "trades", "taker_base_vol",
                "taker_quote_vol", "ignore",
            ],
        )
        df = df.drop_duplicates("open_ms").sort_values("open_ms")
        df = df[df["open_ms"] < end_ms]
        df["datetime"] = pd.to_datetime(
            df["open_ms"], unit="ms", utc=True
        ).dt.tz_convert(None)
        # Binance reports "amount" as quote-asset volume directly; no
        # approximation needed. vwap ≈ amount / volume (close is fine as a
        # placeholder; downstream code only uses vwap for a deviation
        # factor, not for pricing).
        return self.normalise_frame(df[[
            "datetime", "open", "high", "low", "close", "volume", "amount",
        ]])

    # ------------------------------------------------------------------
    # Not-supported-on-this-venue channels
    # ------------------------------------------------------------------
    # The three following methods intentionally raise NotImplementedError.
    # If a user picks binance_vision thinking they can get funding / OI /
    # spot-vs-perp basis, we want them to see an explicit failure rather
    # than a silent-zero feature column that pollutes training.
    def fetch_funding_rate_history(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError(
            "binance_vision only exposes Binance Spot data; funding-rate "
            "history is a Futures (fapi) endpoint. Use OKX or native "
            "Binance Futures for perp-derived features."
        )

    def fetch_open_interest_history(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError(
            "binance_vision only exposes Binance Spot data; open-interest "
            "history is a Futures (fapi) endpoint. Use OKX or native "
            "Binance Futures."
        )

    def fetch_spot_ohlcv(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError(
            "binance_vision already is the spot mirror; call fetch_ohlcv "
            "directly on this backend instead."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _fetch_klines_with_retry(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        limit: int,
        retries: int = 3,
        pause: float = 0.5,
    ) -> list:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                return self._get_json(
                    f"{BASE_URL}/api/v3/klines", params=params
                )
            except Exception as e:  # noqa: BLE001 — ALSO catches network errors
                last_err = e
                log.debug(
                    f"[{symbol}] klines attempt {attempt + 1}/{retries} "
                    f"failed: {e}"
                )
                time.sleep(pause * (2**attempt))
        assert last_err is not None
        raise last_err

    def _get_json(self, url: str, params: Optional[Dict] = None):
        """Thin wrapper around ``requests.get`` with a JSON response.

        Raises ``requests.HTTPError`` on non-2xx so the retry layer above
        gets a clean exception to back off on. Kept on the instance (not a
        module-level function) so tests can patch ``self._session`` and
        observe the requested URL / params deterministically.
        """

        resp = self._session.get(url, params=params, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()


def _factory(config: ExchangeConfig) -> CryptoExchange:
    return BinanceVisionExchange(config)


register_exchange("binance_vision", _factory, overwrite=True)


__all__ = ["BinanceVisionExchange", "BASE_URL"]
