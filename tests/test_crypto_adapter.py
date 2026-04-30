"""Unit tests for the crypto market adapter stack.

These tests never touch the network — the ccxt client is monkey-patched with
an in-memory fake that serves canned OHLCV pages, so they run deterministically
in CI and on every laptop.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# A minimal fake of the ccxt OKX client, injected into sys.modules before the
# adapter gets imported. Only covers the methods Kairos actually calls.
# ---------------------------------------------------------------------------
@dataclass
class _FakeOkx:
    opts: dict
    https_proxy: str | None = None
    http_proxy: str | None = None

    def load_markets(self) -> dict:
        return {
            "BTC/USDT:USDT": {
                "symbol": "BTC/USDT:USDT",
                "active": True,
                "swap": True,
                "linear": True,
                "quote": "USDT",
            },
            "ETH/USDT:USDT": {
                "symbol": "ETH/USDT:USDT",
                "active": True,
                "swap": True,
                "linear": True,
                "quote": "USDT",
            },
            "SOL/USDT:USDT": {
                "symbol": "SOL/USDT:USDT",
                "active": True,
                "swap": True,
                "linear": True,
                "quote": "USDT",
            },
            "BTC/USDT": {
                "symbol": "BTC/USDT",
                "active": True,
                "swap": False,  # spot, should be filtered out by topN resolver
                "linear": True,
                "quote": "USDT",
            },
        }

    def fetch_tickers(self, symbols: List[str]) -> dict:
        volumes = {
            "BTC/USDT:USDT": 5_000_000_000,
            "ETH/USDT:USDT": 2_000_000_000,
            "SOL/USDT:USDT":   800_000_000,
        }
        return {s: {"quoteVolume": volumes.get(s, 0)} for s in symbols}

    options: dict = None  # set post-init so defaultType flip can be tested

    def __post_init__(self):
        # Mirror the real ccxt client shape the adapter reads.
        self.options = {"defaultType": self.opts.get("options", {}).get("defaultType", "swap")}
        self.has = {
            "fetchFundingRateHistory": True,
            "fetchOpenInterestHistory": True,
        }

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int):
        # One bar per `step_ms`, stopping after 2 pages so pagination is exercised.
        step_ms = {"1m": 60_000, "5m": 300_000, "1h": 3_600_000}[timeframe]
        max_bars = 2 * limit
        offset = (since - self._origin_ms(timeframe)) // step_ms
        remaining = max_bars - offset
        if remaining <= 0:
            return []
        n = min(limit, remaining)
        rows = []
        # Spot is 1% below perp so basis is a known non-zero constant.
        price_offset = -1.0 if self.options.get("defaultType") == "spot" else 0.0
        for i in range(n):
            ts = since + i * step_ms
            base = 100.0 + i + price_offset
            rows.append([ts, base, base + 1, base - 1, base + 0.5, 1.0])
        return rows

    def fetch_funding_rate_history(
        self, symbol: str, since: int | None = None, limit: int = 100,
        params: dict | None = None,
    ):
        # The fake has exactly 3 funding settlements starting at a fixed epoch.
        # The real OKX adapter paginates *backwards* via params={"after": T}
        # returning rows with ts < T; mimic that here so the adapter's
        # cursor-advance loop terminates on one pass.
        step_ms = 8 * 3_600_000
        origin = 1_700_000_000_000
        all_rows = [
            {"timestamp": origin + i * step_ms, "fundingRate": 0.0001 * (i + 1)}
            for i in range(3)
        ]
        params = params or {}
        after = params.get("after")
        if after is not None:
            return [r for r in all_rows if r["timestamp"] < int(after)][:limit]
        # ccxt's generic ``since`` path (older tests).
        if since is not None:
            return [r for r in all_rows if r["timestamp"] >= since][:limit]
        return all_rows[:limit]

    def fetch_open_interest_history(
        self, symbol: str, timeframe: str, since: int | None = None,
        limit: int = 100, params: dict | None = None,
    ):
        step_ms = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}[timeframe]
        params = params or {}
        # Real adapter now passes {"begin": start_ms, "end": end_ms}.
        begin = params.get("begin", since if since is not None else 0)
        end = params.get("end", begin + step_ms * 5)
        rows = []
        n_max = min(limit, 5, max(1, (end - begin) // step_ms))
        for i in range(n_max):
            ts = begin + i * step_ms
            if ts >= end:
                break
            rows.append({"timestamp": ts, "openInterestAmount": 1000.0 + i})
        return rows

    def _origin_ms(self, timeframe: str) -> int:
        # Pretend the earliest candle starts at a known epoch so pagination is
        # deterministic regardless of the real clock.
        return 1_700_000_000_000


def _install_fake_ccxt(monkeypatch):
    fake = type(sys)("ccxt")
    fake.okx = lambda opts: _FakeOkx(opts=opts)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ccxt", fake)
    # Drop cached modules so they pick up the fake ccxt.
    for name in list(sys.modules):
        if name.startswith("kairos.data.markets.crypto"):
            monkeypatch.delitem(sys.modules, name, raising=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_sanitize_symbol_roundtrip():
    from kairos.data.markets.base import sanitize_symbol

    assert sanitize_symbol("BTC/USDT:USDT") == "BTC_USDT-USDT"
    assert sanitize_symbol("BTC/USDT") == "BTC_USDT"


def test_crypto_adapter_top_by_volume(monkeypatch):
    _install_fake_ccxt(monkeypatch)

    from kairos.data.markets.crypto import CryptoAdapter

    adapter = CryptoAdapter()
    top2 = adapter.list_symbols("top2")
    assert top2 == ["BTC/USDT:USDT", "ETH/USDT:USDT"]


def test_crypto_adapter_spot_top_by_volume(monkeypatch):
    _install_fake_ccxt(monkeypatch)

    from kairos.data.markets.crypto import CryptoAdapter

    adapter = CryptoAdapter(market_type="spot")
    top1 = adapter.list_symbols("top1")
    assert top1 == ["BTC/USDT"]


def test_crypto_adapter_ad_hoc_symbols(monkeypatch):
    _install_fake_ccxt(monkeypatch)

    from kairos.data.markets.crypto import CryptoAdapter

    adapter = CryptoAdapter()
    got = adapter.list_symbols("BTC/USDT:USDT, ETH/USDT:USDT")
    assert got == ["BTC/USDT:USDT", "ETH/USDT:USDT"]

    # Single symbol (no comma) also works.
    assert adapter.list_symbols("BTC/USDT:USDT") == ["BTC/USDT:USDT"]


def test_crypto_adapter_rejects_unknown_universe(monkeypatch):
    _install_fake_ccxt(monkeypatch)

    from kairos.data.markets.crypto import CryptoAdapter

    adapter = CryptoAdapter()
    with pytest.raises(ValueError, match="unknown crypto universe"):
        adapter.list_symbols("not-a-universe")


def test_crypto_adapter_fetch_ohlcv_paginates(monkeypatch):
    _install_fake_ccxt(monkeypatch)

    from kairos.data.markets.base import FetchTask
    from kairos.data.markets.crypto import CryptoAdapter

    adapter = CryptoAdapter()
    task = FetchTask(
        symbol="BTC/USDT:USDT",
        freq="1min",
        start="2023-11-15",
        end="2023-11-16",
        out_dir=Path("/tmp/unused"),
    )
    df = adapter.fetch_ohlcv(task)
    assert not df.empty
    # The fake serves 2 pages × 300 bars = 600 rows max; since start < origin
    # we expect the full 600 window (or fewer after `<end_ms` trim).
    assert len(df) > 0
    # Canonical schema is preserved.
    from kairos.data.markets.base import STD_COLS

    assert list(df.columns) == STD_COLS
    # Datetime is monotonic and unique.
    assert df["datetime"].is_monotonic_increasing
    assert df["datetime"].is_unique


def test_crypto_adapter_proxy_forwarded(monkeypatch):
    _install_fake_ccxt(monkeypatch)

    from kairos.data.markets.crypto import CryptoAdapter
    from kairos.data.markets.crypto_exchanges import ExchangeConfig

    adapter = CryptoAdapter(config=ExchangeConfig(proxy="http://127.0.0.1:7890"))
    # Touching `.exchange` instantiates the underlying ccxt client.
    assert adapter.exchange._ccxt.https_proxy == "http://127.0.0.1:7890"


def test_trading_calendar_24_7(monkeypatch):
    _install_fake_ccxt(monkeypatch)

    from kairos.data.markets.crypto import CryptoAdapter

    adapter = CryptoAdapter()
    cal = adapter.trading_calendar(
        datetime(2024, 1, 1), datetime(2024, 1, 3), "1h"
    )
    # 48 full hours between Jan-1 00:00 and Jan-3 00:00 inclusive → 49 stamps.
    assert len(cal) == 49
    assert cal[0] == pd.Timestamp("2024-01-01 00:00:00")
    assert cal[-1] == pd.Timestamp("2024-01-03 00:00:00")


def test_dispatcher_forwards_proxy(monkeypatch):
    _install_fake_ccxt(monkeypatch)
    # Reimport markets package so the crypto factory gets registered with the
    # fake ccxt in place.
    importlib.import_module("kairos.data.markets")

    from kairos.data.markets import get_adapter

    adapter = get_adapter(
        "crypto",
        proxy="http://proxy.example:1234",
        market_type="spot",
    )
    assert adapter.exchange._ccxt.https_proxy == "http://proxy.example:1234"
    assert adapter.exchange._ccxt.options["defaultType"] == "spot"


def test_crypto_adapter_fetch_extras_returns_per_kind(monkeypatch):
    _install_fake_ccxt(monkeypatch)

    from kairos.data.markets.base import FetchTask
    from kairos.data.markets.crypto import CryptoAdapter
    from kairos.data import crypto_extras as ce

    adapter = CryptoAdapter()
    task = FetchTask(
        symbol="BTC/USDT:USDT",
        freq="1min",
        start="2023-11-15",
        end="2023-11-16",
        out_dir=Path("/tmp/unused"),
    )
    extras = adapter.fetch_extras(task, kinds=[ce.KIND_FUNDING, ce.KIND_OI, ce.KIND_SPOT])
    assert set(extras.keys()) == {ce.KIND_FUNDING, ce.KIND_OI, ce.KIND_SPOT}

    funding = extras[ce.KIND_FUNDING]
    assert "datetime" in funding.columns and "funding_rate" in funding.columns
    assert len(funding) == 3

    oi = extras[ce.KIND_OI]
    assert "datetime" in oi.columns and "open_interest" in oi.columns
    assert len(oi) >= 1
    # OI payload is the extracted openInterestAmount, not the raw ms timestamp.
    assert oi["open_interest"].iloc[0] == 1000.0

    spot = extras[ce.KIND_SPOT]
    assert "datetime" in spot.columns and "close" in spot.columns
    # Spot is 1 below perp in the fake, verifying the defaultType flip works.
    assert spot["close"].iloc[0] < 100.0  # perp first bar is ~100.5


def test_crypto_extras_io_roundtrip(tmp_path):
    from kairos.data import crypto_extras as ce

    funding_df = pd.DataFrame({
        "datetime": pd.to_datetime([
            "2024-01-01 00:00",
            "2024-01-01 08:00",
            "2024-01-01 16:00",
        ]),
        "funding_rate": [0.0001, -0.00015, 0.00007],
    })
    ce.save_per_symbol(tmp_path, "BTC_USDT-USDT", ce.KIND_FUNDING, funding_df)

    oi_df = pd.DataFrame({
        "datetime": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:05"]),
        "open_interest": [1000.0, 1005.5],
    })
    ce.save_per_symbol(tmp_path, "BTC_USDT-USDT", ce.KIND_OI, oi_df)

    loaded = ce.load_for_symbol(tmp_path, "BTC_USDT-USDT")
    # Adapter-facing keys, not the on-disk kinds.
    assert "funding_rate" in loaded
    assert "open_interest" in loaded
    assert loaded["funding_rate"].iloc[0] == 0.0001
    assert loaded["open_interest"].iloc[1] == 1005.5

    channels = ce.available_channels(tmp_path)
    assert set(channels) == {ce.KIND_FUNDING, ce.KIND_OI}


def test_crypto_extras_missing_parquet_returns_empty(tmp_path):
    from kairos.data import crypto_extras as ce

    # No parquet at all → empty dict, not an error.
    assert ce.load_for_symbol(tmp_path, "BTC_USDT-USDT") == {}
    assert ce.available_channels(tmp_path) == []


def test_collect_resolve_extras_kinds():
    from kairos.data.collect import _resolve_extras_kinds

    assert _resolve_extras_kinds("", "crypto") == []
    assert _resolve_extras_kinds("funding", "crypto") == ["funding"]
    assert _resolve_extras_kinds("funding,open_interest", "crypto") == [
        "funding",
        "open_interest",
    ]
    # 'all' expands; de-dup keeps first occurrence.
    resolved = _resolve_extras_kinds("funding,all", "crypto")
    assert resolved[0] == "funding"
    assert "open_interest" in resolved and "spot" in resolved
    assert "reference" in resolved

    # Non-crypto market silently ignores (with warning).
    assert _resolve_extras_kinds("funding", "other") == []


def test_collect_resolve_extras_kinds_rejects_unknown():
    import pytest as _pytest

    from kairos.data.collect import _resolve_extras_kinds

    with _pytest.raises(SystemExit, match="unknown --crypto-extras"):
        _resolve_extras_kinds("bogus_kind", "crypto")


def test_build_features_reads_extras_via_adapter(monkeypatch):
    """End-to-end: collected extras → build_features → non-zero exog columns."""
    _install_fake_ccxt(monkeypatch)

    from kairos.data.features import build_features

    # Minimal perp OHLCV — 10 bars at 1min spacing.
    start_ts = pd.Timestamp("2024-01-01")
    perp = pd.DataFrame({
        "datetime": pd.date_range(start_ts, periods=10, freq="1min"),
        "open": [100.0 + i for i in range(10)],
        "high": [101.0 + i for i in range(10)],
        "low": [99.0 + i for i in range(10)],
        "close": [100.5 + i for i in range(10)],
        "volume": [1.0] * 10,
        "amount": [100.5 + i for i in range(10)],
        "turnover": [float("nan")] * 10,
        "pct_chg": [float("nan")] * 10,
        "vwap": [100.5 + i for i in range(10)],
    })

    # Spot is 1% below perp, producing a basis ≈ +0.01 after build_features.
    spot_close = pd.Series(
        [(100.5 + i) * 0.99 for i in range(10)],
        index=pd.date_range(start_ts, periods=10, freq="1min"),
        name="close",
    )
    funding = pd.Series(
        [0.0001, 0.0002],
        index=pd.to_datetime(["2023-12-31 16:00", "2024-01-01 00:00"]),
        name="funding_rate",
    )
    oi = pd.Series(
        [1000.0, 1005.0, 1010.0],
        index=pd.date_range(start_ts, periods=3, freq="5min"),
        name="open_interest",
    )

    extras = {
        "funding_rate": funding,
        "open_interest": oi,
        "spot_close": spot_close,
    }
    out = build_features(perp, market="crypto", extras=extras, symbol="BTC_USDT-USDT")

    # The "data-driven" market exog columns are now non-zero (at least one bar).
    assert (out["funding_rate"] != 0.0).any()
    assert (out["basis"] != 0.0).any()
    # Exog vector width is still 32.
    from kairos.data.features import exog_cols_for

    assert len(exog_cols_for("crypto")) == 32
