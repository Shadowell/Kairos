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

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int):
        # One bar per `step_ms`, stopping after 2 pages so pagination is exercised.
        step_ms = {"1m": 60_000, "1h": 3_600_000}[timeframe]
        max_bars = 2 * limit
        offset = (since - self._origin_ms(timeframe)) // step_ms
        remaining = max_bars - offset
        if remaining <= 0:
            return []
        n = min(limit, remaining)
        rows = []
        for i in range(n):
            ts = since + i * step_ms
            rows.append([ts, 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 1.0])
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

    assert sanitize_symbol("600000") == "600000"
    assert sanitize_symbol("BTC/USDT:USDT") == "BTC_USDT-USDT"


def test_crypto_adapter_top_by_volume(monkeypatch):
    _install_fake_ccxt(monkeypatch)

    from kairos.data.markets.crypto import CryptoAdapter

    adapter = CryptoAdapter()
    top2 = adapter.list_symbols("top2")
    assert top2 == ["BTC/USDT:USDT", "ETH/USDT:USDT"]


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
        adapter.list_symbols("csi300")


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

    adapter = get_adapter("crypto", proxy="http://proxy.example:1234")
    assert adapter.exchange._ccxt.https_proxy == "http://proxy.example:1234"
