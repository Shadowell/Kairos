"""Unit tests for the binance_vision public-mirror backend.

These tests are strictly offline — every HTTP call is intercepted by
:func:`_fake_get` so we exercise pagination, symbol normalisation,
top-N ranking, and the explicit NotImplementedError paths without any
network dependency.
"""

from __future__ import annotations

import pytest

pytest.importorskip("requests")

import pandas as pd

from kairos.data.markets import get_adapter
from kairos.data.markets.crypto_exchanges import get_exchange
from kairos.data.markets.crypto_exchanges.binance_vision import (
    BASE_URL,
    BinanceVisionExchange,
)


# ---------------------------------------------------------------------------
# Symbol normalisation
# ---------------------------------------------------------------------------
def test_to_native_symbol_variants():
    assert BinanceVisionExchange.to_native_symbol("BTCUSDT") == "BTCUSDT"
    assert BinanceVisionExchange.to_native_symbol("BTC/USDT") == "BTCUSDT"
    # Perp form gets flattened with a warning (warning itself not asserted;
    # see caplog fixture in the other test).
    assert BinanceVisionExchange.to_native_symbol("BTC/USDT:USDT") == "BTCUSDT"
    assert BinanceVisionExchange.to_native_symbol("eth/usdt") == "ETHUSDT"


def test_to_native_symbol_warns_on_perp(caplog):
    with caplog.at_level("WARNING"):
        BinanceVisionExchange.to_native_symbol("BTC/USDT:USDT")
    assert any("no perp endpoint" in r.message for r in caplog.records)


def test_to_unified_symbol_handles_known_quotes():
    assert BinanceVisionExchange.to_unified_symbol("BTCUSDT") == "BTC/USDT"
    assert BinanceVisionExchange.to_unified_symbol("ETHUSDC") == "ETH/USDC"
    assert BinanceVisionExchange.to_unified_symbol("SOLBUSD") == "SOL/BUSD"


# ---------------------------------------------------------------------------
# HTTP behaviour (mocked)
# ---------------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _FakeSession:
    """Replaces requests.Session in the exchange."""

    def __init__(self):
        self.calls = []
        self.headers = {}
        self.proxies = {}
        self._payloads = []

    def queue(self, payload):
        self._payloads.append(payload)

    def get(self, url, params=None, timeout=None):  # noqa: D401
        self.calls.append((url, dict(params or {})))
        if not self._payloads:
            raise AssertionError(
                f"unexpected request: url={url} params={params}; "
                "no payloads queued"
            )
        return _FakeResponse(self._payloads.pop(0))


def _install_fake_session(ex: BinanceVisionExchange) -> _FakeSession:
    fake = _FakeSession()
    ex._session = fake  # type: ignore[attr-defined]
    return fake


def _make_kline(open_ms: int, step_ms: int = 60_000, price: float = 100.0):
    """Build a 12-field kline row matching Binance's schema."""
    return [
        open_ms,
        str(price),
        str(price + 1),
        str(price - 1),
        str(price + 0.5),
        "1.0",
        open_ms + step_ms - 1,
        str(price * 1.0),
        10,
        "0.5",
        str(price * 0.5),
        "0",
    ]


# ---------------------------------------------------------------------------
# fetch_ohlcv pagination
# ---------------------------------------------------------------------------
def test_fetch_ohlcv_paginates_until_end_ms():
    ex = BinanceVisionExchange()
    fake = _install_fake_session(ex)

    step = 60_000
    start_ms = 0
    end_ms = step * 2500  # 2500 bars → 3 pages at 1000-bar limit

    page1 = [_make_kline(i * step) for i in range(1000)]
    page2 = [_make_kline((1000 + i) * step) for i in range(1000)]
    page3 = [_make_kline((2000 + i) * step) for i in range(500)]
    fake.queue(page1)
    fake.queue(page2)
    fake.queue(page3)

    df = ex.fetch_ohlcv(
        symbol="BTC/USDT", freq="1min",
        start_ms=start_ms, end_ms=end_ms,
    )

    assert len(df) == 2500
    assert list(df.columns) == [
        "datetime", "open", "high", "low", "close", "volume", "amount",
        "turnover", "pct_chg", "vwap",
    ]
    # Cursor should have advanced across all 3 pages
    assert len(fake.calls) == 3
    # First call starts at start_ms; subsequent calls start at cursor > 0
    assert fake.calls[0][1]["startTime"] == 0
    assert fake.calls[1][1]["startTime"] > 0


def test_fetch_ohlcv_trims_bars_past_end_ms():
    ex = BinanceVisionExchange()
    fake = _install_fake_session(ex)

    step = 60_000
    # Server returns 10 bars, but we only want the first 5.
    fake.queue([_make_kline(i * step) for i in range(10)])
    df = ex.fetch_ohlcv(
        symbol="BTCUSDT", freq="1min",
        start_ms=0, end_ms=5 * step,
    )
    assert len(df) == 5


def test_fetch_ohlcv_empty_response_stops_after_three_pages():
    ex = BinanceVisionExchange()
    fake = _install_fake_session(ex)
    for _ in range(3):
        fake.queue([])

    # end_ms needs to be far enough ahead that three empty-page skips
    # (1000 * 60_000 ms each) still fit inside the loop.
    end_ms = 60_000 * 1000 * 5
    df = ex.fetch_ohlcv(
        symbol="BTCUSDT", freq="1min",
        start_ms=0, end_ms=end_ms,
    )
    assert df.empty
    assert len(fake.calls) == 3


def test_fetch_ohlcv_rejects_unknown_freq():
    ex = BinanceVisionExchange()
    _install_fake_session(ex)
    with pytest.raises(ValueError, match="not supported"):
        ex.fetch_ohlcv("BTCUSDT", "7min", 0, 1)


# ---------------------------------------------------------------------------
# list_markets / list_symbols_by_volume
# ---------------------------------------------------------------------------
def test_list_markets_filters_non_trading():
    ex = BinanceVisionExchange()
    fake = _install_fake_session(ex)
    fake.queue({
        "symbols": [
            {"symbol": "BTCUSDT", "status": "TRADING",
             "baseAsset": "BTC", "quoteAsset": "USDT"},
            {"symbol": "FAKEUSDT", "status": "BREAK",
             "baseAsset": "FAKE", "quoteAsset": "USDT"},
            {"symbol": "ETHUSDT", "status": "TRADING",
             "baseAsset": "ETH", "quoteAsset": "USDT"},
        ],
    })
    markets = ex.list_markets()
    assert {m["symbol"] for m in markets} == {"BTC/USDT", "ETH/USDT"}
    assert all(m["spot"] and not m["swap"] for m in markets)


def test_list_symbols_by_volume_ranks_and_filters_quote():
    ex = BinanceVisionExchange()
    fake = _install_fake_session(ex)
    fake.queue([
        {"symbol": "BTCUSDT", "quoteVolume": "100"},
        {"symbol": "ETHUSDT", "quoteVolume": "300"},
        {"symbol": "BNBUSDT", "quoteVolume": "200"},
        {"symbol": "BTCBTC", "quoteVolume": "999"},  # wrong quote, excluded
    ])
    top = ex.list_symbols_by_volume(top_n=2, quote="USDT")
    assert top == ["ETH/USDT", "BNB/USDT"]


# ---------------------------------------------------------------------------
# Unsupported channels raise NotImplementedError
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", [
    "fetch_funding_rate_history",
    "fetch_open_interest_history",
    "fetch_spot_ohlcv",
])
def test_perp_channels_not_supported(method):
    ex = BinanceVisionExchange()
    with pytest.raises(NotImplementedError):
        getattr(ex, method)()


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------
def test_registry_contains_binance_vision():
    ex = get_exchange("binance_vision")
    assert isinstance(ex, BinanceVisionExchange)


def test_crypto_adapter_top_via_binance_vision_uses_hook():
    adapter = get_adapter("crypto", exchange="binance_vision")
    fake = _install_fake_session(adapter.exchange)
    fake.queue([
        {"symbol": "ETHUSDT", "quoteVolume": "300"},
        {"symbol": "BTCUSDT", "quoteVolume": "100"},
    ])
    symbols = adapter.list_symbols("top2")
    assert symbols == ["ETH/USDT", "BTC/USDT"]
    # Exactly one ranked-tickers call, no exchangeInfo load — confirms the
    # adapter took the fast path via list_symbols_by_volume.
    assert len(fake.calls) == 1
    assert fake.calls[0][0].endswith("/api/v3/ticker/24hr")
