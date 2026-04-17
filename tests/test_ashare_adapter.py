"""Smoke tests for the A-share market adapter.

These tests do not hit the network — they verify the pure parts of
:class:`AshareAdapter` that do not depend on akshare's upstream (universe
parsing for ad-hoc symbol lists, symbol-prefix helpers, and the adapter's
presence in the registry).
"""

from __future__ import annotations

from kairos.data.markets import available_adapters, get_adapter
from kairos.data.markets.ashare import AshareAdapter, _sina_symbol


def test_ashare_registered():
    assert "ashare" in available_adapters()
    adapter = get_adapter("ashare")
    assert isinstance(adapter, AshareAdapter)
    assert "daily" in adapter.supported_freqs
    assert "1min" in adapter.supported_freqs


def test_ashare_ad_hoc_universe():
    adapter = get_adapter("ashare")
    codes = adapter.list_symbols("600977,000001,300750")
    assert codes == ["600977", "000001", "300750"]


def test_ashare_zero_padding():
    adapter = get_adapter("ashare")
    # Ad-hoc path should zfill short codes to 6 digits.
    assert adapter.list_symbols("1,2") == ["000001", "000002"]


def test_sina_symbol_prefix_mapping():
    assert _sina_symbol("600000") == "sh600000"
    assert _sina_symbol("688981") == "sh688981"
    assert _sina_symbol("000001") == "sz000001"
    assert _sina_symbol("300750") == "sz300750"
    assert _sina_symbol("831010") == "bj831010"


def test_ashare_ignores_unknown_kwargs(tmp_path):
    # The dispatcher may forward kwargs like proxy=... that the A-share
    # adapter does not understand. Ensure it degrades gracefully instead of
    # crashing with TypeError.
    adapter = get_adapter("ashare", proxy="http://ignored")
    assert isinstance(adapter, AshareAdapter)
