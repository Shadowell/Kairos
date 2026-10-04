"""Regression tests for funding windows and unavailable VWAP measurements."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from kairos.data.common_features import build_common_features
from kairos.data.features import EXOG_COLS, build_features
from kairos.data.markets.base import FeatureContext
from kairos.data.markets.crypto import CryptoAdapter


def _bars(n: int = 180, freq: str = "1min") -> pd.DataFrame:
    close = 100.0 + np.arange(n) / 100.0
    volume = np.full(n, 10.0)
    return pd.DataFrame({
        "datetime": pd.date_range("2026-04-01", periods=n, freq=freq),
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": volume,
        "amount": close * volume,
    })


def _funding_features(bars: pd.DataFrame, rates: pd.Series | None) -> pd.DataFrame:
    return CryptoAdapter().market_features(
        bars, context=FeatureContext(extras={"funding_rate": rates})
    )


def test_funding_signal_survives_between_eight_hour_settlements():
    bars = _bars(5 * 24 * 60)
    rates = pd.Series(
        np.tile([0.0001, 0.0005, -0.0001], 5),
        index=pd.date_range("2026-04-01", periods=15, freq="8h"),
    )

    out = _funding_features(bars, rates)
    late = out["funding_rate_z"].iloc[4 * 24 * 60:]

    assert (late.abs() > 0.1).mean() > 0.95
    assert abs(out["funding_rate_z"].iloc[-1]) > 0.1


def test_funding_window_expires_by_elapsed_time():
    bars = _bars(4)
    bars["datetime"] = pd.Timestamp("2026-04-01") + pd.to_timedelta(
        [0, 8, 16, 80], unit="h"
    )
    rates = pd.Series([0.0001, 0.0002, 0.0003, 0.0004], index=bars["datetime"])

    out = _funding_features(bars, rates)

    # At hour 80 the right-closed three-day window contains only hours 16/80.
    assert out["funding_rate_z"].iloc[-1] == pytest.approx(1 / np.sqrt(2))


def test_missing_funding_history_does_not_become_zero_rate_observations():
    bars = _bars(1000)
    rates = pd.Series([0.0002], index=pd.DatetimeIndex([bars["datetime"].iloc[120]]))

    out = _funding_features(bars, rates)

    assert (out["funding_rate"].iloc[:120] == 0).all()
    assert (out["funding_rate_z"] == 0).all()


@pytest.mark.parametrize("missing", [True, False])
def test_missing_or_constant_funding_is_neutral(missing):
    bars = _bars(200)
    rates = None if missing else pd.Series(
        [0.0001], index=pd.DatetimeIndex([bars["datetime"].iloc[0]])
    )
    assert (_funding_features(bars, rates)["funding_rate_z"] == 0).all()


def test_future_data_does_not_change_historical_features():
    bars = _bars(5000)
    rates = pd.Series(
        np.arange(12, dtype=float) * 0.0001,
        index=pd.date_range("2026-04-01", periods=12, freq="8h"),
    )
    prefix_len = 3000
    prefix = build_features(bars.iloc[:prefix_len], extras={"funding_rate": rates})
    future_rates = rates.copy()
    future_rates.loc[future_rates.index > bars["datetime"].iloc[prefix_len - 1]] = 9.0
    bars.loc[prefix_len:, "amount"] *= 2
    full = build_features(bars, extras={"funding_rate": future_rates})

    pd.testing.assert_frame_equal(prefix[EXOG_COLS], full[EXOG_COLS].iloc[:prefix_len])
    assert len(EXOG_COLS) == 32
    assert np.isfinite(full[EXOG_COLS].to_numpy()).all()


@pytest.mark.parametrize("volume,amount", [
    (0.0, 0.0), (0.0, 100.0), (-1.0, 100.0),
    (np.inf, 100.0), (np.nan, 100.0), (10.0, 0.0),
    (10.0, -1.0), (10.0, np.inf),
])
def test_invalid_vwap_measurements_are_neutral_before_clipping(volume, amount, caplog):
    bars = _bars()
    bars.loc[100, ["volume", "amount"]] = [volume, amount]

    with caplog.at_level(logging.INFO, logger="kairos.common_features"):
        out = build_common_features(bars)

    assert out.loc[100, "vwap_dev"] == 0.0
    assert np.isfinite(out["vwap_dev"]).all()
    assert "unavailable" in caplog.text
    assert build_features(bars).loc[100, "vwap_dev"] == 0.0


@pytest.mark.parametrize("amount_mode", ["missing", "nan", "proxy"])
def test_proxy_amount_has_exactly_zero_vwap_deviation_and_diagnostic(amount_mode, caplog):
    bars = _bars()
    bars.loc[50:70, "volume"] = 1e-12
    bars["amount"] = bars["close"] * bars["volume"]
    if amount_mode == "missing":
        bars = bars.drop(columns="amount")
    elif amount_mode == "nan":
        bars.loc[30:90, "amount"] = np.nan

    with caplog.at_level(logging.INFO, logger="kairos.common_features"):
        out = build_common_features(bars)

    assert (out["vwap_dev"] == 0.0).all()
    assert "amount=close*volume" in caplog.text


def test_valid_quote_amount_preserves_real_vwap_deviation():
    bars = _bars()
    bars.loc[80, "amount"] = bars.loc[80, "volume"] * bars.loc[80, "close"] / 1.1
    bars.loc[100, "amount"] = bars.loc[100, "volume"] * bars.loc[100, "close"] / 0.8

    out = build_common_features(bars)

    assert out.loc[80, "vwap_dev"] == pytest.approx(0.1)
    assert out.loc[100, "vwap_dev"] == pytest.approx(-0.2)


@pytest.mark.parametrize("amount", [-0.5, -1.0, -2.0, np.inf, -np.inf])
def test_invalid_amount_is_excluded_before_log_standardization(amount):
    bars = _bars()
    bars.loc[100, "amount"] = amount

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = build_common_features(bars)

    assert pd.isna(out.loc[100, "amount_z"])
    assert not np.isinf(out["amount_z"]).any()
    assert out.loc[99, "amount_z"] == pytest.approx(build_common_features(_bars()).loc[99, "amount_z"])


def test_zero_quote_amount_remains_a_valid_amount_observation():
    bars = _bars()
    bars.loc[100, "amount"] = 0.0
    out = build_common_features(bars)
    history = np.log1p(bars.loc[41:100, "amount"].to_numpy())
    expected = (0.0 - history.mean()) / (history.std(ddof=1) + 1e-9)
    assert out.loc[100, "amount_z"] == pytest.approx(expected)
