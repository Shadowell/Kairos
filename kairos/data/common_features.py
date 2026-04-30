"""Market-agnostic technical features.

These factors depend only on OHLCV columns that spot and derivatives venues
provide. Crypto-specific factors such as funding rate and basis live behind
:meth:`MarketAdapter.market_features` and are merged in by
:func:`kairos.data.features.build_features`.

All calculations obey the project-wide "no future information leakage" rule:
at index ``t`` we only use data with index ``≤ t``. Exogenous factors are
normalised with a 60-bar rolling z-score where appropriate.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Technical-indicator helpers (private, re-exported via build_features)
# ---------------------------------------------------------------------------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    return macd, signal, macd - signal


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return ma, ma + k * sd, ma - k * sd


def _roc(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(periods=n)


def _parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    ln_hl = np.log(high / low)
    return np.sqrt((ln_hl**2).rolling(window).mean() / (4 * np.log(2)))


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).cumsum()


def _mfi(high, low, close, volume, period: int = 14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    mfr = pos_mf / (neg_mf + 1e-9)
    return 100 - 100 / (1 + mfr)


def rolling_z(s: pd.Series, window: int = 60) -> pd.Series:
    """60-bar rolling z-score used throughout the feature stack."""
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / (sd + 1e-9)


# ---------------------------------------------------------------------------
# Canonical list of common (market-agnostic) feature columns.
# Total: 24. Paired with a market adapter that contributes 8 more columns,
# the full EXOG_COLS vector is 32 long, matching the model's `n_exog`.
# ---------------------------------------------------------------------------
COMMON_EXOG_COLS: List[str] = [
    # Returns
    "log_ret_1", "log_ret_5", "log_ret_20",
    # Momentum
    "rsi_14", "macd_hist", "roc_5", "roc_20",
    # Volatility
    "atr_14", "parkinson_20", "vol_std_20",
    # MA deviation
    "ma5_dev", "ma20_dev", "ma60_dev",
    # Bollinger
    "boll_z",
    # Volume/price
    "obv_z", "mfi_14", "amount_z", "vwap_dev",
    # Micro-structure
    "amplitude", "upper_shadow", "lower_shadow", "body_ratio",
    # Room for common extensions (kept as pads so every market can fill them
    # in consistently if we ever agree on more cross-market factors).
    "pad_common_0", "pad_common_1",
]


def build_common_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append market-agnostic features to ``df`` and return the result.

    Parameters
    ----------
    df : DataFrame with at least ``open``, ``high``, ``low``, ``close``,
        ``volume``, ``amount`` columns, sorted ascending by ``datetime``.

    Returns
    -------
    DataFrame with the input columns plus every name in
    :data:`COMMON_EXOG_COLS`. Values are raw (no clip, no fillna); the top
    level :func:`kairos.data.features.build_features` applies a shared
    clean-up step once all feature groups have been merged.
    """

    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"].astype(float)
    a = df["amount"].astype(float) if "amount" in df.columns else (c * v)

    # Returns
    df["log_ret_1"] = np.log(c / c.shift(1))
    df["log_ret_5"] = np.log(c / c.shift(5))
    df["log_ret_20"] = np.log(c / c.shift(20))

    # Momentum
    df["rsi_14"] = _rsi(c, 14) / 100.0 - 0.5
    _, _, hist = _macd(c)
    df["macd_hist"] = hist / (c.rolling(20).std() + 1e-9)
    df["roc_5"] = _roc(c, 5)
    df["roc_20"] = _roc(c, 20)

    # Volatility
    df["atr_14"] = _atr(h, l, c, 14) / (c + 1e-9)
    df["parkinson_20"] = _parkinson_vol(h, l, 20)
    df["vol_std_20"] = df["log_ret_1"].rolling(20).std()

    # MA deviation
    for n in (5, 20, 60):
        ma = c.rolling(n).mean()
        df[f"ma{n}_dev"] = (c - ma) / (ma + 1e-9)

    # Bollinger position
    ma20, up, lo = _bollinger(c, 20, 2)
    df["boll_z"] = (c - ma20) / (up - lo + 1e-9) * 2

    # Volume / price
    df["obv_z"] = rolling_z(_obv(c, v), 60)
    df["mfi_14"] = _mfi(h, l, c, v, 14) / 100.0 - 0.5
    df["amount_z"] = rolling_z(np.log1p(a), 60)
    vwap = a / (v + 1e-9)
    df["vwap_dev"] = (c - vwap) / (vwap + 1e-9)

    # Microstructure
    df["amplitude"] = (h - l) / (c.shift(1) + 1e-9)
    df["upper_shadow"] = (h - np.maximum(o, c)) / (h - l + 1e-9)
    df["lower_shadow"] = (np.minimum(o, c) - l) / (h - l + 1e-9)
    df["body_ratio"] = (c - o).abs() / (h - l + 1e-9)

    # Pads (kept at 0 until we agree on more cross-market factors)
    df["pad_common_0"] = 0.0
    df["pad_common_1"] = 0.0

    return df


__all__ = [
    "COMMON_EXOG_COLS",
    "build_common_features",
    "rolling_z",
]
