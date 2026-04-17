"""Smoke tests for feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.data import EXOG_COLS, build_features


def _synthetic_df(n: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    close = 10 + np.cumsum(rng.normal(0, 0.2, size=n))
    high = close + rng.uniform(0, 0.3, size=n)
    low = close - rng.uniform(0, 0.3, size=n)
    open_ = close + rng.normal(0, 0.1, size=n)
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({
        "datetime": dates,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
        "amount": volume * close,
        "turnover": rng.uniform(0.5, 5.0, size=n),
    })


def test_build_features_shape():
    df = _synthetic_df()
    out = build_features(df)
    # 所有外生列都在
    for col in EXOG_COLS:
        assert col in out.columns, f"missing {col}"
    assert out[EXOG_COLS].shape[1] == 32


def test_build_features_no_nan_after_warmup():
    df = _synthetic_df()
    out = build_features(df)
    # 跳过前 60 行冷启动窗口后应该没有 NaN
    assert out[EXOG_COLS].iloc[60:].isna().sum().sum() == 0


def test_build_features_clipped():
    df = _synthetic_df()
    out = build_features(df)
    v = out[EXOG_COLS].values
    assert np.all(v >= -5) and np.all(v <= 5)
