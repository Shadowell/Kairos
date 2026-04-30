"""Crypto feature pipeline tests.

Covers:
* :func:`kairos.data.common_features.build_common_features` produces exactly
  the 24-column canonical common block.
* :func:`kairos.data.features.build_features` wires common + adapter blocks
  into a 32-column crypto exog vector.
* ``kairos.data.prepare_dataset.process_symbol`` dispatches through the
  crypto adapter and the resulting exog frame has the expected column schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.common_features import COMMON_EXOG_COLS, build_common_features
from kairos.data.features import build_features, exog_cols_for
from kairos.data.markets import get_adapter
from kairos.data.prepare_dataset import parse_range, process_symbol


def _synthetic_df(n: int = 300, start: str = "2020-01-01", freq: str = "B") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n, freq=freq)
    rng = np.random.default_rng(7)
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


# ---------------------------------------------------------------------------
# common_features
# ---------------------------------------------------------------------------
def test_common_features_has_24_columns():
    assert len(COMMON_EXOG_COLS) == 24


def test_common_features_produces_all_columns():
    df = _synthetic_df()
    out = build_common_features(df)
    for col in COMMON_EXOG_COLS:
        assert col in out.columns, f"missing common feature: {col}"


def test_common_features_pads_are_zero():
    df = _synthetic_df()
    out = build_common_features(df)
    assert (out["pad_common_0"] == 0.0).all()
    assert (out["pad_common_1"] == 0.0).all()


# ---------------------------------------------------------------------------
# Adapter schemas add up to 8
# ---------------------------------------------------------------------------
def test_crypto_market_feats_schema_length():
    adapter = get_adapter("crypto")
    assert len(adapter.MARKET_EXOG_COLS) == 8


def test_common_plus_market_equals_32_for_crypto():
    cols = exog_cols_for("crypto")
    assert len(cols) == 32, (
        f"crypto produces {len(cols)} exog cols, expected 32"
    )


def test_exog_cols_unique_per_market():
    cols = exog_cols_for("crypto")
    assert len(set(cols)) == len(cols), (
        f"crypto has duplicate exog column names: {cols}"
    )


# ---------------------------------------------------------------------------
# build_features orchestrator
# ---------------------------------------------------------------------------
def test_build_features_default_crypto_schema():
    df = _synthetic_df()
    out = build_features(df)
    for col in exog_cols_for("crypto"):
        assert col in out.columns


def test_build_features_crypto_schema():
    df = _synthetic_df(n=500, start="2023-01-01 00:00", freq="1min")
    out = build_features(df, market="crypto")
    for col in exog_cols_for("crypto"):
        assert col in out.columns, f"missing crypto exog col: {col}"
    # hour_sin/hour_cos should have real signal (non-constant) when
    # timestamps cover > 1 hour.
    assert out["hour_sin"].nunique() > 1
    assert out["hour_cos"].nunique() > 1


def test_build_features_crypto_accepts_extras_for_funding():
    df = _synthetic_df(n=300, start="2023-01-01 00:00", freq="1min")
    # Craft a funding-rate series that covers the full window with a
    # constant value so alignment falls on every bar.
    funding = pd.Series(
        0.0001,
        index=pd.date_range(df["datetime"].iloc[0], periods=len(df), freq="1min"),
    )
    out = build_features(
        df, market="crypto",
        extras={"funding_rate": funding},
    )
    assert np.isclose(out["funding_rate"].iloc[-1], 0.0001)


def test_build_features_crypto_clips_and_fills():
    df = _synthetic_df(n=200, start="2023-01-01 00:00", freq="1min")
    out = build_features(df, market="crypto")
    cols = exog_cols_for("crypto")
    v = out[cols].values
    assert np.all(np.isfinite(v))
    assert np.all(v >= -5) and np.all(v <= 5)


# ---------------------------------------------------------------------------
# process_symbol integration (end-to-end feature plumbing)
# ---------------------------------------------------------------------------
def test_process_symbol_crypto_dispatches_through_adapter(tmp_path: Path):
    df = _synthetic_df(n=800, start="2023-01-01 00:00", freq="1min")
    pq = tmp_path / "BTC_USDT-USDT.parquet"
    df.to_parquet(pq)

    # ranges inside the synthetic window so every split gets rows
    ts0 = df["datetime"].iloc[0].strftime("%Y-%m-%d")
    ts_mid1 = df["datetime"].iloc[400].strftime("%Y-%m-%d")
    ts_mid2 = df["datetime"].iloc[600].strftime("%Y-%m-%d")
    ts_end = df["datetime"].iloc[-1].strftime("%Y-%m-%d")

    pieces = process_symbol(
        pq,
        index_df=None,
        train_range=(ts0, ts_mid1),
        val_range=(ts_mid1, ts_mid2),
        test_range=(ts_mid2, ts_end),
        min_len=50,
        market="crypto",
        exog_cols=exog_cols_for("crypto"),
    )
    assert pieces is not None
    for split in ("train", "val", "test"):
        if split in pieces:
            exog = pieces[split]["exog"]
            assert list(exog.columns) == exog_cols_for("crypto")
            assert exog.shape[1] == 32


# ---------------------------------------------------------------------------
# TrainConfig presets
# ---------------------------------------------------------------------------
def test_train_config_presets():
    from kairos.training.config import TrainConfig, available_presets, preset_for

    assert "crypto-1min" in available_presets()

    cfg = TrainConfig(**preset_for("crypto-1min"))
    assert cfg.market == "crypto"
    assert cfg.freq == "1min"
    assert cfg.return_horizon == 30
    assert cfg.lookback_window == 256
    assert cfg.predict_window == 32


def test_parse_range_roundtrip():
    assert parse_range("2018-01-01:2023-12-31") == ("2018-01-01", "2023-12-31")
