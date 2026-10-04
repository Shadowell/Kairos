"""Shared, strict time-series semantics for training and inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


MAIN_COLUMNS = ["open", "high", "low", "close", "vol", "amt"]
TIME_COLUMNS = ["minute", "hour", "weekday", "day", "month"]
_BAR_FREQUENCIES = {
    "1min": "1min", "3min": "3min", "5min": "5min", "15min": "15min",
    "30min": "30min", "60min": "1h", "1h": "1h", "2h": "2h",
    "4h": "4h", "1d": "1D", "daily": "1D",
}


def bar_delta(freq: str) -> pd.Timedelta:
    """Resolve a supported fixed bar frequency; never guess unknown aliases."""
    if freq not in _BAR_FREQUENCIES:
        raise ValueError(f"unsupported bar frequency {freq!r}; expected {list(_BAR_FREQUENCIES)}")
    return pd.Timedelta(_BAR_FREQUENCIES[freq])


def _datetime_index(values) -> pd.DatetimeIndex:
    try:
        index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, format="mixed")).tz_localize(None)
    except (ValueError, TypeError) as exc:
        raise ValueError("datetime must contain valid UTC timestamps") from exc
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("datetime must be strictly ascending, unique, and non-null")
    return index.rename("datetime")


def _indexed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("expected a pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("columns must be unique")
    result = frame.copy()
    if "datetime" in result.columns:
        index = _datetime_index(result.pop("datetime"))
    elif isinstance(result.index, pd.DatetimeIndex):
        index = _datetime_index(result.index)
    else:
        raise ValueError("expected a DatetimeIndex or datetime column")
    result.index = index
    return result


def _finite_values(frame: pd.DataFrame) -> np.ndarray:
    try:
        values = frame.to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise ValueError("bar and exogenous values must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("bar and exogenous values must be finite")
    return values


def validate_main_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return canonical UTC-naive OHLCVA bars, rejecting invalid prices/times."""
    frame = _indexed_frame(df)
    missing = [name for name in MAIN_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"missing main columns: {missing}")
    frame = frame.loc[:, MAIN_COLUMNS]
    values = _finite_values(frame)
    if (values[:, :4] <= 0).any():
        raise ValueError("OHLC prices must be positive")
    if (values[:, 4:] < 0).any():
        raise ValueError("vol and amt must be nonnegative")
    return frame


def validate_exog_frame(
    main: pd.DataFrame, exog: pd.DataFrame, columns: Sequence[str],
) -> pd.DataFrame:
    """Validate exact schema and timestamp alignment without repairing data."""
    frame = _indexed_frame(exog)
    if list(frame.columns) != list(columns):
        raise ValueError("exogenous columns and order must exactly match the schema")
    if not frame.index.equals(main.index):
        raise ValueError("exogenous datetime index must exactly match main bars")
    _finite_values(frame)
    return frame


def contiguous_starts(index, window: int, freq: str) -> np.ndarray:
    """Return starts whose entire window contains consecutive native bars."""
    if not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError("window must be a positive integer")
    dates = _datetime_index(index)
    delta = bar_delta(freq)
    if len(dates) < window:
        return np.empty(0, dtype=np.int64)
    if window == 1:
        return np.arange(len(dates), dtype=np.int64)
    # Prefix counts make the check linear without allocating every window.
    breaks = np.asarray((dates[1:] - dates[:-1]) != delta, dtype=np.int64)
    prefix = np.concatenate(([0], np.cumsum(breaks)))
    valid = prefix[window - 1:] - prefix[:len(dates) - window + 1] == 0
    return np.flatnonzero(valid)


def normalize_window(values, lookback: int, clip: float) -> np.ndarray:
    """Normalize every bar using only the visible history's mean and std."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("normalization expects a finite two-dimensional array")
    if not isinstance(lookback, (int, np.integer)) or not 0 < lookback <= len(array):
        raise ValueError("lookback must select a nonempty visible history")
    if not np.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    history = array[:lookback]
    normalized = (array - history.mean(axis=0)) / (history.std(axis=0) + 1e-5)
    return np.clip(normalized, -clip, clip).astype(np.float32)


def log_return_targets(close, anchor: int, horizon: int) -> np.ndarray:
    """Return h=1..H raw log-price changes measured from the anchor bar."""
    prices = np.asarray(close, dtype=np.float64)
    if prices.ndim != 1 or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("close must be a finite, positive one-dimensional price array")
    if (not isinstance(anchor, (int, np.integer)) or anchor < 0
            or not isinstance(horizon, (int, np.integer)) or horizon <= 0
            or anchor + horizon >= len(prices)):
        raise ValueError("anchor and horizon must reference available future bars")
    # Subtract logs to avoid overflow in a ratio of extreme but valid prices.
    return (np.log(prices[anchor + 1:anchor + horizon + 1])
            - np.log(prices[anchor])).astype(np.float32)
