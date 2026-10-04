"""Helpers for the crypto "extras" channels.

Background
----------
:class:`~kairos.data.markets.crypto.CryptoAdapter.market_features` expects to
read funding, open-interest, spot-close, and reference-close series from
``FeatureContext.extras``. ``kairos-collect`` writes those channels under the
raw data directory, and ``kairos-prepare`` loads them back when packaging a
dataset.

This module closes that gap. It defines a small on-disk layout and the
read/write primitives both sides of the pipeline share.

Directory layout
----------------
Each perp-OHLCV run lives in its own directory, e.g.::

    raw/crypto/perp_1min_top100/
        BTC_USDT-USDT.parquet            # main perp OHLCV (existing)
        ETH_USDT-USDT.parquet
        ...
        _extras/
            funding/
                BTC_USDT-USDT.parquet    # columns: datetime, funding_rate
            open_interest/
                BTC_USDT-USDT.parquet    # columns: datetime, open_interest
            spot/
                BTC_USDT-USDT.parquet    # columns: datetime, close (spot close)
            reference.parquet            # optional BTC/USDT close reference

Why separate parquet per kind
-----------------------------
* Frequencies differ (K-line 1min vs funding 8h vs OI 5m); merging into the
  main parquet would throw away the original cadence.
* Re-collecting one channel (say, OI) shouldn't touch the main OHLCV or the
  funding parquet.
* Back-compat: runs that don't produce ``_extras/`` keep working identically
  to before (extras dict is empty → adapter fills with zeros → nothing
  changes).
"""

from __future__ import annotations

import logging
from numbers import Number
from pathlib import Path
import tempfile
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


log = logging.getLogger("kairos.crypto_extras")


EXTRAS_DIRNAME = "_extras"
"""Sub-directory name that holds every auxiliary channel for a given run."""


# ---------------------------------------------------------------------------
# Canonical (kind, filename, payload column) tuples.
#
# The keys on the left (``funding``, ``open_interest``, ``spot``,
# ``reference``) are what we write on disk and what the adapter reads back
# via ``load_for_symbol``. They are *not* the keys CryptoAdapter.market_features
# ultimately consumes — ``load_for_symbol`` translates them to the adapter's
# expected ``extras`` keys (``funding_rate`` / ``open_interest`` /
# ``spot_close`` / ``reference_close``) at read time.
# ---------------------------------------------------------------------------
KIND_FUNDING = "funding"
KIND_OI = "open_interest"
KIND_SPOT = "spot"
KIND_REFERENCE = "reference"

ALL_KINDS = (KIND_FUNDING, KIND_OI, KIND_SPOT, KIND_REFERENCE)

# Per-symbol channels are stored under _extras/<kind>/<stem>.parquet.
# Market-wide channels live as a single
# _extras/<kind>.parquet.
_PER_SYMBOL_KINDS = (KIND_FUNDING, KIND_OI, KIND_SPOT)
_MARKET_WIDE_KINDS = (KIND_REFERENCE,)

# Column name stored inside each parquet. Kept in one place so writers and
# readers can't drift.
_PAYLOAD_COL: Dict[str, str] = {
    KIND_FUNDING: "funding_rate",
    KIND_OI: "open_interest",
    KIND_SPOT: "close",
    KIND_REFERENCE: "close",
}

# Translation from on-disk kind -> the extras key CryptoAdapter expects.
_EXTRAS_KEY: Dict[str, str] = {
    KIND_FUNDING: "funding_rate",
    KIND_OI: "open_interest",
    KIND_SPOT: "spot_close",
    KIND_REFERENCE: "reference_close",
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def extras_root(raw_dir: Path) -> Path:
    """Return the ``_extras/`` directory under a per-run raw directory."""
    return Path(raw_dir) / EXTRAS_DIRNAME


def per_symbol_path(raw_dir: Path, kind: str, symbol_stem: str) -> Path:
    """Resolve the parquet path for a per-symbol channel."""
    if kind not in _PER_SYMBOL_KINDS:
        raise ValueError(
            f"kind {kind!r} is not a per-symbol extras channel; "
            f"valid: {_PER_SYMBOL_KINDS}"
        )
    return extras_root(raw_dir) / kind / f"{symbol_stem}.parquet"


def market_wide_path(raw_dir: Path, kind: str) -> Path:
    """Resolve the parquet path for a market-wide channel."""
    if kind not in _MARKET_WIDE_KINDS:
        raise ValueError(
            f"kind {kind!r} is not a market-wide extras channel; "
            f"valid: {_MARKET_WIDE_KINDS}"
        )
    return extras_root(raw_dir) / f"{kind}.parquet"


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _utc_time_index(values, label: str) -> pd.DatetimeIndex:
    """Parse real timestamps as UTC, never infer numeric indexes as epochs."""
    index = pd.Index(values)
    if (pd.api.types.is_numeric_dtype(index.dtype)
            or (not isinstance(index, pd.DatetimeIndex)
                and any(isinstance(value, Number) for value in index))):
        raise ValueError(f"{label} requires timestamps, not a numeric index")
    try:
        parsed = pd.DatetimeIndex(pd.to_datetime(index, utc=True, format="mixed"))
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} contains invalid timestamps") from exc
    if parsed.hasnans:
        raise ValueError(f"{label} contains NaT")
    if not parsed.is_unique:
        raise ValueError(f"{label} contains duplicate or conflicting absolute timestamps")
    return parsed.tz_localize(None).rename("datetime")


def _numeric_payload(values) -> np.ndarray:
    """Allow explicit missing measurements, but reject malformed or infinite ones."""
    try:
        numeric = pd.to_numeric(pd.Series(values), errors="raise")
        result = numeric.to_numpy(dtype=float, na_value=np.nan)
    except (ValueError, TypeError) as exc:
        raise ValueError("sidecar payload must be numeric or missing") from exc
    if np.isinf(result).any():
        raise ValueError("sidecar payload must not contain infinite values")
    return result


def _normalise(df: pd.DataFrame, payload_col: str) -> pd.DataFrame:
    """Coerce ``df`` to the canonical ``datetime`` + ``<payload_col>`` schema.

    Accepts a timestamp column or a datetime-indexed single payload. Malformed
    times and duplicate absolute timestamps are rejected without dropping rows.
    """

    try:
        if df is None or (isinstance(df, (pd.DataFrame, pd.Series)) and len(df) == 0):
            return pd.DataFrame({"datetime": pd.Series(dtype="datetime64[ns]"),
                                 payload_col: pd.Series(dtype=float)})
        if isinstance(df, pd.Series):
            df = df.to_frame(name=payload_col)
        if not isinstance(df, pd.DataFrame) or not df.columns.is_unique:
            raise ValueError("sidecar must be a DataFrame or Series with unique columns")
        if payload_col not in df.columns:
            if "datetime" not in df.columns and df.shape[1] == 1:
                df = df.rename(columns={df.columns[0]: payload_col})
            else:
                raise ValueError(f"sidecar is missing required payload column {payload_col!r}")
        times = df["datetime"] if "datetime" in df.columns else df.index
        dates = _utc_time_index(times, "sidecar datetime")
        out = pd.DataFrame({"datetime": dates, payload_col: _numeric_payload(df[payload_col])})
        return out.sort_values("datetime", kind="stable").reset_index(drop=True)
    except ValueError as exc:
        log.error("Invalid sidecar: %s", exc)
        raise


def _read_sidecar(path: Path, payload_col: str) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
        if len(frame) == 0:
            return _normalise(frame, payload_col)
        if "datetime" not in frame.columns or payload_col not in frame.columns:
            raise ValueError(f"expected datetime and {payload_col} columns")
        return _normalise(frame, payload_col)
    except Exception as exc:
        log.error("Invalid existing sidecar %s: %s", path, exc)
        raise ValueError(f"Invalid existing sidecar {path}: {exc}") from exc


def _save_sidecar(path: Path, df, payload_col: str, merge_existing: bool) -> Path:
    normalised = _normalise(df, payload_col)
    if merge_existing and path.exists():
        previous = _read_sidecar(path, payload_col)
        if normalised.empty:
            return path
        if not previous.empty:
            combined = pd.concat([previous, normalised], ignore_index=True)
            # A retried download may repeat an identical observation. A changed
            # value at the same instant requires an explicit historical rewrite.
            combined = combined.drop_duplicates(["datetime", payload_col])
            try:
                normalised = _normalise(combined, payload_col)
            except ValueError as exc:
                log.error("Cannot merge sidecar %s: %s", path, exc)
                raise ValueError(f"Cannot merge sidecar {path}: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        normalised.to_parquet(temporary, index=False)
        temporary.replace(path)
    except Exception:
        log.exception("Failed to write sidecar %s; existing history was preserved", path)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_per_symbol(
    raw_dir: Path,
    symbol_stem: str,
    kind: str,
    df: pd.DataFrame,
    *,
    merge_existing: bool = True,
) -> Path:
    """Write one channel for one symbol, merging with any existing parquet.

    Parameters
    ----------
    raw_dir : Path
        Run directory (e.g. ``raw/crypto/perp_1min_top100/``).
    symbol_stem : str
        ``sanitize_symbol`` output, e.g. ``"BTC_USDT-USDT"``.
    kind : str
        One of :data:`_PER_SYMBOL_KINDS`.
    df : DataFrame
        Must contain a ``datetime`` column and the payload column implied
        by ``kind`` (see :data:`_PAYLOAD_COL`).
    merge_existing : bool
        If True (default), merge with whatever is already on disk and
        dedupe identical observations. Conflicting values fail without changing
        history. Set to False to explicitly overwrite atomically.
    """

    payload = _PAYLOAD_COL[kind]
    out = per_symbol_path(raw_dir, kind, symbol_stem)
    return _save_sidecar(out, df, payload, merge_existing)


def save_market_wide(
    raw_dir: Path,
    kind: str,
    df: pd.DataFrame,
    *,
    merge_existing: bool = True,
) -> Path:
    """Write a market-wide channel such as the BTC/USDT reference close."""

    payload = _PAYLOAD_COL[kind]
    out = market_wide_path(raw_dir, kind)
    return _save_sidecar(out, df, payload, merge_existing)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def _read_datetime_series(path: Path, payload_col: str) -> Optional[pd.Series]:
    if not path.exists():
        return None
    df = _read_sidecar(path, payload_col)
    return df.set_index("datetime")[payload_col]


def load_for_symbol(
    raw_dir: Path,
    symbol_stem: str,
    *,
    kinds: Iterable[str] = ALL_KINDS,
) -> Dict[str, pd.Series]:
    """Build the ``extras`` dict for one symbol from the parquet sidecars.

    Returns a dict keyed as :class:`CryptoAdapter.market_features` expects
    (``"funding_rate"`` / ``"open_interest"`` / ``"spot_close"`` /
    ``"reference_close"``). Channels that have no parquet on disk are
    *absent* from the returned dict (not NaN-filled), so the adapter's
    existing NaN → 0 fallback still triggers for missing inputs.
    """

    raw_dir = Path(raw_dir)
    out: Dict[str, pd.Series] = {}

    for kind in kinds:
        payload = _PAYLOAD_COL[kind]
        extras_key = _EXTRAS_KEY[kind]
        if kind in _PER_SYMBOL_KINDS:
            path = per_symbol_path(raw_dir, kind, symbol_stem)
        elif kind in _MARKET_WIDE_KINDS:
            path = market_wide_path(raw_dir, kind)
        else:
            continue
        series = _read_datetime_series(path, payload)
        if series is not None and len(series) > 0:
            out[extras_key] = series
    return out


def available_channels(raw_dir: Path) -> list[str]:
    """Return the list of extras kinds that have *some* parquet on disk.

    Useful for writing ``meta.json`` without scanning individual symbols.
    """

    raw_dir = Path(raw_dir)
    root = extras_root(raw_dir)
    if not root.exists():
        return []
    found: list[str] = []
    for kind in _PER_SYMBOL_KINDS:
        sub = root / kind
        if sub.exists() and any(sub.glob("*.parquet")):
            found.append(kind)
    for kind in _MARKET_WIDE_KINDS:
        if market_wide_path(raw_dir, kind).exists():
            found.append(kind)
    return found


__all__ = [
    "EXTRAS_DIRNAME",
    "KIND_FUNDING",
    "KIND_OI",
    "KIND_SPOT",
    "KIND_REFERENCE",
    "ALL_KINDS",
    "extras_root",
    "per_symbol_path",
    "market_wide_path",
    "save_per_symbol",
    "save_market_wide",
    "load_for_symbol",
    "available_channels",
]
