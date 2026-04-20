"""Helpers for the crypto "extras" channels (funding rate / OI / spot basis).

Background
----------
:class:`~kairos.data.markets.crypto.CryptoAdapter.market_features` expects to
read funding / open-interest / spot-close / btc-dominance series from
``FeatureContext.extras``. Until now no code path actually produced those
extras — ``kairos-collect`` only dumped OHLCV parquet and
``prepare_dataset`` never forwarded ``extras={...}`` to ``build_features``,
so the five "data-driven" slots in ``MARKET_EXOG_COLS`` were silently zero
in every crypto training run to date.

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
            btc_dominance.parquet        # optional, shared across symbols

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
from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


log = logging.getLogger("kairos.crypto_extras")


EXTRAS_DIRNAME = "_extras"
"""Sub-directory name that holds every auxiliary channel for a given run."""


# ---------------------------------------------------------------------------
# Canonical (kind, filename, payload column) tuples.
#
# The keys on the left (``funding``, ``open_interest``, ``spot``,
# ``btc_dominance``) are what we write on disk and what the adapter reads back
# via ``load_for_symbol``. They are *not* the keys CryptoAdapter.market_features
# ultimately consumes — ``load_for_symbol`` translates them to the adapter's
# expected ``extras`` keys (``funding_rate`` / ``open_interest`` /
# ``spot_close`` / ``btc_dominance``) at read time.
# ---------------------------------------------------------------------------
KIND_FUNDING = "funding"
KIND_OI = "open_interest"
KIND_SPOT = "spot"
KIND_DOMINANCE = "btc_dominance"

ALL_KINDS = (KIND_FUNDING, KIND_OI, KIND_SPOT, KIND_DOMINANCE)

# Per-symbol channels are stored under _extras/<kind>/<stem>.parquet.
# Market-wide channels (currently only btc_dominance) live as a single
# _extras/<kind>.parquet.
_PER_SYMBOL_KINDS = (KIND_FUNDING, KIND_OI, KIND_SPOT)
_MARKET_WIDE_KINDS = (KIND_DOMINANCE,)

# Column name stored inside each parquet. Kept in one place so writers and
# readers can't drift.
_PAYLOAD_COL: Dict[str, str] = {
    KIND_FUNDING: "funding_rate",
    KIND_OI: "open_interest",
    KIND_SPOT: "close",
    KIND_DOMINANCE: "btc_dominance",
}

# Translation from on-disk kind -> the extras key CryptoAdapter expects.
_EXTRAS_KEY: Dict[str, str] = {
    KIND_FUNDING: "funding_rate",
    KIND_OI: "open_interest",
    KIND_SPOT: "spot_close",
    KIND_DOMINANCE: "btc_dominance",
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
def _normalise(df: pd.DataFrame, payload_col: str) -> pd.DataFrame:
    """Coerce ``df`` to the canonical ``datetime`` + ``<payload_col>`` schema.

    Accepts either a DataFrame already in that shape or a single-column
    DataFrame / Series indexed by datetime. Duplicate timestamps are
    dropped (keeping the last observation).
    """

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["datetime", payload_col])

    if isinstance(df, pd.Series):
        df = df.to_frame(name=payload_col)

    if payload_col not in df.columns:
        # Infer: DataFrame indexed by datetime with one unnamed column.
        if df.shape[1] == 1:
            df = df.rename(columns={df.columns[0]: payload_col})
        else:
            raise ValueError(
                f"expected a frame with '{payload_col}' column, got {list(df.columns)}"
            )

    out = df.reset_index() if df.index.name == "datetime" else df.copy()
    if "datetime" not in out.columns:
        # Last resort: treat the index as datetime.
        out = df.reset_index().rename(columns={df.index.name or "index": "datetime"})

    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce").dt.tz_convert(None)
    out = out.dropna(subset=["datetime"])
    out = out[["datetime", payload_col]].drop_duplicates("datetime").sort_values("datetime")
    out[payload_col] = pd.to_numeric(out[payload_col], errors="coerce")
    return out.reset_index(drop=True)


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
        dedupe by timestamp. Set to False to overwrite atomically.
    """

    payload = _PAYLOAD_COL[kind]
    normalised = _normalise(df, payload)
    out = per_symbol_path(raw_dir, kind, symbol_stem)
    out.parent.mkdir(parents=True, exist_ok=True)

    if merge_existing and out.exists():
        try:
            prev = pd.read_parquet(out)
            normalised = pd.concat([prev, normalised], ignore_index=True)
            normalised = normalised.drop_duplicates("datetime").sort_values("datetime")
            normalised = normalised.reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            log.warning(f"failed to merge existing {out}: {e}; overwriting")

    normalised.to_parquet(out, index=False)
    return out


def save_market_wide(
    raw_dir: Path,
    kind: str,
    df: pd.DataFrame,
    *,
    merge_existing: bool = True,
) -> Path:
    """Write a market-wide channel (currently only ``btc_dominance``)."""

    payload = _PAYLOAD_COL[kind]
    normalised = _normalise(df, payload)
    out = market_wide_path(raw_dir, kind)
    out.parent.mkdir(parents=True, exist_ok=True)

    if merge_existing and out.exists():
        try:
            prev = pd.read_parquet(out)
            normalised = pd.concat([prev, normalised], ignore_index=True)
            normalised = normalised.drop_duplicates("datetime").sort_values("datetime")
            normalised = normalised.reset_index(drop=True)
        except Exception as e:  # noqa: BLE001
            log.warning(f"failed to merge existing {out}: {e}; overwriting")

    normalised.to_parquet(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------
def _read_datetime_series(path: Path, payload_col: str) -> Optional[pd.Series]:
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001
        log.warning(f"failed to read {path}: {e}")
        return None
    if "datetime" not in df.columns or payload_col not in df.columns:
        log.warning(
            f"{path} missing expected columns; have {list(df.columns)}, "
            f"need datetime and {payload_col}"
        )
        return None
    df = df[["datetime", payload_col]].dropna(subset=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.drop_duplicates("datetime").sort_values("datetime")
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
    ``"btc_dominance"``). Channels that have no parquet on disk are
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
    "KIND_DOMINANCE",
    "ALL_KINDS",
    "extras_root",
    "per_symbol_path",
    "market_wide_path",
    "save_per_symbol",
    "save_market_wide",
    "load_for_symbol",
    "available_channels",
]
