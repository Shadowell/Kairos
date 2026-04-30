"""Top-level feature builder that orchestrates common + market-specific factors.

This module is the public entry point used by :mod:`kairos.data.prepare_dataset`
and anywhere else we need a *single* call that returns an OHLCV frame enriched
with all 32 exogenous columns.

The heavy lifting is split across two collaborators:

* :func:`kairos.data.common_features.build_common_features` appends the 24
  market-agnostic factors (returns, momentum, volatility, microstructure, ...).
* :meth:`kairos.data.markets.base.MarketAdapter.market_features` appends the
  8 crypto market-specific factors.

The old ``build_features(df, index_df=...)`` signature is preserved so existing
call sites keep working. Kairos is crypto-only now, so the default market is
``"crypto"``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .common_features import COMMON_EXOG_COLS, build_common_features
from .markets import FeatureContext, MarketAdapter, get_adapter


log = logging.getLogger("kairos.features")


def _resolve_adapter(
    market: str | MarketAdapter | None,
) -> MarketAdapter:
    if isinstance(market, MarketAdapter):
        return market
    if market is None:
        market = "crypto"
    return get_adapter(market)


def _build_exog_cols(adapter: MarketAdapter) -> list[str]:
    """Union of common + adapter-specific exog column names."""
    cols = list(COMMON_EXOG_COLS) + list(adapter.MARKET_EXOG_COLS)
    if len(cols) != 32:
        raise RuntimeError(
            f"Adapter {adapter.name!r} produces {len(cols)} exog columns, "
            "expected 32. Adjust MARKET_EXOG_COLS so that "
            "len(COMMON_EXOG_COLS) + len(MARKET_EXOG_COLS) == 32."
        )
    return cols


#: Default 32-column exog schema for crypto. Kept as a module-level constant
#: so imports (``from kairos.data.features import EXOG_COLS``) keep working.
EXOG_COLS: list[str] = _build_exog_cols(_resolve_adapter("crypto"))


def exog_cols_for(market: str | MarketAdapter = "crypto") -> list[str]:
    """Return the 32-entry exog column list for the requested market."""
    return _build_exog_cols(_resolve_adapter(market))


def build_features(
    df: pd.DataFrame,
    index_df: Optional[pd.DataFrame] = None,
    ffill_limit: int = 2,
    *,
    market: str | MarketAdapter = "crypto",
    extras: Optional[dict] = None,
    symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Return ``df`` enriched with the 32-column exogenous feature block.

    Parameters
    ----------
    df : DataFrame with at least ``open, high, low, close, volume, amount``.
        ``datetime`` may be a column or the index; it will be normalised to a
        column named ``datetime`` on output.
    index_df : DataFrame, optional
        Optional reference K-line. Crypto currently relies on ``extras`` for
        market-wide references; this argument remains for API compatibility.
    ffill_limit : int
        Max forward-fill steps applied to the exog columns before the final
        ``fillna(0).clip(-5, 5)``. Intraday crypto callers can bump it higher
        if needed.
    market : str or MarketAdapter
        Which :class:`MarketAdapter` produces the venue-specific factors.
        Defaults to ``"crypto"``.
    extras : dict, optional
        Passed through to :class:`FeatureContext.extras`. Adapters use this
        to pick up auxiliary series they need (e.g. funding-rate history).
    symbol : str, optional
        Symbol being processed (adapter-native form). Adapters that key into
        ``extras`` by symbol should read this.

    Returns
    -------
    DataFrame with the original columns plus every name in
    :data:`EXOG_COLS` (or the adapter-specific equivalent). Exog columns are
    inf-sanitised, forward-filled up to ``ffill_limit``, zero-filled, and
    clipped to ``[-5, 5]`` to keep downstream training stable.
    """

    adapter = _resolve_adapter(market)

    df = df.copy()
    if "datetime" not in df.columns:
        df = df.reset_index().rename(columns={"index": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Step 1 — common features (24)
    df = build_common_features(df)

    # Step 2 — market-specific features (8)
    ctx = FeatureContext(
        symbol=symbol,
        index_df=index_df,
        extras=dict(extras) if extras else {},
    )
    market_df = adapter.market_features(df, context=ctx)
    expected = list(adapter.MARKET_EXOG_COLS)
    missing = [c for c in expected if c not in market_df.columns]
    if missing:
        raise RuntimeError(
            f"Adapter {adapter.name!r} omitted market_features columns: {missing}"
        )
    for col in expected:
        df[col] = market_df[col].values

    exog_cols = list(COMMON_EXOG_COLS) + expected
    if len(exog_cols) != 32:
        raise RuntimeError(
            f"Combined exog columns for market {adapter.name!r} have length "
            f"{len(exog_cols)}, expected 32."
        )

    # Step 3 — shared clean-up (same as the old build_features)
    df[exog_cols] = (
        df[exog_cols]
        .replace([np.inf, -np.inf], np.nan)
        .ffill(limit=ffill_limit)
        .fillna(0.0)
        .clip(-5, 5)
    )

    return df


__all__ = [
    "EXOG_COLS",
    "COMMON_EXOG_COLS",
    "build_features",
    "exog_cols_for",
]


# ---------------------------------------------------------------------------
# CLI self-test (preserved from the pre-refactor version for convenience)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    import pyarrow  # noqa: F401  ensure parquet support

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="per-symbol parquet")
    parser.add_argument("--index", default=None, help="optional index parquet")
    parser.add_argument("--market", default="crypto",
                        help="which MarketAdapter supplies the venue-specific "
                             "columns (default: crypto)")
    args = parser.parse_args()

    df_in = pd.read_parquet(args.input)
    idx_in = pd.read_parquet(args.index) if args.index else None
    out = build_features(df_in, idx_in, market=args.market)
    cols = exog_cols_for(args.market)
    print(out.tail())
    print("NaN rows:", out[cols].isna().any(axis=1).sum())
    print("Exog dim:", len(cols))
