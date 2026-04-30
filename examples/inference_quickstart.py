"""Kairos crypto inference quickstart with caller-supplied OHLCV bars.

Run from the repository root after installing the package:

    pip install -e .
    python examples/inference_quickstart.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kairos import Kronos, KronosPredictor, KronosTokenizer


def synthetic_btc_bars(lookback: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2026-04-01", periods=lookback, freq="1min")
    close = 70_000 + np.cumsum(rng.normal(0, 12, size=lookback))
    open_ = close + rng.normal(0, 4, size=lookback)
    high = np.maximum(open_, close) + rng.uniform(1, 10, size=lookback)
    low = np.minimum(open_, close) - rng.uniform(1, 10, size=lookback)
    volume = rng.uniform(5, 30, size=lookback)
    amount = close * volume
    return pd.DataFrame({
        "datetime": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "amount": amount,
    })


def main(lookback: int = 400, pred_len: int = 30):
    df = synthetic_btc_bars(lookback)

    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tok, max_context=512)

    x_df = df[["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df["datetime"]
    y_ts = pd.Series([
        x_ts.iloc[-1] + pd.Timedelta(minutes=i + 1)
        for i in range(pred_len)
    ])
    pred = predictor.predict(
        x_df,
        x_ts,
        y_ts,
        pred_len=pred_len,
        T=0.6,
        top_p=0.9,
        sample_count=5,
        verbose=False,
    )

    print(pred.head(10))


if __name__ == "__main__":
    main()
