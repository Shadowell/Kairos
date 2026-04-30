"""FastAPI prediction service for Kairos crypto models.

The service is intentionally data-source neutral: callers submit the latest
OHLCV bars in the request body, and the server only runs Kronos inference. This
keeps deployment independent from exchange connectivity and avoids hidden
runtime fetches inside the prediction endpoint.
"""

from __future__ import annotations

import argparse
import logging
from typing import Literal

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kairos.vendor.kronos import Kronos, KronosPredictor, KronosTokenizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("kairos.serve")


_FREQ_TO_DELTA = {
    "1min": pd.Timedelta(minutes=1),
    "3min": pd.Timedelta(minutes=3),
    "5min": pd.Timedelta(minutes=5),
    "15min": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
    "60min": pd.Timedelta(hours=1),
    "1h": pd.Timedelta(hours=1),
    "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
    "daily": pd.Timedelta(days=1),
}


class Bar(BaseModel):
    datetime: str = Field(..., description="Bar timestamp, ISO-8601 preferred")
    open: float
    high: float
    low: float
    close: float
    volume: float
    amount: float | None = Field(
        None,
        description="Quote amount; defaults to close * volume when omitted",
    )


class PredictRequest(BaseModel):
    symbol: str = Field(..., description="Exchange-native symbol, e.g. BTC/USDT")
    market_type: Literal["spot", "swap"] = "spot"
    freq: str = Field("1min", description="1min|3min|5min|15min|30min|1h|4h|1d")
    bars: list[Bar] = Field(..., min_length=32)
    lookback: int = Field(400, ge=32, le=2000)
    pred_len: int = Field(30, ge=1, le=240)
    T: float = Field(0.6, gt=0, le=2.0)
    top_p: float = Field(0.9, gt=0, le=1.0)
    top_k: int = Field(0, ge=0)
    sample_count: int = Field(5, ge=1, le=32)


class PredictResponse(BaseModel):
    symbol: str
    market_type: str
    freq: str
    last_close: float
    pred_close: list[float]
    pred_mean_return: float
    pred_direction_prob_up: float
    forecast: list[dict]


def _request_to_frame(req: PredictRequest) -> pd.DataFrame:
    rows = [b.model_dump() for b in req.bars]
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["datetime"]).sort_values("datetime").tail(req.lookback)
    if len(df) < 32:
        raise HTTPException(400, "not enough valid bars after timestamp parsing")

    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["amount"] = df["amount"].fillna(df["close"] * df["volume"])
    if df[["open", "high", "low", "close", "volume", "amount"]].isna().any().any():
        raise HTTPException(400, "bars contain non-numeric OHLCV values")
    return df.reset_index(drop=True)


def _future_timestamps(last_ts: pd.Timestamp, freq: str, pred_len: int) -> pd.Series:
    if freq not in _FREQ_TO_DELTA:
        raise HTTPException(400, f"unsupported freq={freq!r}")
    step = _FREQ_TO_DELTA[freq]
    return pd.Series([last_ts + step * (i + 1) for i in range(pred_len)])


def _build_app(predictor: KronosPredictor) -> FastAPI:
    app = FastAPI(title="Kairos Crypto Predict API", version="0.2.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": str(predictor.device),
            "max_context": predictor.max_context,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        df = _request_to_frame(req)
        x_df = df[["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df["datetime"]
        y_ts = _future_timestamps(x_ts.iloc[-1], req.freq, req.pred_len)

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=req.pred_len,
            T=req.T,
            top_k=req.top_k,
            top_p=req.top_p,
            sample_count=req.sample_count,
            verbose=False,
        )

        last_close = float(x_df["close"].iloc[-1])
        pred_close = pred_df["close"].astype(float).tolist()
        mean_ret = float(np.mean([c / last_close - 1.0 for c in pred_close]))
        prob_up = float(np.mean([1.0 if c > last_close else 0.0 for c in pred_close]))

        forecast = [
            {
                "time": t.strftime("%Y-%m-%d %H:%M:%S"),
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r["volume"]),
            }
            for t, r in pred_df.iterrows()
        ]

        return PredictResponse(
            symbol=req.symbol,
            market_type=req.market_type,
            freq=req.freq,
            last_close=last_close,
            pred_close=pred_close,
            pred_mean_return=mean_ret,
            pred_direction_prob_up=prob_up,
            forecast=forecast,
        )

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="HF repo id or local checkpoint path")
    ap.add_argument("--predictor", required=True, help="HF repo id or local checkpoint path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-context", type=int, default=512)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    device = args.device or (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    log.info(f"device = {device}")
    tok = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.predictor)
    predictor = KronosPredictor(model, tok, device=device, max_context=args.max_context)
    uvicorn.run(_build_app(predictor), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
