"""FastAPI prediction service for Kairos / Kronos A-share models."""

from __future__ import annotations

import argparse
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kairos.vendor.kronos import Kronos, KronosPredictor, KronosTokenizer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("kairos.serve")


class PredictRequest(BaseModel):
    symbol: str = Field(..., description="六位 A 股代码，如 600977")
    freq: str = Field("daily", description="daily|5min|15min|30min|60min")
    lookback: int = Field(400, ge=32, le=2000)
    pred_len: int = Field(20, ge=1, le=240)
    T: float = Field(0.6, gt=0, le=2.0)
    top_p: float = Field(0.9, gt=0, le=1.0)
    top_k: int = Field(0, ge=0)
    sample_count: int = Field(5, ge=1, le=32)
    adjust: str = Field("qfq")


class PredictResponse(BaseModel):
    symbol: str
    freq: str
    last_close: float
    pred_close: list[float]
    pred_mean_return: float
    pred_direction_prob_up: float
    forecast: list[dict]


def _build_app(predictor: KronosPredictor) -> FastAPI:
    app = FastAPI(title="Kairos A-share Predict API", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok",
                "device": str(predictor.device),
                "max_context": predictor.max_context}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        import akshare as ak

        try:
            if req.freq == "daily":
                df = ak.stock_zh_a_hist(symbol=req.symbol, period="daily",
                                        adjust=req.adjust)
            else:
                period_map = {"5min": "5", "15min": "15",
                              "30min": "30", "60min": "60"}
                if req.freq not in period_map:
                    raise ValueError(f"unsupported freq: {req.freq}")
                df = ak.stock_zh_a_hist_min_em(
                    symbol=req.symbol, period=period_map[req.freq],
                    adjust=req.adjust,
                )
        except Exception as e:
            raise HTTPException(502, f"data source error: {e}")

        if df is None or df.empty:
            raise HTTPException(404, "no data returned")

        df = df.rename(columns={
            "日期": "datetime", "时间": "datetime",
            "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low",
            "成交量": "volume", "成交额": "amount",
        })
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").tail(req.lookback).reset_index(drop=True)
        if len(df) < 32:
            raise HTTPException(400, "not enough history")

        x_df = df[["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df["datetime"]

        if req.freq == "daily":
            y_ts = pd.Series(pd.bdate_range(
                start=x_ts.iloc[-1] + pd.Timedelta(days=1),
                periods=req.pred_len))
        else:
            step = {"5min": 5, "15min": 15, "30min": 30, "60min": 60}[req.freq]
            y_ts = pd.Series([x_ts.iloc[-1] + pd.Timedelta(minutes=step * (i + 1))
                              for i in range(req.pred_len)])

        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=req.pred_len, T=req.T,
            top_k=req.top_k, top_p=req.top_p,
            sample_count=req.sample_count, verbose=False,
        )

        last_close = float(x_df["close"].iloc[-1])
        pred_close = pred_df["close"].astype(float).tolist()
        mean_ret = float(np.mean([c / last_close - 1 for c in pred_close]))
        prob_up = float(np.mean([1.0 if c > last_close else 0.0 for c in pred_close]))

        forecast = [{
            "time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]),   "close": float(r["close"]),
            "volume": float(r["volume"]),
        } for t, r in pred_df.iterrows()]

        return PredictResponse(
            symbol=req.symbol, freq=req.freq,
            last_close=last_close, pred_close=pred_close,
            pred_mean_return=mean_ret, pred_direction_prob_up=prob_up,
            forecast=forecast,
        )

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True,
                    help="HF repo id or local checkpoint path")
    ap.add_argument("--predictor", required=True,
                    help="HF repo id or local checkpoint path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-context", type=int, default=512)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    log.info(f"tokenizer: {args.tokenizer}")
    tok = KronosTokenizer.from_pretrained(args.tokenizer)
    log.info(f"predictor: {args.predictor}")
    model = Kronos.from_pretrained(args.predictor)

    device = args.device or (
        "cuda:0" if torch.cuda.is_available()
        else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        else "cpu"
    )
    log.info(f"device = {device}")
    predictor = KronosPredictor(model, tok, device=device,
                                max_context=args.max_context)
    uvicorn.run(_build_app(predictor), host=args.host, port=args.port,
                log_level="info")


if __name__ == "__main__":
    main()
