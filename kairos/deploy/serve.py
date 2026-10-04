"""Serve version 2 Kairos log-return bundles using the shared inference path."""

from __future__ import annotations

import argparse
import logging
from typing import Literal

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from kairos.data.contracts import bar_delta, contiguous_starts, validate_exog_frame, validate_main_frame
from kairos.inference import KairosPredictor


log = logging.getLogger("kairos.serve")


class Bar(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    datetime: str
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    amount: float | None = Field(None, ge=0)


class PredictRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    symbol: str = Field(min_length=1)
    market_type: Literal["spot", "swap"] = "spot"
    freq: str = "1min"
    bars: list[Bar] = Field(min_length=1, max_length=10000)
    exog_cols: list[str] | None = None
    exog: list[list[float]] | None = None
    lookback: int | None = Field(None, ge=1)
    pred_len: int | None = Field(None, ge=1)


class Forecast(BaseModel):
    time: str
    horizon: int
    log_return_quantiles: list[float]
    median_return: float
    median_close: float
    quantile_crossing: bool


class PredictResponse(BaseModel):
    symbol: str
    market_type: str
    freq: str
    target: Literal["log_return"] = "log_return"
    anchor_time: str
    last_close: float
    quantile_levels: list[float]
    pred_close: list[float]
    forecast: list[Forecast]


def _request_to_frame(req: PredictRequest) -> pd.DataFrame:
    df = pd.DataFrame([b.model_dump() for b in req.bars])
    df["amount"] = df["amount"].fillna(df["close"] * df["volume"])
    return validate_main_frame(df.rename(columns={"volume": "vol", "amount": "amt"}))


def _build_app(predictor: KairosPredictor) -> FastAPI:
    app = FastAPI(title="Kairos Log Return API", version="0.3.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "device": str(predictor.device),
                "max_context": predictor.max_context, "target": "log_return",
                "freq": predictor.manifest["freq"],
                "return_horizon": predictor.manifest["return_horizon"]}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        manifest = predictor.manifest
        try:
            if bar_delta(req.freq) != bar_delta(manifest["freq"]):
                raise ValueError("request frequency differs from checkpoint")
            if manifest.get("market_type") and req.market_type != manifest["market_type"]:
                raise ValueError("request market_type differs from checkpoint")
            lookback = manifest["lookback_window"]
            if req.lookback is not None and req.lookback != lookback:
                raise ValueError("lookback must match checkpoint")
            horizon = req.pred_len or manifest["return_horizon"]
            if horizon > manifest["return_horizon"]:
                raise ValueError("pred_len exceeds trained return_horizon")
            main = _request_to_frame(req)
            if len(main) < lookback:
                raise ValueError(f"at least {lookback} history bars required")
            if len(contiguous_starts(main.index[-lookback:], lookback, req.freq)) != 1:
                raise ValueError("history has gaps or wrong frequency")
            exog = None
            if manifest.get("use_exog", True):
                if req.exog is None or req.exog_cols != manifest["exog_cols"]:
                    raise ValueError("provide exog rows with the checkpoint's ordered exog_cols")
                exog = pd.DataFrame(req.exog, index=main.index, columns=req.exog_cols)
                exog = validate_exog_frame(main, exog, manifest["exog_cols"])
            values = predictor.predict_batch([main], [exog])[0, :horizon]
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

        last_close = float(main["close"].iloc[-1])
        anchor = main.index[-1]
        median_index = int(np.flatnonzero(np.isclose(predictor.quantiles, .5))[0])
        with np.errstate(over="ignore", invalid="ignore"):
            returns = np.expm1(values[:, median_index])
            closes = last_close * np.exp(values[:, median_index])
        if not np.isfinite(values).all() or not np.isfinite(closes).all() or not np.isfinite(returns).all():
            raise HTTPException(500, "model returned non-finite predictions")
        forecasts = [Forecast(
            time=(anchor + bar_delta(req.freq) * (i + 1)).isoformat() + "Z",
            horizon=i + 1, log_return_quantiles=row.tolist(),
            median_return=float(returns[i]), median_close=float(closes[i]),
            quantile_crossing=bool(np.any(np.diff(row) < 0)),
        ) for i, row in enumerate(values)]
        return PredictResponse(symbol=req.symbol, market_type=req.market_type,
                               freq=manifest["freq"], anchor_time=anchor.isoformat() + "Z",
                               last_close=last_close, quantile_levels=predictor.quantiles.tolist(),
                               pred_close=closes.tolist(), forecast=forecasts)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", required=True, help="Local version 2 Kairos checkpoint bundle")
    ap.add_argument("--tokenizer", default=None, help="Optional assertion of the bundle tokenizer path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    try:
        predictor = KairosPredictor.from_checkpoint(args.predictor, device=args.device,
                                                    tokenizer_path=args.tokenizer)
    except ValueError as exc:
        ap.error(str(exc))
    uvicorn.run(_build_app(predictor), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
