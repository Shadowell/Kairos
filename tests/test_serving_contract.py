"""HTTP must expose horizon quantiles from the same predictor as backtests."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from kairos.data.features import EXOG_COLS
from kairos.deploy.serve import _build_app


class Predictor:
    device = "cpu"
    max_context = 32
    quantiles = np.array([.1, .5, .9])
    manifest = dict(freq="1min", market_type="spot", lookback_window=32,
                    return_horizon=2, use_exog=True, exog_cols=EXOG_COLS)

    def predict_batch(self, main_frames, exog_frames):
        # The result depends on the submitted last exogenous row.
        shift = exog_frames[0].iloc[-1, 0]
        return np.array([[[-.2, 0., .2], [-.3, .1, .3]]]) + shift


def request():
    dates = pd.date_range("2026-01-01", periods=32, freq="min")
    return dict(symbol="BTC/USDT", freq="1min", market_type="spot", pred_len=2,
                bars=[dict(datetime=d.isoformat(), open=100, high=101, low=99,
                           close=100, volume=2) for d in dates],
                exog_cols=EXOG_COLS, exog=np.zeros((32, 32)).tolist())


def test_http_uses_bundle_quantiles_and_does_not_invent_probability():
    client = TestClient(_build_app(Predictor()))
    response = client.post("/predict", json=request())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["target"] == "log_return"
    assert body["anchor_time"] == "2026-01-01T00:31:00Z"
    assert body["forecast"][0]["time"] == "2026-01-01T00:32:00Z"
    assert body["forecast"][1]["log_return_quantiles"] == [-.3, .1, .3]
    assert body["pred_close"][1] == pytest.approx(100 * np.exp(.1))
    assert body.get("pred_direction_prob_up") is None
    changed = request()
    changed["exog"][-1][0] = .4
    result = client.post("/predict", json=changed).json()
    assert result["pred_close"][0] == pytest.approx(100 * np.exp(.4))


def test_http_history_length_comes_from_model_contract():
    predictor = Predictor()
    predictor.manifest = {**predictor.manifest, "lookback_window": 4}
    predictor.max_context = 4
    payload = request()
    payload["bars"] = payload["bars"][:4]
    payload["exog"] = payload["exog"][:4]
    response = TestClient(_build_app(predictor)).post("/predict", json=payload)
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("case", ["missing_exog", "wrong_columns", "gap", "duplicate", "bad_date",
                                   "freq", "market", "horizon", "lookback", "zero_price"])
def test_http_rejects_contract_mismatches(case):
    payload = request()
    if case == "missing_exog":
        payload.pop("exog")
    elif case == "wrong_columns":
        payload["exog_cols"] = list(reversed(EXOG_COLS))
    elif case == "gap":
        payload["bars"][-1]["datetime"] = "2026-01-01T01:00:00"
    elif case == "duplicate":
        payload["bars"][-1]["datetime"] = payload["bars"][-2]["datetime"]
    elif case == "bad_date":
        payload["bars"][0]["datetime"] = "bad"
    elif case == "freq":
        payload["freq"] = "5min"
    elif case == "market":
        payload["market_type"] = "swap"
    elif case == "horizon":
        payload["pred_len"] = 3
    elif case == "lookback":
        payload["lookback"] = 40
    elif case == "zero_price":
        payload["bars"][-1]["close"] = 0
    response = TestClient(_build_app(Predictor())).post("/predict", json=payload)
    assert response.status_code in (400, 422), response.text
