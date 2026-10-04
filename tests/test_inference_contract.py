"""Shared serving/backtest inference must use the last observed bar."""

import numpy as np
import pandas as pd
import pytest

from kairos.data.features import EXOG_COLS


def manifest(lookback=4):
    return dict(contract_version=2, target="log_return", feature_cols=["open", "high", "low", "close", "vol", "amt"],
                exog_cols=EXOG_COLS, freq="1min", market="crypto", market_type="spot",
                lookback_window=lookback, clip=5.0, return_horizon=2, n_quantiles=3, use_exog=True,
                training_config={"time_feature_list": ["minute", "hour", "weekday", "day", "month"]})


def frames(n=4):
    index = pd.date_range("2026-01-01", periods=n, freq="min", name="datetime")
    close = np.arange(n, dtype=float) + 100
    main = pd.DataFrame(dict(open=close, high=close+1, low=close-1, close=close, vol=2., amt=close*2), index=index)
    exog = pd.DataFrame(0., index=index, columns=EXOG_COLS)
    exog.iloc[-1, 0] = .25
    return main, exog


class Tokenizer:
    def eval(self):
        return self

    def to(self, device):
        return self

    def encode(self, x, half):
        # Real input normalization is exercised; no temporal slicing in this fake.
        return x[:, :, 3], x[:, :, 3]


class ReturnModel:
    n_exog = 32
    return_horizon = 2
    n_quantiles = 3
    use_return_head = True

    def eval(self):
        return self

    def to(self, device):
        return self

    def decode_s1(self, s1, s2, stamp, exog):
        return None, exog

    def return_head(self, hidden):
        return hidden[:, :, :1, None].repeat(1, 1, 2, 3)


def test_prediction_uses_last_observation_and_exogenous_features():
    from kairos.inference import KairosPredictor
    main, exog = frames()
    predictor = KairosPredictor(ReturnModel(), Tokenizer(), manifest())
    result = predictor.predict_batch([main], [exog])
    np.testing.assert_allclose(result, np.full((1, 2, 3), .25))
    exog.iloc[-1, 0] = -.5
    np.testing.assert_allclose(predictor.predict_batch([main], [exog]), -.5)


def test_prediction_rejects_short_or_gapped_context():
    from kairos.inference import KairosPredictor
    predictor = KairosPredictor(ReturnModel(), Tokenizer(), manifest())
    main, exog = frames(3)
    with pytest.raises(ValueError, match="history|lookback|bars"):
        predictor.predict_batch([main], [exog])
    main, exog = frames(5)
    with pytest.raises(ValueError, match="continu|gap|frequency"):
        predictor.predict_batch([main.drop(main.index[2])], [exog.drop(exog.index[2])])


def test_prediction_rejects_wrong_schema_and_legacy_target():
    from kairos.inference import KairosPredictor
    main, exog = frames()
    predictor = KairosPredictor(ReturnModel(), Tokenizer(), manifest())
    with pytest.raises(ValueError):
        predictor.predict_batch([main], [exog.iloc[:, ::-1]])
    old = manifest()
    old["target"] = "normalized_delta"
    with pytest.raises(ValueError, match="log_return"):
        KairosPredictor(ReturnModel(), Tokenizer(), old)
    old = manifest()
    old["training_config"]["time_feature_list"].reverse()
    with pytest.raises(ValueError, match="time feature"):
        KairosPredictor(ReturnModel(), Tokenizer(), old)


def test_normalization_and_inference_are_history_only():
    from kairos.inference import KairosPredictor

    class CloseModel(ReturnModel):
        def decode_s1(self, s1, s2, stamp, exog):
            return None, s1[:, :, None]

    predictor = KairosPredictor(CloseModel(), Tokenizer(), manifest())
    main, exog = frames()
    expected = (103. - 101.5) / (np.std([100., 101., 102., 103.]) + 1e-5)
    np.testing.assert_allclose(predictor.predict_batch([main], [exog]), expected, rtol=1e-5)
