"""Regression coverage for IC methodology and the common prediction contract."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from kairos.training import backtest_ic as bt
from kairos.training.config import TrainConfig


def test_cross_section_is_calculated_per_timestamp_before_daily_average():
    # Pooling the two timestamps gives a positive IC; both true sections are -1.
    records = []
    for minute, offset in [(0, 0), (1, 100)]:
        for symbol, score, ret in zip("ABC", [1, 2, 3], [3, 2, 1]):
            records.append(dict(symbol=symbol, date=pd.Timestamp("2026-01-01")
                                + pd.Timedelta(minutes=minute),
                                score_h1=score + offset, ret_h1=ret + offset))
    report = bt.summarize_records(records, [1], "date")
    assert report["pooled"]["h1"]["pearson"] > 0.9
    assert report["cross_sectional"]["h1"]["summary"]["ic"] == pytest.approx(-1)
    assert report["cross_sectional"]["h1"]["summary"]["n_timestamps"] == 2
    assert report["by_date_mean"]["h1"]["n_dates"] == 1
    assert report["by_date_mean"]["h1"]["icir"] is None
    assert report["time_series"]["A"]["h1"]["pearson"] == pytest.approx(1)


def test_two_symbols_never_become_cross_section_by_pooling_minutes():
    records = [dict(symbol=s, date=pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
                    score_h1=i + j, ret_h1=i + j)
               for i in range(4) for j, s in enumerate("AB")]
    report = bt.summarize_records(records, [1], "date")
    assert report["by_date_mean"]["h1"]["ic"] is None
    assert report["by_date_mean"]["h1"]["n_timestamps"] == 0
    assert report["pooled"]["h1"]["n"] == 8


def test_duplicate_symbol_timestamp_cannot_inflate_cross_section():
    record = dict(symbol="A", date=pd.Timestamp("2026-01-01"), score_h1=1, ret_h1=1)
    with pytest.raises(ValueError, match="duplicate"):
        bt.summarize_records([record, record], [1], "date")


def test_empty_and_constant_reports_are_valid_json():
    empty = bt.summarize_records([], [1, 3], "hour")
    assert empty["n_records"] == 0
    assert empty["date_range"] == [None, None]
    assert empty["by_date_mean"]["h3"]["ic"] is None
    constant = [dict(symbol=s, date=pd.Timestamp("2026-01-01"), score_h1=1, ret_h1=2)
                for s in "ABC"]
    report = bt.summarize_records(constant, [1], "date")
    assert report["pooled"]["h1"]["pearson"] is None
    json.dumps([empty, report], allow_nan=False)


def _dataset(tmp_path, *, rows=8, gap=False):
    index = pd.date_range("2026-01-01", periods=rows, freq="min")
    if gap:
        index = index[:4].append(index[4:] + pd.Timedelta(minutes=1))
    main = {}
    exog = {}
    for i, sym in enumerate("ABC"):
        close = (100 + i * 10) * np.exp(np.arange(rows) * (i + 1) * 0.01)
        main[sym] = pd.DataFrame({k: close if k not in {"vol", "amt"} else np.ones(rows)
                                  for k in ["open", "high", "low", "close", "vol", "amt"]},
                                 index=index)
        exog[sym] = pd.DataFrame(0., index=index, columns=[f"f{i}" for i in range(32)])
    for name, data in [("test_data.pkl", main), ("exog_test.pkl", exog)]:
        with (tmp_path / name).open("wb") as fh:
            pickle.dump(data, fh)
    (tmp_path / "meta.json").write_text(json.dumps({
        "schema_version": 2, "freq": "1min", "market": "crypto",
        "feature_cols": ["open", "high", "low", "close", "vol", "amt"],
        "exog_cols": [f"f{i}" for i in range(32)],
    }))
    return TrainConfig(dataset_path=str(tmp_path), lookback_window=3, return_horizon=2), main, exog


class _ContractPredictor:
    """Loader boundary double: forecasts the fixture's constant log trend."""

    device = "cpu"
    max_context = 512
    manifest = {"contract_version": 2, "target": "log_return", "freq": "1min",
                "market": "crypto", "lookback_window": 3, "return_horizon": 2,
                "n_quantiles": 9, "feature_cols": ["open", "high", "low", "close", "vol", "amt"],
                "exog_cols": [f"f{i}" for i in range(32)], "clip": 5.0}

    def predict_batch(self, main, exog):
        return np.array([[[h * np.log(frame.close.iloc[-1] / frame.close.iloc[-2])]*9
                          for h in [1, 2]] for frame in main])


def test_trained_backtest_uses_bundle_lookback_and_final_history_bar(tmp_path, monkeypatch):
    cfg, _, _ = _dataset(tmp_path)
    cfg.lookback_window = 7  # A mutable training default cannot override the saved contract.
    monkeypatch.setattr(bt, "_load_trained_predictor", lambda *a, **kw: _ContractPredictor())
    report = bt.run_backtest("bundle", cfg, horizons=[1, 2], batch_size=4, seed=9)
    assert report["n_records"] == 12  # 8 - 3 - 2 + 1 = 4 anchors, three symbols.
    assert pd.Timestamp(report["date_range"][0]) == pd.Timestamp("2026-01-01 00:02", tz="UTC")
    assert pd.Timestamp(report["date_range"][1]) == pd.Timestamp("2026-01-01 00:05", tz="UTC")
    assert report["by_date_mean"]["h2"]["rank_ic"] == pytest.approx(1)
    assert report["methodology"]["target"] == "log_return"
    assert report["seed"] == 9


def test_window_cannot_cross_history_or_future_gap(tmp_path, monkeypatch):
    cfg, _, _ = _dataset(tmp_path, gap=True)
    monkeypatch.setattr(bt, "_load_trained_predictor", lambda *a, **kw: _ContractPredictor())
    report = bt.run_backtest("bundle", cfg, horizons=[1, 2])
    assert report["n_records"] == 0  # Each continuous piece has only four bars; five are needed.


@pytest.mark.parametrize("horizons", [[], [0], [-1], [3], [1, 1]])
def test_horizon_must_be_distinct_and_inside_model_contract(tmp_path, monkeypatch, horizons):
    cfg, _, _ = _dataset(tmp_path)
    monkeypatch.setattr(bt, "_load_trained_predictor", lambda *a, **kw: _ContractPredictor())
    with pytest.raises(ValueError, match="horizon"):
        bt.run_backtest("bundle", cfg, horizons=horizons)


def test_dataset_frequency_must_match_model_bundle(tmp_path, monkeypatch):
    cfg, _, _ = _dataset(tmp_path)
    meta_path = tmp_path / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["freq"] = "5min"
    meta_path.write_text(json.dumps(meta))
    monkeypatch.setattr(bt, "_load_trained_predictor", lambda *a, **kw: _ContractPredictor())
    with pytest.raises(ValueError, match="freq"):
        bt.run_backtest("bundle", cfg, horizons=[1])


class _OriginalPredictor:
    def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, **kwargs):
        # Stochastic original-Kronos boundary; run_backtest owns seeding and conversion.
        return [pd.DataFrame({"close": frame.close.iloc[-1]
                              * np.exp(np.arange(1, pred_len + 1) * np.random.normal(0.01, 0.01))},
                             index=ts)
                for frame, ts in zip(df_list, y_timestamp_list)]


def test_baseline_repeat_seeds_are_reproducible_and_report_dispersion(tmp_path, monkeypatch):
    cfg, _, _ = _dataset(tmp_path)
    monkeypatch.setattr(bt, "_load_baseline_predictor", lambda *a, **kw: _OriginalPredictor())
    first = bt.run_backtest(None, cfg, horizons=[1], use_baseline=True, seeds=[7, 11])
    second = bt.run_backtest(None, cfg, horizons=[1], use_baseline=True, seeds=[7, 11])
    assert first == second
    assert [r["seed"] for r in first["runs"]] == [7, 11]
    assert first["seed_summary"]["pooled"]["h1"]["spearman"]["std"] > 0
    assert first["runs"][0]["model"]["mode"] == "original_kronos"
    json.dumps(first, allow_nan=False)


def test_baseline_price_conversion_uses_last_observed_close(tmp_path, monkeypatch):
    cfg, _, _ = _dataset(tmp_path)

    class TrendPredictor:
        def predict_batch(self, df_list, x_timestamp_list, y_timestamp_list, pred_len, **kwargs):
            return [pd.DataFrame({"close": frame.close.iloc[-1] * np.exp(
                    np.arange(1, pred_len + 1) * np.log(frame.close.iloc[-1] / frame.close.iloc[-2]))},
                    index=ts) for frame, ts in zip(df_list, y_timestamp_list)]

    monkeypatch.setattr(bt, "_load_baseline_predictor", lambda *a, **kw: TrendPredictor())
    report = bt.run_backtest(None, cfg, horizons=[1, 2], use_baseline=True)
    assert report["n_records"] == 12
    assert report["by_date_mean"]["h2"]["rank_ic"] == pytest.approx(1)


def test_baseline_scores_are_raw_log_returns_at_each_future_horizon():
    frame = pd.DataFrame({"close": [10., 40., 100.]},
                         index=pd.date_range("2026-01-01", periods=3, freq="min"))

    class PricePredictor:
        def predict_batch(self, *args, **kwargs):
            return [pd.DataFrame({"close": [200., 400.]})]

    assert bt._baseline_scores(PricePredictor(), [frame], 2, "1min")[0].tolist() == pytest.approx(
        [0.6931471805599453, 1.3862943611198906])


@pytest.mark.parametrize("change", ["columns", "index", "missing", "nonfinite"])
def test_exogenous_data_is_never_repaired_silently(tmp_path, monkeypatch, change):
    cfg, _, exog = _dataset(tmp_path)
    if change == "columns":
        exog["A"] = exog["A"].iloc[:, ::-1]
    elif change == "index":
        exog["A"].index += pd.Timedelta(minutes=1)
    elif change == "missing":
        del exog["A"]
    else:
        exog["A"].iloc[0, 0] = np.nan
    with (tmp_path / "exog_test.pkl").open("wb") as fh:
        pickle.dump(exog, fh)
    monkeypatch.setattr(bt, "_load_trained_predictor", lambda *a, **kw: _ContractPredictor())
    with pytest.raises(ValueError):
        bt.run_backtest("bundle", cfg, horizons=[1])


def test_shared_limit_preserves_exact_time_sections_with_different_listing_dates(tmp_path, monkeypatch):
    cfg, main, exog = _dataset(tmp_path, rows=12)
    for i, symbol in enumerate("ABC"):
        main[symbol] = main[symbol].iloc[i:]
        exog[symbol] = exog[symbol].iloc[i:]
    for name, data in [("test_data.pkl", main), ("exog_test.pkl", exog)]:
        with (tmp_path / name).open("wb") as fh:
            pickle.dump(data, fh)
    monkeypatch.setattr(bt, "_load_trained_predictor", lambda *a, **kw: _ContractPredictor())
    report = bt.run_backtest("bundle", cfg, horizons=[1], stride=2, per_symbol_limit=3)
    sections = report["cross_sectional"]["h1"]["timestamps"]
    assert len(sections) == 2
    assert all(item["n_symbols"] == 3 for item in sections)
    assert all(pd.Timestamp(item["timestamp"]).minute % 2 == 0 for item in sections)
    assert all(result["h1"]["n"] <= 3 for result in report["time_series"].values())


def test_real_original_kronos_baseline_generates_without_return_head(tmp_path):
    cfg, _, _ = _dataset(tmp_path, rows=5)
    torch.manual_seed(42)
    tokenizer = bt.KronosTokenizer(
        d_in=6, d_model=32, n_heads=4, ff_dim=64, n_enc_layers=1, n_dec_layers=1,
        ffn_dropout_p=0., attn_dropout_p=0., resid_dropout_p=0., s1_bits=2, s2_bits=2,
        beta=0.1, gamma0=1., gamma=1., zeta=1., group_size=2,
    )
    model = bt.Kronos(s1_bits=2, s2_bits=2, n_layers=1, d_model=32, n_heads=4, ff_dim=64,
                      ffn_dropout_p=0., attn_dropout_p=0., resid_dropout_p=0.,
                      token_dropout_p=0., learn_te=True)
    cfg.pretrained_tokenizer_path = str(tmp_path / "original-tokenizer")
    cfg.pretrained_predictor_path = str(tmp_path / "original-predictor")
    tokenizer.save_pretrained(cfg.pretrained_tokenizer_path)
    model.save_pretrained(cfg.pretrained_predictor_path)
    # A broken local fine-tuned tokenizer must not displace the explicitly configured original.
    cfg.save_path = str(tmp_path / "other-artifacts")
    local = tmp_path / "other-artifacts" / "tokenizer" / "checkpoints" / "best_model"
    local.mkdir(parents=True)
    (tmp_path / "exog_test.pkl").unlink()  # Original Kronos has no exogenous channel.
    first = bt.run_backtest(None, cfg, horizons=[1, 2], batch_size=3, seed=17)
    second = bt.run_backtest(None, cfg, horizons=[1, 2], batch_size=3, seed=17)
    assert first == second
    assert first["n_records"] == 3
    assert first["model"]["tokenizer_source"] == cfg.pretrained_tokenizer_path


@pytest.mark.parametrize("sidecar", ["missing", "invalid"])
def test_trained_model_without_exog_does_not_require_or_read_sidecar(tmp_path, monkeypatch, sidecar):
    cfg, _, _ = _dataset(tmp_path)
    if sidecar == "missing":
        (tmp_path / "exog_test.pkl").unlink()
    else:
        (tmp_path / "exog_test.pkl").write_bytes(b"Not a pickle: this disabled channel must not be opened")

    class NoExogPredictor(_ContractPredictor):
        manifest = {**_ContractPredictor.manifest, "use_exog": False}

        def predict_batch(self, main, exog):
            if len(main) != len(exog) or any(frame is not None for frame in exog):
                raise ValueError("disabled exog must have one None per history")
            return super().predict_batch(main, exog)

    monkeypatch.setattr(bt, "_load_trained_predictor", lambda *a, **kw: NoExogPredictor())
    report = bt.run_backtest("bundle", cfg, horizons=[1, 2], batch_size=4)
    assert report["n_records"] == 12
    assert report["by_date_mean"]["h2"]["rank_ic"] == pytest.approx(1)


@pytest.mark.parametrize("schema_version", [None, 1, 3, "2", 2.0, True])
def test_unknown_dataset_schema_version_is_rejected(tmp_path, monkeypatch, schema_version):
    cfg, _, _ = _dataset(tmp_path)
    meta_path = tmp_path / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["schema_version"] = schema_version
    meta_path.write_text(json.dumps(meta))
    monkeypatch.setattr(bt, "_load_trained_predictor", lambda *a, **kw: _ContractPredictor())
    with pytest.raises(ValueError, match="schema_version"):
        bt.run_backtest("bundle", cfg, horizons=[1])


@pytest.mark.parametrize("missing", ["file", "market", "freq", "exog_cols"])
def test_trained_backtest_requires_dataset_provenance(tmp_path, monkeypatch, missing):
    cfg, _, _ = _dataset(tmp_path)
    meta_path = tmp_path / "meta.json"
    if missing == "file":
        meta_path.unlink()
    else:
        meta = json.loads(meta_path.read_text())
        del meta[missing]
        meta_path.write_text(json.dumps(meta))
    monkeypatch.setattr(bt, "_load_trained_predictor", lambda *a, **kw: _ContractPredictor())
    with pytest.raises(ValueError, match="meta|repack"):
        bt.run_backtest("bundle", cfg, horizons=[1])


def test_trained_market_type_requires_matching_dataset_declaration(tmp_path, monkeypatch):
    cfg, _, _ = _dataset(tmp_path)

    class PerpetualPredictor(_ContractPredictor):
        manifest = {**_ContractPredictor.manifest, "market_type": "swap"}

    monkeypatch.setattr(bt, "_load_trained_predictor", lambda *a, **kw: PerpetualPredictor())
    with pytest.raises(ValueError, match="market_type"):
        bt.run_backtest("bundle", cfg, horizons=[1])


def test_baseline_can_evaluate_legacy_dataset_without_metadata(tmp_path, monkeypatch):
    cfg, _, _ = _dataset(tmp_path)
    (tmp_path / "meta.json").unlink()
    monkeypatch.setattr(bt, "_load_baseline_predictor", lambda *a, **kw: _OriginalPredictor())
    assert bt.run_backtest(None, cfg, horizons=[1])["n_records"] == 15
