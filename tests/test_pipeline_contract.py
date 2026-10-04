"""Offline CLI training -> saved bundle -> shared inference/backtest/HTTP."""

import json
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from fastapi.testclient import TestClient

from kairos.data.features import EXOG_COLS
from kairos.deploy.serve import _build_app
from kairos.inference import KairosPredictor
from kairos.training.backtest_ic import run_backtest
from kairos.training.config import TrainConfig
from kairos.vendor.kronos import Kronos, KronosTokenizer


def test_real_cli_bundle_backtest_and_http_share_predictions(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = Kronos(s1_bits=2, s2_bits=2, n_layers=1, d_model=16, n_heads=4, ff_dim=32,
                   ffn_dropout_p=0., attn_dropout_p=0., resid_dropout_p=0.,
                   token_dropout_p=0., learn_te=True)
    tokenizer = KronosTokenizer(d_in=6, d_model=16, n_heads=4, ff_dim=32,
                                n_enc_layers=2, n_dec_layers=2, ffn_dropout_p=0.,
                                attn_dropout_p=0., resid_dropout_p=0., s1_bits=2, s2_bits=2,
                                beta=.25, gamma0=1., gamma=1., zeta=1., group_size=2)
    model.save_pretrained(str(tmp_path / "base"))
    tokenizer.save_pretrained(str(tmp_path / "tokenizer"))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for day, split in enumerate(("train", "val", "test"), 1):
        main, exog = {}, {}
        for symbol_index, symbol in enumerate(("BTC", "ETH", "SOL")):
            dates = pd.date_range(f"2026-01-0{day}", periods=330, freq="min", name="datetime")
            close = 100 + np.sin(np.arange(330) / (5 + symbol_index)) + symbol_index * 20
            main[symbol] = pd.DataFrame(dict(open=close, high=close+1, low=close-1,
                                             close=close, vol=2., amt=close*2), index=dates)
            exog[symbol] = pd.DataFrame(.1, index=dates, columns=EXOG_COLS)
        for name, content in ((f"{split}_data.pkl", main), (f"exog_{split}.pkl", exog)):
            with (data_dir / name).open("wb") as stream:
                pickle.dump(content, stream)
    (data_dir / "meta.json").write_text(json.dumps(dict(
        schema_version=2, market="crypto", market_type="spot", freq="1min", exog_cols=EXOG_COLS)))
    env = {**os.environ, "KAIROS_SMOKE": "1", "KAIROS_DATASET": str(data_dir),
           "KAIROS_PRETRAINED_PREDICTOR": str(tmp_path / "base"),
           "KAIROS_PRETRAINED_TOKENIZER": str(tmp_path / "tokenizer"),
           "KAIROS_SAVE_PATH": str(tmp_path / "runs"), "KAIROS_RUN_ID": "cli-smoke",
           "KAIROS_N_TRAIN_ITER": "8", "KAIROS_N_VAL_ITER": "4",
           "KAIROS_ACCUM_STEPS": "2", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    # No inherited torchrun context or user's prior override may alter the smoke.
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "KAIROS_PRESET", "KAIROS_BATCH_SIZE",
                "KAIROS_NUM_WORKERS", "KAIROS_EPOCHS", "KAIROS_LR"):
        env.pop(key, None)
    result = subprocess.run([sys.executable, "-m", "kairos.training.train_predictor"],
                            env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    bundle = tmp_path / "runs/predictor/cli-smoke/checkpoints/best_model"
    engine = KairosPredictor.from_checkpoint(bundle, device="cpu")
    history = main["BTC"].iloc[:256]
    factors = exog["BTC"].iloc[:256]
    expected = engine.predict_batch([history], [factors])[0]
    bars = history.rename(columns={"vol": "volume", "amt": "amount"}).reset_index()
    bars["datetime"] = bars["datetime"].map(lambda value: value.isoformat())
    response = TestClient(_build_app(engine)).post("/predict", json=dict(
        symbol="BTC", market_type="spot", freq="1min", pred_len=2,
        bars=bars.to_dict("records"), exog_cols=EXOG_COLS, exog=factors.values.tolist()))
    assert response.status_code == 200, response.text
    np.testing.assert_allclose(response.json()["forecast"][0]["log_return_quantiles"], expected[0])
    cfg = TrainConfig(dataset_path=str(data_dir))
    report = run_backtest(str(bundle), cfg, horizons=[1, 2], device="cpu", per_symbol_limit=2)
    assert report["n_records"] == 6
    assert report["n_symbols"] == 3
    assert report["methodology"]["version"] == 2
    # Upload validation reads the complete bound bundle, without network or mutation.
    original_files = {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    uploaded = subprocess.run([sys.executable, "-m", "kairos.deploy.push_to_hf", "--predictor-ckpt",
                               str(bundle), "--repo-predictor", "test/model", "--dry-run"],
                              capture_output=True, text=True, timeout=30)
    assert uploaded.returncode == 0, uploaded.stdout + uploaded.stderr
    assert {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()} == original_files
