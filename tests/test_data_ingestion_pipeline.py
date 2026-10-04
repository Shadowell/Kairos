"""Resume intraday bars, merge timezone-aware sidecars, then package splits."""

import pickle

import numpy as np
import pandas as pd

from kairos.data import crypto_extras
from kairos.data.collect import fetch_one
from kairos.data.markets.base import FetchTask
from kairos.data.prepare_dataset import main


def _bars(dates):
    return pd.DataFrame(dict(datetime=dates, open=100., high=101., low=99.,
                             close=100., volume=2., amount=200.))


class Exchange:
    name = "crypto"

    def fetch_ohlcv(self, task):
        self.start = pd.to_datetime(task.start, utc=True)
        dates = pd.date_range(self.start, "2026-04-04T00:00:00Z", freq="min", inclusive="left")
        return _bars(dates.tz_convert("Asia/Taipei"))

    def fetch_extras(self, task, kinds):
        assert pd.to_datetime(task.start, utc=True) == self.start
        dates = pd.date_range("2026-04-01T16:00:00Z", periods=7, freq="8h")
        return {"funding": pd.DataFrame(dict(datetime=dates.tz_convert("America/New_York"),
                                            funding_rate=np.arange(7) * .0001 + .0003))}


def test_resume_and_sidecar_timezone_survive_actual_dataset_packaging(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    task = FetchTask(symbol="BTC/USDT:USDT", freq="1min", start="2026-04-01",
                     end="2026-04-03", out_dir=raw)
    _bars(pd.date_range("2026-04-01", periods=841, freq="min")).to_parquet(task.out_path)
    crypto_extras.save_per_symbol(raw, task.out_path.stem, "funding", pd.DataFrame({
        "datetime": pd.date_range("2026-04-01", periods=2, freq="8h"),
        "funding_rate": [.0001, .0002],
    }))
    exchange = Exchange()
    assert fetch_one(exchange, task, daily_append=True, pause=0, extras_kinds=["funding"]) == "ok"
    assert exchange.start == pd.Timestamp("2026-04-01T14:01:00Z")
    collected = pd.read_parquet(task.out_path)
    assert len(collected) == 4320
    assert collected.datetime.diff().dropna().eq(pd.Timedelta(minutes=1)).all()
    monkeypatch.setattr("sys.argv", ["kairos-prepare", "--raw", str(raw), "--out", str(out),
                                    "--market-type", "swap", "--train", "2026-04-01:2026-04-01",
                                    "--val", "2026-04-02:2026-04-02", "--test", "2026-04-03:2026-04-03"])
    main()
    for split in ("train", "val", "test"):
        with (out / f"{split}_data.pkl").open("rb") as stream:
            bars = pickle.load(stream)[task.out_path.stem]
        with (out / f"exog_{split}.pkl").open("rb") as stream:
            factors = pickle.load(stream)[task.out_path.stem]
        assert len(bars) == 1440
        assert factors.shape == (1440, 32)
        assert factors.index.equals(bars.index)
        assert np.isfinite(factors.values).all()
        assert (factors.funding_rate > 0).all()
    assert factors.funding_rate.iloc[-1] == np.float32(.0009)
