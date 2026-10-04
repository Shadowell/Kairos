"""Offline regressions for date boundaries and dataset packaging failures."""

from datetime import datetime
import os
import pickle
import time

import numpy as np
import pandas as pd
import pytest

from kairos.data.markets.crypto import _to_unix_ms
from kairos.data.prepare_dataset import _slice, main, parse_range, process_symbol


@pytest.mark.parametrize("start,end", [
    ("2026-04-13", "2026-04-14"),
    ("2026-04-13 08:00", "2026-04-14 08:00"),
    ("2026-04-13T08:00:00+08:00", "2026-04-14T00:00:00Z"),
])
def test_parse_timestamp_ranges(start, end):
    assert parse_range(f" {start}:{end} ") == (start, end)


@pytest.mark.parametrize("value", [
    "2026-04-14:2026-04-13", "bad:date", ":", "2026-02-30:2026-03-01",
])
def test_reject_invalid_ranges(value):
    with pytest.raises(ValueError):
        parse_range(value)


def test_date_end_includes_whole_day_but_timestamp_is_precise():
    df = pd.DataFrame({"datetime": pd.date_range("2026-04-13", periods=1441, freq="min")})
    assert len(_slice(df, "2026-04-13", "2026-04-13")) == 1440
    assert len(_slice(df, "2026-04-13", "2026-04-13T00:00:00")) == 1


def test_slice_normalizes_timezone_offsets():
    df = pd.DataFrame({"datetime": pd.date_range("2026-04-13", periods=3, freq="h")})
    result = _slice(df, "2026-04-13T08:00:00+08:00", "2026-04-13T09:00:00+08:00")
    assert result["datetime"].tolist() == [pd.Timestamp("2026-04-13"), pd.Timestamp("2026-04-13 01:00")]


def test_collection_timestamps_are_independent_of_host_timezone():
    original = os.environ.get("TZ")
    try:
        for zone in ("UTC", "Asia/Taipei", "America/New_York"):
            os.environ["TZ"] = zone
            time.tzset()
            assert _to_unix_ms("1970-01-01") == 0
            assert _to_unix_ms("1970-01-01", end_of_day=True) == 86400000
            assert _to_unix_ms("1970-01-01T00:00:00", end_of_day=True) == 0
            assert _to_unix_ms(datetime(1970, 1, 1), end_of_day=True) == 0
            assert _to_unix_ms("1970-01-01T08:00:00+08:00") == 0
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()


def _bars():
    return pd.DataFrame({
        "datetime": pd.date_range("2026-04-13", periods=3 * 1440, freq="min"),
        "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0,
        "volume": 2.0,
    })


@pytest.mark.parametrize("amount", ["absent", "partial", "all_nan"])
def test_packaging_fills_missing_amount_without_dropping_bars(tmp_path, amount):
    df = _bars()
    if amount != "absent":
        df["amount"] = np.nan
        if amount == "partial":
            df.loc[0, "amount"] = 777.0
    path = tmp_path / "BTC_USDT.parquet"
    df.to_parquet(path)
    pieces = process_symbol(path, None, ("2026-04-13", "2026-04-13"),
                            ("2026-04-14", "2026-04-14"), ("2026-04-15", "2026-04-15"))
    assert [len(pieces[s]["main"]) for s in ("train", "val", "test")] == [1440] * 3
    assert pieces["train"]["main"]["amt"].iloc[0] == (777.0 if amount == "partial" else 202.0)
    assert pieces["train"]["main"]["amt"].iloc[1] == 202.0
    for part in pieces.values():
        assert part["exog"].shape == (1440, 32)
        assert part["main"].index.equals(part["exog"].index)


@pytest.mark.parametrize("case", ["empty", "invalid", "overlap", "missing_split", "corrupt"])
def test_cli_failure_preserves_existing_output(tmp_path, monkeypatch, case):
    raw = tmp_path / "raw"
    raw.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    sentinel = out / "train_data.pkl"
    sentinel.write_bytes(b"existing dataset")
    if case not in ("empty", "corrupt"):
        _bars().to_parquet(raw / "BTC_USDT.parquet")
    elif case == "corrupt":
        (raw / "BTC_USDT.parquet").write_bytes(b"invalid parquet")
    train = "bad:range" if case == "invalid" else "2026-04-13:2026-04-13"
    val = "2026-04-13:2026-04-14" if case == "overlap" else "2026-04-14:2026-04-14"
    test = "2027-01-01:2027-01-01" if case == "missing_split" else "2026-04-15:2026-04-15"
    monkeypatch.setattr("sys.argv", ["kairos-prepare", "--raw", str(raw), "--out", str(out),
                                    "--train", train, "--val", val, "--test", test])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code != 0
    assert sentinel.read_bytes() == b"existing dataset"
    assert list(out.iterdir()) == [sentinel]


def test_cli_packages_real_parquet(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    _bars().to_parquet(raw / "BTC_USDT.parquet")
    out = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["kairos-prepare", "--raw", str(raw), "--out", str(out),
                                    "--train", "2026-04-13:2026-04-13",
                                    "--val", "2026-04-14:2026-04-14",
                                    "--test", "2026-04-15:2026-04-15", "--limit", "1"])
    main()
    for split in ("train", "val", "test"):
        with (out / f"{split}_data.pkl").open("rb") as stream:
            data = pickle.load(stream)
        assert len(data["BTC_USDT"]) == 1440
    assert (out / "meta.json").is_file()
