"""Offline smoke test for the crypto extras pipeline (Phase 4).

Writes synthetic perp OHLCV + funding + OI + spot parquet into a temp
directory laid out like ``kairos-collect --crypto-extras all`` would
produce, then runs ``prepare_dataset.main`` against it and checks that
the five data-driven market exog columns in the resulting pickles are
actually non-zero for at least one bar.

Usage
-----
    python scripts/smoke_crypto_extras.py

Exits non-zero with a loud error if the pipeline silently zeroes out the
extras (which is exactly the regression this commit line is fighting).
"""
from __future__ import annotations

import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data import crypto_extras as ce


SYMBOLS = ["BTC_USDT-USDT", "ETH_USDT-USDT", "SOL_USDT-USDT"]
# Need enough bars so that prepare_dataset's min-len=200 check passes
# and the time-based split has populated all three windows.
N_BARS = 4 * 60 * 24  # 4 days of 1-minute bars


def _synthetic_ohlcv(symbol: str, start: pd.Timestamp, n: int) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
    drift = rng.normal(0, 0.0005, size=n).cumsum()
    price = 100.0 * np.exp(drift)
    volume = rng.uniform(0.5, 1.5, size=n)
    idx = pd.date_range(start, periods=n, freq="1min")
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": price,
            "high": price * (1 + rng.uniform(0, 0.001, size=n)),
            "low": price * (1 - rng.uniform(0, 0.001, size=n)),
            "close": price * (1 + rng.normal(0, 0.0003, size=n)),
            "volume": volume,
            "amount": volume * price,
        }
    )


def _synthetic_funding(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = pd.date_range(start, end, freq="8h")
    return pd.DataFrame({
        "datetime": ts,
        "funding_rate": np.sin(np.arange(len(ts)) / 3.0) * 0.0002,
    })


def _synthetic_oi(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    ts = pd.date_range(start, end, freq="5min")
    # Random walk so oi_change is non-zero bar to bar.
    vals = 10_000.0 * (1 + np.cumsum(np.random.default_rng(0).normal(0, 0.001, len(ts))))
    return pd.DataFrame({"datetime": ts, "open_interest": vals})


def _synthetic_spot(perp: pd.DataFrame) -> pd.DataFrame:
    # Spot trades ~20bps below perp on average so basis is clearly non-zero.
    spot_close = perp["close"].values * (1 - 0.002)
    return pd.DataFrame({"datetime": perp["datetime"].values, "close": spot_close})


def build_fixture(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2024-01-01")
    for sym in SYMBOLS:
        perp = _synthetic_ohlcv(sym, start, N_BARS)
        perp.to_parquet(raw_dir / f"{sym}.parquet", index=False)

        ce.save_per_symbol(raw_dir, sym, ce.KIND_FUNDING,
                           _synthetic_funding(start, perp["datetime"].iloc[-1]))
        ce.save_per_symbol(raw_dir, sym, ce.KIND_OI,
                           _synthetic_oi(start, perp["datetime"].iloc[-1]))
        ce.save_per_symbol(raw_dir, sym, ce.KIND_SPOT, _synthetic_spot(perp))
    print(f"[fixture] wrote {len(SYMBOLS)} symbols under {raw_dir}")
    print(f"[fixture] extras channels: {ce.available_channels(raw_dir)}")


def run_prepare(raw_dir: Path, out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m", "kairos.data.prepare_dataset",
        "--raw", str(raw_dir),
        "--out", str(out_dir),
        "--market", "crypto",
        "--train", "2024-01-01:2024-01-02",
        "--val", "2024-01-03:2024-01-03",
        "--test", "2024-01-04:2024-01-04",
        "--split-mode", "time",
        "--min-len", "100",
    ]
    print("[prepare]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def assert_non_zero(out_dir: Path) -> None:
    with open(out_dir / "exog_train.pkl", "rb") as f:
        exog_train: dict[str, pd.DataFrame] = pickle.load(f)

    if not exog_train:
        raise SystemExit("[FAIL] exog_train.pkl is empty; prepare_dataset produced nothing")

    driven = ["funding_rate", "funding_rate_z", "oi_change", "basis"]
    print(f"[check] inspecting {len(exog_train)} symbols; target cols: {driven}")
    bad: list[str] = []
    for sym, df in exog_train.items():
        nz = {c: bool((df[c].abs() > 1e-9).any()) for c in driven}
        print(f"  {sym}: {nz}")
        if not all(nz.values()):
            bad.append(sym)
    if bad:
        raise SystemExit(
            f"[FAIL] these symbols still have all-zero extras: {bad}. "
            "The wiring from crypto_extras -> build_features is broken."
        )

    import json
    with open(out_dir / "meta.json") as f:
        meta = json.load(f)
    print(f"[meta] extras_channels recorded: {meta.get('extras_channels')}")
    expected = [ce.KIND_FUNDING, ce.KIND_OI, ce.KIND_SPOT]
    if sorted(meta.get("extras_channels", [])) != sorted(expected):
        raise SystemExit(
            f"[FAIL] meta.json extras_channels={meta.get('extras_channels')}, "
            f"expected {expected}"
        )
    print("[PASS] all data-driven exog columns are non-zero and meta.json is tagged.")


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="kairos_smoke_crypto_"))
    raw = tmp / "raw"
    out = tmp / "out"
    try:
        build_fixture(raw)
        run_prepare(raw, out)
        assert_non_zero(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
