"""Offline regressions for the shared bar, split, and target contract."""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

from kairos.data.features import EXOG_COLS
from kairos.data.prepare_dataset import _interleave_trainval, main, process_symbol


def bars(index):
    values = np.arange(len(index), dtype=float) + 100
    return pd.DataFrame({
        "open": values, "high": values + 1, "low": values - 1,
        "close": values, "vol": 2.0, "amt": values * 2,
    }, index=pd.DatetimeIndex(index, name="datetime"))


def test_interleave_uses_complete_calendar_days_instead_of_rows():
    dates = pd.date_range("2026-01-01", periods=8 * 1440, freq="min")
    frame = pd.DataFrame({"datetime": dates})
    train, val = _interleave_trainval(frame, .25, 2, np.random.default_rng(42))
    assert len(val) == 2 * 1440
    assert len(train) == 6 * 1440
    assert set(train.datetime.dt.normalize()).isdisjoint(val.datetime.dt.normalize())


def test_main_contract_normalizes_timezone_without_reordering():
    from kairos.data.contracts import validate_main_frame

    frame = bars(pd.date_range("2026-01-01 08:00", periods=3, freq="min", tz="Asia/Taipei"))
    validated = validate_main_frame(frame.reset_index())
    assert validated.index.equals(pd.date_range("2026-01-01", periods=3, freq="min", name="datetime"))
    assert list(validated.columns) == ["open", "high", "low", "close", "vol", "amt"]
    assert frame.index.tz is not None


@pytest.mark.parametrize("case", ["duplicate", "reverse", "nat", "zero", "negative_volume", "inf"])
def test_main_contract_rejects_invalid_input(case):
    from kairos.data.contracts import validate_main_frame

    frame = bars(pd.date_range("2026-01-01", periods=4, freq="min"))
    if case == "duplicate":
        frame.index = frame.index.take([0, 1, 1, 3])
    elif case == "reverse":
        frame = frame.iloc[::-1]
    elif case == "nat":
        frame.index = pd.DatetimeIndex([pd.NaT, *frame.index[1:]])
    elif case == "zero":
        frame.iloc[0, 3] = 0
    elif case == "negative_volume":
        frame.iloc[0, 4] = -1
    else:
        frame.iloc[0, 0] = np.inf
    with pytest.raises(ValueError):
        validate_main_frame(frame)


@pytest.mark.parametrize("case", ["order", "missing", "extra", "shifted", "nan"])
def test_exog_contract_never_repairs_invalid_channels(case):
    from kairos.data.contracts import validate_exog_frame

    frame = bars(pd.date_range("2026-01-01", periods=4, freq="min"))
    exog = pd.DataFrame(0., index=frame.index, columns=EXOG_COLS)
    if case == "order":
        exog = exog[EXOG_COLS[::-1]]
    elif case == "missing":
        exog = exog.iloc[:, :-1]
    elif case == "extra":
        exog["unknown"] = 0
    elif case == "shifted":
        exog.index = exog.index + pd.Timedelta(minutes=1)
    else:
        exog.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_exog_frame(frame, exog, EXOG_COLS)


def test_continuous_windows_exclude_gap_and_short_segments():
    from kairos.data.contracts import contiguous_starts

    index = pd.date_range("2026-01-01", periods=9, freq="min").delete(4)
    assert contiguous_starts(index, 3, "1min").tolist() == [0, 1, 4, 5]
    assert contiguous_starts(index, 10, "1min").tolist() == []


def test_bar_frequency_is_explicit_and_known():
    from kairos.data.contracts import bar_delta

    assert bar_delta("5min") == pd.Timedelta(minutes=5)
    assert bar_delta("1h") == bar_delta("60min")
    assert bar_delta("daily") == pd.Timedelta(days=1)
    for invalid in ("1month", "invalid", "0min", "-1min"):
        with pytest.raises(ValueError):
            bar_delta(invalid)


def test_normalization_uses_only_history_and_return_targets_remain_raw():
    from kairos.data.contracts import log_return_targets, normalize_window

    raw = np.array([[100., 4.], [110., 4.], [121., 4.], [220., 4.]])
    normalized = normalize_window(raw, 2, 2.)
    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized[:, 0], [-.999998, .999998, 2, 2], atol=1e-6)
    np.testing.assert_array_equal(normalized[:, 1], 0)
    changed_future = raw.copy()
    changed_future[2:] *= 100
    np.testing.assert_array_equal(normalize_window(changed_future, 2, 2.)[:2], normalized[:2])
    targets = log_return_targets(raw[:, 0], 1, 2)
    assert targets.dtype == np.float32
    np.testing.assert_allclose(targets, [np.log(1.1), np.log(2.)])


@pytest.mark.parametrize("anchor,horizon", [(0, 0), (-1, 1), (3, 1), (1, 4)])
def test_return_targets_require_real_future_bars(anchor, horizon):
    from kairos.data.contracts import log_return_targets

    with pytest.raises(ValueError):
        log_return_targets(np.array([100., 110., 120., 130.]), anchor, horizon)


def test_interleave_cli_shares_blocks_despite_missing_symbol_days(tmp_path, monkeypatch):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    dates = pd.date_range("2026-01-01", periods=10 * 24, freq="h")
    for symbol, index in (("A", dates), ("B", dates[24:])):
        frame = bars(index).rename(columns={"vol": "volume", "amt": "amount"})
        frame.reset_index().to_parquet(raw / f"{symbol}.parquet")
    monkeypatch.setattr("sys.argv", [
        "kairos-prepare", "--raw", str(raw), "--out", str(out), "--freq", "1h",
        "--train", "2026-01-01:2026-01-04", "--val", "2026-01-05:2026-01-08",
        "--test", "2026-01-09:2026-01-10", "--split-mode", "interleave",
        "--val-ratio", ".25", "--block-days", "2", "--min-len", "1",
    ])
    main()
    splits = {}
    for split in ("train", "val", "test"):
        with (out / f"{split}_data.pkl").open("rb") as handle:
            splits[split] = pickle.load(handle)
    for split in ("train", "val"):
        shared = splits[split]["A"].index.intersection(dates[24:])
        assert shared.equals(splits[split]["B"].index)
        assert splits[split]["A"].index.intersection(splits["test"]["A"].index).empty
    assert len(splits["val"]["A"]) == 48
    manifest = json.loads((out / "meta.json").read_text())
    assert manifest["freq"] == "1h"
    assert manifest["schema_version"] == 2
    assert manifest["split_parameters"]["block_days"] == 2
    assert manifest["split_parameters"]["seed"] == 42


def test_process_symbol_rejects_overlapping_fit_and_test(tmp_path):
    frame = bars(pd.date_range("2026-01-01", periods=72, freq="h"))
    path = tmp_path / "A.parquet"
    frame.rename(columns={"vol": "volume", "amt": "amount"}).reset_index().to_parquet(path)
    with pytest.raises(ValueError, match="overlap"):
        process_symbol(path, None, ("2026-01-01", "2026-01-01"),
                       ("2026-01-02", "2026-01-02"), ("2026-01-02", "2026-01-03"),
                       min_len=1, split_mode="interleave", interleave_block_days=1)


def write_dataset(tmp_path, *, invalid=None, gap=False):
    from kairos.training.config import TrainConfig

    dates = pd.date_range("2026-01-01", periods=40, freq="min")
    if gap:
        dates = dates.delete(20)
    frame = bars(dates)
    exog = pd.DataFrame(np.repeat(np.arange(len(frame))[:, None], 32, axis=1),
                        index=frame.index, columns=EXOG_COLS, dtype=float)
    if invalid == "order":
        exog = exog[EXOG_COLS[::-1]]
    elif invalid == "index":
        exog.index = exog.index + pd.Timedelta(minutes=1)
    elif invalid == "nan":
        exog.iloc[0, 0] = np.nan
    for split in ("train", "val"):
        with (tmp_path / f"{split}_data.pkl").open("wb") as handle:
            pickle.dump({"A": frame}, handle)
        if invalid != "missing":
            with (tmp_path / f"exog_{split}.pkl").open("wb") as handle:
                pickle.dump({"A": exog}, handle)
    return TrainConfig(dataset_path=str(tmp_path), lookback_window=4, predict_window=2,
                       return_horizon=2, n_train_iter=12, n_val_iter=8)


def test_dataset_only_includes_full_continuous_windows(tmp_path):
    from kairos.training.dataset import KronosSequenceDataset

    cfg = write_dataset(tmp_path, gap=True)
    dataset = KronosSequenceDataset("train", cfg)
    assert dataset.indices == [("A", start) for start in [*range(14), *range(20, 33)]]


@pytest.mark.parametrize("invalid", ["missing", "order", "index", "nan"])
def test_dataset_rejects_invalid_exog_instead_of_repairing(tmp_path, invalid):
    from kairos.training.dataset import KronosSequenceDataset

    cfg = write_dataset(tmp_path, invalid=invalid)
    with pytest.raises((ValueError, FileNotFoundError)):
        KronosSequenceDataset("train", cfg)


def sample_ids(dataset):
    return [int(dataset[index][2][0, 0]) for index in range(len(dataset))]


def test_dataset_epoch_selection_is_distinct_and_index_deterministic(tmp_path):
    import torch
    from torch.utils.data import DistributedSampler
    from kairos.training.dataset import KronosSequenceDataset

    cfg = write_dataset(tmp_path)
    dataset = KronosSequenceDataset("train", cfg)
    replica = KronosSequenceDataset("train", cfg)
    first = sample_ids(dataset)
    assert len(first) == len(set(first)) == 12
    assert sample_ids(dataset) == first
    assert sample_ids(replica) == first
    dataset.set_epoch_seed(2)
    replica.set_epoch_seed(2)
    assert sample_ids(dataset) == sample_ids(replica)
    assert sample_ids(dataset) != first
    shards = []
    for rank in range(2):
        sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=False)
        shards.append({int(dataset[index][2][0, 0]) for index in sampler})
    assert shards[0].isdisjoint(shards[1])
    for left, right in zip(dataset[0], dataset[0]):
        assert torch.equal(left, right)
    for index in (-1, len(dataset)):
        with pytest.raises(IndexError):
            dataset[index]


def test_validation_selection_is_fixed_and_full_pool_has_no_repeats(tmp_path):
    from kairos.training.dataset import KronosSequenceDataset

    cfg = write_dataset(tmp_path)
    dataset = KronosSequenceDataset("val", cfg)
    expected = sample_ids(dataset)
    dataset.set_epoch_seed(8)
    assert sample_ids(dataset) == expected
    cfg.n_train_iter = 1000
    full = KronosSequenceDataset("train", cfg)
    assert set(sample_ids(full)) == set(range(34))
    cfg.n_val_iter = 0
    empty = KronosSequenceDataset("val", cfg)
    assert len(empty) == 0
    with pytest.raises(IndexError):
        empty[0]


def test_dataset_targets_use_raw_anchor_prices_and_triple_stays_compatible(tmp_path):
    from kairos.training.dataset import KronosSequenceDataset

    cfg = write_dataset(tmp_path)
    legacy = KronosSequenceDataset("train", cfg)
    dataset = KronosSequenceDataset("train", cfg, include_targets=True)
    assert len(legacy[0]) == 3
    normalized, stamps, exog, target = dataset[0]
    start = int(exog[0, 0])
    assert normalized.shape == (7, 6)
    assert stamps.shape == (7, 5)
    np.testing.assert_allclose(target, [np.log((104 + start) / (103 + start)),
                                       np.log((105 + start) / (103 + start))])


def test_dataset_rejects_frequency_mismatch_from_manifest(tmp_path):
    from kairos.training.dataset import KronosSequenceDataset

    cfg = write_dataset(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps({"freq": "5min", "exog_cols": EXOG_COLS}))
    with pytest.raises(ValueError, match="freq"):
        KronosSequenceDataset("train", cfg)


@pytest.mark.parametrize("columns", [
    ["hour", "minute", "weekday", "day", "month"],
    ["minute", "hour", "weekday", "day"],
])
def test_dataset_rejects_time_features_that_differ_from_model_slots(tmp_path, columns):
    from kairos.training.dataset import KronosSequenceDataset

    cfg = write_dataset(tmp_path)
    cfg.time_feature_list = columns
    with pytest.raises(ValueError, match="time_feature_list"):
        KronosSequenceDataset("train", cfg)


def test_dataset_samples_do_not_depend_on_worker_count(tmp_path):
    import torch
    from torch.utils.data import DataLoader
    from kairos.training.dataset import KronosSequenceDataset

    cfg = write_dataset(tmp_path)
    dataset = KronosSequenceDataset("train", cfg, include_targets=True)
    dataset.set_epoch_seed(3)
    serial = DataLoader(dataset, batch_size=4, num_workers=0)
    parallel = DataLoader(dataset, batch_size=4, num_workers=2, multiprocessing_context="spawn")
    for serial_batch, parallel_batch in zip(serial, parallel):
        for expected, actual in zip(serial_batch, parallel_batch):
            assert torch.equal(actual, expected)
