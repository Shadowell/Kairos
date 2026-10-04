"""UTC sidecars must align causally and reject supplied malformed data."""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kairos.data import crypto_extras as ce
from kairos.data.features import EXOG_COLS, build_features
from kairos.data.markets.base import FetchTask
from kairos.data.markets.crypto import CryptoAdapter, _align_series


def target_times():
    return pd.Series(pd.date_range("2026-04-01", periods=4, freq="h"))


def test_offset_sidecars_align_to_absolute_utc_without_future_fill():
    source = pd.Series([.1, .2], index=pd.DatetimeIndex([
        "2026-04-01T09:00:00+08:00", "2026-04-01T11:00:00+08:00",
    ]))
    original = source.copy(deep=True)
    for times in (target_times(), target_times().dt.tz_localize("UTC"),
                  target_times().dt.tz_localize("UTC").dt.tz_convert("Asia/Taipei")):
        np.testing.assert_allclose(_align_series(source, times), [np.nan, .1, .1, .2], equal_nan=True)
    pd.testing.assert_series_equal(source, original)


def test_mixed_string_times_are_sorted_stably_and_input_is_unchanged():
    source = pd.Series([.2, .1], index=["2026-04-01T10:00:00+08:00", "2026-04-01 00:00"])
    original = source.copy(deep=True)
    np.testing.assert_allclose(_align_series(source, target_times()), [.1, .1, .2, .2])
    pd.testing.assert_series_equal(source, original)


def test_datetime_column_frame_has_one_explicit_payload():
    source = pd.DataFrame({"datetime": ["2026-04-01T00:00:00Z"], "funding_rate": [.1]})
    np.testing.assert_allclose(_align_series(source, target_times()), [.1] * 4)


@pytest.mark.parametrize("source", [None, pd.Series(dtype=float), pd.DataFrame(), np.array([])])
def test_missing_and_empty_channels_remain_missing(source):
    assert _align_series(source, target_times()).isna().all()


def test_positional_array_remains_supported_but_nan_is_missing():
    values = np.array([.1, np.nan, .2, .3])
    np.testing.assert_allclose(_align_series(values, target_times()), values, equal_nan=True)
    source = pd.Series([np.nan, .2], index=pd.date_range("2026-04-01", periods=2, freq="2h"))
    np.testing.assert_allclose(_align_series(source, target_times()), [np.nan, np.nan, .2, .2], equal_nan=True)


@pytest.mark.parametrize("source", [
    pd.Series([.1, .2]),
    pd.Series([.1], index=["not-a-date"]),
    pd.Series([.1], index=pd.DatetimeIndex([pd.NaT])),
    pd.Series([.1, .2], index=["2026-04-01T00:00:00Z", "2026-04-01T08:00:00+08:00"]),
    pd.DataFrame({"first": [.1], "second": [.2]}, index=pd.date_range("2026-04-01", periods=1)),
    pd.DataFrame(index=pd.date_range("2026-04-01", periods=1)),
    np.array([[.1, .2], [.3, .4]]),
    np.array([.1]),
    {"funding_rate": .1},
    pd.Series(["bad"], index=pd.date_range("2026-04-01", periods=1)),
    pd.Series([np.inf], index=pd.date_range("2026-04-01", periods=1)),
])
def test_supplied_invalid_sidecars_raise_and_log(source, caplog):
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            _align_series(source, target_times())
    assert caplog.records


@pytest.mark.parametrize("times", [pd.Series([0, 1]), pd.Series([pd.NaT]),
                                    pd.Series(["2026-04-01", "not-a-date"])])
def test_invalid_main_timestamps_are_not_interpreted_as_epoch(times):
    with pytest.raises(ValueError):
        _align_series(None, times)


def test_normalise_preserves_mixed_time_observations_and_nan_payload():
    frame = pd.DataFrame({"datetime": ["2026-04-01T10:00:00+08:00", "2026-04-01 00:00"],
                          "funding_rate": [.2, np.nan]})
    original = frame.copy(deep=True)
    result = ce._normalise(frame, "funding_rate")
    assert result.datetime.tolist() == [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-01 02:00")]
    assert pd.isna(result.funding_rate.iloc[0])
    assert result.funding_rate.iloc[1] == .2
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"funding_rate": [.1]}),
    pd.DataFrame({"datetime": ["bad"], "funding_rate": [.1]}),
    pd.DataFrame({"datetime": [pd.NaT], "funding_rate": [.1]}),
    pd.DataFrame({"datetime": ["2026-04-01", "2026-04-01T08:00:00+08:00"], "funding_rate": [.1, .2]}),
    pd.DataFrame({"datetime": ["2026-04-01"], "funding_rate": ["bad"]}),
    pd.DataFrame({"datetime": ["2026-04-01"], "funding_rate": [np.inf]}),
])
def test_normalise_rejects_bad_sidecars_instead_of_dropping_rows(frame):
    with pytest.raises(ValueError):
        ce._normalise(frame, "funding_rate")


@pytest.mark.parametrize("case", ["corrupt", "missing_columns", "nat", "duplicates", "bad_payload"])
def test_existing_invalid_sidecar_is_visible_and_preserved(tmp_path, case, caplog):
    path = ce.per_symbol_path(tmp_path, ce.KIND_FUNDING, "BTC")
    path.parent.mkdir(parents=True)
    if case == "corrupt":
        path.write_bytes(b"corrupt parquet data")
    else:
        frame = pd.DataFrame({"datetime": ["2026-04-01"], "funding_rate": [.1]})
        if case == "missing_columns":
            frame = frame.drop(columns="datetime")
        elif case == "nat":
            frame["datetime"] = pd.NaT
        elif case == "duplicates":
            frame = pd.concat([frame, frame], ignore_index=True)
        else:
            frame["funding_rate"] = "bad"
        frame.to_parquet(path, index=False)
    before = path.read_bytes()
    with caplog.at_level(logging.ERROR):
        with pytest.raises(ValueError):
            ce.load_for_symbol(tmp_path, "BTC")
    assert str(path) in caplog.text
    assert path.read_bytes() == before


@pytest.mark.parametrize("columns", [[], ["datetime", "funding_rate"]])
def test_empty_sidecar_file_remains_a_missing_channel(tmp_path, columns):
    path = ce.per_symbol_path(tmp_path, ce.KIND_FUNDING, "BTC")
    path.parent.mkdir(parents=True)
    pd.DataFrame(columns=columns).to_parquet(path, index=False)
    assert ce.load_for_symbol(tmp_path, "BTC") == {}


@pytest.mark.parametrize("market_wide", [False, True])
def test_merge_refuses_bad_existing_files_instead_of_overwriting(tmp_path, market_wide):
    kind = ce.KIND_REFERENCE if market_wide else ce.KIND_FUNDING
    payload = "close" if market_wide else "funding_rate"
    path = ce.market_wide_path(tmp_path, kind) if market_wide else ce.per_symbol_path(tmp_path, kind, "BTC")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"existing corrupt data must survive")
    frame = pd.DataFrame({"datetime": ["2026-04-01"], payload: [.1]})
    before = path.read_bytes()
    with pytest.raises(ValueError):
        if market_wide:
            ce.save_market_wide(tmp_path, kind, frame)
        else:
            ce.save_per_symbol(tmp_path, "BTC", kind, frame)
    assert path.read_bytes() == before


def test_partial_write_failure_does_not_modify_existing_sidecar(tmp_path, monkeypatch):
    frame = pd.DataFrame({"datetime": ["2026-04-01"], "funding_rate": [.1]})
    path = ce.save_per_symbol(tmp_path, "BTC", ce.KIND_FUNDING, frame)
    before = path.read_bytes()
    def interrupted_write(self, filename, **kwargs):
        filename.write_bytes(b"incomplete output")
        raise OSError("simulated write failure")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupted_write)
    new = pd.DataFrame({"datetime": ["2026-04-02"], "funding_rate": [.2]})
    with pytest.raises(OSError, match="simulated"):
        ce.save_per_symbol(tmp_path, "BTC", ce.KIND_FUNDING, new)
    assert path.read_bytes() == before
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("market_wide", [False, True])
def test_merge_deduplicates_equal_observations_but_rejects_conflicts(tmp_path, market_wide):
    kind = ce.KIND_REFERENCE if market_wide else ce.KIND_FUNDING
    payload = "close" if market_wide else "funding_rate"
    frame = pd.DataFrame({"datetime": ["2026-04-01"], payload: [.1]})
    def save(data):
        return (ce.save_market_wide(tmp_path, kind, data) if market_wide
                else ce.save_per_symbol(tmp_path, "BTC", kind, data))
    path = save(frame)
    same = frame.copy()
    same["datetime"] = "2026-04-01T08:00:00+08:00"
    save(same)
    assert len(pd.read_parquet(path)) == 1
    before = path.read_bytes()
    same[payload] = .2
    with pytest.raises(ValueError, match="duplicate|conflict"):
        save(same)
    assert path.read_bytes() == before


def test_real_sidecar_roundtrip_and_features_are_timezone_invariant(tmp_path):
    rates = pd.DataFrame({"datetime": ["2026-04-01 00:00", "2026-04-01T16:00:00+08:00",
                                       "2026-04-01T16:00:00Z"], "funding_rate": [.0001, .0002, -.0001]})
    ce.save_per_symbol(tmp_path, "BTC", ce.KIND_FUNDING, rates)
    extras = ce.load_for_symbol(tmp_path, "BTC")
    assert len(extras["funding_rate"]) == 3
    times = pd.date_range("2026-04-01", periods=24, freq="h", tz="UTC")
    close = np.arange(24, dtype=float) + 100
    bars = pd.DataFrame({"datetime": times, "open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 2., "amount": close * 2})
    utc = build_features(bars, extras=extras)
    bars["datetime"] = bars.datetime.dt.tz_convert("Asia/Taipei")
    offset = build_features(bars, extras=extras)
    pd.testing.assert_frame_equal(utc[EXOG_COLS], offset[EXOG_COLS])
    np.testing.assert_allclose(utc.funding_rate.iloc[[0, 8, 16]], [.0001, .0002, -.0001])
    assert len(EXOG_COLS) == 32
    assert np.isfinite(utc[EXOG_COLS].values).all()


def _exchange_adapter(kind, hook):
    hook_name = {ce.KIND_FUNDING: "fetch_funding_rate_history",
                 ce.KIND_OI: "fetch_open_interest_history"}.get(kind, "fetch_spot_ohlcv")
    adapter = CryptoAdapter(market_type="swap")
    adapter._exchange = SimpleNamespace(name="offline-test", **{hook_name: hook})
    return adapter


def _extras_task(tmp_path):
    return FetchTask(symbol="BTC/USDT:USDT", freq="1min", start="2026-04-01",
                     end="2026-04-02", out_dir=tmp_path)


@pytest.mark.parametrize("kind", ce.ALL_KINDS)
@pytest.mark.parametrize("exception", [ValueError, TypeError, KeyError])
def test_real_adapter_propagates_invalid_exchange_payloads(tmp_path, caplog, kind, exception):
    def hook(*args, **kwargs):
        raise exception("invalid exchange payload")
    adapter = _exchange_adapter(kind, hook)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(exception):
            adapter.fetch_extras(_extras_task(tmp_path), [kind])
    assert "invalid" in caplog.text.lower()


@pytest.mark.parametrize("kind", [ce.KIND_SPOT, ce.KIND_REFERENCE])
@pytest.mark.parametrize("column", ["datetime", "close"])
def test_real_adapter_logs_missing_spot_columns_before_raising(tmp_path, caplog, kind, column):
    frame = pd.DataFrame({"datetime": [pd.Timestamp("2026-04-01")], "close": [100.]}).drop(columns=column)
    adapter = _exchange_adapter(kind, lambda *args, **kwargs: frame)
    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyError):
            adapter.fetch_extras(_extras_task(tmp_path), [kind])
    assert "invalid" in caplog.text.lower()


@pytest.mark.parametrize("kind", ce.ALL_KINDS)
def test_real_adapter_still_skips_network_failures(tmp_path, kind):
    def hook(*args, **kwargs):
        raise TimeoutError("remote endpoint unavailable")
    assert _exchange_adapter(kind, hook).fetch_extras(_extras_task(tmp_path), [kind]) == {}


@pytest.mark.parametrize("kind", [ce.KIND_FUNDING, ce.KIND_OI])
def test_funding_and_oi_missing_payload_columns_fail_when_saved(tmp_path, kind):
    frame = pd.DataFrame({"datetime": [pd.Timestamp("2026-04-01")], "unexpected": [.1]})
    fetched = _exchange_adapter(kind, lambda *args, **kwargs: frame).fetch_extras(_extras_task(tmp_path), [kind])
    with pytest.raises(ValueError, match="payload column"):
        ce.save_per_symbol(tmp_path, "BTC", kind, fetched[kind])
    assert not ce.per_symbol_path(tmp_path, kind, "BTC").exists()
