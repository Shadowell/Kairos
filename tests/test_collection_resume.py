"""Real parquet regressions for frequency-aware, UTC-safe collection resumes."""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from kairos.data import collect
from kairos.data.markets.base import FetchTask


class OfflineAdapter:
    name = "offline"

    def __init__(self, rows):
        self.rows = rows
        self.requests = []
        self.extra_requests = []

    def fetch_ohlcv(self, task):
        self.requests.append(task)
        return self.rows.copy()

    def fetch_extras(self, task, kinds):
        self.extra_requests.append((task, kinds))
        return {}


def _bars(dates, prices=None):
    return pd.DataFrame({"datetime": dates, "close": prices if prices is not None else range(100, 100 + len(dates))})


def _task(tmp_path, *, freq="1min", end="2026-04-13", start="2026-04-01"):
    return FetchTask(symbol="BTC/USDT", freq=freq, start=start, end=end, out_dir=tmp_path)


@pytest.mark.parametrize("freq,last,expected", [
    ("1min", "2026-04-13 14:00", "2026-04-13 14:01"),
    ("5min", "2026-04-13 14:00", "2026-04-13 14:05"),
    ("1h", "2026-04-13 14:00", "2026-04-13 15:00"),
    ("daily", "2026-04-12", "2026-04-13"),
    ("1d", "2026-04-12", "2026-04-13"),
])
def test_resume_advances_one_native_bar_and_includes_remainder_of_end_day(tmp_path, freq, last, expected):
    task = _task(tmp_path, freq=freq)
    history = _bars([pd.Timestamp(last)], [101.])
    history.to_parquet(task.out_path, index=False)
    adapter = OfflineAdapter(_bars([pd.Timestamp(expected)], [102.]))

    assert collect.fetch_one(adapter, task, daily_append=True, pause=0, extras_kinds=["funding"]) == "ok"
    assert pd.Timestamp(adapter.requests[0].start) == pd.Timestamp(expected, tz="UTC")
    assert adapter.requests[0].end == task.end
    assert adapter.extra_requests == [(adapter.requests[0], ["funding"])]
    saved = pd.read_parquet(task.out_path)
    assert saved.datetime.tolist() == [pd.Timestamp(last), pd.Timestamp(expected)]
    assert saved.close.tolist() == [101., 102.]
    assert task.start == "2026-04-01"  # Caller-owned request is not mutated.


@pytest.mark.parametrize("last,end", [
    ("2026-04-13 23:59", "2026-04-13"),
    ("2026-04-13 14:00", "2026-04-13T14:01:00Z"),
    ("2026-04-13 23:59", "2026-04-14T08:00:00+08:00"),
])
def test_at_exclusive_end_boundary_skips_without_network_or_file_rewrite(tmp_path, last, end):
    task = _task(tmp_path, end=end)
    _bars([pd.Timestamp(last)]).to_parquet(task.out_path, index=False)
    original = task.out_path.read_bytes()
    adapter = OfflineAdapter(pd.DataFrame())

    assert collect.fetch_one(adapter, task, daily_append=True, pause=0) == "skip_up_to_date"
    assert adapter.requests == []
    assert task.out_path.read_bytes() == original


def test_resume_preserves_seconds_and_merges_aware_history_with_naive_utc(tmp_path):
    task = _task(tmp_path, end="2026-04-13T15:00:37+08:00")
    old = pd.Timestamp("2026-04-13T14:00:37+08:00")
    _bars([old], [123.]).to_parquet(task.out_path, index=False)
    adapter = OfflineAdapter(_bars([pd.Timestamp("2026-04-13 06:01:37")], [124.]))

    assert collect.fetch_one(adapter, task, daily_append=True, pause=0) == "ok"
    assert pd.Timestamp(adapter.requests[0].start) == pd.Timestamp("2026-04-13T06:01:37Z")
    saved = pd.read_parquet(task.out_path)
    assert saved.datetime.tolist() == [pd.Timestamp("2026-04-13 06:00:37"), pd.Timestamp("2026-04-13 06:01:37")]
    assert saved.close.tolist() == [123., 124.]


def test_merge_filters_new_window_only_and_preserves_old_rows(tmp_path):
    task = _task(tmp_path, end="2026-04-13T14:03:00Z")
    history = _bars(pd.to_datetime(["2026-04-13 14:00", "2026-04-01 00:00"]), [100., 90.])
    history.to_parquet(task.out_path, index=False)
    adapter = OfflineAdapter(_bars(pd.to_datetime([
        "2026-04-13 14:03", "2026-04-13 14:02", "2026-04-13 14:00",
        "2026-04-13 14:01", "2026-04-13 14:01",
    ]), [103., 102., 999., 101., 888.]))

    assert collect.fetch_one(adapter, task, daily_append=True, pause=0) == "ok"
    saved = pd.read_parquet(task.out_path)
    assert saved.datetime.is_monotonic_increasing
    assert saved.datetime.is_unique
    assert saved.close.tolist() == [90., 100., 101., 102.]


@pytest.mark.parametrize("response", ["empty", "outside"])
def test_no_new_in_range_data_leaves_history_bytes_unchanged(tmp_path, response):
    task = _task(tmp_path, end="2026-04-13T14:03:00Z")
    _bars([pd.Timestamp("2026-04-13 14:00")]).to_parquet(task.out_path, index=False)
    original = task.out_path.read_bytes()
    rows = pd.DataFrame() if response == "empty" else _bars([pd.Timestamp("2026-04-13 14:00")])
    adapter = OfflineAdapter(rows)
    assert collect.fetch_one(adapter, task, daily_append=True, pause=0, extras_kinds=["funding"]) == "empty"
    assert task.out_path.read_bytes() == original
    assert adapter.extra_requests == []


@pytest.mark.parametrize("history", ["corrupt", "invalid_datetime", "nat", "missing_datetime", "numeric_datetime"])
def test_bad_history_fails_before_fetch_and_cannot_be_overwritten(tmp_path, history, caplog):
    task = _task(tmp_path)
    if history == "corrupt":
        task.out_path.write_bytes(b"not parquet; preserve these exact bytes")
    elif history == "invalid_datetime":
        _bars(["not-a-date"]).to_parquet(task.out_path, index=False)
    elif history == "nat":
        _bars([pd.NaT]).to_parquet(task.out_path, index=False)
    elif history == "missing_datetime":
        pd.DataFrame({"close": [100.]}).to_parquet(task.out_path, index=False)
    else:
        _bars([1_776_084_000_000]).to_parquet(task.out_path, index=False)
    original = task.out_path.read_bytes()
    adapter = OfflineAdapter(_bars([pd.Timestamp("2026-04-13 14:01")]))
    assert collect.fetch_one(adapter, task, daily_append=True, pause=0) == "fail"
    assert adapter.requests == []
    assert task.out_path.read_bytes() == original
    assert "existing" in caplog.text.lower()


def test_user_start_is_not_moved_backwards_when_history_is_older(tmp_path):
    task = _task(tmp_path, start="2026-04-13T14:00:00+08:00")
    _bars([pd.Timestamp("2026-04-12")]).to_parquet(task.out_path, index=False)
    adapter = OfflineAdapter(_bars([pd.Timestamp("2026-04-13 06:00")]))
    assert collect.fetch_one(adapter, task, daily_append=True, pause=0) == "ok"
    assert pd.Timestamp(adapter.requests[0].start) == pd.Timestamp("2026-04-13T06:00:00Z")


def test_cli_default_end_uses_current_utc_date(tmp_path, monkeypatch):
    class LocalNextDay(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 13, 20, tzinfo=timezone.utc) if tz is not None else cls(2026, 4, 14, 4)

    monkeypatch.setattr(collect, "datetime", LocalNextDay)
    monkeypatch.setattr("sys.argv", ["kairos-collect"])
    assert collect.parse_args().end == "2026-04-13"


def test_partial_main_parquet_write_failure_preserves_history(tmp_path, monkeypatch, caplog):
    task = _task(tmp_path)
    _bars([pd.Timestamp("2026-04-13 14:00")]).to_parquet(task.out_path, index=False)
    original = task.out_path.read_bytes()
    adapter = OfflineAdapter(_bars([pd.Timestamp("2026-04-13 14:01")]))

    def partial_write(self, path, **kwargs):
        Path(path).write_bytes(b"partial parquet write")
        raise OSError("injected disk write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", partial_write)
    assert collect.fetch_one(adapter, task, daily_append=True, pause=0, extras_kinds=["funding"]) == "fail"
    assert task.out_path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [task.out_path]
    assert len(adapter.extra_requests) == 1
    assert "write failure" in caplog.text


def test_in_progress_write_is_not_discoverable_as_a_formal_parquet(tmp_path, monkeypatch):
    task = _task(tmp_path)
    _bars([pd.Timestamp("2026-04-13 14:00")], [100.]).to_parquet(task.out_path, index=False)
    adapter = OfflineAdapter(_bars([pd.Timestamp("2026-04-13 14:01")], [101.]))
    real_write = pd.DataFrame.to_parquet
    discovered = []

    def inspect_while_writing(self, path, **kwargs):
        discovered.extend(tmp_path.glob("*.parquet"))
        return real_write(self, path, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", inspect_while_writing)
    assert collect.fetch_one(adapter, task, daily_append=True, pause=0) == "ok"
    assert discovered == [task.out_path]
    assert pd.read_parquet(task.out_path).close.tolist() == [100., 101.]


@pytest.mark.parametrize("existing", ["corrupt", "conflict"])
def test_requested_sidecar_validation_failure_reports_fail_and_preserves_sidecar(tmp_path, existing, caplog):
    from kairos.data import crypto_extras
    from kairos.data.markets.base import sanitize_symbol

    task = _task(tmp_path)
    _bars([pd.Timestamp("2026-04-13 14:00")], [100.]).to_parquet(task.out_path, index=False)
    sidecar = crypto_extras.per_symbol_path(tmp_path, "funding", sanitize_symbol(task.symbol))
    sidecar.parent.mkdir(parents=True)
    if existing == "corrupt":
        sidecar.write_bytes(b"bad existing sidecar: preserve")
    else:
        pd.DataFrame({"datetime": [pd.Timestamp("2026-04-13 14:01")], "funding_rate": [0.001]}).to_parquet(sidecar, index=False)
    original = sidecar.read_bytes()

    class FundingAdapter(OfflineAdapter):
        def fetch_extras(self, task, kinds):
            return {"funding": pd.DataFrame({"datetime": [pd.Timestamp("2026-04-13 14:01")],
                                             "funding_rate": [0.002]})}

    adapter = FundingAdapter(_bars([pd.Timestamp("2026-04-13 14:01")], [101.]))
    assert collect.fetch_one(adapter, task, daily_append=True, pause=0, extras_kinds=["funding"]) == "fail"
    assert sidecar.read_bytes() == original
    assert pd.read_parquet(task.out_path).close.tolist() == [100.]
    assert "OHLCV unchanged" in caplog.text


def test_sidecar_persistence_failure_prevents_main_cursor_advance(tmp_path, monkeypatch):
    task = _task(tmp_path)
    _bars([pd.Timestamp("2026-04-13 14:00")], [100.]).to_parquet(task.out_path, index=False)
    real_write = pd.DataFrame.to_parquet

    def sidecar_write_error(self, path, **kwargs):
        if "_extras" in Path(path).parts:
            raise OSError("sidecar storage unavailable")
        return real_write(self, path, **kwargs)

    class FundingAdapter(OfflineAdapter):
        def fetch_extras(self, task, kinds):
            return {"funding": pd.DataFrame({"datetime": [pd.Timestamp("2026-04-13 14:01")],
                                             "funding_rate": [0.002]})}

    monkeypatch.setattr(pd.DataFrame, "to_parquet", sidecar_write_error)
    adapter = FundingAdapter(_bars([pd.Timestamp("2026-04-13 14:01")], [101.]))
    assert collect.fetch_one(adapter, task, daily_append=True, pause=0, extras_kinds=["funding"]) == "fail"
    assert pd.read_parquet(task.out_path).close.tolist() == [100.]


@pytest.mark.parametrize("error,status", [(ValueError("invalid sidecar timestamps"), "fail"),
                                        (ConnectionError("optional endpoint unavailable"), "ok")])
def test_optional_fetch_failure_is_distinct_from_sidecar_validation_failure(tmp_path, error, status):
    task = _task(tmp_path)

    class UnavailableExtras(OfflineAdapter):
        def fetch_extras(self, task, kinds):
            raise error

    adapter = UnavailableExtras(_bars([pd.Timestamp("2026-04-13 14:01")], [101.]))
    assert collect.fetch_one(adapter, task, pause=0, extras_kinds=["funding"]) == status
    if status == "ok":
        assert pd.read_parquet(task.out_path).close.tolist() == [101.]
    else:
        assert not task.out_path.exists()


def test_sidecar_failure_keeps_main_cursor_so_identical_append_can_retry(tmp_path):
    from kairos.data import crypto_extras
    from kairos.data.markets.base import sanitize_symbol

    task = _task(tmp_path, end="2026-04-13T14:02:00Z")
    _bars([pd.Timestamp("2026-04-13 14:00")], [100.]).to_parquet(task.out_path, index=False)
    main_before = task.out_path.read_bytes()
    sidecar = crypto_extras.per_symbol_path(tmp_path, "funding", sanitize_symbol(task.symbol))
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"broken existing sidecar")

    class FundingAdapter(OfflineAdapter):
        def fetch_extras(self, task, kinds):
            self.extra_requests.append((task, kinds))
            return {"funding": pd.DataFrame({"datetime": [pd.Timestamp("2026-04-13 14:01")],
                                             "funding_rate": [0.002]})}

    adapter = FundingAdapter(_bars([pd.Timestamp("2026-04-13 14:01")], [101.]))
    assert collect.fetch_one(adapter, task, daily_append=True, pause=0, extras_kinds=["funding"]) == "fail"
    assert task.out_path.read_bytes() == main_before
    assert sidecar.read_bytes() == b"broken existing sidecar"
    pd.DataFrame({"datetime": [pd.Timestamp("2026-04-13 14:00")], "funding_rate": [0.001]}).to_parquet(sidecar, index=False)
    assert collect.fetch_one(adapter, task, daily_append=True, pause=0, extras_kinds=["funding"]) == "ok"
    assert len(adapter.requests) == 2
    assert adapter.requests[0].start == adapter.requests[1].start
    assert pd.read_parquet(task.out_path).close.tolist() == [100., 101.]
    assert pd.read_parquet(sidecar).funding_rate.tolist() == [0.001, 0.002]


@pytest.mark.parametrize("broken", [False, True])
def test_up_to_date_main_can_backfill_requested_sidecar_without_main_fetch(tmp_path, broken):
    from kairos.data import crypto_extras
    from kairos.data.markets.base import sanitize_symbol

    task = _task(tmp_path, start="2026-04-13T14:00:00Z", end="2026-04-13T14:02:00Z")
    _bars(pd.to_datetime(["2026-04-13 14:00", "2026-04-13 14:01"]), [100., 101.]).to_parquet(task.out_path, index=False)
    main_before = task.out_path.read_bytes()
    sidecar = crypto_extras.per_symbol_path(tmp_path, "funding", sanitize_symbol(task.symbol))
    if broken:
        sidecar.parent.mkdir(parents=True)
        sidecar.write_bytes(b"broken old sidecar")

    class FundingAdapter(OfflineAdapter):
        def fetch_extras(self, request, kinds):
            self.extra_requests.append((request, kinds))
            return {"funding": pd.DataFrame({"datetime": [pd.Timestamp("2026-04-13 14:01")],
                                             "funding_rate": [0.002]})}

    adapter = FundingAdapter(pd.DataFrame())
    status = collect.fetch_one(adapter, task, daily_append=True, pause=0, extras_kinds=["funding"])
    assert status == ("fail" if broken else "ok")
    assert adapter.requests == []
    assert adapter.extra_requests == [(task, ["funding"])]
    assert task.out_path.read_bytes() == main_before
    if broken:
        assert sidecar.read_bytes() == b"broken old sidecar"
        pd.DataFrame({"datetime": [pd.Timestamp("2026-04-13 14:00")], "funding_rate": [0.001]}).to_parquet(sidecar, index=False)
        assert collect.fetch_one(adapter, task, daily_append=True, pause=0, extras_kinds=["funding"]) == "ok"
        assert adapter.requests == []
        assert task.out_path.read_bytes() == main_before
    assert pd.read_parquet(sidecar).funding_rate.iloc[-1] == 0.002


def test_sidecar_hook_key_error_is_a_visible_schema_failure(tmp_path):
    task = _task(tmp_path)
    _bars([pd.Timestamp("2026-04-13 14:00")], [100.]).to_parquet(task.out_path, index=False)
    original = task.out_path.read_bytes()

    class BrokenSchema(OfflineAdapter):
        def fetch_extras(self, task, kinds):
            raise KeyError("funding_rate")

    adapter = BrokenSchema(_bars([pd.Timestamp("2026-04-13 14:01")], [101.]))
    assert collect.fetch_one(adapter, task, daily_append=True, pause=0, extras_kinds=["funding"]) == "fail"
    assert task.out_path.read_bytes() == original


class CLIAdapter(OfflineAdapter):
    supported_freqs = ("1min",)

    def list_symbols(self, universe):
        return universe.split(",")


def _configure_cli(tmp_path, monkeypatch, adapter):
    monkeypatch.setattr(collect, "get_adapter", lambda *args, **kwargs: adapter)
    monkeypatch.setattr("sys.argv", [
        "kairos-collect", "--universe", "BTC/USDT", "--freq", "1min",
        "--start", "2026-04-01", "--end", "2026-04-13", "--daily-append",
        "--out", str(tmp_path), "--workers", "1",
    ])


def test_cli_exits_nonzero_for_bad_history_without_network(tmp_path, monkeypatch):
    task = _task(tmp_path)
    original = b"broken parquet must not appear successful to shell automation"
    task.out_path.write_bytes(original)
    adapter = CLIAdapter(_bars([pd.Timestamp("2026-04-13 14:01")]))
    _configure_cli(tmp_path, monkeypatch, adapter)

    with pytest.raises(SystemExit) as error:
        collect.main()
    assert error.value.code == 1
    assert adapter.requests == []
    assert task.out_path.read_bytes() == original


def test_cli_successful_batch_returns_normally_for_exit_zero(tmp_path, monkeypatch):
    adapter = CLIAdapter(_bars([pd.Timestamp("2026-04-13 14:01")], [101.]))
    _configure_cli(tmp_path, monkeypatch, adapter)
    assert collect.main() is None
    assert pd.read_parquet(_task(tmp_path).out_path).close.tolist() == [101.]
