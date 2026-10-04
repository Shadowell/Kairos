"""Crypto K-line collection entrypoint.

This module delegates venue-specific work to the crypto
:class:`~kairos.data.markets.base.MarketAdapter`. It supports spot and
USDT-margined perpetual swap collection through ``--market-type``.

Examples
--------
::

    # OKX spot
    kairos-collect --market-type spot --universe "BTC/USDT,ETH/USDT" \\
        --freq 1min --start 2026-04-01 --out ./raw/crypto/spot_1min

    # OKX USDT perpetual swaps with representative sidecars
    kairos-collect --market-type swap \\
        --universe "BTC/USDT:USDT,ETH/USDT:USDT" --freq 1min \\
        --start 2026-04-01 --out ./raw/crypto/swap_1min \\
        --crypto-extras funding,open_interest,spot,reference
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from .contracts import bar_delta
from .markets import (
    FetchTask,
    MarketAdapter,
    available_adapters,
    get_adapter,
)
from .markets.crypto import _to_unix_ms


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect")


# ---------------------------------------------------------------------------
# Per-task fetch + write
# ---------------------------------------------------------------------------
def _load_existing(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    # An unreadable history is not an absent history: callers must not overwrite it.
    return _normalize_datetimes(pd.read_parquet(path))


def _normalize_datetimes(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical UTC-naive timestamps without changing caller-owned values."""
    if not isinstance(frame, pd.DataFrame) or list(frame.columns).count("datetime") != 1:
        raise ValueError("collected bars must have exactly one datetime column")
    values = frame["datetime"]
    if pd.api.types.is_numeric_dtype(values.dtype) or (
        values.dtype == object and values.map(lambda value: isinstance(value, Number)).any()
    ):
        raise ValueError("datetime must contain timestamps, not numeric epoch values")
    dates = pd.to_datetime(values, utc=True, format="mixed")
    if dates.isna().any():
        raise ValueError("datetime must not contain NaT or missing values")
    result = frame.copy()
    result["datetime"] = dates.dt.tz_localize(None)
    return result


def _save_main_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Publish a complete parquet atomically; failed writes leave history intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".parquet.tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_one(
    adapter: MarketAdapter,
    task: FetchTask,
    daily_append: bool = False,
    retries: int = 3,
    pause: float = 0.5,
    extras_kinds: Optional[List[str]] = None,
) -> str:
    """Fetch one symbol, merge with any existing file, return a status tag.

    When ``extras_kinds`` is non-empty and the adapter exposes
    ``fetch_extras`` (currently only the crypto adapter does), we also
    fetch auxiliary channels (funding / OI / spot basis) and drop them
    into the ``_extras/`` sidecar directory next to the main parquet.
    Optional endpoint unavailability remains best effort. Requested sidecars
    are validated and saved before publishing new OHLCV so a strict failure
    does not advance the main cursor. Already-current OHLCV can backfill
    requested sidecars over the original request without rewriting history.
    """

    try:
        existing = _load_existing(task.out_path) if daily_append else None
    except Exception as exc:  # noqa: BLE001
        log.error(f"[{task.symbol}] invalid existing history; refusing overwrite: {exc}")
        return "fail"

    original_task = task
    try:
        new_start = pd.Timestamp(_to_unix_ms(task.start), unit="ms", tz="UTC")
        end_exclusive = pd.Timestamp(_to_unix_ms(task.end, end_of_day=True), unit="ms", tz="UTC")
        if existing is not None and not existing.empty:
            next_bar = existing["datetime"].max().tz_localize("UTC") + bar_delta(task.freq)
            new_start = max(new_start, next_bar)
            task = replace(task, start=new_start.isoformat())
        if new_start >= end_exclusive:
            if extras_kinds and existing is not None and not existing.empty:
                if not _fetch_and_save_extras(adapter, original_task, extras_kinds):
                    log.error(f"[{task.symbol}] OHLCV unchanged; requested sidecar repair failed")
                    return "fail"
                return "ok"
            return "skip_up_to_date"
    except (TypeError, ValueError, OverflowError) as exc:
        log.error(f"[{task.symbol}] invalid collection window: {exc}")
        return "fail"

    last_err: Optional[Exception] = None
    df: Optional[pd.DataFrame] = None
    for attempt in range(retries):
        try:
            df = adapter.fetch_ohlcv(task)
            break
        except Exception as e:
            last_err = e
            time.sleep(pause * (2**attempt))
    else:
        log.error(f"[{task.symbol}] give up: {last_err}")
        return "fail"

    if df is None or df.empty:
        return "empty"

    try:
        df = _normalize_datetimes(df)
        # Filter only newly fetched rows; a narrower request must not trim history.
        dates = df["datetime"]
        df = df.loc[(dates >= new_start.tz_localize(None)) & (dates < end_exclusive.tz_localize(None))]
        if df.empty:
            return "empty"
        if existing is not None and not existing.empty:
            df = pd.concat([existing, df], ignore_index=True)
        # Stable ordering makes an existing row win over any repeated response row.
        df = df.sort_values("datetime", kind="stable").drop_duplicates("datetime", keep="first")
    except (TypeError, ValueError, OverflowError) as exc:
        log.error(f"[{task.symbol}] invalid fetched bars; preserving history: {exc}")
        return "fail"

    if extras_kinds and not _fetch_and_save_extras(adapter, task, extras_kinds):
        log.error(f"[{task.symbol}] OHLCV unchanged; requested sidecar validation or persistence failed")
        return "fail"

    try:
        _save_main_parquet(df, task.out_path)
    except Exception as exc:  # noqa: BLE001
        log.error(f"[{task.symbol}] OHLCV save failed; existing history preserved: {exc}")
        return "fail"

    return "ok"


def _fetch_and_save_extras(
    adapter: MarketAdapter,
    task: FetchTask,
    kinds: List[str],
) -> bool:
    """Fetch optional channels; report validation and persistence failures."""

    hook = getattr(adapter, "fetch_extras", None)
    if hook is None:
        log.debug(
            f"adapter {adapter.name!r} has no fetch_extras; "
            f"ignoring --crypto-extras={kinds}"
        )
        return True

    try:
        extras = hook(task, kinds=kinds)
    except (TypeError, ValueError, KeyError) as exc:
        log.error(f"[{task.symbol}] extras validation failed: {exc}")
        return False
    except Exception as e:  # noqa: BLE001
        log.warning(f"[{task.symbol}] extras fetch failed: {e}")
        return True

    if not extras:
        return True

    from .markets.base import sanitize_symbol
    from . import crypto_extras as _ce

    stem = sanitize_symbol(task.symbol)
    succeeded = True
    for kind, df in extras.items():
        try:
            if kind in _ce._PER_SYMBOL_KINDS:  # type: ignore[attr-defined]
                _ce.save_per_symbol(task.out_dir, stem, kind, df)
            elif kind in _ce._MARKET_WIDE_KINDS:  # type: ignore[attr-defined]
                _ce.save_market_wide(task.out_dir, kind, df)
        except Exception as e:  # noqa: BLE001
            log.error(f"[{task.symbol}] save {kind} extras failed: {e}")
            succeeded = False
    return succeeded


# ---------------------------------------------------------------------------
# Batch driver
# ---------------------------------------------------------------------------
def run_batch(
    adapter: MarketAdapter,
    symbols: Iterable[str],
    freq: str,
    start: str,
    end: str,
    adjust: str,
    out_dir: Path,
    workers: int = 4,
    daily_append: bool = False,
    extras_kinds: Optional[List[str]] = None,
) -> dict[str, int]:
    symbols = list(symbols)
    extras_note = f" extras={extras_kinds}" if extras_kinds else ""
    log.info(
        f"Collecting {len(symbols)} symbols | market={adapter.name} | "
        f"freq={freq} | {start} → {end} | adjust={adjust or 'none'} | "
        f"out={out_dir}{extras_note}"
    )

    tasks = [
        FetchTask(
            symbol=sym,
            freq=freq,
            start=start,
            end=end,
            adjust=adjust,
            out_dir=out_dir,
        )
        for sym in symbols
    ]

    counter = {"ok": 0, "fail": 0, "empty": 0, "skip_up_to_date": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                fetch_one, adapter, t, daily_append, extras_kinds=extras_kinds
            ): t
            for t in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), ncols=100):
            status = fut.result()
            counter[status] = counter.get(status, 0) + 1

    log.info(f"Done: {counter}")
    return counter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Collect crypto OHLCV data from OKX-compatible adapters"
    )
    p.add_argument(
        "--market",
        default="crypto",
        help=f"Market adapter, default crypto; available: {available_adapters() or '<none>'}",
    )
    p.add_argument(
        "--universe",
        default="BTC/USDT:USDT,ETH/USDT:USDT",
        help="Universe name such as top10 or a comma-separated symbol list",
    )
    p.add_argument("--freq", default="1min")
    p.add_argument("--start", default="2026-04-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument(
        "--adjust",
        default="",
        help="Reserved compatibility option; ignored for crypto",
    )
    p.add_argument(
        "--proxy",
        default=None,
        help="HTTP(S) proxy URL, e.g. http://127.0.0.1:7890 (crypto only; "
        "falls back to HTTPS_PROXY/HTTP_PROXY env vars)",
    )
    p.add_argument(
        "--exchange",
        default=None,
        help="crypto venue override, e.g. okx (default) / binance (when added)",
    )
    p.add_argument(
        "--market-type",
        choices=["spot", "swap"],
        default="swap",
        help="Crypto instrument type: spot or USDT-margined perpetual swap",
    )
    p.add_argument("--out", default="./raw/crypto/okx_swap_1min")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--daily-append",
        action="store_true",
        help="Resume/append mode",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only fetch the first N symbols when >0, for smoke tests",
    )
    p.add_argument(
        "--crypto-extras",
        default="",
        help="Comma-separated subset of {funding,open_interest,spot,reference,all} "
        "to fetch alongside OHLCV. funding/open_interest/spot are swap-only; "
        "reference is a market-wide BTC/USDT close sidecar under <out>/_extras/.",
    )
    return p.parse_args()


def _resolve_extras_kinds(flag: str, market: str) -> List[str]:
    """Parse ``--crypto-extras`` into a list of canonical kind names.

    Accepts comma-separated tokens including the special ``all`` that
    expands to every known per-symbol channel. Unknown tokens raise
    ``SystemExit`` so typos surface immediately. Returns an empty list
    when the flag is blank or the market isn't crypto.
    """

    flag = (flag or "").strip()
    if not flag:
        return []
    if market != "crypto":
        log.warning(
            f"--crypto-extras is crypto-only; ignoring for market={market!r}"
        )
        return []

    from . import crypto_extras as _ce

    tokens = [t.strip() for t in flag.split(",") if t.strip()]
    resolved: List[str] = []
    for t in tokens:
        if t == "all":
            resolved.extend(_ce.ALL_KINDS)
            continue
        if t not in _ce.ALL_KINDS:
            raise SystemExit(
                f"unknown --crypto-extras value {t!r}; "
                f"allowed: {list(_ce.ALL_KINDS) + ['all']}"
            )
        resolved.append(t)
    # de-dup, keep order of first appearance
    seen: set[str] = set()
    unique = []
    for t in resolved:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def main() -> None:
    args = parse_args()

    adapter_kwargs = {}
    if args.proxy:
        adapter_kwargs["proxy"] = args.proxy
    if args.exchange:
        adapter_kwargs["exchange"] = args.exchange
    if args.market == "crypto":
        adapter_kwargs["market_type"] = args.market_type
    adapter = get_adapter(args.market, **adapter_kwargs)

    if args.freq not in adapter.supported_freqs:
        raise SystemExit(
            f"market={adapter.name} does not support freq={args.freq}; "
            f"available: {list(adapter.supported_freqs)}"
        )

    symbols = adapter.list_symbols(args.universe)
    if args.limit > 0:
        symbols = symbols[: args.limit]

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extras_kinds = _resolve_extras_kinds(args.crypto_extras, args.market)

    counter = run_batch(
        adapter=adapter,
        symbols=symbols,
        freq=args.freq,
        start=args.start,
        end=args.end,
        adjust=args.adjust,
        out_dir=out_dir,
        workers=args.workers,
        daily_append=args.daily_append,
        extras_kinds=extras_kinds,
    )
    if counter["fail"] > 0:
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Convenience helper for programmatic universe resolution.
# ---------------------------------------------------------------------------
def get_universe(name: str) -> List[str]:
    """Resolve a crypto universe using the default adapter."""
    return get_adapter("crypto").list_symbols(name)


if __name__ == "__main__":
    main()
