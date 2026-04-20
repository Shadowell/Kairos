"""Unified K-line collection entrypoint.

Previously this module was hard-wired to akshare / A-share. It is now a thin
dispatcher that delegates the market-specific work to a
:class:`~kairos.data.markets.base.MarketAdapter` chosen via ``--market``.

Backwards compatibility
-----------------------
Calls that omit ``--market`` continue to behave exactly like before (A-shares
via akshare), so existing docs and scripts keep working. The default universe,
frequency, adjust option and output path are unchanged.

Examples
--------
::

    # A-share daily (default, identical to previous behaviour)
    kairos-collect --universe csi300 --freq daily \\
        --start 2018-01-01 --end 2026-04-17 --out ./raw/daily

    # A-share 1min with explicit market flag
    kairos-collect --market ashare --universe csi300 --freq 1min \\
        --daily-append --out ./raw/1min

    # Crypto (once the crypto adapter is installed)
    kairos-collect --market crypto --universe top10 --freq 1min \\
        --start 2023-01-01 --out ./raw/crypto/1min
"""

from __future__ import annotations

import argparse
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

from .markets import (
    FetchTask,
    MarketAdapter,
    available_adapters,
    get_adapter,
)


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
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


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
    Failures in the extras path never break the main OHLCV save — they
    are logged and the task still returns ``"ok"`` if the primary fetch
    succeeded.
    """

    existing = _load_existing(task.out_path) if daily_append else None

    if existing is not None and not existing.empty:
        last_dt = pd.to_datetime(existing["datetime"].max())
        new_start = (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        if new_start > task.end:
            return "skip_up_to_date"
        task = FetchTask(**{**task.__dict__, "start": new_start})

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
        log.error(f"[{task.symbol}] 放弃: {last_err}")
        return "fail"

    if df is None or df.empty:
        return "empty"

    if existing is not None and not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)
        df = df.sort_values("datetime").drop_duplicates("datetime")

    task.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(task.out_path, index=False)

    if extras_kinds:
        _fetch_and_save_extras(adapter, task, extras_kinds)

    return "ok"


def _fetch_and_save_extras(
    adapter: MarketAdapter,
    task: FetchTask,
    kinds: List[str],
) -> None:
    """Best-effort auxiliary-channel fetch for crypto.

    Exceptions are logged and swallowed: the main OHLCV save is already
    on disk, and a missing extras parquet degrades gracefully to the
    adapter's NaN → 0 fallback at training time.
    """

    hook = getattr(adapter, "fetch_extras", None)
    if hook is None:
        log.debug(
            f"adapter {adapter.name!r} has no fetch_extras; "
            f"ignoring --crypto-extras={kinds}"
        )
        return

    try:
        extras = hook(task, kinds=kinds)
    except Exception as e:  # noqa: BLE001
        log.warning(f"[{task.symbol}] extras fetch failed: {e}")
        return

    if not extras:
        return

    from .markets.base import sanitize_symbol
    from . import crypto_extras as _ce

    stem = sanitize_symbol(task.symbol)
    for kind, df in extras.items():
        try:
            if kind in _ce._PER_SYMBOL_KINDS:  # type: ignore[attr-defined]
                _ce.save_per_symbol(task.out_dir, stem, kind, df)
            elif kind in _ce._MARKET_WIDE_KINDS:  # type: ignore[attr-defined]
                _ce.save_market_wide(task.out_dir, kind, df)
        except Exception as e:  # noqa: BLE001
            log.warning(f"[{task.symbol}] save {kind} extras failed: {e}")


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
) -> None:
    symbols = list(symbols)
    extras_note = f" extras={extras_kinds}" if extras_kinds else ""
    log.info(
        f"开始采集 {len(symbols)} 个标的 | market={adapter.name} | "
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

    log.info(f"完成: {counter}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="采集各市场 K 线数据（A 股 / 加密货币 / ...）"
    )
    p.add_argument(
        "--market",
        default="ashare",
        help=f"市场 adapter，默认 ashare；可用: {available_adapters() or '<none>'}",
    )
    p.add_argument(
        "--universe",
        default="csi300",
        help="标的池名或逗号分隔列表；具体取值由所选 market 的 adapter 决定",
    )
    p.add_argument("--freq", default="daily")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument(
        "--adjust",
        default="qfq",
        help="复权方式（A 股专用），非 A 股市场可忽略",
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
    p.add_argument("--out", default="./raw/daily")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--daily-append",
        action="store_true",
        help="断点续传 / 每日累积模式",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help=">0 时只抓前 N 个，用于 smoke test",
    )
    p.add_argument(
        "--crypto-extras",
        default="",
        help="crypto-only; comma-separated subset of "
        "{funding,open_interest,spot,btc_dominance,all} to also fetch "
        "alongside the main perp OHLCV. Sidecar parquet land under "
        "<out>/_extras/<kind>/. Empty (default) keeps the previous "
        "OHLCV-only behaviour.",
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
    adapter = get_adapter(args.market, **adapter_kwargs)

    if args.freq not in adapter.supported_freqs:
        raise SystemExit(
            f"market={adapter.name} 不支持 freq={args.freq}；"
            f"可用: {list(adapter.supported_freqs)}"
        )

    symbols = adapter.list_symbols(args.universe)
    if args.limit > 0:
        symbols = symbols[: args.limit]

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    extras_kinds = _resolve_extras_kinds(args.crypto_extras, args.market)

    run_batch(
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


# ---------------------------------------------------------------------------
# Backwards-compatible helpers (keep old imports working).
# ---------------------------------------------------------------------------
def get_universe(name: str) -> List[str]:
    """Deprecated: use ``get_adapter("ashare").list_symbols(name)`` instead."""
    return get_adapter("ashare").list_symbols(name)


if __name__ == "__main__":
    main()
