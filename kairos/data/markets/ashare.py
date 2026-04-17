"""A-share market adapter.

Wraps the ``akshare`` API with a multi-source fallback (EastMoney → Tencent →
Sina) that has been battle-tested against flaky upstream data providers. The
concrete fetching / standardisation logic is ported as-is from the original
``kairos.data.collect`` module; only the glue that exposes it through the
:class:`MarketAdapter` interface is new.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

from .base import STD_COLS, FetchTask, MarketAdapter, register_adapter


log = logging.getLogger("kairos.market.ashare")


try:
    import akshare as ak
except ImportError:  # pragma: no cover - optional dep, surfaced on first use
    ak = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Universes
# ---------------------------------------------------------------------------
UNIVERSE_FUNCS = {
    "csi300": ("index_stock_cons_csindex", {"symbol": "000300"}),
    "csi500": ("index_stock_cons_csindex", {"symbol": "000905"}),
    "csi800": ("index_stock_cons_csindex", {"symbol": "000906"}),
    "csi1000": ("index_stock_cons_csindex", {"symbol": "000852"}),
}


# ---------------------------------------------------------------------------
# Column renaming tables (shared between daily and min-bar rails)
# ---------------------------------------------------------------------------
DAILY_RENAME = {
    "日期": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "涨跌幅": "pct_chg",
}
MIN_RENAME = {
    "时间": "datetime",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
}


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename → sort → dedupe → compute vwap → enforce schema."""
    rename = {}
    for src, dst in {**DAILY_RENAME, **MIN_RENAME}.items():
        if src in df.columns:
            rename[src] = dst
    df = df.rename(columns=rename).copy()

    if "datetime" not in df.columns:
        raise ValueError("原始数据缺少时间列")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime")

    if "amount" in df.columns and "volume" in df.columns:
        vol = df["volume"].replace(0, pd.NA)
        df["vwap"] = (df["amount"] / vol).astype(float)
    else:
        df["vwap"] = df.get("close", pd.NA)

    for col in STD_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    numeric = [c for c in STD_COLS if c != "datetime"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df[STD_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# EastMoney circuit breaker — shared across all fetches in a single process.
# If the upstream fails too many times in a row we stop trying so we don't
# waste minutes waiting on timeouts for every remaining symbol.
# ---------------------------------------------------------------------------
@dataclass
class _EmBreaker:
    disabled: bool = False
    fails: int = 0
    max_fails: int = 3


_EM = _EmBreaker()


def _sina_symbol(code: str) -> str:
    """6-digit code → Sina-prefixed symbol (sh600000 / sz000001 / bj831010)."""
    if code.startswith(("60", "68", "90", "11", "13")):
        return "sh" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sz" + code


def _fetch_daily(task: FetchTask) -> pd.DataFrame:
    """Daily bars with EastMoney → Tencent → Sina fallback."""
    if ak is None:
        raise RuntimeError("akshare 未安装；`pip install akshare` 后重试")

    start_compact = task.start.replace("-", "")
    end_compact = task.end.replace("-", "")
    sina_sym = _sina_symbol(task.symbol)
    adjust = task.adjust or ""

    def from_eastmoney() -> pd.DataFrame:
        return ak.stock_zh_a_hist(
            symbol=task.symbol,
            period="daily",
            start_date=start_compact,
            end_date=end_compact,
            adjust=adjust,
        )

    def from_tencent() -> pd.DataFrame:
        df = ak.stock_zh_a_hist_tx(
            symbol=sina_sym,
            start_date=start_compact,
            end_date=end_compact,
            adjust=adjust,
        )
        if df is None or df.empty:
            return df
        return df.rename(
            columns={
                "date": "日期",
                "open": "开盘",
                "close": "收盘",
                "high": "最高",
                "low": "最低",
                "amount": "成交量",
            }
        )

    def from_sina() -> pd.DataFrame:
        df = ak.stock_zh_a_daily(
            symbol=sina_sym,
            adjust=adjust,
            start_date=task.start,
            end_date=task.end,
        )
        if df is None or df.empty:
            return df
        out = df.rename(
            columns={
                "date": "日期",
                "open": "开盘",
                "high": "最高",
                "low": "最低",
                "close": "收盘",
                "volume": "成交量",
                "amount": "成交额",
            }
        )
        if "turnover" in out.columns:
            out["换手率"] = out["turnover"] * 100.0
        return out

    sources: list[tuple] = []
    if not _EM.disabled:
        sources.append(("eastmoney", from_eastmoney))
    sources += [("tencent", from_tencent), ("sina", from_sina)]

    last_err: Optional[Exception] = None
    for name, fn in sources:
        try:
            df = fn()
            if df is None or df.empty:
                continue
            if name == "eastmoney":
                _EM.fails = 0
            return _standardize(df)
        except Exception as e:
            last_err = e
            log.debug(f"[{task.symbol}] {name} 源失败: {e}")
            if name == "eastmoney":
                _EM.fails += 1
                if _EM.fails >= _EM.max_fails:
                    _EM.disabled = True
                    log.warning(
                        f"东财连续失败 {_EM.fails} 次，本次运行禁用该源，降级到腾讯/新浪"
                    )
            continue

    if last_err is not None:
        raise last_err
    return pd.DataFrame(columns=STD_COLS)


def _fetch_min(task: FetchTask) -> pd.DataFrame:
    """Intraday bars (1/5/15/30/60 min) via EastMoney, Sina as fallback."""
    if ak is None:
        raise RuntimeError("akshare 未安装；`pip install akshare` 后重试")

    period_map = {
        "1min": "1",
        "5min": "5",
        "15min": "15",
        "30min": "30",
        "60min": "60",
    }
    period = period_map[task.freq]

    start = f"{task.start} 09:30:00"
    end = f"{task.end} 15:00:00"
    try:
        df = ak.stock_zh_a_hist_min_em(
            symbol=task.symbol,
            period=period,
            start_date=start,
            end_date=end,
            adjust=task.adjust or "",
        )
    except Exception as e:
        log.warning(f"[{task.symbol}] 东财分钟源失败: {e}，尝试新浪源")
        sina_symbol = (
            "sh" if task.symbol.startswith(("6", "9")) else "sz"
        ) + task.symbol
        df = ak.stock_zh_a_minute(
            symbol=sina_symbol,
            period=period,
            adjust=task.adjust or "",
        )

    if df is None or df.empty:
        return pd.DataFrame(columns=STD_COLS)
    return _standardize(df)


_FETCHERS = {
    "daily": _fetch_daily,
    "1min": _fetch_min,
    "5min": _fetch_min,
    "15min": _fetch_min,
    "30min": _fetch_min,
    "60min": _fetch_min,
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class AshareAdapter(MarketAdapter):
    """A-share adapter built on top of ``akshare``."""

    name = "ashare"
    supported_freqs = tuple(_FETCHERS.keys())

    def list_symbols(self, universe: str) -> List[str]:
        if ak is None:
            raise RuntimeError("akshare 未安装；`pip install akshare` 后重试")

        if universe == "all_ashare":
            df = ak.stock_info_a_code_name()
            return sorted(df["code"].astype(str).str.zfill(6).tolist())

        if universe in UNIVERSE_FUNCS:
            fn_name, kwargs = UNIVERSE_FUNCS[universe]
            df = getattr(ak, fn_name)(**kwargs)
            preferred = ["成分券代码", "成份券代码", "证券代码", "股票代码"]
            col = next((c for c in preferred if c in df.columns), None)
            if col is None:
                col = next(
                    (c for c in df.columns if "代码" in c and "指数" not in c),
                    None,
                )
            if col is None:
                raise RuntimeError(
                    f"无法识别成分股列名: {df.columns.tolist()}"
                )
            codes = df[col].astype(str).str.zfill(6).tolist()
            return sorted(set(codes))

        # Ad-hoc list: "600977,000001,300750"
        if "," in universe or universe.isdigit():
            return [c.strip().zfill(6) for c in universe.split(",") if c.strip()]

        raise ValueError(f"未知 universe: {universe}")

    def fetch_ohlcv(self, task: FetchTask) -> pd.DataFrame:
        if task.freq not in _FETCHERS:
            raise ValueError(
                f"freq {task.freq!r} 不支持，可用: {list(_FETCHERS)}"
            )
        return _FETCHERS[task.freq](task)

    def trading_calendar(
        self, start: datetime, end: datetime, freq: str
    ) -> pd.DatetimeIndex:
        """Best-effort A-share trading calendar.

        We avoid introducing a hard dependency on pandas_market_calendars here
        and fall back to business days if akshare's calendar endpoint is
        unreachable. Downstream code still validates against actual fetched
        bars, so this is only used for smoke checks.
        """
        if ak is not None:
            try:
                df = ak.tool_trade_date_hist_sina()
                dates = pd.to_datetime(df["trade_date"])
                mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
                return pd.DatetimeIndex(dates[mask].sort_values().values)
            except Exception as e:  # pragma: no cover
                log.debug(f"akshare 交易日历获取失败，降级为 B 日: {e}")
        return pd.bdate_range(start, end)


register_adapter("ashare", AshareAdapter, overwrite=True)


__all__ = ["AshareAdapter"]
