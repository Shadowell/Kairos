"""A 股 K 线采集脚本（基于 akshare，多源 + 断点续传 + 每日累积）。

用法示例
--------
# 全历史日线（前复权）
python data_pipeline/collect_ashare_kline.py \
    --universe csi300 --freq daily \
    --start 2015-01-01 --end 2026-04-17 \
    --adjust qfq --out ./raw/daily

# 5 分钟（能拉多长就拉多长，akshare 限制 3~6 月）
python data_pipeline/collect_ashare_kline.py \
    --universe csi300 --freq 5min --adjust qfq --out ./raw/5min

# 每日 cron 累积 1 分钟（15:30 以后跑）
python data_pipeline/collect_ashare_kline.py \
    --universe csi300 --freq 1min --daily-append --out ./raw/1min

依赖: akshare>=1.14, pandas, pyarrow, tqdm
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from tqdm import tqdm

try:
    import akshare as ak
except ImportError as e:
    raise SystemExit("请先 `pip install akshare pyarrow tqdm`") from e


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("collect")


# ---------------------------------------------------------------------------
# 股票池
# ---------------------------------------------------------------------------
UNIVERSE_FUNCS = {
    # name -> (fetcher, kwargs)
    "csi300":  ("index_stock_cons_csindex", {"symbol": "000300"}),
    "csi500":  ("index_stock_cons_csindex", {"symbol": "000905"}),
    "csi800":  ("index_stock_cons_csindex", {"symbol": "000906"}),
    "csi1000": ("index_stock_cons_csindex", {"symbol": "000852"}),
}


def get_universe(name: str) -> List[str]:
    """返回六位纯数字代码列表，如 ['600000', '000001', ...]。"""
    if name == "all_ashare":
        df = ak.stock_info_a_code_name()
        codes = df["code"].astype(str).str.zfill(6).tolist()
        return sorted(codes)

    if name in UNIVERSE_FUNCS:
        fn_name, kwargs = UNIVERSE_FUNCS[name]
        df = getattr(ak, fn_name)(**kwargs)
        # csindex 返回的列同时包含 '指数代码' 和 '成分券代码'，必须挑后者
        preferred = ["成分券代码", "成份券代码", "证券代码", "股票代码"]
        col = next((c for c in preferred if c in df.columns), None)
        if col is None:
            # 回退：挑带 '代码' 但不是 '指数代码' 的列
            col = next(
                (c for c in df.columns if "代码" in c and "指数" not in c),
                None,
            )
        if col is None:
            raise RuntimeError(f"无法识别成分股列名: {df.columns.tolist()}")
        codes = df[col].astype(str).str.zfill(6).tolist()
        # 去重，防止同一只股票因多交易所上市出现多次
        codes = sorted(set(codes))
        return codes

    # 支持直接传入逗号分隔的股票列表: "600977,000001,300750"
    if "," in name or name.isdigit():
        return [c.strip().zfill(6) for c in name.split(",") if c.strip()]

    raise ValueError(f"未知 universe: {name}")


# ---------------------------------------------------------------------------
# 字段标准化
# ---------------------------------------------------------------------------
STD_COLS = [
    "datetime", "open", "high", "low", "close",
    "volume", "amount", "turnover", "pct_chg", "vwap",
]

DAILY_RENAME = {
    "日期": "datetime", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume",
    "成交额": "amount", "换手率": "turnover", "涨跌幅": "pct_chg",
}
MIN_RENAME = {
    "时间": "datetime", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume",
    "成交额": "amount",
}


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    """字段统一 + 补全 + 类型转换。"""
    rename = {}
    for src, dst in {**DAILY_RENAME, **MIN_RENAME}.items():
        if src in df.columns:
            rename[src] = dst
    df = df.rename(columns=rename).copy()

    if "datetime" not in df.columns:
        raise ValueError("原始数据缺少时间列")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime")

    # VWAP = amount / volume (防 0 除)
    if "amount" in df.columns and "volume" in df.columns:
        vol = df["volume"].replace(0, pd.NA)
        df["vwap"] = (df["amount"] / vol).astype(float)
    else:
        df["vwap"] = df.get("close", pd.NA)

    # 补齐缺失字段
    for col in STD_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    # 类型
    numeric = [c for c in STD_COLS if c != "datetime"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df[STD_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 单只股票抓取
# ---------------------------------------------------------------------------
@dataclass
class FetchTask:
    symbol: str
    freq: str         # daily / 1min / 5min / 15min / 30min / 60min
    start: str        # YYYY-MM-DD
    end: str          # YYYY-MM-DD
    adjust: str       # qfq / hfq / ""
    out_dir: Path

    @property
    def out_path(self) -> Path:
        return self.out_dir / f"{self.symbol}.parquet"


# 进程级状态：东财连续失败阈值后自动禁用本次运行的 eastmoney 源
_EASTMONEY_DISABLED = False
_EASTMONEY_CONSEC_FAILS = 0
_EASTMONEY_MAX_FAILS = 3


def _sina_symbol(code: str) -> str:
    """六位代码 -> 新浪前缀代码 (sh600000 / sz000001 / bj831010)。"""
    if code.startswith(("60", "68", "90", "11", "13")):
        return "sh" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sz" + code


def _fetch_daily(t: FetchTask) -> pd.DataFrame:
    """日线抓取：优先东财，失败降级到腾讯 / 新浪。"""
    start_compact = t.start.replace("-", "")
    end_compact = t.end.replace("-", "")
    sina_sym = _sina_symbol(t.symbol)

    def from_eastmoney() -> pd.DataFrame:
        return ak.stock_zh_a_hist(
            symbol=t.symbol, period="daily",
            start_date=start_compact, end_date=end_compact, adjust=t.adjust,
        )

    def from_tencent() -> pd.DataFrame:
        df = ak.stock_zh_a_hist_tx(
            symbol=sina_sym,
            start_date=start_compact, end_date=end_compact, adjust=t.adjust,
        )
        if df is None or df.empty:
            return df
        # 腾讯返回: date/open/close/high/low/amount (这里 amount 实为成交量/手)
        return df.rename(columns={
            "date": "日期", "open": "开盘", "close": "收盘",
            "high": "最高", "low": "最低", "amount": "成交量",
        })

    def from_sina() -> pd.DataFrame:
        df = ak.stock_zh_a_daily(
            symbol=sina_sym, adjust=t.adjust or "",
            start_date=t.start, end_date=t.end,
        )
        if df is None or df.empty:
            return df
        # 新浪返回: date/open/high/low/close/volume/amount/outstanding_share/turnover
        out = df.rename(columns={
            "date": "日期", "open": "开盘", "high": "最高", "low": "最低",
            "close": "收盘", "volume": "成交量", "amount": "成交额",
        })
        if "turnover" in out.columns:
            out["换手率"] = out["turnover"] * 100.0
        return out

    global _EASTMONEY_DISABLED, _EASTMONEY_CONSEC_FAILS
    sources: List[tuple] = []
    if not _EASTMONEY_DISABLED:
        sources.append(("eastmoney", from_eastmoney))
    sources += [("tencent", from_tencent), ("sina", from_sina)]

    last_err: Optional[Exception] = None
    for name, fn in sources:
        try:
            df = fn()
            if df is None or df.empty:
                continue
            if name == "eastmoney":
                _EASTMONEY_CONSEC_FAILS = 0
            return _standardize(df)
        except Exception as e:
            last_err = e
            log.debug(f"[{t.symbol}] {name} 源失败: {e}")
            if name == "eastmoney":
                _EASTMONEY_CONSEC_FAILS += 1
                if _EASTMONEY_CONSEC_FAILS >= _EASTMONEY_MAX_FAILS:
                    _EASTMONEY_DISABLED = True
                    log.warning(
                        f"东财连续失败 {_EASTMONEY_CONSEC_FAILS} 次，本次运行禁用该源，降级到腾讯/新浪"
                    )
            continue

    if last_err is not None:
        raise last_err
    return pd.DataFrame(columns=STD_COLS)


def _fetch_min(t: FetchTask) -> pd.DataFrame:
    """5/15/30/60 min 用东方财富源（stock_zh_a_hist_min_em）。"""
    period_map = {"1min": "1", "5min": "5", "15min": "15",
                  "30min": "30", "60min": "60"}
    period = period_map[t.freq]

    start = f"{t.start} 09:30:00"
    end = f"{t.end} 15:00:00"
    try:
        df = ak.stock_zh_a_hist_min_em(
            symbol=t.symbol, period=period,
            start_date=start, end_date=end, adjust=t.adjust,
        )
    except Exception as e:
        log.warning(f"[{t.symbol}] 东财分钟源失败: {e}，尝试新浪源")
        sina_symbol = ("sh" if t.symbol.startswith(("6", "9")) else "sz") + t.symbol
        df = ak.stock_zh_a_minute(symbol=sina_symbol, period=period, adjust=t.adjust)

    if df is None or df.empty:
        return pd.DataFrame(columns=STD_COLS)
    return _standardize(df)


FETCHERS = {
    "daily": _fetch_daily,
    "1min": _fetch_min,
    "5min": _fetch_min,
    "15min": _fetch_min,
    "30min": _fetch_min,
    "60min": _fetch_min,
}


def _load_existing(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def fetch_one(t: FetchTask, daily_append: bool = False,
              retries: int = 3, pause: float = 0.5) -> str:
    """返回状态字符串用于统计。"""
    existing = _load_existing(t.out_path) if daily_append else None

    # 断点续传：如果已有数据，只抓新增部分
    if existing is not None and not existing.empty:
        last_dt = pd.to_datetime(existing["datetime"].max())
        t = FetchTask(**{**t.__dict__, "start": (last_dt + timedelta(days=1)).strftime("%Y-%m-%d")})
        if t.start > t.end:
            return "skip_up_to_date"

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            df = FETCHERS[t.freq](t)
            break
        except Exception as e:
            last_err = e
            time.sleep(pause * (2 ** attempt))
    else:
        log.error(f"[{t.symbol}] 放弃: {last_err}")
        return "fail"

    if df.empty:
        return "empty"

    if existing is not None and not existing.empty:
        df = pd.concat([existing, df], ignore_index=True)
        df = df.sort_values("datetime").drop_duplicates("datetime")

    t.out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(t.out_path, index=False)
    return "ok"


# ---------------------------------------------------------------------------
# 批量并行
# ---------------------------------------------------------------------------
def run_batch(symbols: Iterable[str], freq: str, start: str, end: str,
              adjust: str, out_dir: Path, workers: int = 4,
              daily_append: bool = False) -> None:
    symbols = list(symbols)
    log.info(f"开始采集 {len(symbols)} 只股票 | freq={freq} | "
             f"{start} → {end} | adjust={adjust or 'none'} | out={out_dir}")

    tasks = [
        FetchTask(sym, freq, start, end, adjust, out_dir)
        for sym in symbols
    ]

    counter = {"ok": 0, "fail": 0, "empty": 0, "skip_up_to_date": 0}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_one, t, daily_append): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), ncols=100):
            status = fut.result()
            counter[status] = counter.get(status, 0) + 1

    log.info(f"完成: {counter}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--universe", default="csi300",
                   help="csi300|csi500|csi800|csi1000|all_ashare|'600977,000001,...'")
    p.add_argument("--freq", default="daily",
                   choices=["daily", "1min", "5min", "15min", "30min", "60min"])
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    p.add_argument("--adjust", default="qfq", choices=["qfq", "hfq", ""])
    p.add_argument("--out", default="./raw/daily")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--daily-append", action="store_true",
                   help="断点续传 / 每日累积模式")
    p.add_argument("--limit", type=int, default=0,
                   help=">0 时只抓前 N 只，用于 smoke test")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    symbols = get_universe(args.universe)
    if args.limit > 0:
        symbols = symbols[: args.limit]

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_batch(
        symbols=symbols,
        freq=args.freq,
        start=args.start,
        end=args.end,
        adjust=args.adjust,
        out_dir=out_dir,
        workers=args.workers,
        daily_append=args.daily_append,
    )


if __name__ == "__main__":
    main()
