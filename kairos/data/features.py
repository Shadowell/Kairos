"""特征/因子工程模块。

设计原则
--------
1. **无未来信息泄漏**：所有指标只使用 t 时刻及之前的数据。
2. **与 Kronos 原生特征分离**：Kronos Tokenizer 只吃
   ``[open, high, low, close, volume, amount]``；其余作为 "外生变量 exog"
   单独返回，供 ``KronosWithExogenous`` 的旁路通道使用。
3. **归一化**：外生变量做 **滚动 z-score**（窗口 60），避免不同量纲。

用法
----
    from data_pipeline.build_features import build_features, EXOG_COLS
    df_feat = build_features(df, index_df=csi300_df)
    kline = df_feat[["open","high","low","close","volume","amount"]]
    exog  = df_feat[EXOG_COLS]
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 技术指标
# ---------------------------------------------------------------------------
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0).rolling(period).mean()
    loss = (-delta.clip(upper=0.0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    return ma, upper, lower


def _roc(series: pd.Series, n: int) -> pd.Series:
    return series.pct_change(periods=n)


def _parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    ln_hl = np.log(high / low)
    return np.sqrt((ln_hl ** 2).rolling(window).mean() / (4 * np.log(2)))


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff().fillna(0))
    return (sign * volume).cumsum()


def _mfi(high, low, close, volume, period: int = 14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    mfr = pos_mf / (neg_mf + 1e-9)
    return 100 - 100 / (1 + mfr)


# ---------------------------------------------------------------------------
# 外生变量列清单（严格与 model/kronos_ext.py 的 n_exog 对齐）
# ---------------------------------------------------------------------------
EXOG_COLS: list[str] = [
    # 收益/对数收益
    "log_ret_1", "log_ret_5", "log_ret_20",
    # 动量
    "rsi_14", "macd_hist", "roc_5", "roc_20",
    # 波动
    "atr_14", "parkinson_20", "vol_std_20",
    # 均线偏离
    "ma5_dev", "ma20_dev", "ma60_dev",
    # 布林带
    "boll_z",
    # 量价
    "obv_z", "mfi_14", "amount_z", "vwap_dev",
    # 微结构
    "amplitude", "upper_shadow", "lower_shadow", "body_ratio",
    # 换手率
    "turnover", "turnover_z",
    # 日历
    "is_quarter_end", "days_to_holiday",
    # 相对指数
    "excess_ret_index", "index_ret",
    # 占位预留（便于后续扩展时保持 embedding 维度不变）
    "pad_0", "pad_1", "pad_2", "pad_3",
]


def _rolling_z(s: pd.Series, window: int = 60) -> pd.Series:
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / (sd + 1e-9)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def build_features(
    df: pd.DataFrame,
    index_df: Optional[pd.DataFrame] = None,
    ffill_limit: int = 2,
) -> pd.DataFrame:
    """给单只股票的 OHLCV(+turnover) DataFrame 加因子。

    Parameters
    ----------
    df : 必含 ``open, high, low, close, volume, amount`` 列，且已按 datetime 升序。
         如果有 ``turnover`` 列（换手率 %）会被使用。索引或列里必须能拿到 datetime。
    index_df : 可选。同频率的指数 K 线（如 CSI300），用来计算超额收益。
               需要至少 ``datetime, close`` 两列。
    ffill_limit : 技术指标 NaN 前向填充的最大步数（日线用 2 够了）。

    Returns
    -------
    与 ``df`` 行对齐的 DataFrame，追加了 ``EXOG_COLS`` 中的所有列。
    """
    df = df.copy()
    if "datetime" not in df.columns:
        df = df.reset_index().rename(columns={"index": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    v = df["volume"].astype(float)
    a = df["amount"].astype(float) if "amount" in df.columns else (c * v)

    # -------- 收益 --------
    df["log_ret_1"] = np.log(c / c.shift(1))
    df["log_ret_5"] = np.log(c / c.shift(5))
    df["log_ret_20"] = np.log(c / c.shift(20))

    # -------- 动量 --------
    df["rsi_14"] = _rsi(c, 14) / 100.0 - 0.5  # 居中到 [-0.5, 0.5]
    _, _, hist = _macd(c)
    df["macd_hist"] = hist / (c.rolling(20).std() + 1e-9)
    df["roc_5"] = _roc(c, 5)
    df["roc_20"] = _roc(c, 20)

    # -------- 波动 --------
    df["atr_14"] = _atr(h, l, c, 14) / (c + 1e-9)
    df["parkinson_20"] = _parkinson_vol(h, l, 20)
    df["vol_std_20"] = df["log_ret_1"].rolling(20).std()

    # -------- 均线偏离 --------
    for n in (5, 20, 60):
        ma = c.rolling(n).mean()
        df[f"ma{n}_dev"] = (c - ma) / (ma + 1e-9)

    # -------- 布林带位置 --------
    ma20, up, lo = _bollinger(c, 20, 2)
    df["boll_z"] = (c - ma20) / (up - lo + 1e-9) * 2  # 约在 [-1, 1]

    # -------- 量价 --------
    df["obv_z"] = _rolling_z(_obv(c, v), 60)
    df["mfi_14"] = _mfi(h, l, c, v, 14) / 100.0 - 0.5
    df["amount_z"] = _rolling_z(np.log1p(a), 60)
    vwap = a / (v + 1e-9)
    df["vwap_dev"] = (c - vwap) / (vwap + 1e-9)

    # -------- 微结构 --------
    df["amplitude"] = (h - l) / (c.shift(1) + 1e-9)
    df["upper_shadow"] = (h - np.maximum(o, c)) / (h - l + 1e-9)
    df["lower_shadow"] = (np.minimum(o, c) - l) / (h - l + 1e-9)
    df["body_ratio"] = (c - o).abs() / (h - l + 1e-9)

    # -------- 换手率 --------
    if "turnover" in df.columns:
        df["turnover"] = df["turnover"].astype(float) / 100.0  # % → 小数
    else:
        df["turnover"] = np.nan
    df["turnover_z"] = _rolling_z(df["turnover"].fillna(0), 60)

    # -------- 日历 --------
    dt = df["datetime"]
    df["is_quarter_end"] = (dt.dt.month.isin([3, 6, 9, 12]) &
                            (dt.dt.day >= 25)).astype(float)
    df["days_to_holiday"] = _days_to_holiday(dt).astype(float)

    # -------- 相对指数 --------
    if index_df is not None and not index_df.empty:
        idx = index_df[["datetime", "close"]].copy()
        idx["datetime"] = pd.to_datetime(idx["datetime"])
        idx = idx.rename(columns={"close": "_idx_close"})
        df = df.merge(idx, on="datetime", how="left")
        df["index_ret"] = np.log(df["_idx_close"] / df["_idx_close"].shift(1))
        df["excess_ret_index"] = df["log_ret_1"] - df["index_ret"]
        df = df.drop(columns=["_idx_close"])
    else:
        df["index_ret"] = 0.0
        df["excess_ret_index"] = df["log_ret_1"]

    # -------- 占位 --------
    for p in ("pad_0", "pad_1", "pad_2", "pad_3"):
        df[p] = 0.0

    # 填充 & 截断
    df[EXOG_COLS] = (
        df[EXOG_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .ffill(limit=ffill_limit)
        .fillna(0.0)
        .clip(-5, 5)
    )

    assert len(EXOG_COLS) == 32, (
        f"EXOG_COLS 数量变了（{len(EXOG_COLS)}），"
        "记得同步修改 model/kronos_ext.py 的 n_exog"
    )
    return df


# ---------------------------------------------------------------------------
# 节假日距离（粗糙但够用）
# ---------------------------------------------------------------------------
_CN_HOLIDAYS_APPROX = [
    # (月, 日) 粗估，精确版请换 chinese_calendar 库
    (1, 1), (5, 1), (10, 1), (10, 2), (10, 3),
]


def _days_to_holiday(dt: pd.Series) -> pd.Series:
    """返回到下一个固定假期（元旦/劳动节/国庆）的天数。"""
    out = []
    for d in dt:
        diffs = []
        for m, day in _CN_HOLIDAYS_APPROX:
            try:
                target = d.replace(month=m, day=day)
            except ValueError:
                continue
            if target < d:
                target = target.replace(year=d.year + 1)
            diffs.append((target - d).days)
        out.append(min(diffs) if diffs else 30)
    return pd.Series(out, index=dt.index)


# ---------------------------------------------------------------------------
# CLI 自测
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import pyarrow  # noqa: F401  ensure parquet support
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="某只股票的 parquet")
    parser.add_argument("--index", default=None, help="指数 parquet（可选）")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    idx = pd.read_parquet(args.index) if args.index else None
    out = build_features(df, idx)
    print(out.tail())
    print("NaN 行数:", out[EXOG_COLS].isna().any(axis=1).sum())
    print("Exog 维度:", len(EXOG_COLS))
