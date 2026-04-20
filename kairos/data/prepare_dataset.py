"""把 ``collect_ashare_kline.py`` 产出的 per-symbol parquet
转换成 Kronos finetune 期望的 pkl 格式（与 ``finetune/dataset.py`` 对齐）。

产出目录结构::

    <out>/
        train_data.pkl
        val_data.pkl
        test_data.pkl
        exog_train.pkl   (可选，用于 KronosWithExogenous)
        exog_val.pkl
        exog_test.pkl

原版 ``QlibDataset`` 读取的 ``feature_list`` 默认是 ``['open','high','low','close','vol','amt']``，
这里我们对齐这一顺序，并额外导出外生 DataFrame 以便旁路通道使用。

用法
----
    python data_pipeline/prepare_kronos_dataset.py \
        --raw ./raw/daily \
        --index 000300 \
        --raw-index ./raw/index_daily/000300.parquet \
        --train 2018-01-01:2023-12-31 \
        --val   2024-01-01:2024-12-31 \
        --test  2025-01-01:2026-04-17 \
        --out   ./finetune/data/processed_datasets
"""
from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from kairos.data.features import build_features, exog_cols_for
from kairos.data.markets import sanitize_symbol


KRONOS_FEATURES = ["open", "high", "low", "close", "vol", "amt"]


def parse_range(s: str) -> tuple[str, str]:
    a, b = s.split(":")
    return a.strip(), b.strip()


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    return df.loc[mask].reset_index(drop=True)


def _interleave_trainval(
    df: pd.DataFrame,
    val_ratio: float,
    block_days: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Block-level interleaved train/val split inside the train period.

    按 ``block_days`` 将时序切成连续块，再**整块**抽取 ``val_ratio`` 做 val。
    这样既打破"val=单一年份"导致的风格偏移，又避免逐日随机导致 train/val
    高度相关（泄漏）。

    返回 (train_df, val_df)，均保留 datetime 索引。
    """
    if df.empty:
        return df, df
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    n_blocks = max(1, (n + block_days - 1) // block_days)
    block_ids = np.minimum(np.arange(n) // block_days, n_blocks - 1)
    all_blocks = np.arange(n_blocks)
    n_val = max(1, int(round(n_blocks * val_ratio)))
    val_blocks = set(rng.choice(all_blocks, size=n_val, replace=False).tolist())

    is_val = np.array([b in val_blocks for b in block_ids])
    return df.loc[~is_val].reset_index(drop=True), df.loc[is_val].reset_index(drop=True)


def process_symbol(
    path: Path,
    index_df: Optional[pd.DataFrame],
    train_range: tuple[str, str],
    val_range: tuple[str, str],
    test_range: tuple[str, str],
    min_len: int = 200,
    split_mode: str = "time",
    interleave_val_ratio: float = 0.15,
    interleave_block_days: int = 20,
    rng: Optional[np.random.Generator] = None,
    market: str = "ashare",
    exog_cols: Optional[list[str]] = None,
    extras: Optional[dict] = None,
):
    """Build per-symbol train/val/test frames.

    split_mode
    ----------
    - ``"time"``: 传统切法。train/val/test 各自是 ``[train_range]``、``[val_range]``、
      ``[test_range]``. val 集中在一个时段，容易出现风格偏移。
    - ``"interleave"``: **推荐**。把 ``[train_range ∪ val_range]`` 合并后再按
      ``block_days``（默认 20 日）切块，随机抽 ``val_ratio`` 的 block 做 val，
      其余为 train；``[test_range]`` 维持独立。避免 val 单一年份偏差。
    """
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        df = df.reset_index()
    if len(df) < min_len:
        return None

    # 部分数据源不返回 amount（成交额），用 close*volume 作为近似，避免下游 dropna
    # 全表清空。OHLCV 已归一化后 Kronos 的 amt 只用来提供量能尺度，近似是可接受的。
    if "amount" in df.columns and df["amount"].isna().all():
        df["amount"] = df["close"] * df["volume"]

    df = build_features(
        df, index_df, market=market, symbol=path.stem, extras=extras,
    )

    # Kronos 期待列名 vol/amt（对齐 finetune/config.py feature_list）
    df = df.rename(columns={"volume": "vol", "amount": "amt"})

    main_cols = KRONOS_FEATURES
    if exog_cols is None:
        exog_cols = exog_cols_for(market)

    df = df.set_index("datetime")

    if split_mode == "interleave":
        # 合并 train_range + val_range 作为 "fit 区间"，再块级抽 val
        fit_start = min(train_range[0], val_range[0])
        fit_end = max(train_range[1], val_range[1])
        fit_df = _slice(df.reset_index(), fit_start, fit_end)
        test_df = _slice(df.reset_index(), *test_range)
        if rng is None:
            rng = np.random.default_rng(0)
        train_df, val_df = _interleave_trainval(
            fit_df, interleave_val_ratio, interleave_block_days, rng)
        splits = {
            "train": train_df.set_index("datetime") if not train_df.empty else train_df,
            "val":   val_df.set_index("datetime") if not val_df.empty else val_df,
            "test":  test_df.set_index("datetime") if not test_df.empty else test_df,
        }
    else:
        splits = {
            "train": _slice(df.reset_index(), *train_range).set_index("datetime"),
            "val":   _slice(df.reset_index(), *val_range).set_index("datetime"),
            "test":  _slice(df.reset_index(), *test_range).set_index("datetime"),
        }

    out = {}
    for name, part in splits.items():
        if part.empty:
            continue
        if part[main_cols].isna().any().any():
            part = part.dropna(subset=main_cols)
        if part.empty:
            continue
        out[name] = {
            "main": part[main_cols].astype("float32"),
            "exog": part[exog_cols].astype("float32"),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="per-symbol parquet 目录")
    ap.add_argument("--raw-index", default=None,
                    help="用于相对收益的指数 parquet（可选）")
    ap.add_argument("--train", default="2018-01-01:2023-12-31")
    ap.add_argument("--val",   default="2024-01-01:2024-12-31")
    ap.add_argument("--test",  default="2025-01-01:2026-04-17")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-len", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 时只处理前 N 只，用于 smoke test")
    ap.add_argument("--split-mode", choices=["time", "interleave"], default="time",
                    help="'time'=按时间段切（原版）；'interleave'=把 train+val 区间块级"
                         "随机交错，val 不再集中在单一年份")
    ap.add_argument("--val-ratio", type=float, default=0.15,
                    help="interleave 模式下 val block 占比")
    ap.add_argument("--block-days", type=int, default=20,
                    help="interleave 模式下每块的天数（约一个月交易日）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--market", default="ashare",
                    help="which MarketAdapter to use for the feature builder "
                         "(default: ashare; set to 'crypto' for OKX perps, "
                         "etc.). The adapter dictates which 8 market-specific "
                         "columns join the 24 common ones in the exog vector.")
    args = ap.parse_args()

    exog_cols = exog_cols_for(args.market)
    print(f"[market] {args.market}; exog_dim={len(exog_cols)}")

    raw_dir = Path(args.raw).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    index_df = None
    if args.raw_index:
        idx_path = Path(args.raw_index).expanduser()
        if idx_path.exists():
            index_df = pd.read_parquet(idx_path)
            print(f"已加载指数: {idx_path} ({len(index_df)} 行)")
        else:
            print(f"指数文件不存在，跳过: {idx_path}")

    # 过滤 macOS 打包/传输时产生的 ._xxx.parquet 元数据伴生文件
    paths = sorted(
        p for p in raw_dir.glob("*.parquet") if not p.name.startswith("._")
    )
    if args.limit > 0:
        paths = paths[: args.limit]
    print(f"共 {len(paths)} 只股票待处理")

    # Crypto: pick up the sidecar funding/OI/spot parquet that
    # ``kairos-collect --crypto-extras ...`` dropped next to the OHLCV.
    # For every other market this stays an empty dict and changes nothing.
    extras_channels: list[str] = []
    if args.market == "crypto":
        from kairos.data import crypto_extras as _ce

        extras_channels = _ce.available_channels(raw_dir)
        if extras_channels:
            print(f"[extras] discovered channels under {raw_dir}: {extras_channels}")
        else:
            print(
                f"[extras] no _extras/ sidecar found under {raw_dir}; "
                "funding/OI/basis columns will fall back to zero."
            )

    train_data, val_data, test_data = {}, {}, {}
    exog_train, exog_val, exog_test = {}, {}, {}

    master_rng = np.random.default_rng(args.seed)
    print(f"[split] mode={args.split_mode}", end="")
    if args.split_mode == "interleave":
        print(f" val_ratio={args.val_ratio} block_days={args.block_days}", end="")
    print()

    for p in tqdm(paths, ncols=100):
        sym = p.stem
        # 每只股票用独立的子 rng，保证可复现且相互独立
        sub_rng = np.random.default_rng(master_rng.integers(0, 2**31 - 1))
        sym_extras: Optional[dict] = None
        if extras_channels:
            from kairos.data import crypto_extras as _ce

            # ``p.stem`` is already the sanitized-symbol form that
            # ``kairos-collect`` wrote both for OHLCV and sidecars.
            sym_extras = _ce.load_for_symbol(
                raw_dir, p.stem, kinds=extras_channels
            )
        try:
            pieces = process_symbol(
                p, index_df,
                parse_range(args.train),
                parse_range(args.val),
                parse_range(args.test),
                min_len=args.min_len,
                split_mode=args.split_mode,
                interleave_val_ratio=args.val_ratio,
                interleave_block_days=args.block_days,
                rng=sub_rng,
                market=args.market,
                exog_cols=exog_cols,
                extras=sym_extras,
            )
        except Exception as e:
            print(f"[{sym}] 失败: {e}")
            continue
        if not pieces:
            continue
        if "train" in pieces:
            train_data[sym] = pieces["train"]["main"]
            exog_train[sym] = pieces["train"]["exog"]
        if "val" in pieces:
            val_data[sym] = pieces["val"]["main"]
            exog_val[sym] = pieces["val"]["exog"]
        if "test" in pieces:
            test_data[sym] = pieces["test"]["main"]
            exog_test[sym] = pieces["test"]["exog"]

    def _dump(obj, name):
        with open(out_dir / name, "wb") as f:
            pickle.dump(obj, f)
        print(f"已保存 {name}: {len(obj)} symbols")

    _dump(train_data, "train_data.pkl")
    _dump(val_data, "val_data.pkl")
    _dump(test_data, "test_data.pkl")
    _dump(exog_train, "exog_train.pkl")
    _dump(exog_val, "exog_val.pkl")
    _dump(exog_test, "exog_test.pkl")

    # Lightweight dataset manifest so downstream consumers (training,
    # backtest) can recover what market/exog schema produced this bundle.
    import json

    meta = {
        "market": args.market,
        "exog_cols": exog_cols,
        "split_mode": args.split_mode,
        "ranges": {
            "train": args.train,
            "val": args.val,
            "test": args.test,
        },
        "extras_channels": extras_channels,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"完成: {out_dir}")


if __name__ == "__main__":
    main()
