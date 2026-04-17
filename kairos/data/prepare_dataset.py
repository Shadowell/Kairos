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

import pandas as pd
from tqdm import tqdm

from kairos.data.features import EXOG_COLS, build_features


KRONOS_FEATURES = ["open", "high", "low", "close", "vol", "amt"]


def parse_range(s: str) -> tuple[str, str]:
    a, b = s.split(":")
    return a.strip(), b.strip()


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    return df.loc[mask].reset_index(drop=True)


def process_symbol(
    path: Path,
    index_df: Optional[pd.DataFrame],
    train_range: tuple[str, str],
    val_range: tuple[str, str],
    test_range: tuple[str, str],
    min_len: int = 200,
):
    df = pd.read_parquet(path)
    if "datetime" not in df.columns:
        df = df.reset_index()
    if len(df) < min_len:
        return None

    df = build_features(df, index_df)

    # Kronos 期待列名 vol/amt（对齐 finetune/config.py feature_list）
    df = df.rename(columns={"volume": "vol", "amount": "amt"})

    main_cols = KRONOS_FEATURES
    exog_cols = EXOG_COLS

    # 统一索引为 datetime
    df = df.set_index("datetime")

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
            # 丢弃含 NaN 的样本（通常出现在窗口冷启动）
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
    args = ap.parse_args()

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

    paths = sorted(raw_dir.glob("*.parquet"))
    if args.limit > 0:
        paths = paths[: args.limit]
    print(f"共 {len(paths)} 只股票待处理")

    train_data, val_data, test_data = {}, {}, {}
    exog_train, exog_val, exog_test = {}, {}, {}

    for p in tqdm(paths, ncols=100):
        sym = p.stem
        try:
            pieces = process_symbol(
                p, index_df,
                parse_range(args.train),
                parse_range(args.val),
                parse_range(args.test),
                min_len=args.min_len,
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
    print(f"完成: {out_dir}")


if __name__ == "__main__":
    main()
