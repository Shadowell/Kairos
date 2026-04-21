"""IC backtest for the fine-tuned Kronos-with-exogenous predictor.

流程
----
1. 加载 fine-tuned ``best_model`` (KronosWithExogenous) + Kronos-Tokenizer.
2. 遍历 ``test_data.pkl`` / ``exog_test.pkl`` 中每只股票的每一个滑动窗口.
3. 模型 forward 一次，取 ``quantiles[:, -1, :, mid_q]`` 作为"最后一个 token 对未来 5 步
   差分收益的中位预测"作为 score.
4. 真值用窗口结尾对应的 *原始 close* 的未来 H 步 log-return.
5. 按日期聚合，计算 cross-sectional IC (Pearson) / rank-IC (Spearman).

用法::

    python -m kairos.training.backtest_ic \
        --ckpt artifacts/checkpoints/predictor/checkpoints/best_model \
        --out  artifacts/backtest_report.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from kairos.models import KronosWithExogenous
from kairos.training.config import TrainConfig, preset_for
from kairos.vendor.kronos import KronosTokenizer


def _load_dataset_meta(dataset_path: str | Path) -> dict:
    """Load ``meta.json`` produced by ``kairos-prepare``.

    Returns an empty dict if no manifest is present (e.g. for legacy
    A-share bundles created before the multi-market refactor), which keeps
    the backtest backward-compatible.
    """
    path = Path(dataset_path) / "meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


_BUCKET_ALIASES = {
    "date": "date",
    "day": "date",
    "daily": "date",
    "hour": "hour",
    "hourly": "hour",
    "minute": "minute",
    "minutely": "minute",
    "none": "none",
    "pool": "none",
}


def _bucket_label(date: pd.Timestamp, bucket: str) -> pd.Timestamp | str:
    if bucket == "date":
        return date.normalize()
    if bucket == "hour":
        return date.floor("h")
    if bucket == "minute":
        return date.floor("min")
    return "pool"


def _build_window_features(
    df: pd.DataFrame, feature_list: List[str], time_feature_list: List[str]
) -> pd.DataFrame:
    df = df.reset_index()
    if "datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["minute"] = df["datetime"].dt.minute
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.weekday
    df["day"] = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    return df


@torch.no_grad()
def run_backtest(
    ckpt_path: str | None,
    cfg: TrainConfig,
    horizons: List[int] = (1, 5),
    batch_size: int = 64,
    max_symbols: int | None = None,
    device: str | None = None,
    use_baseline: bool = False,
    aggregation: str = "auto",
    stride: int = 1,
    per_symbol_limit: int | None = None,
    tokenizer_path: str | None = None,
) -> Dict:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device_t}")

    # Pick an aggregation bucket. "auto" picks the bucket that keeps the
    # cross-section non-trivial: daily bars on A-share → by calendar date;
    # 1-minute crypto with a handful of symbols → also by calendar date
    # (aggregating by minute would leave 1-2 rows per bucket). Advanced
    # users can override this from the CLI.
    bucket_key = _BUCKET_ALIASES.get(aggregation.lower(), aggregation.lower())
    if bucket_key == "auto":
        if cfg.freq and cfg.freq.lower() in {"1min", "3min", "5min", "15min"}:
            bucket_key = "date"  # intraday bars → daily cross-section
        else:
            bucket_key = "date"
    if bucket_key not in {"date", "hour", "minute", "none"}:
        raise ValueError(
            f"Unknown aggregation bucket {aggregation!r}; "
            "use one of: auto, date, hour, minute, none"
        )
    print(f"[bucket] {bucket_key} (market={cfg.market}, freq={cfg.freq})")

    tok_src = tokenizer_path
    if tok_src is None:
        tok_local = Path(cfg.save_path) / cfg.tokenizer_save_folder_name / "checkpoints" / "best_model"
        tok_src = str(tok_local) if tok_local.exists() else cfg.pretrained_tokenizer_path
    print(f"[load] tokenizer: {tok_src}")
    tok = KronosTokenizer.from_pretrained(tok_src).eval().to(device_t)

    if use_baseline or ckpt_path is None:
        print(f"[load] baseline Kronos-small + random exog/return heads")
        model = KronosWithExogenous.from_kronos_pretrained(
            cfg.pretrained_predictor_path,
            n_exog=cfg.n_exog,
            use_return_head=cfg.use_return_head,
            return_horizon=cfg.return_horizon,
            n_quantiles=cfg.n_quantiles,
        ).eval().to(device_t)
    else:
        print(f"[load] model ckpt: {ckpt_path}")
        model = KronosWithExogenous.from_pretrained(ckpt_path).eval().to(device_t)

    lookback = cfg.lookback_window
    n_quantiles = cfg.n_quantiles
    mid_q = n_quantiles // 2

    root = Path(cfg.dataset_path)
    with open(root / "test_data.pkl", "rb") as f:
        test_data: Dict[str, pd.DataFrame] = pickle.load(f)
    with open(root / "exog_test.pkl", "rb") as f:
        test_exog: Dict[str, pd.DataFrame] = pickle.load(f)

    symbols = list(test_data.keys())
    if max_symbols:
        symbols = symbols[:max_symbols]
    print(f"[data] {len(symbols)} symbols")

    close_idx = cfg.feature_list.index("close")

    records: List[Dict] = []

    buf_x, buf_stamp, buf_exog, buf_meta = [], [], [], []

    def flush():
        if not buf_x:
            return
        x = torch.from_numpy(np.stack(buf_x)).to(device_t)
        stamp = torch.from_numpy(np.stack(buf_stamp)).to(device_t)
        exog = torch.from_numpy(np.stack(buf_exog)).to(device_t)

        s1_ids, s2_ids = tok.encode(x, half=True)
        _, _, q_pred = model(
            s1_ids[:, :-1], s2_ids[:, :-1],
            stamp=stamp[:, :-1], exog=exog[:, :-1],
        )
        # q_pred: [B, T-1, horizon, n_quantiles]  —— 取最后时刻的中位数
        med = q_pred[:, -1, :, mid_q].cpu().numpy()  # [B, horizon]

        for meta, score_vec in zip(buf_meta, med):
            r = dict(meta)
            for hi, s in enumerate(score_vec):
                r[f"score_h{hi + 1}"] = float(s)
            records.append(r)

        buf_x.clear(); buf_stamp.clear(); buf_exog.clear(); buf_meta.clear()

    flushes = 0

    for si, sym in enumerate(symbols):
        print(f"[sym {si + 1}/{len(symbols)}] {sym} (records so far: {len(records)})")
        df = _build_window_features(test_data[sym], cfg.feature_list, cfg.time_feature_list)
        edf = test_exog.get(sym)
        if edf is None:
            continue
        if "datetime" in edf.columns:
            ei = edf.set_index("datetime")
        else:
            ei = edf

        total = len(df)
        H = max(horizons)
        window = lookback + 1  # 用 lookback+1 个 bar，最后一根用来预测未来 (模型 forward 是 T-1)

        step = max(1, int(stride))
        starts = list(range(0, total - window - H + 1, step))
        if per_symbol_limit and per_symbol_limit > 0:
            # Take an evenly spaced slice so we still cover the whole test
            # window; useful for CPU smoke checks without biasing to the
            # start/end of the series.
            if len(starts) > per_symbol_limit:
                idx = np.linspace(0, len(starts) - 1, per_symbol_limit).astype(int)
                starts = [starts[i] for i in idx]

        for start in starts:
            end = start + window
            win = df.iloc[start:end]

            x = win[cfg.feature_list].values.astype(np.float32)
            past = x[:lookback]
            mu, sd = past.mean(0), past.std(0)
            x_norm = np.clip((x - mu) / (sd + 1e-5), -cfg.clip, cfg.clip)

            stamp = win[cfg.time_feature_list].values.astype(np.float32)

            dates = pd.to_datetime(win["datetime"].values)
            ex = ei.reindex(dates).fillna(0.0).values.astype(np.float32)
            if ex.shape[1] != cfg.n_exog:
                t = cfg.n_exog
                if ex.shape[1] < t:
                    pad = np.zeros((ex.shape[0], t - ex.shape[1]), dtype=np.float32)
                    ex = np.concatenate([ex, pad], axis=1)
                else:
                    ex = ex[:, :t]

            # True future log-returns
            close_raw = df["close"].values
            pivot_idx = end - 1
            c0 = close_raw[pivot_idx]
            pivot_dt = pd.Timestamp(df["datetime"].iloc[pivot_idx])
            meta = {
                "symbol": sym,
                "date": pivot_dt,
                "bucket": _bucket_label(pivot_dt, bucket_key),
            }
            for h in horizons:
                cf = close_raw[pivot_idx + h]
                meta[f"ret_h{h}"] = float(np.log(cf / c0))

            buf_x.append(x_norm)
            buf_stamp.append(stamp)
            buf_exog.append(ex)
            buf_meta.append(meta)

            if len(buf_x) >= batch_size:
                flush()
                flushes += 1
                if flushes % 50 == 0:
                    print(f"  [batch] {flushes} flushes, {len(records)} records")

        if (si + 1) % 30 == 0:
            print(f"[progress] {si + 1}/{len(symbols)} symbols, "
                  f"records so far: {len(records)}")

    flush()
    df_rec = pd.DataFrame.from_records(records)
    print(f"[done] total records: {len(df_rec)}")

    # --- Compute ICs ---
    report: Dict = {
        "n_records": int(len(df_rec)),
        "n_symbols": int(df_rec["symbol"].nunique()),
        "date_range": [str(df_rec["date"].min()), str(df_rec["date"].max())],
        "overall": {},
        "by_date_mean": {},
    }

    for h in horizons:
        score_col = f"score_h{h}"
        ret_col = f"ret_h{h}"
        sub = df_rec[[score_col, ret_col]].dropna()
        if len(sub) < 2:
            continue

        # 全体 pool IC
        p_all = pearsonr(sub[score_col], sub[ret_col])
        s_all = spearmanr(sub[score_col], sub[ret_col])
        report["overall"][f"h{h}"] = {
            "pearson": float(p_all.statistic),
            "pearson_p": float(p_all.pvalue),
            "spearman": float(s_all.statistic),
            "spearman_p": float(s_all.pvalue),
            "n": int(len(sub)),
            "hit_rate": float(((sub[score_col] > 0) == (sub[ret_col] > 0)).mean()),
        }

        # Cross-sectional IC per bucket (date / hour / minute / pool),
        # then averaged — this is the number that matters in production
        # because it reflects ability to *rank* names at one point in time.
        bucket_ic = df_rec.dropna(subset=[score_col, ret_col]).groupby("bucket").apply(
            lambda g: pd.Series({
                "pearson": pearsonr(g[score_col], g[ret_col]).statistic
                if len(g) > 2 and g[score_col].std() > 0 and g[ret_col].std() > 0 else np.nan,
                "spearman": spearmanr(g[score_col], g[ret_col]).statistic
                if len(g) > 2 else np.nan,
                "n": len(g),
            }), include_groups=False,
        )
        ic_mean = bucket_ic["pearson"].mean()
        ric_mean = bucket_ic["spearman"].mean()
        ic_std = bucket_ic["pearson"].std()
        icir = ic_mean / (ic_std + 1e-9)
        report["by_date_mean"][f"h{h}"] = {
            "ic": float(ic_mean),
            "rank_ic": float(ric_mean),
            "icir": float(icir),
            "n_dates": int(bucket_ic["pearson"].notna().sum()),
            "bucket": bucket_key,
        }

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=False, default=None,
                    help="path to best_model directory; omit to use Kronos-small baseline")
    ap.add_argument("--baseline", action="store_true",
                    help="use KronosWithExogenous.from_kronos_pretrained (random heads) baseline")
    ap.add_argument("--out", default="artifacts/backtest_report.json")
    ap.add_argument("--horizons", default="1,5", help="comma-separated list")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-symbols", type=int, default=None)
    ap.add_argument("--market", default=None,
                    help="override market preset (ashare/crypto/...); "
                         "default: read from dataset meta.json or fall back "
                         "to the TrainConfig default (ashare).")
    ap.add_argument("--dataset-path", default=None,
                    help="override TrainConfig.dataset_path when pointing "
                         "the backtest at a specific prepared bundle")
    ap.add_argument("--preset", default=None,
                    help="named preset from kairos.training.config.preset_for "
                         "(e.g. 'crypto-1min'); overrides --market/freq")
    ap.add_argument("--aggregation", default="auto",
                    help="cross-sectional bucket: auto / date / hour / minute / none")
    ap.add_argument("--stride", type=int, default=1,
                    help="只对每 N 根 bar 取一个窗口用于评估（CPU smoke 友好；"
                         "GPU 全量回测保持默认 1）")
    ap.add_argument("--per-symbol-limit", type=int, default=0,
                    help=">0 时每个 symbol 最多评估 N 个窗口（等距抽样覆盖全区间）")
    ap.add_argument("--tokenizer", default=None,
                    help="override tokenizer checkpoint / repo；默认优先本地 artifacts/checkpoints/tokenizer/checkpoints/best_model")
    ap.add_argument("--predictor", default=None,
                    help="override baseline predictor source repo / path；默认使用 cfg.pretrained_predictor_path")
    args = ap.parse_args()

    overrides: dict = {}
    if args.preset:
        overrides.update(preset_for(args.preset))
    if args.dataset_path:
        overrides["dataset_path"] = args.dataset_path
    if args.market:
        overrides["market"] = args.market
    if args.predictor:
        overrides["pretrained_predictor_path"] = args.predictor

    cfg = TrainConfig(**overrides) if overrides else TrainConfig()

    # Auto-hydrate market/freq from the dataset manifest when the user
    # didn't override them. This keeps backtest_ic usable as
    #     python -m kairos.training.backtest_ic --ckpt ...
    # regardless of which market produced the bundle.
    meta = _load_dataset_meta(cfg.dataset_path)
    if meta:
        if "market" in meta and "market" not in overrides:
            cfg.market = meta["market"]
        if "freq" in meta and "freq" not in overrides:
            cfg.freq = meta["freq"]

    horizons = [int(h) for h in args.horizons.split(",")]
    report = run_backtest(
        args.ckpt, cfg,
        horizons=horizons,
        batch_size=args.batch_size,
        max_symbols=args.max_symbols,
        use_baseline=args.baseline,
        aggregation=args.aggregation,
        stride=args.stride,
        per_symbol_limit=args.per_symbol_limit or None,
        tokenizer_path=args.tokenizer,
    )

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {out_p}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
