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
from kairos.training.config import TrainConfig
from kairos.vendor.kronos import KronosTokenizer


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
) -> Dict:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device_t}")

    print(f"[load] tokenizer: {cfg.pretrained_tokenizer_path}")
    tok = KronosTokenizer.from_pretrained(cfg.pretrained_tokenizer_path).eval().to(device_t)

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

    for si, sym in enumerate(symbols):
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

        for start in range(0, total - window - H + 1):
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
            meta = {"symbol": sym, "date": pd.Timestamp(df["datetime"].iloc[pivot_idx])}
            for h in horizons:
                cf = close_raw[pivot_idx + h]
                meta[f"ret_h{h}"] = float(np.log(cf / c0))

            buf_x.append(x_norm)
            buf_stamp.append(stamp)
            buf_exog.append(ex)
            buf_meta.append(meta)

            if len(buf_x) >= batch_size:
                flush()

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

        # 每日 cross-sectional IC，再求均值
        daily_ic = df_rec.dropna(subset=[score_col, ret_col]).groupby("date").apply(
            lambda g: pd.Series({
                "pearson": pearsonr(g[score_col], g[ret_col]).statistic
                if len(g) > 2 and g[score_col].std() > 0 and g[ret_col].std() > 0 else np.nan,
                "spearman": spearmanr(g[score_col], g[ret_col]).statistic
                if len(g) > 2 else np.nan,
                "n": len(g),
            }), include_groups=False,
        )
        ic_mean = daily_ic["pearson"].mean()
        ric_mean = daily_ic["spearman"].mean()
        ic_std = daily_ic["pearson"].std()
        icir = ic_mean / (ic_std + 1e-9)
        report["by_date_mean"][f"h{h}"] = {
            "ic": float(ic_mean),
            "rank_ic": float(ric_mean),
            "icir": float(icir),
            "n_dates": int(daily_ic["pearson"].notna().sum()),
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
    args = ap.parse_args()

    cfg = TrainConfig()
    horizons = [int(h) for h in args.horizons.split(",")]
    report = run_backtest(
        args.ckpt, cfg,
        horizons=horizons,
        batch_size=args.batch_size,
        max_symbols=args.max_symbols,
        use_baseline=args.baseline,
    )

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[save] {out_p}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
